// src/models/qwen4exp/kernels/dfly_kv.cu - a draft layer's context cells out of its paired layer's K/V cache (dfly_kv.hpp).
#include "basalt/models/qwen4exp/kernels/dfly_kv.hpp"

#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 128;   // a block copies one cell
constexpr int kUnit = 16;       // bytes a thread moves at once

struct GatherArgs {
    int n_head_kv, head_dim, page_size;
    long long stride;
};

// the cell this block copies and its page in the source pools; false past the block's context
__device__ __forceinline__ bool gather_cell(int j, const QsaAttnPools* src, const int32_t* ids, const int32_t* blk,
                                            const GatherArgs& a, QsaAttnPools& p, int& cell, long long& page) {
    if (j >= __ldg(blk + kDflyNctx)) return false;
    p = *src;
    cell = __ldg(ids + (long long) __ldg(blk + kDflyRow) * a.stride + j);
    page = cell >= 0 ? (long long) __ldg(p.page_table + cell / a.page_size) : -1;
    return true;
}

// a pool's row of KV head h of a cell on `page`
__device__ __forceinline__ long long pool_row(long long page, int cell, int h, const GatherArgs& a) {
    return (page * a.n_head_kv + h) * a.page_size + cell % a.page_size;
}

// every KV head's row of `row_bytes`, of K then V, from the cell's rows to cell j's
__device__ __forceinline__ void gather_rows_stage(int j, int t, long long page, int cell, const GatherArgs& a,
                                                  const unsigned char* sk, const unsigned char* sv, unsigned char* dk,
                                                  unsigned char* dv, int row_bytes) {
    const int per = row_bytes / kUnit;        // units of a head's row
    const int units = a.n_head_kv * per;      // of the cell's K rows (and of its V rows)
    for (int i = t; i < 2 * units; i += kThreads) {
        const int u = i % units, h = u / per, o = u % per;
        const bool is_k = i < units;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (page >= 0) {
            const unsigned char* sb = is_k ? sk : sv;
            val = __ldg(reinterpret_cast<const uint4*>(sb + pool_row(page, cell, h, a) * row_bytes + (long long) o * kUnit));
        }
        // the identity table: cell j lives on page j / page_size
        *reinterpret_cast<uint4*>((is_k ? dk : dv) + pool_row(j / a.page_size, j, h, a) * row_bytes + (long long) o * kUnit) = val;
    }
}

__device__ __forceinline__ void gather_f16_stage(int j, int t, const QsaAttnPools* src, const int32_t* ids, const int32_t* blk,
                                                 const GatherArgs& a, uint16_t* dk, uint16_t* dv) {
    QsaAttnPools p;
    int cell;
    long long page;
    if (!gather_cell(j, src, ids, blk, a, p, cell, page)) return;
    gather_rows_stage(j, t, page, cell, a, (const unsigned char*) p.k_pool, (const unsigned char*) p.v_pool, (unsigned char*) dk,
                      (unsigned char*) dv, a.head_dim * 2);
}

__device__ __forceinline__ void gather_q8_stage(int j, int t, const QsaAttnPools* src, const int32_t* ids, const int32_t* blk,
                                                const GatherArgs& a, int8_t* dk, int8_t* dv, uint16_t* dks, uint16_t* dvs) {
    QsaAttnPools p;
    int cell;
    long long page;
    if (!gather_cell(j, src, ids, blk, a, p, cell, page)) return;
    gather_rows_stage(j, t, page, cell, a, (const unsigned char*) p.k_q, (const unsigned char*) p.v_q, (unsigned char*) dk,
                      (unsigned char*) dv, a.head_dim);
    // the codes' scales, one per group of a head's row
    const int nsc = a.head_dim / KV_Q8_GROUP, scales = a.n_head_kv * nsc;
    for (int i = t; i < 2 * scales; i += kThreads) {
        const int u = i % scales, h = u / nsc, o = u % nsc;
        const bool is_k = i < scales;
        const uint16_t val = page >= 0 ? __ldg((is_k ? p.k_scale : p.v_scale) + pool_row(page, cell, h, a) * nsc + o) : (uint16_t) 0;
        (is_k ? dks : dvs)[pool_row(j / a.page_size, j, h, a) * nsc + o] = val;
    }
}

__global__ void __launch_bounds__(kThreads) gather_f16_kernel(const QsaAttnPools* src, const int32_t* ids, const int32_t* blk,
                                                              GatherArgs a, uint16_t* dk, uint16_t* dv) {
    gather_f16_stage((int) blockIdx.x, (int) threadIdx.x, src, ids, blk, a, dk, dv);
}

__global__ void __launch_bounds__(kThreads) gather_q8_kernel(const QsaAttnPools* src, const int32_t* ids, const int32_t* blk,
                                                             GatherArgs a, int8_t* dk, int8_t* dv, uint16_t* dks, uint16_t* dvs) {
    gather_q8_stage((int) blockIdx.x, (int) threadIdx.x, src, ids, blk, a, dk, dv, dks, dvs);
}

}  // namespace

void dfly_kv_gather(const QsaAttnPools* src, const int32_t* ids, int64_t stride, const int32_t* blk, int64_t cap,
                    const QsaShapes& s, const DflyKvDst& dst, void* stream) {
    if (!src || !ids || !blk) throw std::invalid_argument("dfly_kv_gather: a null buffer");
    const bool f16 = dst.k_pool != nullptr && dst.v_pool != nullptr;
    const bool q8 = dst.k_q != nullptr && dst.v_q != nullptr && dst.k_scale != nullptr && dst.v_scale != nullptr;
    if (f16 == q8) throw std::invalid_argument("dfly_kv_gather: the destination is fp16 pools or int8 pools with scales");
    // a head's row is whole 16-byte units in either form, and an int8 row whole scale groups
    if (cap <= 0 || stride < cap || s.n_head_kv <= 0 || s.page_size <= 0 || s.head_dim <= 0 || s.head_dim % KV_Q8_GROUP != 0)
        throw std::invalid_argument("dfly_kv_gather: cap, stride or shapes");
    const GatherArgs a = {(int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, (long long) stride};
    const cudaStream_t cs = (cudaStream_t) stream;
    if (f16) gather_f16_kernel<<<(unsigned) cap, kThreads, 0, cs>>>(src, ids, blk, a, dst.k_pool, dst.v_pool);
    else gather_q8_kernel<<<(unsigned) cap, kThreads, 0, cs>>>(src, ids, blk, a, dst.k_q, dst.v_q, dst.k_scale, dst.v_scale);
}

}  // namespace basalt::kernels
