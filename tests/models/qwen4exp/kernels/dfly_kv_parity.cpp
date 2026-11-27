// tests/models/qwen4exp/kernels/dfly_kv_parity.cpp - dfly_kv_gather against a CPU copy through a permuted page table: fp16
// rows and int8 codes with their scales, byte for byte, for 0, 1 and the widest count of cells; a cell's bytes do not
// depend on its place in the list. Prints the time of a call. GPU, synthetic, no model.
#include "basalt/models/qwen4exp/kernels/dfly_kv.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty()) ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// one K or V array of a pool: `row` bytes per (page, KV head, cell of the page)
struct Arr {
    std::vector<uint8_t> src, want, got;
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    size_t row = 0;
};

constexpr int kRows = 3;   // selection rows the id table holds; the block reads one of them

bool run_case(bool q8, int n, int64_t cells, std::mt19937& rng, bool bench) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t ps = s.page_size, nkv = s.n_head_kv, pages = (cells + ps - 1) / ps, dst_pages = (cap + ps - 1) / ps;
    const size_t nsc = (size_t) (s.head_dim / k::KV_Q8_GROUP);
    // the arrays of the form: K and V rows, and for int8 their scales
    std::vector<Arr> arrs(q8 ? 4 : 2);
    arrs[0].row = arrs[1].row = (size_t) s.head_dim * (q8 ? 1 : 2);
    if (q8) arrs[2].row = arrs[3].row = nsc * 2;
    std::vector<int32_t> table((size_t) pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    for (Arr& a : arrs) {
        a.src.resize((size_t) (pages * nkv * ps) * a.row);
        for (uint8_t& b : a.src) b = (uint8_t) rng();
        a.want.assign((size_t) (dst_pages * nkv * ps) * a.row, 0xA5);
        a.got = a.want;
        a.d_src = to_device(a.src);
        a.d_dst = to_device(a.want);
    }
    // each selection row: distinct cells in ascending order, as the target's top-k leaves them
    std::vector<int32_t> ids((size_t) (kRows * cap), -1);
    std::vector<int32_t> all((size_t) cells);
    std::iota(all.begin(), all.end(), 0);
    for (int r = 0; r < kRows; ++r) {
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + n);
        std::copy(all.begin(), all.begin() + n, ids.begin() + (size_t) (r * cap));
    }
    const int row = 1;
    const std::vector<int32_t> blk = {7, row, n};
    // the CPU copy: cell ids[row][j] through the permuted table into cell j under the identity table
    auto phys = [&](int64_t page, int64_t cell, int64_t h) { return (size_t) ((page * nkv + h) * ps + cell % ps); };
    for (Arr& a : arrs)
        for (int j = 0; j < n; ++j) {
            const int64_t cell = ids[(size_t) (row * cap + j)];
            for (int64_t h = 0; h < nkv; ++h)
                std::memcpy(a.want.data() + phys(j / ps, j, h) * a.row, a.src.data() + phys(table[(size_t) (cell / ps)], cell, h) * a.row,
                            a.row);
        }
    k::QsaAttnPools pools;
    k::DflyKvDst dst;
    if (q8) {
        pools.k_q = (const int8_t*) arrs[0].d_src; pools.v_q = (const int8_t*) arrs[1].d_src;
        pools.k_scale = (const uint16_t*) arrs[2].d_src; pools.v_scale = (const uint16_t*) arrs[3].d_src;
        dst.k_q = (int8_t*) arrs[0].d_dst; dst.v_q = (int8_t*) arrs[1].d_dst;
        dst.k_scale = (uint16_t*) arrs[2].d_dst; dst.v_scale = (uint16_t*) arrs[3].d_dst;
    } else {
        pools.k_pool = (const uint16_t*) arrs[0].d_src; pools.v_pool = (const uint16_t*) arrs[1].d_src;
        dst.k_pool = (uint16_t*) arrs[0].d_dst; dst.v_pool = (uint16_t*) arrs[1].d_dst;
    }
    int32_t* d_table = to_device(table);
    pools.page_table = d_table;
    const std::vector<k::QsaAttnPools> pv = {pools};
    k::QsaAttnPools* d_pools = to_device(pv);
    int32_t* d_ids = to_device(ids);
    int32_t* d_blk = to_device(blk);
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs));
    k::dfly_kv_gather(d_pools, d_ids, cap, d_blk, cap, s, dst, cs);
    ck(cudaStreamSynchronize(cs));
    bool ok = true;
    for (size_t i = 0; i < arrs.size(); ++i) {
        ck(cudaMemcpy(arrs[i].got.data(), arrs[i].d_dst, arrs[i].got.size(), cudaMemcpyDeviceToHost));
        // equal bytes: the copied cells and the untouched rest
        if (arrs[i].got != arrs[i].want) {
            std::fprintf(stderr, "dfly_kv_parity: FAILED %s array %zu, %d cells\n", q8 ? "int8" : "fp16", i, n);
            ok = false;
        }
    }
    if (bench) {
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) k::dfly_kv_gather(d_pools, d_ids, cap, d_blk, cap, s, dst, cs);
        ck(cudaStreamSynchronize(cs));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        size_t bytes = 0;
        for (const Arr& a : arrs) bytes += (size_t) n * (size_t) nkv * a.row;
        std::printf("dfly_kv_parity: %s, %d cells: %.1f us per call, %.2f GB/s\n", q8 ? "int8" : "fp16", n, us,
                    (double) bytes / us / 1e3);
    }
    for (Arr& a : arrs) { cudaFree(a.d_src); cudaFree(a.d_dst); }
    cudaFree(d_table); cudaFree(d_pools); cudaFree(d_ids); cudaFree(d_blk);
    cudaStreamDestroy(cs);
    return ok;
}

// a cell's bytes in the private pool do not depend on where the list names it: the same cell first and last
bool run_place(bool q8) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), ps = s.page_size, nkv = s.n_head_kv;
    const size_t row = (size_t) s.head_dim * (q8 ? 1 : 2), nsc = (size_t) (s.head_dim / k::KV_Q8_GROUP);
    const int64_t cells = 4096, pages = cells / ps, dst_pages = (cap + ps - 1) / ps;
    std::mt19937 rng(99);
    std::vector<uint8_t> kv((size_t) (pages * nkv * ps) * row), sc((size_t) (pages * nkv * ps) * nsc * 2);
    for (uint8_t& b : kv) b = (uint8_t) rng();
    for (uint8_t& b : sc) b = (uint8_t) rng();
    std::vector<int32_t> table((size_t) pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    const int n = 9;
    const int32_t cell = 1234;
    std::vector<int32_t> ids((size_t) (2 * cap), -1);
    for (int j = 0; j < n; ++j) ids[(size_t) j] = cell + j;                 // row 0: the cell first
    for (int j = 0; j < n; ++j) ids[(size_t) (cap + j)] = cell - (n - 1) + j;   // row 1: the cell last
    uint8_t *d_k = to_device(kv), *d_v = to_device(kv), *d_s = to_device(sc);
    int32_t* d_table = to_device(table);
    int32_t* d_ids = to_device(ids);
    k::QsaAttnPools pools;
    pools.page_table = d_table;
    if (q8) { pools.k_q = (const int8_t*) d_k; pools.v_q = (const int8_t*) d_v; pools.k_scale = pools.v_scale = (const uint16_t*) d_s; }
    else { pools.k_pool = (const uint16_t*) d_k; pools.v_pool = (const uint16_t*) d_v; }
    const std::vector<k::QsaAttnPools> pv = {pools};
    k::QsaAttnPools* d_pools = to_device(pv);
    const size_t out_bytes = (size_t) (dst_pages * nkv * ps) * row;
    std::vector<uint8_t> got[2];
    for (int r = 0; r < 2; ++r) {
        const std::vector<uint8_t> zero(out_bytes, 0);
        uint8_t *o_k = to_device(zero), *o_v = to_device(zero), *o_ks = to_device(zero), *o_vs = to_device(zero);
        k::DflyKvDst dst;
        if (q8) { dst.k_q = (int8_t*) o_k; dst.v_q = (int8_t*) o_v; dst.k_scale = (uint16_t*) o_ks; dst.v_scale = (uint16_t*) o_vs; }
        else { dst.k_pool = (uint16_t*) o_k; dst.v_pool = (uint16_t*) o_v; }
        const std::vector<int32_t> blk = {0, r, n};
        int32_t* d_blk = to_device(blk);
        k::dfly_kv_gather(d_pools, d_ids, cap, d_blk, cap, s, dst, nullptr);
        ck(cudaDeviceSynchronize());
        // the cell's K row of head 0 at its place in this list
        const int j = r == 0 ? 0 : n - 1;
        got[r].resize(row);
        ck(cudaMemcpy(got[r].data(), o_k + (size_t) (((j / ps) * nkv) * ps + j % ps) * row, row, cudaMemcpyDeviceToHost));
        cudaFree(o_k); cudaFree(o_v); cudaFree(o_ks); cudaFree(o_vs); cudaFree(d_blk);
    }
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_s); cudaFree(d_table); cudaFree(d_ids); cudaFree(d_pools);
    if (got[0] != got[1]) {
        std::fprintf(stderr, "dfly_kv_parity: FAILED a cell's bytes depend on its place (%s)\n", q8 ? "int8" : "fp16");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("dfly_kv_parity: no CUDA device, skipped\n");
        return 77;
    }
    const k::QsaShapes s = k::qsa_real_shapes();
    const int widest = (int) k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(20);
    bool ok = true;
    for (const bool q8 : {false, true}) {
        for (const int n : {0, 1, widest}) ok = run_case(q8, n, 8192, rng, n == widest) && ok;
        ok = run_place(q8) && ok;
    }
    if (!ok) return 1;
    std::printf("dfly_kv_parity: ok\n");
    return 0;
}
