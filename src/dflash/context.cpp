// src/dflash/context.cpp - the block drafter's context: the feature projection of a target layer's rows, and the rows
// that become K and V in a sequence's rings.
#include "basalt/dflash/core.hpp"

#include "basalt/log.hpp"
#include "products.hpp"

#include <cuda_runtime.h>

namespace basalt::dflash {

kernels::RopeSpec block_rope(const Meta& m) {
    kernels::RopeSpec r;
    r.rot = m.head_dim;
    r.theta = m.theta;
    return r;
}

void tap(const BlockDrafter& bd, const Scratch& sc, const wg::Workspace& ws, size_t j, const uint16_t* rows16, int n, float* acc,
         void* stream) {
    if (j >= bd.fc.size() || !rows16 || !acc || !sc.tmp || n < 1 || n > wg::kRows) {
        basalt::log::error("basalt dflash", "feature projection %zu of %zu over %d rows: not run", j, bd.fc.size(), n);
        return;
    }
    // BF16 rows are the window gemm's activations as they lie
    const Linear& fc = bd.fc[j];
    prod(ws, fc, rows16, j == 0 ? acc : sc.tmp, n, stream);
    if (j > 0) kernels::add_scaled(acc, sc.tmp, nullptr, (int64_t) n * bd.n_embd, stream);
}

bool context_rows(const BlockDrafter& bd, const Scratch& sc, const wg::Workspace& ws, Seq& seq, const float* acc, int n, int64_t pos0,
                  void* stream, std::string& err) {
    if (n < 1 || n > sc.rows || n > wg::kRows || pos0 < 0) {
        err = "block drafter: " + std::to_string(n) + " context rows at position " + std::to_string((long long) pos0);
        return false;
    }
    if (!acc || seq.cap <= 0 || seq.k.size() != bd.layers.size() || seq.v.size() != bd.layers.size()) {
        err = "block drafter: a sequence without its context rings";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int H = bd.n_embd;
    // the rows' positions: the same on every rope axis
    int32_t p3[wg::kRows * 3];
    for (int t = 0; t < n; ++t) p3[t * 3] = p3[t * 3 + 1] = p3[t * 3 + 2] = (int32_t) (pos0 + t);
    cudaMemcpyAsync(sc.bpos, p3, (size_t) n * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const void* xq = norm_act(sc, acc, bd.hidden_norm, H, n, bd.eps, stream);
    const kernels::RopeSpec rope = block_rope(bd);
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        prod_group(sc, ws, sc.xn, xq, D.k.type, {{&D.k, sc.bk}, {&D.v, sc.bv}}, n, stream);
        kernels::attn_qk_prepare(nullptr, sc.bk, sc.bv, nullptr, D.k_norm, bd.eps, rope, sc.bpos, n, 0, bd.n_kv, nullptr, seq.k[l],
                                 seq.v[l], pos0, stream, bd.head_dim, false, seq.cap);
    }
    seq.extend(pos0, n);
    return true;
}

bool copy_context(const BlockDrafter& bd, const Seq& src, Seq& dst, void* stream, std::string& err) {
    if (src.cap != dst.cap || src.cap != ring_cells(bd) || src.k.size() != bd.layers.size() || dst.k.size() != src.k.size() ||
        src.v.size() != src.k.size() || dst.v.size() != src.k.size()) {
        err = "block drafter: two sequences whose context rings differ";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const size_t bytes = ring_bytes(bd);
    for (size_t l = 0; l < src.k.size(); ++l)
        if (cudaMemcpyAsync(dst.k[l], src.k[l], bytes, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(dst.v[l], src.v[l], bytes, cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
            err = std::string("block drafter: copying a context ring: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    dst.take_range(src);
    return true;
}

}  // namespace basalt::dflash
