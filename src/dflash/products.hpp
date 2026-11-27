// src/dflash/products.hpp - the block drafter's products, all on the window gemm: its activations go to the scratch's xq,
// and a matrix's tensor scale is applied by its product.
#pragma once

#include "basalt/dflash/core.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/window_gemm.hpp"

#include <initializer_list>

namespace basalt::dflash {

using model::Linear;
namespace wg = kernels::wgemm;

// rows 0..T-1 of x readied for a product
inline const void* act(const Scratch& sc, const float* x, int64_t cols, int T, void* s) {
    wg::pack(x, cols, cols, T, sc.xq, s);
    return sc.xq;
}
// the RMS norm of x's rows into sc.xn, readied for a product (one kernel); FP32 rows, or BF16 rows (the residual)
inline const void* norm_act(const Scratch& sc, const float* x, const float* w, int64_t cols, int T, float eps, void* s) {
    wg::norm_pack(x, w, eps, cols, T, sc.xq, sc.xn, s);
    return sc.xq;
}
inline const void* norm_act(const Scratch& sc, const uint16_t* x, const float* w, int64_t cols, int T, float eps, void* s) {
    wg::norm_pack(x, w, eps, cols, T, sc.xq, sc.xn, s);
    return sc.xq;
}

// y [T][rows] = scale W x
inline void prod(const wg::Workspace& ws, const Linear& l, const void* xq, float* y, int T, void* s) {
    wg::Call c = model::wcall(l);
    c.m[0] = model::wmat(l, y);
    wg::gemm(c, xq, T, ws, s);
}
// Several matrices of one input (x, readied in xq for `type`): one call when they are all of one format and width.
struct Out {
    const Linear* l;
    float* y;
};
inline void prod_group(const Scratch& sc, const wg::Workspace& ws, const float* x, const void* xq, int type,
                       std::initializer_list<Out> outs, int T, void* s) {
    bool one = outs.size() <= (size_t) wg::kMaxMatrices;
    for (const Out& o : outs) one = one && o.l->type == type && o.l->cols == outs.begin()->l->cols;
    if (one) {
        wg::Call c = model::wcall(*outs.begin()->l);
        c.n = 0;
        for (const Out& o : outs) c.m[c.n++] = model::wmat(*o.l, o.y);
        wg::gemm(c, xq, T, ws, s);
        return;
    }
    for (const Out& o : outs) {
        if (o.l->type != type) {
            type = o.l->type;
            xq = act(sc, x, o.l->cols, T, s);
        }
        prod(ws, *o.l, xq, o.y, T, s);
    }
}
// h [T][rows] = silu(s_g gate x) (s_u up x): one call with the SwiGLU epilogue for NVFP4 planes, else two products and
// the step after them
inline void prod_swiglu(const Scratch& sc, const wg::Workspace& ws, const Linear& gate, const Linear& up, const float* x,
                        const void* xq, float* h, int T, void* s) {
    if (gate.type == model::kTypeNvfp4Planes && up.type == gate.type) {
        wg::Call c = model::wcall(gate, wg::Epilogue::SwiGLU);
        c.n = 2;
        c.m[0] = model::wmat(gate, h);
        c.m[1] = model::wmat(up, nullptr);
        wg::gemm(c, xq, T, ws, s);
        return;
    }
    prod_group(sc, ws, x, xq, gate.type, {{&gate, sc.g}, {&up, sc.u}}, T, s);
    kernels::swiglu_scaled(sc.g, sc.u, nullptr, nullptr, h, T, gate.rows, gate.rows, s);
}

}  // namespace basalt::dflash
