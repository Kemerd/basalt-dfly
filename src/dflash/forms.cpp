// src/dflash/forms.cpp - the block drafter's decode forms, made at load, and what its passes need: the window gemm's
// workspace and the drafter's own scratch.
#include "basalt/dflash/core.hpp"

#include "basalt/core/device_copy.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <initializer_list>

namespace basalt::dflash {

using model::Linear;
namespace wg = kernels::wgemm;

Forms::~Forms() {
    for (void* p : owned) cudaFree(p);
}

// The drafter as decode passes run it, on the window gemm: q, o, gate, up and down as NVFP4 or Q8 planes in place and the
// convolutions repacked to Q8 planes in place; k and v (as the big products) and the feature projection (Q8 planes) get
// copies when a prompt path reads their blocks, and are converted in place otherwise.
bool decode_forms(BlockDrafter& bd, bool fp4, bool keep_blocks, Forms& forms, std::string& err) {
    if (!bd.present) {
        err = "no block drafter to lay out";
        return false;
    }
    // the memory and the new forms now (later sizing sees them); the kernels that make them once the weights have landed
    // (make_forms), in this order, through one scratch buffer of the largest matrix
    std::vector<std::function<bool(void*)>>& steps = forms.steps;
    size_t& tmp_bytes = forms.tmp_bytes;
    auto q8 = [](const Linear& l) { return l.type == 8 && l.cols % 256 == 0 && l.rows > 0; };
    auto bytes = [](const Linear& l) { return (size_t) l.rows * (size_t) l.cols / 32 * 34; };
    auto in_place = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        const size_t n = bytes(l);
        tmp_bytes = std::max(tmp_bytes, n);
        void* w = const_cast<void*>(l.w);
        const int64_t rows = l.rows, cols = l.cols;
        steps.push_back([=](void* tmp) {
            if (basalt::core::device_copy(tmp, w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
            wg::q8_planes(tmp, rows, cols, w, nullptr);
            return cudaDeviceSynchronize() == cudaSuccess;
        });
        l.type = model::kTypeQ8P;
        return true;
    };
    auto copy = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes(l)) != cudaSuccess) return false;
        forms.owned.push_back(p);
        const void* w = l.w;
        const int64_t rows = l.rows, cols = l.cols;
        steps.push_back([=](void*) {
            wg::q8_planes(w, rows, cols, p, nullptr);
            return true;
        });
        l.w = p;
        l.type = model::kTypeQ8P;
        return true;
    };
    // The drafter's big products (gate, up, down, q, o) in NVFP4 planes, rounded from its Q8_0 at load: the drafts' precision
    // only (every draft is verified exactly, so the output does not change); about half the bytes a pass reads.  Without
    // `fp4` they stay Q8 planes.
    auto to_fp4 = [&](Linear& l) -> bool {
        if (!fp4 || !q8(l) || !kernels::nvfp4p::fits(l.rows, l.cols)) return in_place(l);
        const size_t n = bytes(l);
        tmp_bytes = std::max(tmp_bytes, n);
        void* sc = nullptr;
        if (cudaMalloc(&sc, 16) != cudaSuccess) return false;
        forms.owned.push_back(sc);
        void* w = const_cast<void*>(l.w);
        const int64_t rows = l.rows, cols = l.cols;
        steps.push_back([=](void* tmp) {
            if (basalt::core::device_copy(tmp, w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
            kernels::nvfp4p::from_q8(tmp, rows, cols, w, (float*) sc, nullptr);
            return cudaDeviceSynchronize() == cudaSuccess;
        });
        l.scale = (const float*) sc;
        l.type = model::kTypeNvfp4Planes;
        return true;
    };
    bool ok = true;
    for (BlockDrafter::Layer& D : bd.layers) {
        for (Linear* l : {&D.gate, &D.up, &D.down, &D.q, &D.o}) ok = ok && to_fp4(*l);
        for (Linear* l : {&D.conv_a, &D.conv_f}) ok = ok && in_place(*l);
        if (!keep_blocks) {
            // nothing reads the blocks again: k and v as the big products, so q, k and v stay one call
            ok = ok && to_fp4(D.k) && to_fp4(D.v);
        } else if (fp4 && kernels::nvfp4p::fits(D.k.rows, D.k.cols) && kernels::nvfp4p::fits(D.v.rows, D.v.cols)) {
            // k and v as NVFP4 copies (prompt chunks read their blocks), so q, k and v stay one call
            for (Linear* l : {&D.k, &D.v}) {
                void* p = nullptr;
                void* sc = nullptr;
                if (!q8(*l) || cudaMalloc(&p, kernels::nvfp4p::bytes(l->rows, l->cols)) != cudaSuccess) {
                    ok = false;
                    break;
                }
                forms.owned.push_back(p);
                if (cudaMalloc(&sc, 16) != cudaSuccess) {
                    ok = false;
                    break;
                }
                forms.owned.push_back(sc);
                const void* w = l->w;
                const int64_t rows = l->rows, cols = l->cols;
                steps.push_back([=](void*) {
                    kernels::nvfp4p::from_q8(w, rows, cols, p, (float*) sc, nullptr);
                    return true;
                });
                l->w = p;
                l->scale = (const float*) sc;
                l->type = model::kTypeNvfp4Planes;
            }
        } else {
            ok = ok && copy(D.k) && copy(D.v);
        }
    }
    for (Linear& l : bd.fc) ok = ok && (keep_blocks ? copy(l) : in_place(l));
    if (!ok) {
        err = "laying out the block drafter for decode windows failed";
        return false;
    }
    // every product of a pass runs on the window gemm: a matrix left in another form has no path
    for (const BlockDrafter::Layer& D : bd.layers)
        for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) ok = ok && model::on_wg(l->type);
    for (const Linear& l : bd.fc) ok = ok && model::on_wg(l.type);
    if (!ok) {
        err = "the block drafter has a projection the window products do not read (Q8_0 rows in whole 256-value blocks, "
              "NVFP4 planes or F8)";
        return false;
    }
    return true;
}

bool make_forms(Forms& forms, std::string& err) {
    if (forms.steps.empty()) return true;
    void* tmp = nullptr;
    bool ok = forms.tmp_bytes == 0 || cudaMalloc(&tmp, forms.tmp_bytes) == cudaSuccess;
    for (const auto& step : forms.steps) ok = ok && step(tmp);
    if (tmp) cudaFree(tmp);
    forms.steps.clear();
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
        err = "laying out the block drafter for decode windows failed";
        return false;
    }
    return true;
}

// the workspace for the drafter's calls (each matrix, q + k + v, k + v, gate + up, the selector's projection)
void workspace_need(const BlockDrafter& bd, size_t& partial_floats, size_t& tickets) {
    auto raise = [&](const wg::Call& c) {
        size_t p, t;
        wg::workspace_need(c, p, t);
        partial_floats = std::max(partial_floats, p);
        tickets = std::max(tickets, t);
    };
    auto need = [&](std::initializer_list<const Linear*> ls) {
        wg::Call c;
        c.fmt = wg::Format::Q8;
        c.n = 0;
        for (const Linear* l : ls) {
            if (l->type != model::kTypeQ8P && l->type != model::kTypeNvfp4Planes) return;
            c.fmt = l->type == model::kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
            c.K = l->cols;
            c.m[c.n++].rows = l->rows;
        }
        if (ls.size() == 2 && c.fmt == wg::Format::Nvfp4) c.epi = wg::Epilogue::SwiGLU;   // the FFN pair
        raise(c);
    };
    for (const BlockDrafter::Layer& D : bd.layers) {
        for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) need({l});
        need({&D.q, &D.k, &D.v});
        need({&D.k, &D.v});
        need({&D.gate, &D.up});
    }
    for (const Linear& l : bd.fc) need({&l});
    if (bd.rank > 0 && bd.n_embd > 0) {   // the selector's rows (BF16 weights)
        wg::Call c;
        c.fmt = wg::Format::Bf16;
        c.K = bd.n_embd;
        c.m[0].rows = bd.rank;
        raise(c);
    }
}

size_t scratch_floats(const BlockDrafter& bd, int rows) {
    if (!bd.present || rows < 1) return 0;
    const size_t W = (size_t) rows, AQ = (size_t) bd.n_head * bd.head_dim, KV = (size_t) bd.n_kv * bd.head_dim;
    return W * (3 * AQ + 2 * KV + (size_t) conv_rows(bd) + (size_t) bd.n_embd + kernels::kLogprobStride + (size_t) bd.rank + 3) +
           kernels::gqa_decode_scratch_floats(ring_cells(bd), rows, bd.n_head, bd.head_dim) + 64 * 64;
}

bool take_scratch(const BlockDrafter& bd, int rows, const std::function<void*(size_t)>& take, Scratch& sc) {
    if (!take || !bd.present || rows < 1 || rows > wg::kRows) return false;
    const size_t W = (size_t) rows, AQ = (size_t) bd.n_head * bd.head_dim, KV = (size_t) bd.n_kv * bd.head_dim;
    const auto floats = [&](size_t n) { return static_cast<float*>(take(n * sizeof(float))); };
    sc.bq = floats(W * AQ);
    sc.bk = floats(W * KV);
    sc.bv = floats(W * KV);
    sc.bqo = floats(W * AQ);
    sc.bao = floats(W * AQ);
    sc.dyn = floats(W * (size_t) conv_rows(bd));
    sc.bh = floats(W * (size_t) bd.n_embd);
    sc.blp = floats(W * kernels::kLogprobStride);
    sc.bsel = floats(W * (size_t) bd.rank);
    sc.bpos = static_cast<int32_t*>(take(W * 3 * sizeof(int32_t)));
    sc.battn = floats(kernels::gqa_decode_scratch_floats(ring_cells(bd), rows, bd.n_head, bd.head_dim));
    sc.rows = rows;
    return sc.bq && sc.bk && sc.bv && sc.bqo && sc.bao && sc.dyn && sc.bh && sc.blp && sc.bsel && sc.bpos && sc.battn;
}

}  // namespace basalt::dflash
