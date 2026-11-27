// src/dflash/draft.cpp - the block drafter's pass: the blocks' ids, its layers over their embedded rows, and the path
// its selector walks through the head's candidates.
#include "basalt/dflash/core.hpp"

#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "products.hpp"

#include <cuda_runtime.h>

namespace basalt::dflash {
namespace {

// Sequence k's rows of a pass: r0[k] .. r0[k] + widths[k] - 1 (r0[count]: the rows in all).  False with `err` for a pass
// the scratch or a sequence cannot hold.
bool pass_rows(const BlockDrafter& bd, const Scratch& sc, Seq* const* seqs, const int64_t* pos, const int* widths, int count, int* r0,
               std::string& err) {
    if (!bd.present || !seqs || !pos || !widths || count < 1 || count > sc.rows || sc.rows > wg::kRows) {
        err = "block drafter: a pass of " + std::to_string(count) + " blocks";
        return false;
    }
    r0[0] = 0;
    for (int k = 0; k < count; ++k) {
        if (widths[k] < bd.block || widths[k] > sc.rows || pos[k] < 0) {
            err = "block drafter: a block of " + std::to_string(widths[k]) + " rows";
            return false;
        }
        const Seq* q = seqs[k];
        if (!q || q->cap <= 0 || q->k.size() != bd.layers.size() || q->v.size() != bd.layers.size() || !q->anchor || !q->ids || !q->conf) {
            err = "block drafter: a sequence without its buffers";
            return false;
        }
        r0[k + 1] = r0[k] + widths[k];
        if (r0[k + 1] > sc.rows) {
            err = "block drafter: a batch of more than " + std::to_string(sc.rows) + " rows";
            return false;
        }
    }
    return true;
}

}  // namespace

bool block_tokens(const BlockDrafter& bd, const Scratch& sc, Seq* const* seqs, const int32_t* anchors, const int64_t* pos,
                  const int* widths, int count, void* stream, std::string& err) {
    int r0[wg::kRows + 1];
    if (!pass_rows(bd, sc, seqs, pos, widths, count, r0, err)) return false;
    if (!anchors || !sc.tok || !sc.bpos) {
        err = "block drafter: no room for a block's ids";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int R = r0[count];
    // each block: its anchor, then mask tokens, at positions pos ..
    int32_t tok[wg::kRows];
    int32_t p3[wg::kRows * 3];
    for (int k = 0; k < count; ++k) {
        tok[r0[k]] = anchors[k];
        for (int i = 1; i < widths[k]; ++i) tok[r0[k] + i] = bd.mask_id;
        for (int i = 0; i < widths[k]; ++i) {
            int32_t* q = &p3[(size_t) (r0[k] + i) * 3];
            q[0] = q[1] = q[2] = (int32_t) (pos[k] + i);
        }
        cudaMemcpyAsync(seqs[k]->anchor, &tok[r0[k]], sizeof(int32_t), cudaMemcpyHostToDevice, s);
    }
    cudaMemcpyAsync(sc.tok, tok, (size_t) R * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(sc.bpos, p3, (size_t) R * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    return true;
}

bool block_layers(const BlockDrafter& bd, const Scratch& sc, const wg::Workspace& ws, Seq* const* seqs, const int64_t* pos,
                  const int* widths, int count, uint16_t* x, bool pack, void* stream, std::string& err) {
    int r0[wg::kRows + 1];
    if (!pass_rows(bd, sc, seqs, pos, widths, count, r0, err)) return false;
    if (!x) {
        err = "block drafter: no rows to run";
        return false;
    }
    const int S = count, R = r0[count];
    const int H = bd.n_embd, AQ = bd.n_head * bd.head_dim, KV = bd.n_kv * bd.head_dim;
    const int groups = H / bd.conv_group;
    const kernels::RopeSpec rope = block_rope(bd);
    // the dynamic convolution runs along each block's own rows into sc.bh
    const auto conv = [&](const float* in, const float* dyn, int64_t ld_dyn, const float* base) {
        for (int k = 0; k < S; ++k)
            kernels::dyn_conv2(in + (size_t) r0[k] * H, dyn + (size_t) r0[k] * ld_dyn, ld_dyn, base, sc.bh + (size_t) r0[k] * H,
                               widths[k], H, bd.conv_group, stream);
    };
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        // attention, its input and output through the dynamic convolution
        prod(ws, D.conv_a, norm_act(sc, x, D.attn_norm, H, R, bd.eps, stream), sc.dyn, R, stream);
        conv(sc.xn, sc.dyn, D.conv_a.rows, D.base_a);
        prod_group(sc, ws, sc.bh, act(sc, sc.bh, H, R, stream), D.q.type, {{&D.q, sc.bq}, {&D.k, sc.bk}, {&D.v, sc.bv}}, R, stream);
        for (int k = 0; k < S; ++k) {
            Seq& st = *seqs[k];
            const size_t q0 = (size_t) r0[k];
            const int B = widths[k];
            kernels::attn_qk_prepare(sc.bq + q0 * AQ, sc.bk + q0 * KV, sc.bv + q0 * KV, D.q_norm, D.k_norm, bd.eps, rope,
                                     sc.bpos + q0 * 3, B, bd.n_head, bd.n_kv, sc.bqo + q0 * AQ, st.k[l], st.v[l], pos[k], stream,
                                     bd.head_dim, false, st.cap);
            // each row sees the drafter's window behind it, but never below the sequence's first position: a context
            // shorter than the window is seen whole by every row (the rows share the anchor's reach back to it)
            const int64_t window = st.window(bd.window, pos[k]);
            kernels::gqa_window_attn(sc.bqo + q0 * AQ, st.k[l], st.v[l], bd.n_head, bd.n_kv, bd.head_dim, pos[k], B, pos[k] + B,
                                     window, st.cap, sc.battn, st.cap, sc.bao + q0 * AQ, stream, window < bd.window);
        }
        prod(ws, D.o, act(sc, sc.bao, AQ, R, stream), sc.tmp, R, stream);
        conv(sc.tmp, sc.dyn + 2 * groups, D.conv_a.rows, D.base_a + 2 * H);
        kernels::add_scaled(x, sc.bh, nullptr, (int64_t) R * H, stream);
        // the FFN, likewise
        prod(ws, D.conv_f, norm_act(sc, x, D.ffn_norm, H, R, bd.eps, stream), sc.dyn, R, stream);
        conv(sc.xn, sc.dyn, D.conv_f.rows, D.base_f);
        prod_swiglu(sc, ws, D.gate, D.up, sc.bh, act(sc, sc.bh, H, R, stream), sc.h, R, stream);
        prod(ws, D.down, act(sc, sc.h, D.gate.rows, R, stream), sc.tmp, R, stream);
        conv(sc.tmp, sc.dyn + 2 * groups, D.conv_f.rows, D.base_f + 2 * H);
        kernels::add_scaled(x, sc.bh, nullptr, (int64_t) R * H, stream);
    }
    // each block's own rows took cells of its sequence's rings
    for (int k = 0; k < S; ++k) seqs[k]->mark(pos[k], widths[k]);
    // the final norm: the rows the head scores and the selector projects
    if (pack) wg::norm_pack(x, bd.norm, bd.eps, H, R, sc.xq, sc.xn, stream);
    else kernels::rms_norm_rows(x, bd.norm, sc.xn, R, H, bd.eps, stream);
    return true;
}

bool block_select(const BlockDrafter& bd, const Scratch& sc, const wg::Workspace& ws, Seq* const* seqs, const int64_t* pos,
                  const int* widths, int count, const float* logits, int64_t n_score, void* stream, std::string& err) {
    int r0[wg::kRows + 1];
    if (!pass_rows(bd, sc, seqs, pos, widths, count, r0, err)) return false;
    if (!logits || n_score < bd.top_k) {
        err = "block drafter: a head of " + std::to_string((long long) n_score) + " rows to pick from";
        return false;
    }
    const int S = count, R = r0[count];
    const int H = bd.n_embd;
    {   // the selector's projection of every row (BF16 weights) on the window gemm
        wg::Call c;
        c.fmt = wg::Format::Bf16;
        c.K = H;
        c.m[0] = {(const uint8_t*) bd.sel_h, bd.rank, nullptr, sc.bsel, bd.rank};
        wg::pack(sc.xn, H, H, R, sc.xq, stream);
        wg::gemm(c, sc.xq, R, ws, stream);
    }
    // slots 1 .. B-1 of each block: the head's candidates, then one path through them
    for (int k = 0; k < S; ++k) {
        const Seq& st = *seqs[k];
        const size_t q0 = (size_t) r0[k];
        const int B = widths[k];
        kernels::row_logprobs(logits + (q0 + 1) * n_score, B - 1, (int) n_score, sc.tok + q0 + 1, bd.top_k, sc.blp, sc.lp_scratch, stream);
        kernels::CandidateDraw draw;
        draw.temperature = st.temperature;
        draw.seed = st.seed;
        draw.pos0 = pos[k] + 1;
        draw.q_ids = st.q_ids;
        draw.q_p = st.q_p;
        kernels::candidate_path(sc.blp, kernels::kLogprobStride, bd.top_k, sc.bsel + (q0 + 1) * bd.rank, bd.pred, bd.succ, bd.rank,
                                st.anchor, B - 1, st.ids, st.conf, draw, stream);
    }
    return true;
}

}  // namespace basalt::dflash
