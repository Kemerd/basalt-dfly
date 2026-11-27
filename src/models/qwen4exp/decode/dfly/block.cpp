// src/models/qwen4exp/decode/dfly/block.cpp - the DFly2 block pass: one captured graph over the block's rows, bound to the
// last window's rows, and the sequences that queue blocks on it.
#include "basalt/models/qwen4exp/decode/dfly.hpp"

#include "basalt/core/device_copy.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/log.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/native_qsa.hpp"
#include "basalt/models/qwen4exp/kernels/native_rope.hpp"
#include "basalt/models/qwen4exp/kernels/q5k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q6k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q80_proxy.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_attn.hpp"
#include "basalt/models/qwen4exp/kernels/rope_scaling.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <numeric>

namespace basalt::core {

namespace bk = basalt::kernels;

namespace {

constexpr int kQ8_0 = 8;            // ggml's type id of the projections
constexpr int kOut = kDflyBlockMax;   // words of each output row: ids, confidence logits, head probabilities

// where the parts of a block's record are: int32 words, then the paired layers' pools on an 8-byte boundary
struct RecLayout {
    size_t step, pos_q, pos_k;   // int32 offsets
    size_t pools, bytes;         // byte offsets
};
RecLayout layout_of(const DflyDims& d, int rows) {
    RecLayout l;
    l.step = (size_t) bk::kDflyBlkCount + 1;
    l.pos_q = l.step + (size_t) rows * bk::kStepCount;
    l.pos_k = l.pos_q + (size_t) rows * (size_t) d.n_head;
    const size_t ints = l.pos_k + (size_t) rows * (size_t) d.n_head_kv;
    l.pools = (ints * 4 + 7) & ~(size_t) 7;
    l.bytes = l.pools + (size_t) d.layers * sizeof(bk::QsaAttnPools);
    return l;
}

bool ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("dfly: ") + what + ": " + cudaGetErrorString(e);
    cudaGetLastError();
    return false;
}

}  // namespace

uint64_t DflyDrafter::record_bytes(const DflyDims& d, int rows) { return layout_of(d, rows).bytes; }

bk::QsaShapes DflyDrafter::shapes_of(const DflyDims& d) {
    bk::QsaShapes s = bk::qsa_real_shapes();
    s.n_head = d.n_head;
    s.n_head_kv = d.n_head_kv;
    s.head_dim = d.head_dim;
    s.n_rot = d.n_rot;
    return s;
}

bool DflyDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* head_rows, const int32_t* const* selections,
                       int64_t cap, int win_rows, std::string& err) {
    if (!loaded()) { err = "dfly: bind before load"; return false; }
    if (bound() || proxy_ != nullptr) { err = "dfly: the drafter is already bound"; return false; }
    if (head == nullptr || !head->loaded()) { err = "dfly: the drafter needs the native head"; return false; }
    if (!NativeHead::has_draft_top1(head->type())) { err = "dfly: the head's type has no two-stage argmax"; return false; }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr || wo->ne1 != d_.vocab) { err = "dfly: the head's vocabulary is not the drafter's"; return false; }
    const NativeEmbed* embed = native_embed();
    if (embed == nullptr) { err = "dfly: the drafter reads the embedding table in its GGUF form"; return false; }
    const OnDevice on(device_);
    const int64_t n_vocab = d_.vocab;
    // ---- the 2-bit copy of the whole head its drafts are picked with, and the map of its rows
    const size_t pb = bk::q5k_proxy_bytes(n_vocab), sb = bk::q5k_proxy_scratch_bytes(n_vocab);
    if (cudaMalloc(&proxy_, pb) != cudaSuccess || cudaMalloc(&proxy_scratch_, sb) != cudaSuccess ||
        cudaMalloc((void**) &map_, (size_t) n_vocab * 4) != cudaSuccess) {
        cudaGetLastError();
        err = "dfly: the draft head's copy (" + std::to_string((pb + sb) >> 20) + " MiB) does not fit";
        return false;
    }
    std::vector<int32_t> map((size_t) n_vocab);
    std::iota(map.begin(), map.end(), 0);
    if (!ok(cudaMemset(proxy_, 0, pb), "the head copy", err) || !ok(cudaMemset(proxy_scratch_, 0, sb), "the head scratch", err) ||
        !ok(device_copy(map_, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "the head's row map", err))
        return false;
    // each type packs its own rows; the copy's row r is the head's row r
    if (head->type() == 13) bk::q5k_proxy_pack(head->weights(), 0, n_vocab, proxy_, n_vocab, nullptr);
    else if (head->type() == kQ8_0) bk::q80_proxy_pack(head->weights(), 0, n_vocab, proxy_, n_vocab, nullptr);
    else bk::q6k_proxy_pack(head->weights(), 0, n_vocab, proxy_, n_vocab, nullptr);
    if (!ok(cudaDeviceSynchronize(), "packing the head copy", err)) return false;
    vram_ += pb + sb + (uint64_t) n_vocab * 4;
    Target t;
    t.n_vocab = n_vocab;
    t.embed = [embed](const int32_t* tokens, int64_t n, float* out, void* stream) { embed->gather_dev(tokens, n, out, stream); };
    t.pick = [this, head, n_vocab](const void* x_q8_1, int* out_id, float* out_prob, void* stream) {
        return head->draft_top1(proxy_, head->weights(), x_q8_1, (int) n_vocab, n_vocab, map_, out_id, out_prob, proxy_scratch_,
                                stream, false);
    };
    return bind(t, head_rows, selections, cap, win_rows, err);
}

bool DflyDrafter::bind(const Target& target, const float* head_rows, const int32_t* const* selections, int64_t cap,
                       int win_rows, std::string& err) {
    if (!loaded()) { err = "dfly: bind before load"; return false; }
    if (bound()) { err = "dfly: the drafter is already bound"; return false; }
    if (!target.embed || !target.pick || target.n_vocab <= d_.mask_id || head_rows == nullptr || selections == nullptr ||
        cap < d_.sel_width || win_rows < 1 || win_rows > dfly2::kTapRows) {
        err = "dfly: the drafter needs the target's embedding and pick, the window's head rows and its kept selections";
        return false;
    }
    for (int i = 0; i < d_.layers; ++i)
        if (selections[i] == nullptr) { err = "dfly: no kept selection for draft layer " + std::to_string(i); return false; }
    const OnDevice on(device_);
    target_ = target;
    head_rows_ = head_rows;
    sel_.assign(selections, selections + d_.layers);
    sel_cap_ = cap;
    win_rows_ = win_rows;
    const bk::QsaShapes s = shapes_of(d_);
    // ---- the constants of every block: the private pool's identity tables, the mask ids, the bias
    {
        const int64_t cap_p = d_.sel_width + kDflyBlockMax, pages = (cap_p + s.page_size - 1) / s.page_size;
        std::vector<int32_t> iota((size_t) (rows_ * cap_p)), ident((size_t) pages), mask((size_t) kDflyBlockMax, d_.mask_id);
        for (int r = 0; r < rows_; ++r) std::iota(iota.begin() + (size_t) (r * cap_p), iota.begin() + (size_t) ((r + 1) * cap_p), 0);
        std::iota(ident.begin(), ident.end(), 0);
        if (!ok(device_copy(b_.iota, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice), "the cell ids", err) ||
            !ok(device_copy(b_.ident, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice), "the page table", err) ||
            !ok(device_copy(b_.mask_ids, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice), "the mask ids", err) ||
            !ok(cudaMemcpy(&conf_bias_, conf_b_, sizeof(float), cudaMemcpyDeviceToHost), "the confidence bias", err))
            return false;
    }
    cudaStream_t ds = nullptr;
    cudaEvent_t done = nullptr;
    if (!ok(cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking), "the stream", err) ||
        !ok(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "the event", err))
        return false;
    ds_ = ds;
    done_ = done;
    // the mask token's row for every row after the anchor's
    target_.embed(b_.mask_ids, rows_ - 1, b_.mask_rows, ds);
    if (!ok(cudaStreamSynchronize(ds), "the mask rows", err)) return false;
    // ---- the graph
    if (!ok(cudaStreamBeginCapture(ds, cudaStreamCaptureModeThreadLocal), "begin capture", err)) return false;
    const bool recorded = record(err);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(ds, &graph);
    cudaGraphExec_t exec = nullptr;
    if (!recorded || ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        if (recorded) err = std::string("dfly: the block graph: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaGraphDestroy(graph);
    exec_ = exec;
    // uploaded now, so the first block does not pay for it
    cudaGraphUpload(exec, ds);
    return ok(cudaStreamSynchronize(ds), "the block graph's upload", err);
}

// One block, every per-block value read from the record on the device.
bool DflyDrafter::record(std::string& err) {
    const cudaStream_t cs = (cudaStream_t) ds_;
    const bk::PdlScope plain(false);   // its launches depend on nothing but the stream's order
    const bk::QsaShapes s = shapes_of(d_);
    const RecLayout lay = layout_of(d_, rows_);
    const int R = rows_, N = d_.n_embd, I = d_.ffn, NH = d_.n_head, NKV = d_.n_head_kv, HD = d_.head_dim;
    const int Q = NH * HD, KV = NKV * HD;
    const int64_t cap_p = d_.sel_width + kDflyBlockMax;
    const int32_t* blk = (const int32_t*) b_.rec;
    const int32_t* step = blk + lay.step;
    const int32_t *pos_q = blk + lay.pos_q, *pos_k = blk + lay.pos_k;
    const bk::QsaAttnPools* src = (const bk::QsaAttnPools*) (b_.rec + lay.pools);
    int32_t* out_id = b_.out;
    float *out_raw = (float*) (b_.out + kOut), *out_prob = (float*) (b_.out + 2 * kOut);
    // the private pool, in the cache's form with the identity table
    bk::QsaAttnPools priv;
    bk::DflyKvDst dst;
    priv.page_table = b_.ident;
    if (kv_int8_) {
        priv.k_q = (const int8_t*) b_.pk; priv.v_q = (const int8_t*) b_.pv;
        priv.k_scale = (const uint16_t*) b_.pks; priv.v_scale = (const uint16_t*) b_.pvs;
        dst.k_q = (int8_t*) b_.pk; dst.v_q = (int8_t*) b_.pv;
        dst.k_scale = (uint16_t*) b_.pks; dst.v_scale = (uint16_t*) b_.pvs;
    } else {
        priv.k_pool = (const uint16_t*) b_.pk; priv.v_pool = (const uint16_t*) b_.pv;
        dst.k_pool = (uint16_t*) b_.pk; dst.v_pool = (uint16_t*) b_.pv;
    }
    if (!bk::qsa_dec_attn_supported(priv, s)) { err = "dfly: the attention does not take the drafter's head shape"; return false; }
    try {
        // ---- inputs: the source row's hidden state, the anchor's embedding, the mask rows
        bk::gather_rows((const uint8_t*) head_rows_, (int64_t) N * 4, blk + bk::kDflyRow, 1, (uint8_t*) b_.hs, cs);
        target_.embed(blk + bk::kDflyAnchor, 1, b_.emb0, cs);
        dfly2::gated_fusion(fusion_, b_.emb0, b_.hs, N, d_.rms_eps, b_.cat, b_.cat_q8, b_.fused, b_.x, cs);
        if (cudaMemcpyAsync(b_.x + N, b_.mask_rows, (size_t) (R - 1) * N * 4, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
            err = "dfly: the mask rows' copy failed";
            return false;
        }
        for (int i = 0; i < d_.layers; ++i) {
            const Layer& l = layer_[(size_t) i];
            // ---- the attention's projections from one q8_1 image of the normed rows
            bk::rms_norm_rows(b_.x, l.in_ln, b_.y, R, N, d_.rms_eps, cs);
            bk::native_quantize_q8_1(b_.y, b_.xq, N, R, cs);
            bk::native_mmvq(kQ8_0, l.q, b_.xq, b_.qfull, N, 2 * Q, R, cs);
            bk::native_mmvq(kQ8_0, l.k, b_.xq, b_.kcur, N, KV, R, cs);
            bk::native_mmvq(kQ8_0, l.v, b_.xq, b_.vcur, N, KV, R, cs);
            // the query half of each head's row, normed and rotated; the gate half stays in qfull for the merge
            if (cudaMemcpy2DAsync(b_.qcur, (size_t) HD * 4, b_.qfull, (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) (R * NH),
                                  cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "dfly: the q/gate split failed";
                return false;
            }
            bk::native_qsa_rms_norm_weighted(b_.qcur, l.q_norm, b_.qcur, HD, R * NH, d_.attn_eps, cs);
            bk::native_rope_apply(b_.qcur, b_.qcur, R * NH, HD, d_.n_rot, bk::rope_scaling(), pos_q, cs);
            bk::native_qsa_rms_norm_weighted(b_.kcur, l.k_norm, b_.kcur, HD, R * NKV, d_.attn_eps, cs);
            bk::native_rope_apply(b_.kcur, b_.kcur, R * NKV, HD, d_.n_rot, bk::rope_scaling(), pos_k, cs);
            // ---- the context: the paired layer's selected cells, then the block's own rows behind them
            bk::dfly_kv_gather(src + i, sel_[(size_t) i], sel_cap_, blk, d_.sel_width, s, dst, cs);
            for (int r = 0; r < R; ++r) {
                const int32_t* st = step + (size_t) r * bk::kStepCount;
                if (kv_int8_)
                    bk::kv_append_q8_step(dst.k_q, dst.v_q, dst.k_scale, dst.v_scale, b_.ident, st, b_.kcur + (size_t) r * KV,
                                          b_.vcur + (size_t) r * KV, s, cs);
                else
                    bk::kv_append_step(dst.k_pool, dst.v_pool, b_.ident, st, b_.kcur + (size_t) r * KV, b_.vcur + (size_t) r * KV, s, cs);
            }
            // row r attends cells 0 .. nctx + r in one softmax; the merge gates the rows and writes their q8_1 image
            bk::qsa_dec_attn(b_.qcur, priv, b_.iota, step, cap_p, s, b_.attn_scratch, b_.attn, R, cs, b_.qfull, b_.attn_gated,
                             b_.xq);
            bk::native_mmvq(kQ8_0, l.o, b_.xq, b_.o, Q, N, R, cs);
            bk::add_inplace(b_.x, b_.o, (int64_t) R * N, cs);
            // ---- the MLP
            bk::rms_norm_rows(b_.x, l.post_ln, b_.y, R, N, d_.rms_eps, cs);
            bk::native_quantize_q8_1(b_.y, b_.xq, N, R, cs);
            bk::native_mmvq(kQ8_0, l.gate, b_.xq, b_.g, N, I, R, cs);
            bk::native_mmvq(kQ8_0, l.up, b_.xq, b_.u, N, I, R, cs);
            bk::swiglu_scaled(b_.g, b_.u, nullptr, nullptr, b_.h, R, I, I, cs);
            bk::native_quantize_q8_1(b_.h, b_.xq, I, R, cs);
            bk::native_mmvq(kQ8_0, l.down, b_.xq, b_.o, I, N, R, cs);
            bk::add_inplace(b_.x, b_.o, (int64_t) R * N, cs);
        }
        bk::rms_norm_rows(b_.x, final_norm_, b_.z, R, N, d_.rms_eps, cs);
        // ---- the drafts in order: each row's correction reads the token drafted before it
        for (int j = 0; j + 1 < R; ++j) {
            const float* zr = b_.z + (size_t) (j + 1) * N;
            float* hp = b_.hp + (size_t) j * N;
            target_.embed(j == 0 ? blk + bk::kDflyAnchor : out_id + (j - 1), 1, b_.pe, cs);
            dfly2::correction_mlp(corr_, zr, b_.pe, N, d_.rms_eps, b_.cat, b_.cat_q8, b_.cg, b_.cu, b_.ch, b_.ch_q8, b_.delta, cs);
            if (cudaMemcpyAsync(hp, zr, (size_t) N * 4, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "dfly: a draft row's copy failed";
                return false;
            }
            bk::add_inplace(hp, b_.delta, N, cs);
            bk::f32_rows_dot(hp, conf_w_, out_raw + j, 1, 1, N, cs);
            bk::native_quantize_q8_1(hp, b_.hq, N, 1, cs);
            if (!target_.pick(b_.hq, out_id + j, out_prob + j, cs)) {
                err = "dfly: the head's two-stage argmax refused the head";
                return false;
            }
        }
    } catch (const std::exception& e) {
        err = std::string("dfly: ") + e.what();
        return false;
    }
    return true;
}

bool dfly_session_pools(const SessionState& ss, const ModelGeometry& g, const DflyDims& d, bool kv_int8, bk::QsaAttnPools* out,
                        std::string& err) {
    if (out == nullptr || d.layers < 1 || (size_t) d.layers != d.target_layers.size() || ss.qsa_states == nullptr) {
        err = "dfly: no session state for the drafter's paired layers";
        return false;
    }
    for (int i = 0; i < d.layers; ++i) {
        const int64_t l = d.target_layers[(size_t) i];
        // the session's states are indexed by a full-attention layer's number among them
        const int64_t ord = l / g.qsa_interval;
        if (l < ss.layer_lo || l >= ss.layer_hi || !is_qsa_layer(g, l) || ord < ss.qsa_ord0 || ord >= ss.qsa_ord0 + ss.qsa_alloc) {
            err = "dfly: the session does not run target layer " + std::to_string(l);
            return false;
        }
        const QsaState& st = ss.qsa_states[ord];
        // the gather copies cells as stored, through the page table of a cache that is whole in VRAM
        const bool form = kv_int8 ? (st.kv_int8 && st.k_q && st.v_q && st.k_scale && st.v_scale) : (!st.kv_int8 && st.k_pool && st.v_pool);
        if (st.kv_mode != 0 || st.kv_q4 || st.kv_hybrid || !form || st.page_table == nullptr) {
            err = "dfly: target layer " + std::to_string(l) + "'s K/V is not whole in VRAM as " + (kv_int8 ? "int8" : "fp16");
            return false;
        }
        out[i] = qsa_attn_pools(st);
    }
    return true;
}

int DflyDrafter::add_sequence(const bk::QsaAttnPools* pools, std::string& err) {
    if (!loaded()) { err = "dfly: a sequence before load"; return -1; }
    if (pools == nullptr) { err = "dfly: a sequence needs its paired layers' K/V pools"; return -1; }
    const RecLayout lay = layout_of(d_, rows_);
    for (int i = 0; i < d_.layers; ++i) {
        const bk::QsaAttnPools& p = pools[i];
        const bool form = kv_int8_ ? (p.k_q && p.v_q && p.k_scale && p.v_scale) : (p.k_pool && p.v_pool);
        if (!form || p.page_table == nullptr) {
            err = "dfly: draft layer " + std::to_string(i) + "'s K/V pools are not in the form the drafter was loaded for";
            return -1;
        }
    }
    Sequence q;
    const size_t rb = lay.bytes, ob = (size_t) 3 * kOut * 4;
    // page-locked, so a block's copies are queued without a staging copy
    if (cudaHostAlloc((void**) &q.rec, rb, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &q.out, ob, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        if (q.rec) cudaFreeHost(q.rec);
        err = "dfly: a sequence's page-locked record does not fit";
        return -1;
    }
    std::memset(q.rec, 0, rb);
    std::memset(q.out, 0, ob);
    // the pools do not move, so the record names them once
    std::memcpy(q.rec + lay.pools, pools, (size_t) d_.layers * sizeof(bk::QsaAttnPools));
    seqs_.push_back(q);
    return (int) seqs_.size() - 1;
}

bool DflyDrafter::draft(int seq, int32_t anchor, int64_t pos, int64_t win_pos0, int win_row0, int win_rows, bool& queued,
                        std::string& err) {
    queued = false;
    if (!bound()) { err = "dfly: a block before bind"; return false; }
    if (seq < 0 || seq >= (int) seqs_.size() || win_row0 < 0 || win_rows < 0 || win_row0 + win_rows > win_rows_) {
        err = "dfly: a block's sequence or its rows of the window are out of range";
        return false;
    }
    // every row of a window keeps its head row and each paired layer's selection
    unsigned sel_ok[dfly2::kTapRows] = {};
    for (int r = 0; r < win_rows; ++r) sel_ok[r] = (1u << d_.layers) - 1u;
    const int row = dfly2::source_row(win_pos0, (1u << win_rows) - 1u, sel_ok, d_.layers, pos);
    if (row < 0) {
        ++skipped;
        return true;
    }
    if (!launch(seq, anchor, win_row0 + row, pos, err)) return false;
    queued = true;
    return true;
}

bool DflyDrafter::launch(int seq, int32_t anchor, int row, int64_t pos, std::string& err) {
    Sequence& q = seqs_[(size_t) seq];
    if (pos < 1 || pos > INT32_MAX - kDflyBlockMax || anchor < 0 || anchor >= target_.n_vocab) {
        err = "dfly: a block's anchor or position is out of range";
        return false;
    }
    // the copy of the record is queued, so the record is the block's until the round's wait
    if (q.queued) { err = "dfly: the sequence's last block is not waited for"; return false; }
    const bk::QsaShapes s = shapes_of(d_);
    const RecLayout lay = layout_of(d_, rows_);
    // ---- the block's record: every value its kernels read
    int32_t* w = (int32_t*) q.rec;
    const int32_t nctx = (int32_t) bk::qsa_selection_width(pos, s);
    w[bk::kDflyAnchor] = anchor;
    w[bk::kDflyRow] = row;
    w[bk::kDflyNctx] = nctx;
    for (int r = 0; r < rows_; ++r) {
        // row r is cell nctx + r of the private pool and attends the cells up to itself
        int32_t* st = w + lay.step + (size_t) r * bk::kStepCount;
        st[bk::kStepPos] = nctx + r;
        st[bk::kStepNKv] = nctx + r + 1;
        st[bk::kStepNBid] = (int32_t) ((nctx + r + 1) / s.idx_block);
        st[bk::kStepWidth] = nctx + r + 1;
        for (int h = 0; h < d_.n_head; ++h) w[lay.pos_q + (size_t) r * (size_t) d_.n_head + (size_t) h] = (int32_t) pos + r;
        for (int h = 0; h < d_.n_head_kv; ++h) w[lay.pos_k + (size_t) r * (size_t) d_.n_head_kv + (size_t) h] = (int32_t) pos + r;
    }
    // ---- three queued operations: the record in, the graph, the output rows out
    const OnDevice on(device_);
    const cudaStream_t cs = (cudaStream_t) ds_;
    const std::chrono::steady_clock::time_point queued_at = std::chrono::steady_clock::now();
    if (!ok(cudaMemcpyAsync(b_.rec, q.rec, lay.bytes, cudaMemcpyHostToDevice, cs), "a block's record", err) ||
        !ok(cudaGraphLaunch((cudaGraphExec_t) exec_, cs), "a block's launch", err) ||
        !ok(cudaMemcpyAsync(q.out, b_.out, (size_t) 3 * kOut * 4, cudaMemcpyDeviceToHost, cs), "a block's drafts", err) ||
        !ok(cudaEventRecord((cudaEvent_t) done_, cs), "a block's event", err))
        return false;
    q.queued = true;
    // a round is timed from its first block
    if (!pending_) round0_ = queued_at;
    pending_ = true;
    return true;
}

bool DflyDrafter::wait(std::string& err) {
    if (!bound()) { err = "dfly: a wait before bind"; return false; }
    if (!pending_) return true;
    const bool done = ok(cudaEventSynchronize((cudaEvent_t) done_), "a block", err);
    for (Sequence& q : seqs_) q.queued = false;
    pending_ = false;
    if (!done) return false;
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - round0_).count();
    ++rounds;
    return true;
}

void DflyDrafter::read(int seq, int32_t* ids, float* conf) const {
    if (seq < 0 || seq >= (int) seqs_.size() || ids == nullptr || conf == nullptr) return;
    const int32_t* out = seqs_[(size_t) seq].out;
    const float* raw = (const float*) (out + kOut);
    for (int j = 0; j < drafts(); ++j) {
        ids[j] = out[j];
        conf[j] = 1.0f / (1.0f + std::exp(-(raw[j] + conf_bias_)));
    }
}

const float* DflyDrafter::raw_conf(int seq) const {
    return seq < 0 || seq >= (int) seqs_.size() ? nullptr : (const float*) (seqs_[(size_t) seq].out + kOut);
}

const float* DflyDrafter::head_prob(int seq) const {
    return seq < 0 || seq >= (int) seqs_.size() ? nullptr : (const float*) (seqs_[(size_t) seq].out + 2 * kOut);
}

}  // namespace basalt::core
