// src/models/qwen4exp/serve/block_draft.cpp - the pack's DFlash 2 block drafter on this engine: its memory, its
// context from windows and prompt chunks, and its draft pass around the target's embedding and head.
#include "block_draft.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/log.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/kernels/stream_mean.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <exception>

namespace basalt::models::qwen4exp::serve {

namespace dflash = basalt::dflash;
namespace bk = basalt::kernels;
namespace wg = basalt::kernels::wgemm;

namespace {

constexpr const char* kArch = "qwen4exp";           // the metadata keys' prefix
constexpr const char* kTensors = "dflash.";         // the drafter's tensors' prefix
constexpr int kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBf16 = 30;   // ggml type ids
constexpr uint64_t kAlign = 256;
constexpr int kSeqRows = bk::kVerifyMaxT;           // a verify window's rows: what one window adds to the context
constexpr int kPassRows = wg::kRows;                // the rows a pass may hold: a batch of a prompt chunk's rows

constexpr uintptr_t kSizing = kAlign;               // a base for a layout that is only measured: nothing reads its addresses

uint64_t aligned(uint64_t n) { return (n + kAlign - 1) & ~(kAlign - 1); }

/// Hands a region out in aligned pieces from `base` on.
struct Bump {
    uintptr_t base = 0;
    uint64_t used = 0;
    void* take(size_t bytes) {
        void* p = (void*) (base + (uintptr_t) used);
        used += aligned(bytes);
        return p;
    }
};

/// The drafter's shape from the pack, checked against the target it sits on.
bool read_shape(const basalt::GgufModel& model, int64_t n_embd, dflash::Meta& meta, std::string& err) {
    if (n_embd <= 0 || n_embd > INT32_MAX) {
        err = "block drafter: a target of no width";
        return false;
    }
    return dflash::read_meta(model.meta().metadata(), kArch, (int) n_embd, meta, err);
}

}  // namespace

/// The working memory's buffers, carved from one allocation.
struct BlockDraft::Work {
    dflash::Scratch sc;
    wg::Workspace ws;
    dflash::Seq seq;
    float* mean = nullptr;        ///< [kPassRows][n_embd]: a batch of prompt rows' stream means
    uint16_t* rows16 = nullptr;   ///< [kPassRows][n_embd] bf16: a batch of rows as the feature projection reads them
    float* pacc = nullptr;        ///< [window][n_embd]: a prompt chunk's features
    float* emb = nullptr;         ///< [block][n_embd]: a block's embedded ids
    uint16_t* x = nullptr;        ///< [block][n_embd] bf16: the block's residual
    void* hq = nullptr;           ///< the head's input: the block's normed rows as q8_1 columns
    float* logits = nullptr;      ///< [block][n_vocab]: row i scores slot i (row 0, the anchor's, is not scored)
};

BlockDraft::BlockDraft() : w_(std::make_unique<Work>()) {}

BlockDraft::~BlockDraft() {
    const basalt::core::OnDevice on(device_);
    if (ds_) cudaStreamDestroy((cudaStream_t) ds_);
    if (h_ids_) cudaFreeHost(h_ids_);
    if (h_chance_) cudaFreeHost(h_chance_);
    if (arena_) cudaFree(arena_);
    if (weights_) cudaFree(weights_);
}

bool BlockDraft::plan_bytes(const std::vector<std::string>& shards, uint64_t& out, std::string& err) {
    out = 0;
    try {
        const basalt::GgufModel model(shards);
        // each tensor on its own boundary, as load() places them
        for (size_t s = 0; s < model.size(); ++s)
            for (const basalt::TensorInfo& t : model.shard(s).tensors())
                if (t.name.starts_with(kTensors)) out += aligned(basalt::tensor_payload_bytes(t));
    } catch (const std::exception& e) {
        err = std::string("block drafter: ") + e.what();
        return false;
    }
    if (out == 0) {
        err = "block drafter: the pack holds none of its tensors";
        return false;
    }
    return true;
}

// The working memory in one order, counted and carved by the same code: the workspace of the drafter's products, its
// scratch (the shared rows first, then its own), the conversation's rings, then the host's rows around the passes.
bool BlockDraft::carve(const dflash::BlockDrafter& bd, int64_t n_vocab, uintptr_t base, Work& w, uint64_t& bytes) {
    Bump b{base, 0};
    const auto take = [&](size_t n) { return b.take(n); };
    const size_t N = (size_t) bd.n_embd, FF = (size_t) bd.n_ff, R = (size_t) kPassRows, B = (size_t) bd.block;
    const size_t widest = std::max({N, (size_t) bd.n_head * (size_t) bd.head_dim, FF});
    w = Work{};
    // the window gemm's split sums for every call the drafter makes
    size_t partial = 0, tickets = 0;
    dflash::workspace_need(bd, partial, tickets);
    w.ws.partial = (float*) take(std::max<size_t>(partial, 1) * sizeof(float));
    w.ws.tickets = (int32_t*) take(std::max<size_t>(tickets, 1) * sizeof(int32_t));
    w.ws.partial_floats = partial;
    w.ws.tickets_n = tickets;
    // the rows a pass works in
    w.sc.xn = (float*) take(R * N * sizeof(float));
    w.sc.tmp = (float*) take(R * N * sizeof(float));
    w.sc.xq = take(wg::act_bytes((int64_t) widest));
    w.sc.g = (float*) take(R * FF * sizeof(float));
    w.sc.u = (float*) take(R * FF * sizeof(float));
    w.sc.h = (float*) take(R * FF * sizeof(float));
    w.sc.tok = (int32_t*) take(R * sizeof(int32_t));
    w.sc.lp_scratch = (float*) take(bk::row_logprobs_scratch_floats((int) B, (int) n_vocab) * sizeof(float));
    if (!dflash::take_scratch(bd, kPassRows, take, w.sc) || !dflash::take_seq(bd, kSeqRows, take, w.seq)) return false;
    // a batch of rows on its way into the feature projection, and a prompt chunk's features
    w.mean = (float*) take(R * N * sizeof(float));
    w.rows16 = (uint16_t*) take(R * N * sizeof(uint16_t));
    w.pacc = (float*) take((size_t) bd.window * N * sizeof(float));
    // a block: its embedded ids, its residual, the head's input and the head's rows
    w.emb = (float*) take(B * N * sizeof(float));
    w.x = (uint16_t*) take(B * N * sizeof(uint16_t));
    w.hq = take(bk::native_q8_1_bytes((int) N, (int) B));
    w.logits = (float*) take(B * (size_t) n_vocab * sizeof(float));
    bytes = b.used;
    return true;
}

bool BlockDraft::load(const std::vector<std::string>& shards, int64_t n_layers, int64_t n_embd, int64_t n_vocab,
                      basalt::core::LoadPlan& reads, std::string& err) {
    if (on() || weights_ != nullptr) {
        err = "block drafter: already loaded";
        return false;
    }
    if (n_layers <= 0 || n_vocab <= 0 || n_vocab > INT32_MAX) {
        err = "block drafter: a target without layers or a vocabulary";
        return false;
    }
    if (cudaGetDevice(&device_) != cudaSuccess) device_ = 0;
    try {
        const basalt::GgufModel model(shards);
        dflash::BlockDrafter bd;
        if (!read_shape(model, n_embd, bd, err)) return false;
        // what the host itself needs of the shape: target layers this model has, in the order their rows come by; a
        // mask id the embedding holds; a block a verify window can check; a head row per candidate
        bool layers_ok = true;
        for (size_t j = 0; j < bd.target_layers.size(); ++j)
            layers_ok = layers_ok && bd.target_layers[j] >= 0 && bd.target_layers[j] < n_layers &&
                        (j == 0 || bd.target_layers[j] > bd.target_layers[j - 1]);
        if (!layers_ok || bd.mask_id >= n_vocab || bd.block > kSeqRows || bd.top_k > n_vocab || bd.n_embd % 256 != 0) {
            err = std::string(kArch) + ": the pack's block drafter does not fit this model (its target layers, mask token or block)";
            return false;
        }
        // ---- the weights: one allocation, each tensor on its own boundary, read in the caller's pass
        uint64_t wb = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const basalt::TensorInfo& t : model.shard(s).tensors())
                if (t.name.starts_with(kTensors)) wb += aligned(basalt::tensor_payload_bytes(t));
        if (wb == 0) {
            err = "block drafter: the pack holds none of its tensors";
            return false;
        }
        if (cudaMalloc(&weights_, wb) != cudaSuccess) {
            (void) cudaGetLastError();
            weights_ = nullptr;
            err = "block drafter: its weights (" + std::to_string(wb >> 20) + " MiB) do not fit";
            return false;
        }
        weight_bytes_ = wb;
        uint64_t used = 0;
        // one tensor of the pack: checked, given its place in the block, its read queued
        const auto place = [&](const std::string& name, int type, int64_t elems, int64_t dim0) -> void* {
            size_t at = 0;
            const basalt::TensorInfo* t = model.find(name, &at);
            if (t == nullptr) {
                if (err.empty()) err = "block drafter: the pack has no " + name;
                return nullptr;
            }
            const uint64_t bytes = basalt::tensor_payload_bytes(*t);
            if ((int) t->type != type || (int64_t) t->elements() != elems || bytes == 0 ||
                (dim0 > 0 && (t->shape.size() != 2 || (int64_t) t->shape[0] != dim0))) {
                if (err.empty()) err = "block drafter: " + name + " is not the expected type and shape";
                return nullptr;
            }
            const basalt::GgufFile& gg = model.shard(at);
            const uint64_t payload = gg.file_size() - gg.data_start();
            if (t->offset > payload || bytes > payload - t->offset || used + aligned(bytes) > wb) {
                if (err.empty()) err = "block drafter: " + name + " runs past the file or the weights' block";
                return nullptr;
            }
            void* d = (uint8_t*) weights_ + used;
            used += aligned(bytes);
            basalt::core::LoadItem it;
            it.file = reads.file(gg.path());
            it.off = gg.data_start() + t->offset;
            it.bytes = bytes;
            it.dev = d;
            it.device = device_;
            reads.add(it);
            return d;
        };
        dflash::Loaders loaders;
        loaders.floats = [&](const std::string& name, int64_t elems) { return (const float*) place(name, kTypeF32, elems, 0); };
        loaders.bf16 = [&](const std::string& name, int64_t elems) { return (const uint16_t*) place(name, kTypeBf16, elems, 0); };
        // a projection as the converter writes it: Q8_0 blocks, `cols` values to a row
        loaders.linear = [&](const std::string& base, int64_t rows, int64_t cols, basalt::model::Linear& l) {
            l.w = place(base + ".weight", kTypeQ8_0, rows * cols, cols);
            l.type = kTypeQ8_0;
            l.rows = rows;
            l.cols = cols;
            return l.w != nullptr;
        };
        if (!dflash::read_weights(bd, n_vocab, loaders)) {
            if (err.empty()) err = "block drafter: the pack's drafter did not load";
            return false;
        }
        // ---- the forms its passes read (made in finish(), once the bytes are there), then the working memory
        std::string why;
        // the pack's Q8_0 values: a verify window here costs far more than the pass, so the drafts' acceptance outweighs
        // the bytes NVFP4 would save
        if (!dflash::decode_forms(bd, false, false, forms_, why)) {
            err = "block drafter: " + why;
            return false;
        }
        Work counted;
        uint64_t ab = 0;
        if (!carve(bd, n_vocab, kSizing, counted, ab)) {
            err = "block drafter: its working memory could not be laid out";
            return false;
        }
        if (cudaMalloc(&arena_, ab) != cudaSuccess) {
            (void) cudaGetLastError();
            arena_ = nullptr;
            err = "block drafter: its working memory (" + std::to_string(ab >> 20) + " MiB) does not fit";
            return false;
        }
        arena_bytes_ = ab;
        // zeroed: the workspace's tickets start at 0, and no ring cell is read before it is written
        if (cudaMemset(arena_, 0, ab) != cudaSuccess || !carve(bd, n_vocab, (uintptr_t) arena_, *w_, ab)) {
            err = "block drafter: clearing its working memory failed";
            return false;
        }
        // its own stream, so a pass runs beside the window's commit; the drafts come back through page-locked memory
        cudaStream_t s = nullptr;
        void *hi = nullptr, *hc = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess ||
            cudaHostAlloc(&hi, (size_t) bd.block * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&hc, (size_t) bd.block * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            if (s) cudaStreamDestroy(s);
            if (hi) cudaFreeHost(hi);
            if (hc) cudaFreeHost(hc);
            err = "block drafter: no stream or page-locked memory for its drafts";
            return false;
        }
        ds_ = s;
        h_ids_ = (int32_t*) hi;
        h_chance_ = (float*) hc;
        n_vocab_ = n_vocab;
        w_->seq.reset(0);
        bd_ = std::move(bd);
    } catch (const std::exception& e) {
        err = std::string("block drafter: ") + e.what();
        return false;
    }
    std::string layers;
    for (int l : bd_.target_layers) layers += (layers.empty() ? "" : ",") + std::to_string(l);
    basalt::log::info("basalt generate", "block drafter: %d layers reading the target's layers %s, blocks of %d, a context of "
                      "%d positions; %.0f MiB of weights and %.0f MiB of working memory on the GPU",
                      bd_.n_layers, layers.c_str(), bd_.block, bd_.window, (double) weight_bytes_ / 1048576.0,
                      (double) arena_bytes_ / 1048576.0);
    return true;
}

bool BlockDraft::finish(std::string& err) {
    if (!on()) return true;
    std::string why;
    if (dflash::make_forms(forms_, why)) return true;
    err = "block drafter: " + why;
    return false;
}

bool BlockDraft::bind(const basalt::core::NativeHead* head, const basalt::core::Verifier& ver, std::string& err) {
    if (!on()) {
        err = "block drafter: bind before load";
        return false;
    }
    if (head == nullptr || !head->loaded()) {
        err = "block drafter: it scores its rows with the native head, which is not loaded";
        return false;
    }
    const basalt::core::NativeEmbed* embed = basalt::core::native_embed();
    if (embed == nullptr) {
        err = "block drafter: it reads the embedding table in its pack form, which is not loaded";
        return false;
    }
    // the rows the verifier keeps for it: one buffer per target layer, a window's rows each
    std::vector<const float*> means;
    for (size_t j = 0; j < bd_.target_layers.size(); ++j) {
        const float* m = ver.stream_means((int) j);
        if (m == nullptr) {
            err = "block drafter: the verifier keeps no rows of target layer " + std::to_string(bd_.target_layers[j]);
            return false;
        }
        means.push_back(m);
    }
    if (ver.max_t() < 1 || ver.max_t() > kSeqRows) {
        err = "block drafter: verify windows of " + std::to_string(ver.max_t()) + " rows";
        return false;
    }
    head_ = head;
    embed_ = embed;
    means_ = std::move(means);
    ver_rows_ = ver.max_t();
    return true;
}

// ---- the conversation's context -------------------------------------------------------------------------------------

void BlockDraft::begin(int64_t resume, int64_t prompt_len) {
    if (!on()) return;
    w_->seq.truncate(std::max<int64_t>(resume, 0));
    prompt_len_ = std::max<int64_t>(prompt_len, 0);
    acc_next_ = -1;
}

void BlockDraft::forget() {
    if (!on()) return;
    w_->seq.reset(0);
    acc_next_ = -1;
}

bool BlockDraft::own(void* stream, std::string& err) {
    if (busy_ != nullptr && busy_ != stream && cudaStreamSynchronize((cudaStream_t) busy_) != cudaSuccess) {
        err = std::string("block drafter: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    busy_ = stream;
    return true;
}

void BlockDraft::context_at(int64_t pos) {
    dflash::Seq& seq = w_->seq;
    if (pos > seq.upto) seq.reset(pos);          // a gap: what lies before it is no context of what follows
    else if (pos < seq.upto) seq.truncate(pos);  // rows written again
}

int BlockDraft::target_index(int64_t layer) const {
    for (size_t j = 0; j < bd_.target_layers.size(); ++j)
        if (bd_.target_layers[j] == layer) return (int) j;
    return -1;
}

bool BlockDraft::prompt_rows(int64_t layer, int64_t pos0, int64_t T, const float* R_rows, int64_t hc, void* stream,
                             std::string& err) {
    if (!on()) return true;
    const int j = target_index(layer);
    if (j < 0) return true;
    if (R_rows == nullptr || T <= 0 || pos0 < 0 || hc <= 0) {
        err = "block drafter: a prompt chunk without rows";
        return false;
    }
    const int64_t N = bd_.n_embd, W = bd_.window;
    // the chunk's rows the window can still reach when the prompt has ended; at most a window of them
    int64_t r0 = std::max<int64_t>(0, prompt_len_ - W - pos0);
    if (T - r0 > W) r0 = T - W;
    const int64_t n = T - r0;
    if (j == 0) {
        acc_pos0_ = pos0 + r0;
        acc_n_ = n;
        acc_next_ = n > 0 ? 0 : -1;
    }
    // a chunk whose target layers do not all come by here in order (some ran on another card) gives no context
    if (n <= 0 || acc_next_ != j || acc_pos0_ != pos0 + r0 || acc_n_ != n) {
        acc_next_ = -1;
        return true;
    }
    if (!own(stream, err)) return false;
    Work& w = *w_;
    const bk::PdlScope plain(false);   // each batch follows a copy or another path's kernels
    try {
        for (int64_t b = 0; b < n; b += kPassRows) {
            const int nb = (int) std::min<int64_t>(kPassRows, n - b);
            // the layer's output as the drafter was trained on it: the mean of the residual's streams
            bk::stream_mean(R_rows + (size_t) (r0 + b) * (size_t) (hc * N), nb, hc * N, N, hc, nullptr, 0, nullptr, 0, w.mean, stream);
            bk::to_bf16(w.mean, w.rows16, (int64_t) nb * N, stream);
            dflash::tap(bd_, w.sc, w.ws, (size_t) j, w.rows16, nb, w.pacc + (size_t) b * (size_t) N, stream);
        }
    } catch (const std::exception& e) {
        err = std::string("block drafter: ") + e.what();
        return false;
    }
    acc_next_ = j + 1;
    if ((size_t) acc_next_ < bd_.target_layers.size()) return true;
    // every target layer's features of these rows are in: their K and V
    acc_next_ = -1;
    context_at(acc_pos0_);
    for (int64_t b = 0; b < n; b += kPassRows) {
        const int nb = (int) std::min<int64_t>(kPassRows, n - b);
        if (!dflash::context_rows(bd_, w.sc, w.ws, w.seq, w.pacc + (size_t) b * (size_t) N, nb, acc_pos0_ + b, stream, err)) return false;
    }
    return true;
}

bool BlockDraft::window_rows(int n, int64_t pos0, std::string& err) {
    if (!on()) return true;
    if (head_ == nullptr || means_.size() != bd_.target_layers.size()) {
        err = "block drafter: context before bind";
        return false;
    }
    if (n < 1 || n > ver_rows_ || pos0 < 0) {
        err = "block drafter: " + std::to_string(n) + " committed rows at position " + std::to_string((long long) pos0);
        return false;
    }
    if (!own(ds_, err)) return false;
    Work& w = *w_;
    const int64_t N = bd_.n_embd;
    const bk::PdlScope plain(false);   // the first launch follows the window on another stream, the rest a copy or each other
    // each target layer's rows through its block of the feature projection, then the rows' K and V
    for (size_t j = 0; j < means_.size(); ++j) {
        bk::to_bf16(means_[j], w.rows16, (int64_t) n * N, ds_);
        dflash::tap(bd_, w.sc, w.ws, j, w.rows16, n, w.seq.acc, ds_);
    }
    context_at(pos0);
    return dflash::context_rows(bd_, w.sc, w.ws, w.seq, w.seq.acc, n, pos0, ds_, err);
}

bool BlockDraft::settle(std::string& err) {
    if (busy_ == nullptr || cudaStreamSynchronize((cudaStream_t) busy_) == cudaSuccess) return true;
    err = std::string("block drafter: ") + cudaGetErrorString(cudaGetLastError());
    return false;
}

// ---- a draft --------------------------------------------------------------------------------------------------------

bool BlockDraft::draft_launch(int32_t anchor, int64_t pos, std::string& err) {
    n_ready_ = 0;
    if (!on() || head_ == nullptr || embed_ == nullptr) {
        err = "block drafter: a draft before bind";
        return false;
    }
    if (anchor < 0 || anchor >= n_vocab_ || pos < 0) {
        err = "block drafter: anchor " + std::to_string(anchor) + " at position " + std::to_string((long long) pos);
        return false;
    }
    if (!own(ds_, err)) return false;
    Work& w = *w_;
    // the block reads the context up to its anchor: where that does not reach the anchor, it reads none
    if (w.seq.upto != pos) w.seq.reset(pos);
    const int64_t N = bd_.n_embd;
    const int width = bd_.block;
    dflash::Seq* seqs[1] = {&w.seq};
    w.seq.temperature = 0.0f;   // the selector's best path: the target checks every draft
    if (!dflash::block_tokens(bd_, w.sc, seqs, &anchor, &pos, &width, 1, ds_, err)) return false;
    try {
        const bk::PdlScope pdl(true, true);   // after the uploads: each product stages its weights during the kernel before
        // the target's own embedding of the anchor and the mask token, as the residual the layers add into
        embed_->gather_dev(w.sc.tok, width, w.emb, ds_);
        bk::to_bf16(w.emb, w.x, (int64_t) width * N, ds_);
        if (!dflash::block_layers(bd_, w.sc, w.ws, seqs, &pos, &width, 1, w.x, false, ds_, err)) return false;
        // the target's head over the drafts' rows, every vocabulary row with the head's own kernel (slot 0 is the anchor)
        bk::native_quantize_q8_1(w.sc.xn + N, w.hq, (int) N, width - 1, ds_);
        head_->score_rows(head_->weights(), w.hq, w.logits + (size_t) n_vocab_, (int) n_vocab_, width - 1, ds_);
        if (!dflash::block_select(bd_, w.sc, w.ws, seqs, &pos, &width, 1, w.logits, n_vocab_, ds_, err)) return false;
    } catch (const std::exception& e) {
        err = std::string("block drafter: ") + e.what();
        return false;
    }
    const cudaStream_t s = (cudaStream_t) ds_;
    if (cudaMemcpyAsync(h_ids_, w.seq.ids, (size_t) (width - 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(h_chance_, w.seq.conf, (size_t) (width - 1) * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess) {
        err = std::string("block drafter: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    n_ready_ = width - 1;
    return true;
}

bool BlockDraft::draft_read(int32_t* ids, float* chance, int max_n, int& n, std::string& err) {
    n = 0;
    if (!on() || ids == nullptr || chance == nullptr || max_n < 0) {
        err = "block drafter: nowhere to put its drafts";
        return false;
    }
    const int ready = n_ready_;
    n_ready_ = 0;
    if (cudaStreamSynchronize((cudaStream_t) ds_) != cudaSuccess) {
        err = std::string("block drafter: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // a draft that is no token ends the block there: the window takes the drafts before it
    const int most = std::min(ready, max_n);
    while (n < most && h_ids_[n] >= 0 && h_ids_[n] < n_vocab_) {
        ids[n] = h_ids_[n];
        const float c = h_chance_[n];
        chance[n] = c >= 0.0f ? std::min(c, 1.0f) : 0.0f;   // a chance that is no number counts as none
        ++n;
    }
    return true;
}

}  // namespace basalt::models::qwen4exp::serve
