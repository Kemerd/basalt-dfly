// src/models/qwen4exp/serve/block_draft.hpp - the pack's DFlash 2 block drafter on this engine.
//
// The drafter's passes are the shared ones (basalt/dflash/core.hpp).  This is their host: the weights on the card, the
// working memory, the one conversation's context rings, and the two things the core leaves to a model - the target's
// embedding of a block's ids and the target's head over the block's rows.
//
//   context   a committed row's features are the stream means of the target's residual after the drafter's target
//             layers: a verify window's from the verifier's kept buffers, a prompt chunk's from the chunk's rows as
//             each of those layers ends.  Their K and V go to the rings.
//   draft     one block [anchor, mask, ...] behind the last kept row: the drafter's layers, the head over the whole
//             vocabulary, the selector's path.  The drafts go to the host with the chance each states.
//
// Drafts only choose which tokens the target checks: every window's rows are the target's own.
#pragma once

#include "basalt/dflash/core.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {
class LoadPlan;
class NativeEmbed;
class NativeHead;
class Verifier;
}  // namespace basalt::core

namespace basalt::models::qwen4exp::serve {

/// The block drafter of one engine: one conversation's context, one pass at a time.
class BlockDraft {
public:
    BlockDraft();
    ~BlockDraft();
    BlockDraft(const BlockDraft&) = delete;
    BlockDraft& operator=(const BlockDraft&) = delete;

    /// The device bytes of the pack's drafter weights as load() places them, from the pack's headers only.  False with
    /// `err` when the pack holds none.
    static bool plan_bytes(const std::vector<std::string>& shards, uint64_t& out, std::string& err);

    /// Reads the drafter's shape, allocates its weights and working memory on the current device and queues the
    /// weights' reads in `reads` (the caller's pass places them).  `n_layers`, `n_embd`, `n_vocab`: the target's.
    /// False with `err` when the pack's drafter does not fit this target or the memory is short.
    bool load(const std::vector<std::string>& shards, int64_t n_layers, int64_t n_embd, int64_t n_vocab,
              basalt::core::LoadPlan& reads, std::string& err);
    /// Once the weights have landed: the forms the passes read are made from them.
    bool finish(std::string& err);
    /// The target's head and the verifier whose windows it follows (its kept stream means, keep_stream_means of
    /// target_layers()).  After the verifier's init.
    bool bind(const basalt::core::NativeHead* head, const basalt::core::Verifier& ver, std::string& err);

    /// Whether the pack's drafter is loaded.
    bool on() const { return bd_.present; }
    /// The target layers whose outputs it reads, ascending.
    const std::vector<int>& target_layers() const { return bd_.target_layers; }
    /// The drafts one block gives.
    int max_drafts() const { return bd_.block - 1; }
    /// Device bytes held: the weights, and the working memory with the context rings.
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t work_bytes() const { return arena_bytes_; }

    // ---- the conversation's context ----------------------------------------------------------------------------------

    /// A request starts: the conversation goes on from position `resume` (0: a new one) and its prompt ends at
    /// `prompt_len`.  What the rings hold from `resume` on is dropped.
    void begin(int64_t resume, int64_t prompt_len);
    /// The session now holds a conversation the rings never saw (restored from an image): they hold nothing of it.
    void forget();
    /// Prefill::on_layer_rows: a prompt chunk's rows after `layer` (`R_rows`, device, T rows of hc streams of n_embd).
    /// The rows the drafter's window can still reach at the prompt's end become context once every target layer of the
    /// chunk has come by; a chunk some of whose target layers ran elsewhere gives none.  Queued on `stream`.
    bool prompt_rows(int64_t layer, int64_t pos0, int64_t T, const float* R_rows, int64_t hc, void* stream, std::string& err);
    /// The first `n` rows of the verifier's last window were committed at positions pos0 ..: their context.  Queued on
    /// the drafter's stream; nothing is waited for.  The verifier's kept rows are read by that work: before the next
    /// window runs, either a draft is read (draft_read) or the work is waited for (settle).
    bool window_rows(int n, int64_t pos0, std::string& err);
    /// Waits for the work queued so far.
    bool settle(std::string& err);

    // ---- a draft ----------------------------------------------------------------------------------------------------

    /// Queues one block behind `anchor`, the token at position `pos` (the context ends just before it), and the copy
    /// of its drafts to the host.  Returns once everything is launched.
    bool draft_launch(int32_t anchor, int64_t pos, std::string& err);
    /// Waits for the block and gives its drafts: up to `max_n` ids and the chance each states, in order.  `n`: how many.
    bool draft_read(int32_t* ids, float* chance, int max_n, int& n, std::string& err);

private:
    struct Work;
    /// Lays the working memory out from address `base` on (a region, or any address to size one): `w`'s buffers and
    /// the bytes they take.  False when the core has no layout for the drafter.
    static bool carve(const basalt::dflash::BlockDrafter& bd, int64_t n_vocab, uintptr_t base, Work& w, uint64_t& bytes);
    /// One pass at a time uses the scratch and the workspace: work for another stream first waits for the last one's.
    bool own(void* stream, std::string& err);
    /// The context goes on at `pos`: what the rings hold from there on is dropped, all of it when `pos` leaves a gap.
    void context_at(int64_t pos);
    /// The index of `layer` among the target layers, or -1.
    int target_index(int64_t layer) const;

    basalt::dflash::BlockDrafter bd_;
    basalt::dflash::Forms forms_;
    std::unique_ptr<Work> w_;           ///< the working memory's buffers: the scratch, the workspace, the rings
    const basalt::core::NativeHead* head_ = nullptr;
    const basalt::core::NativeEmbed* embed_ = nullptr;
    std::vector<const float*> means_;   ///< per target layer: the verifier's kept rows
    int ver_rows_ = 0;                  ///< rows of a verify window at most
    int64_t n_vocab_ = 0;
    int device_ = 0;

    void* weights_ = nullptr;           ///< every tensor of the drafter, one allocation
    void* arena_ = nullptr;             ///< the working memory, one allocation
    uint64_t weight_bytes_ = 0, arena_bytes_ = 0;
    void* ds_ = nullptr;                ///< the drafter's stream: its passes run beside a window's commit
    void* busy_ = nullptr;              ///< the stream the last pass was queued on

    int32_t* h_ids_ = nullptr;          ///< page-locked host: the last block's drafts ...
    float* h_chance_ = nullptr;         ///< ... and their chances
    int n_ready_ = 0;                   ///< drafts the last launch will give

    int64_t prompt_len_ = 0;            ///< the request's prompt length (0: unknown)
    // the prompt chunk whose features are being gathered: its rows' first position and count, and the target layer due
    int64_t acc_pos0_ = 0, acc_n_ = 0;
    int acc_next_ = -1;
};

}  // namespace basalt::models::qwen4exp::serve
