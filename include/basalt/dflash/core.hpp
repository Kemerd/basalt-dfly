// include/basalt/dflash/core.hpp - the DFlash 2 block drafter's passes, for every model that hosts one.
//
// A host gives the drafter (weights.hpp) with its weights on the device, a window gemm workspace sized with
// workspace_need, a scratch for its widest pass and one Seq per sequence.  The passes then are:
//
//   context   tap() projects a target layer's rows into a window's features as the layer ends; context_rows() norms
//             the features of the rows that were kept and writes their K and V into the sequence's rings.
//   draft     block_tokens() lays out [anchor, mask, ...] per sequence; the host embeds the ids with its own table;
//             block_layers() runs the drafter's layers and the final norm; the host scores the normed rows against
//             its own head; block_select() walks one path through each row's candidates.
//
// The pass is cut where models differ (the embedding, the head), so the core calls nothing of the host's.  Every
// product runs on the window gemm: a row's bits do not depend on the rows beside it.  Drafts are never output: the
// target checks each one.
#pragma once

#include "basalt/dflash/weights.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/window_gemm.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::dflash {

/// Cells of a sequence's ring: the drafter's window, a block's own rows and room to spare, so that a block's cells
/// never land on a position its window reads.
inline int64_t ring_cells(const Meta& m) { return (int64_t) m.window + 256; }
/// Bytes of one ring (one layer's K or V).
inline size_t ring_bytes(const Meta& m) { return (size_t) ring_cells(m) * (size_t) m.n_kv * (size_t) m.head_dim * sizeof(uint16_t); }

/// The drafter's rope: plain positions over the whole head.
kernels::RopeSpec block_rope(const Meta& m);

// ---- a sequence ---------------------------------------------------------------------------------------------------

/// One sequence's drafter state on the device, and the positions its context holds.
///
/// The rings are cyclic: position p lies in cell p % cap of each layer's K and V.  The context is the positions
/// [from, upto): a pass at `pos` reads at most the drafter's window behind it and nothing below `from`.  The host keeps
/// the range true with the rules below; the core's passes record what they write.
struct Seq {
    std::vector<uint16_t*> k, v;   ///< per drafter layer: [cap][n_kv][head_dim] fp16
    int64_t cap = 0;               ///< cells of each ring
    int64_t from = 0, upto = 0;    ///< the context: positions [from, upto)
    int64_t hi = -1;               ///< the highest position a cell was written for (a block's own rows count)
    float* acc = nullptr;          ///< [rows][n_embd]: the last window's features (tap), read by context_rows
    int32_t* anchor = nullptr;     ///< [1]: the last block's anchor, where the selector reads it
    int32_t* ids = nullptr;        ///< [rows]: the last block's drafts
    float* conf = nullptr;         ///< [rows]: their confidences
    int32_t* q_ids = nullptr;      ///< [rows][kDraftQ]: the candidates each draft was picked from
    float* q_p = nullptr;          ///< [rows][kDraftQ]: and the distribution over them
    float temperature = 0.0f;      ///< how the selector picks: 0 its best candidates; above, drawn
    uint64_t seed = 0;             ///< the draws' seed

    /// The context is empty from position q: no cell is this sequence's (a new or a restored conversation).
    void reset(int64_t q);
    /// The conversation goes on from q, at or below where it was: the positions from q on are dropped, and so are the
    /// positions whose cells later ones have taken since.  Nothing is left when q lies past the context or below it.
    void truncate(int64_t q);
    /// Context rows for positions [pos, pos + n) are in the rings: the context ends after them.
    void extend(int64_t pos, int64_t n);
    /// Cells were written for positions [pos, pos + n) (a block's own rows, which are no context).
    void mark(int64_t pos, int64_t n);
    /// The positions a pass at `pos` reads behind it, itself included: `limit` at most, none below `from`, one at least.
    int64_t window(int64_t limit, int64_t pos) const;
    /// Another sequence's range, for rings that were copied from it (copy_context).
    void take_range(const Seq& other);
};

/// Carves a sequence's buffers for windows of up to `rows` rows: each layer's K then V ring, the features, the drafts
/// and their candidates.  `take(bytes)` gives each buffer from the host's memory (null: out of room).  False then.
bool take_seq(const BlockDrafter& bd, int rows, const std::function<void*(size_t)>& take, Seq& seq);

// ---- load -----------------------------------------------------------------------------------------------------------

/// The kernels that give a drafter its decode forms, run once its weights have landed, and the device memory the forms
/// live in (freed with it).
struct Forms {
    std::vector<std::function<bool(void*)>> steps;   ///< in order, each through one scratch buffer
    size_t tmp_bytes = 0;                            ///< that buffer: the largest matrix
    std::vector<void*> owned;                        ///< device buffers made for the forms
    Forms() = default;
    ~Forms();
    Forms(const Forms&) = delete;
    Forms& operator=(const Forms&) = delete;
};

/// Lays `bd` (the pack's Q8_0 blocks, device memory still to be read) out for the window gemm, the convolutions repacked
/// to Q8 planes in place.  `fp4`: q, o, gate, up and down rounded to NVFP4 planes in place (the drafts' precision only:
/// every draft is verified exactly), about half the bytes a pass reads, for a host whose rounds are short; otherwise
/// they keep the pack's values as Q8 planes in place, for a host whose verify windows cost far more than the pass.
/// `keep_blocks`: k, v and the feature projection keep their blocks for a host whose prompt path reads them, and get
/// copies (k and v as the big products, the projection Q8 planes); otherwise they are converted in place too.  Reserves
/// the memory and sets the new types now; the kernels run in make_forms.  False with `err` when the memory is short or a
/// projection has a form the window gemm does not read.
bool decode_forms(BlockDrafter& bd, bool fp4, bool keep_blocks, Forms& forms, std::string& err);
/// Runs the forms' kernels (the weights have landed) on the current device.
bool make_forms(Forms& forms, std::string& err);

/// Raises `partial_floats` and `tickets` to what the drafter's window gemm calls need: each matrix of its decode forms
/// alone, q + k + v, k + v, gate + up, and the selector's projection.  The host adds its own head's call.
void workspace_need(const BlockDrafter& bd, size_t& partial_floats, size_t& tickets);

// ---- scratch --------------------------------------------------------------------------------------------------------

/// The rows a pass works in, on the device.  The first group is the host's to share with its own passes (one pass runs
/// at a time); the second is the drafter's alone, carved by take_scratch.
struct Scratch {
    int rows = 0;                              ///< the rows a pass may hold (every sequence's block together)
    float *xn = nullptr, *tmp = nullptr;       ///< [rows][n_embd]: the normed rows; a product's rows
    void* xq = nullptr;                        ///< the window gemm's activations: wgemm::act_bytes of the widest of n_embd,
                                               ///< n_head * head_dim and n_ff
    float *g = nullptr, *u = nullptr, *h = nullptr;   ///< [rows][n_ff]: the FFN's gate, up and hidden rows
    int32_t* tok = nullptr;                    ///< [rows]: the blocks' ids, which the host embeds
    float* lp_scratch = nullptr;               ///< row_logprobs_scratch_floats(rows, the scored vocabulary)

    float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bqo = nullptr, *bao = nullptr;   // q, k, v, rotated q, attention
    float *dyn = nullptr, *bh = nullptr;       // the convolutions' coefficients and their output rows
    float *battn = nullptr, *blp = nullptr, *bsel = nullptr;   // attention partial sums, candidates, the selector's rows
    int32_t* bpos = nullptr;                   // [rows][3] positions
};

/// Floats of the drafter's own buffers for passes of up to `rows` rows.
size_t scratch_floats(const BlockDrafter& bd, int rows);
/// Carves them (`take` as take_seq's) and records `rows`.  False when a buffer is missing or `rows` is more than a
/// window gemm call takes.
bool take_scratch(const BlockDrafter& bd, int rows, const std::function<void*(size_t)>& take, Scratch& sc);

// ---- context --------------------------------------------------------------------------------------------------------

/// The features of target layer j: `rows16` (n BF16 rows [n][n_embd], the target's rows after that layer) through its
/// block of the projection into `acc` (j == 0 writes it, later ones add).  Capturable.
void tap(const BlockDrafter& bd, const Scratch& sc, const kernels::wgemm::Workspace& ws, size_t j, const uint16_t* rows16, int n,
         float* acc, void* stream);

/// The context cells of positions pos0 .. pos0 + n - 1 from n rows of features `acc`: normed, then each layer's K and V
/// into the sequence's rings.  The range then ends after them.
bool context_rows(const BlockDrafter& bd, const Scratch& sc, const kernels::wgemm::Workspace& ws, Seq& seq, const float* acc, int n,
                  int64_t pos0, void* stream, std::string& err);

/// `src`'s rings and range into `dst` (a sequence of the same drafter), on `stream`.
bool copy_context(const BlockDrafter& bd, const Seq& src, Seq& dst, void* stream, std::string& err);

// ---- draft ----------------------------------------------------------------------------------------------------------
// One pass drafts for `count` sequences: sequence k's block is widths[k] rows (the drafter's block or wider: its anchor,
// then mask tokens) at positions pos[k] .., the blocks packed in order (at most Scratch::rows rows in all).

/// The blocks' ids into sc.tok and their positions into sc.bpos; each anchor to its sequence.
bool block_tokens(const BlockDrafter& bd, const Scratch& sc, Seq* const* seqs, const int32_t* anchors, const int64_t* pos,
                  const int* widths, int count, void* stream, std::string& err);

/// The drafter's layers over the blocks' embedded rows `x` (BF16 [rows][n_embd], the residual: the host's embedding of
/// sc.tok), then the final norm into sc.xn; with `pack`, the normed rows also as the window gemm's activations in sc.xq
/// (one kernel).  Each block's own K and V go to its sequence's rings at its positions.
bool block_layers(const BlockDrafter& bd, const Scratch& sc, const kernels::wgemm::Workspace& ws, Seq* const* seqs, const int64_t* pos,
                  const int* widths, int count, uint16_t* x, bool pack, void* stream, std::string& err);

/// One path through the candidates of slots 1 .. width - 1 of each block: `logits` are the host's head over the normed
/// rows, [rows][n_score] (the first n_score vocabulary rows).  Each sequence's drafts go to its ids and conf (and the
/// candidates to q_ids and q_p), picked as its temperature and seed say.
bool block_select(const BlockDrafter& bd, const Scratch& sc, const kernels::wgemm::Workspace& ws, Seq* const* seqs, const int64_t* pos,
                  const int* widths, int count, const float* logits, int64_t n_score, void* stream, std::string& err);

}  // namespace basalt::dflash
