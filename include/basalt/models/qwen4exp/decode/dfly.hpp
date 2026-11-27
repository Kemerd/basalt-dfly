// include/basalt/models/qwen4exp/decode/dfly.hpp - the DFly2 drafter of a pack: its record and shapes, its weights and the
// block pass.
//
// A block is the anchor token (the last pick, at a position the target has not run) and up to seven mask tokens. The
// drafter is dense with one residual stream; each of its layers is paired with one full-attention layer of the target and
// attends that layer's own K/V cache at the cells the layer selected for the row before the anchor, plus the block's own
// rows causally. It writes nothing into the target's state. The target's embedding and output head are shared.
//
// The forward, for the anchor `a` at position `pos` and rows r = 0..R-1 at pos + r; `s` = pos - 1 is the source row, e()
// the target's token embedding, rms(x, w, eps) = x * rsqrt(mean(x^2) + eps) * w:
//   1. Row 0: x = e(a) + gate * eh_proj(rms(e(a), enorm) :: rms(h_s, hnorm)), h_s the head's input at row s. Rows 1..: e(mask).
//   2. Each layer: y = rms(x, input_layernorm); x += o_proj(attention(y)); x += mlp(rms(x, post_attention_layernorm)),
//      mlp(u) = down(silu(gate(u)) * up(u)). Then z = rms(x, final_norm).
//   3. Attention: q_proj(y) is, per head, a query half and a gate half. q = rope(rms(query, q_norm)), the row's own key
//      rope(rms(k_proj(y), k_norm)) and value v_proj(y), both stored as the cache stores a cell. One softmax of
//      q.k / sqrt(head_dim) over the selected cells of the paired layer and the block's rows 0..r; the result times
//      sigmoid(gate). Query head h reads K/V head h / (heads / kv_heads).
//   4. Draft j = 0..R-2 from row j + 1 and the token before it (a, then draft j - 1):
//      c = rms(z[j+1], hidden_norm) :: rms(e(prev), embed_norm); h' = z[j+1] + down(silu(gate(c)) * up(c));
//      the draft is the head's argmax over h' and its confidence sigmoid(confidence_head(h')).
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/dfly2/record.hpp"
#include "basalt/dfly2/ops.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/kernels/dfly_kv.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::core {

class LoadPlan;
class NativeHead;
struct SessionState;

constexpr int kDflyBlockMax = 8;   ///< rows of the widest block: the anchor and seven masks

/// The drafter's shape, from the pack's meta.dfly2.
struct DflyDims {
    int layers = 0;
    int n_embd = 0, ffn = 0, corr_ffn = 0;                    ///< the stream, the MLP's and the correction's inner widths
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;   ///< the attention's, as the paired layers'
    int vocab = 0, mask_id = 0;
    int block = 0;                                            ///< rows of a block the drafter was trained with
    int sel_width = 0;                                        ///< the most cells a paired layer selects for a row
    float rms_eps = 0, attn_eps = 0, rope_base = 0;
    int64_t step = 0;                                         ///< the training step of the checkpoint
    std::vector<int> target_layers;                           ///< the target layer each draft layer is paired with
};

/// One tensor of the section: stored BF16; held as FP32 (the norms, the fusion's gate, the confidence head) or rounded
/// to Q8_0 as it loads (every projection).
struct DflySpec {
    std::string name;
    std::vector<uint64_t> shape;   ///< outermost first
    bool f32 = false;
    uint64_t elems() const;
    uint64_t bytes() const { return f32 ? elems() * 4 : elems() / 32 * 34; }   ///< on the card
};
/// Every tensor of a drafter of these dims, by the name the record gives it.
std::vector<DflySpec> dfly_specs(const DflyDims& d);

/// What of the target a record is checked against beside its geometry.
struct DflyTarget {
    int64_t n_vocab = 0;
    int sel_width = 0;   ///< ids a full-attention layer selects per row at most
    int n_rot = 0;       ///< rotary dims of a head
};

/// Fills `d` from the pack's record: `meta` is meta.dfly2 and `rec` its shared part. False with `err` for a record this
/// drafter does not run: another format, layer lists that disagree or name a layer without full attention, widths or
/// head counts that are not the target's, a mask token outside the vocabulary, a block wider than kDflyBlockMax,
/// another selection width, an inner width that is not whole Q8_0 blocks, a near window of the context, or a tensor that
/// is missing, not BF16 or of another shape.
bool dfly_check_record(const Json& meta, const dfly2::Record& rec, const ModelGeometry& g, const DflyTarget& t, DflyDims& d,
                       std::string& err);

/// What of a start decides whether the drafter can run.
struct DflyConfig {
    bool serve = false;              ///< the persistent engine
    bool kv_q4 = false;              ///< --kv q4_0
    bool kv_hybrid = false;          ///< --kv k8v4
    bool kv_resident = false;        ///< --kv-resident
    bool layer_split = false;
    bool control_vector = false;     ///< a control vector file is applied
    bool rope_scaled = false;        ///< the process rotary scaling is not none
    float rope_base = 0;             ///< the process rotary base
    bool head_two_stage = false;     ///< the output head's type has the two-stage argmax
};
/// Why a start like `c` cannot run a drafter of `d`, as the words of the one warning; empty when it can.
std::string dfly_pass_over(const DflyDims& d, const DflyConfig& c);

/// A pack's drafter as read: `present` false when the pack carries none.
struct DflyPack {
    bool present = false;
    DflyDims dims;
    dfly2::Record rec;
};
/// Reads the drafter of `file` (its meta loaded) and checks it as dfly_check_record does, and that every tensor lies
/// inside the `dfly` section. A pack with neither the section nor the record has no drafter, which is not an error.
bool dfly_read_pack(const ModelFile& file, const ModelGeometry& g, const DflyTarget& t, DflyPack& out, std::string& err);

/// The K/V pools of the target layers a drafter of `d` is paired with, in its layers' order, as session `ss` holds
/// them: `out` gets d.layers of them. False with `err` unless the session holds each of those layers' caches whole in
/// VRAM, as fp16 rows or (`kv_int8`) as int8 codes with their scales: the forms the block pass gathers.
bool dfly_session_pools(const SessionState& ss, const ModelGeometry& g, const DflyDims& d, bool kv_int8,
                        basalt::kernels::QsaAttnPools* out, std::string& err);

/// The drafter on the main card: its weights, one captured graph of the block pass and the sequences that draft with
/// it. A block is queued on the drafter's own stream and reads the last window's rows and the sequence's K/V pools; it
/// keeps nothing between blocks.
class DflyDrafter {
public:
    DflyDrafter() = default;
    ~DflyDrafter();
    DflyDrafter(const DflyDrafter&) = delete;
    DflyDrafter& operator=(const DflyDrafter&) = delete;

    /// The device bytes load() takes for a drafter of `d` whose blocks have `rows` rows: the weights as they are held
    /// and the working memory, a private K/V pool in the cache's form among it.
    static uint64_t plan_bytes(const DflyDims& d, int rows, bool kv_int8);
    /// Allocates the weights of `pack` (read from `path`, their reads queued in `reads`: the caller's pass places and
    /// rounds them) and the working memory for blocks of `rows` rows (2..kDflyBlockMax) over K/V of the given form.
    /// On the current device; call before the expert tier is sized.
    bool load(const DflyPack& pack, const std::string& path, int rows, bool kv_int8, LoadPlan& reads, std::string& err);
    /// The device bytes bind() takes for a head of this type over `n_vocab` rows: the 2-bit copy of the head its drafts
    /// are picked with, its scratch and its row map.
    static uint64_t bind_bytes(int head_type, int64_t n_vocab);
    /// The buffers the block pass reads, which the verify window leaves at fixed addresses: its head rows (`win_rows`
    /// rows of n_embd floats) and each draft layer's kept selections (`win_rows` rows of `cap` ids), and the head its
    /// drafts are picked with. Captures the graph.
    bool bind(const WeightTable& wt, const NativeHead* head, const float* head_rows, const int32_t* const* selections,
              int64_t cap, int win_rows, std::string& err);
    /// What the block pass takes from the target beside the window's rows, as calls that only enqueue on `stream` and
    /// read their per-block values from device memory: the embedding rows of `n` device token ids, and one q8_1
    /// column's draft with its probability (false: no pick was queued).
    struct Target {
        std::function<void(const int32_t* tokens, int64_t n, float* out, void* stream)> embed;
        std::function<bool(const void* x_q8_1, int* out_id, float* out_prob, void* stream)> pick;
        int64_t n_vocab = 0;
    };
    /// bind() over the target's two calls as given: the form the parity programs bind, with rows of their own.
    bool bind(const Target& target, const float* head_rows, const int32_t* const* selections, int64_t cap, int win_rows,
              std::string& err);

    bool loaded() const { return weights_ != nullptr; }
    bool bound() const { return exec_ != nullptr; }
    const DflyDims& dims() const { return d_; }
    int rows() const { return rows_; }            ///< rows of a block: the anchor and drafts()
    int drafts() const { return rows_ - 1; }
    bool kv_int8() const { return kv_int8_; }     ///< the form of the K/V pools its sequences hold
    uint64_t vram_bytes() const { return vram_; }

    /// A sequence that drafts (the solo session, a batch slot) over the K/V pools of its paired layers (dims().layers
    /// of them, in the record's order: dfly_session_pools), which must stay where they are while it drafts. It holds a
    /// page-locked record and output rows. Its id, or -1 with `err`.
    int add_sequence(const basalt::kernels::QsaAttnPools* pools, std::string& err);
    /// Queues the block that follows a window for sequence `seq`, without waiting: `anchor` is the token at `pos`, the
    /// pick of the window's row at pos - 1. The sequence's rows of the last window are `win_rows` rows from its row
    /// `win_row0`, the first of them at position `win_pos0` (a window of one sequence: row 0 and every row). `queued`
    /// is false, and no block is queued, when those rows do not hold the row at pos - 1: the round then has no drafts
    /// for the sequence.
    bool draft(int seq, int32_t anchor, int64_t pos, int64_t win_pos0, int win_row0, int win_rows, bool& queued,
               std::string& err);
    /// Waits for every block queued so far: one wait a round, whatever the sequences that drafted.
    bool wait(std::string& err);
    /// After wait(): the drafts of the sequence's last block, drafts() ids with each one's confidence (the confidence
    /// head's, a stated chance that the draft is accepted).
    void read(int seq, int32_t* ids, float* conf) const;
    /// The same block's confidence logits before the bias, and the draft head's own probability of each pick.
    const float* raw_conf(int seq) const;
    const float* head_prob(int seq) const;
    /// The last block's rows after the final norm (rows() of n_embd) and its drafts' corrected rows (drafts() of
    /// n_embd), on the device until the next block.
    const float* block_rows() const { return b_.z; }
    const float* corrected_rows() const { return b_.hp; }

    double ms_draft = 0;      ///< host time of the rounds, from a round's first block queued to its wait's end
    int64_t rounds = 0;       ///< rounds that drafted
    int64_t skipped = 0;      ///< blocks not queued: the last window did not hold their source row

private:
    struct Layer {
        const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
        const float *in_ln = nullptr, *q_norm = nullptr, *k_norm = nullptr, *post_ln = nullptr;
    };
    // the working memory of a block, carved from one arena
    struct Buffers {
        uint8_t* rec = nullptr;      // the block's record: blk, steps, positions, the paired layers' pools
        int32_t* out = nullptr;      // the block's output rows: ids, confidence logits, head probabilities
        int32_t *iota = nullptr, *ident = nullptr, *mask_ids = nullptr;
        float *mask_rows = nullptr, *hs = nullptr, *emb0 = nullptr, *cat = nullptr, *fused = nullptr;
        float *x = nullptr, *y = nullptr, *o = nullptr, *z = nullptr, *hp = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr, *attn_gated = nullptr;
        float *attn_scratch = nullptr, *g = nullptr, *u = nullptr, *h = nullptr;
        float *pe = nullptr, *cg = nullptr, *cu = nullptr, *ch = nullptr, *delta = nullptr;
        uint8_t *xq = nullptr, *cat_q8 = nullptr, *ch_q8 = nullptr, *hq = nullptr;
        uint8_t *pk = nullptr, *pv = nullptr, *pks = nullptr, *pvs = nullptr;   // the private pool
    };
    struct Sequence {
        uint8_t* rec = nullptr;   // page-locked, the record's layout: its pools are written once
        int32_t* out = nullptr;   // page-locked, the output rows' layout
        bool queued = false;      // its block's copies are queued: the record is not rewritten before the wait
    };
    static uint64_t carve(const DflyDims& d, int rows, bool kv_int8, uint8_t* base, Buffers& b);
    static uint64_t record_bytes(const DflyDims& d, int rows);
    static basalt::kernels::QsaShapes shapes_of(const DflyDims& d);   // the target's shapes with the drafter's heads
    bool record(std::string& err);
    bool launch(int seq, int32_t anchor, int row, int64_t pos, std::string& err);

    DflyDims d_;
    int rows_ = 0, device_ = -1;
    bool kv_int8_ = false;
    uint64_t vram_ = 0;
    void *weights_ = nullptr, *arena_ = nullptr;
    std::vector<Layer> layer_;
    dfly2::FusionWeights fusion_;
    dfly2::CorrectionWeights corr_;
    const float *final_norm_ = nullptr, *conf_w_ = nullptr, *conf_b_ = nullptr;
    float conf_bias_ = 0;                      // read at bind, once the weights have landed
    Buffers b_;
    // bind()
    Target target_;
    const float* head_rows_ = nullptr;
    std::vector<const int32_t*> sel_;
    int64_t sel_cap_ = 0;
    int win_rows_ = 0;                         // rows the bound window buffers hold
    bool pending_ = false;                     // a block is queued and not yet waited for
    std::chrono::steady_clock::time_point round0_;   // when the round's first block was queued
    void *proxy_ = nullptr, *proxy_scratch_ = nullptr;
    int32_t* map_ = nullptr;
    void *ds_ = nullptr, *exec_ = nullptr, *done_ = nullptr;   // the drafter's stream, its graph, the event after a round
    std::vector<Sequence> seqs_;
};

}  // namespace basalt::core
