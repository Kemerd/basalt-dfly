// include/basalt/models/qwen4exp/decode/dfly.hpp - the DFly2 drafter of a pack: its record and shapes.
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
#include "basalt/models/qwen4exp/decode/layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

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

}  // namespace basalt::core
