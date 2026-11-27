// include/basalt/artifact/dfly2_checkpoint.hpp - the drafter checkpoint the DFly2 trainer (tools/dfly2-train) writes and
// a runtime or a converter reads: one safetensors file, its metadata keys, and every tensor's name, shape and dtype.
// Header-only and standard C++ only, so the trainer, the converter and the engine share one definition.
//
// The file is a safetensors file: its `__metadata__` holds the keys below (every value a string) and its tensors are
// exactly drafter_tensors(shape), all F32, shapes outermost first, a projection as [out][in] (row-major). The values
// are the trainer's master weights; `dfly2.weight_dtype` names the type the trained forward rounded them to (BF16:
// round to nearest even), which is what a runtime rounds to before any further quantisation.
// The target's token embedding and output head are not in the file: the drafter uses the pack's.
//
// The forward the weights were trained for. A block is the anchor token `a` at rotary position `pos` and n mask
// tokens, rows r = 0..n at positions pos + r; `s` = pos - 1 is the row whose output chose `a`; e() is the target's
// token embedding; rms(x, w, eps) = x * rsqrt(mean(x^2) + eps) * w in FP32.
//   1. Inputs: x[0] = e(a) + mtp_fusion.gate * eh_proj(rms(e(a), enorm) :: rms(h_s, hnorm)), h_s the vector that
//      entered the target's output head at row s (after its output mix). x[r] = e(mask_token_id) for r >= 1.
//      `::` joins along the feature axis; the norms outside the attention use `dfly2.rms_norm_eps`.
//   2. `dfly2.num_layers` pre-norm layers, one residual stream:
//      y = rms(x, input_layernorm); x += o_proj(attention(y)); x += mlp(rms(x, post_attention_layernorm));
//      mlp(u) = down_proj(silu(gate_proj(u)) * up_proj(u)). Then z = rms(x, final_norm).
//   3. Attention of layer i, paired with target layer `dfly2.target_layers`[i]:
//      qg = q_proj(y) is [num_heads][2 * head_dim]: head h's first head_dim values are its query, the next head_dim
//      its gate. q = rope(rms(query, q_norm, attn_norm_eps)); a block row's own key is
//      rope(rms(k_proj(y)[kv head], k_norm, attn_norm_eps)) and its value v_proj(y)[kv head], [num_kv_heads][head_dim].
//      rope turns the first `dfly2.rope_dims` values of a head: pair p of rope_dims / 2 is (u[p], u[p + rope_dims/2])
//      -> (u[p] cos t - u[p + rope_dims/2] sin t, u[p] sin t + u[p + rope_dims/2] cos t), t = position *
//      rope_base^(-2p / rope_dims), the position being the row's (pos + r).
//      Row r attends the cells the paired target layer selected for row s, read from that layer's cache (keys already
//      normed and rotated, at most `dfly2.selection_width` of them), and the block's rows 0..r. Query head h reads
//      key/value head h / (num_heads / num_kv_heads); score = q . k / sqrt(head_dim); one softmax over both sets;
//      out[h] = sum(w * v) * sigmoid(gate[h]); o_proj takes the heads joined in order.
//   4. Hidden-Correction, draft j from row j + 1 and the token before it (a, then draft j - 1):
//      c = rms(z[j+1], hidden_norm) :: rms(e(prev), embed_norm); h' = z[j+1] + down_proj(silu(gate_proj(c)) * up_proj(c)).
//   5. The target's output head over h' gives the draft; sigmoid(confidence_head.proj(h')) is the drafter's estimate
//      of the chance the target accepts it. Row 0's output is unused.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace basalt::dfly2 {

constexpr const char* kCheckpointFormat = "basalt-dfly2-drafter";   ///< value of the `format` key
constexpr const char* kCheckpointVersion = "1";                     ///< value of the `version` key
constexpr int kCheckpointMaxLayers = 16;                            ///< draft layers a checkpoint can name

// The metadata keys. Every value is a string: an integer in decimal, a real as printf's %.9g, a list comma-separated.
constexpr const char* kKeyFormat = "format";
constexpr const char* kKeyVersion = "version";
constexpr const char* kKeyHidden = "dfly2.hidden_size";                  ///< residual width
constexpr const char* kKeyIntermediate = "dfly2.intermediate_size";      ///< a layer MLP's inner width
constexpr const char* kKeyCorrection = "dfly2.correction_intermediate_size";   ///< Hidden-Correction's inner width
constexpr const char* kKeyLayers = "dfly2.num_layers";
constexpr const char* kKeyHeads = "dfly2.num_heads";                     ///< query heads
constexpr const char* kKeyKvHeads = "dfly2.num_kv_heads";                ///< key/value heads of a cached cell
constexpr const char* kKeyHeadDim = "dfly2.head_dim";
constexpr const char* kKeyRopeDims = "dfly2.rope_dims";                  ///< rotated values at the start of a head
constexpr const char* kKeyRopeBase = "dfly2.rope_base";
constexpr const char* kKeyAttnNormEps = "dfly2.attn_norm_eps";           ///< eps of q_norm and k_norm (the target's)
constexpr const char* kKeyRmsNormEps = "dfly2.rms_norm_eps";             ///< eps of every other norm
constexpr const char* kKeyVocab = "dfly2.vocab_size";                    ///< rows of the target's embedding and head
constexpr const char* kKeyBlock = "dfly2.block_size";                    ///< rows of a trained block: the anchor and the drafts
constexpr const char* kKeyMaskToken = "dfly2.mask_token_id";
constexpr const char* kKeySelectionWidth = "dfly2.selection_width";      ///< most cells a row's selection holds
constexpr const char* kKeyTargetLayers = "dfly2.target_layers";          ///< the target layer each draft layer reads
constexpr const char* kKeyWeightDtype = "dfly2.weight_dtype";            ///< BF16 or F32: what the forward rounded to
constexpr const char* kKeyStep = "dfly2.step";                           ///< optimiser steps behind the weights (0: untrained)
constexpr const char* kKeyPackName = "dfly2.pack_name";                  ///< file name of the pack the drafter was built on
constexpr const char* kKeyPackBytes = "dfly2.pack_bytes";                ///< that pack's size
constexpr const char* kKeyPackMetaSha256 = "dfly2.pack_meta_sha256";     ///< its `meta` section's digest, 64 hex digits

/// A drafter's shape and settings: everything the tensors' shapes and the forward depend on.
struct DrafterShape {
    int64_t hidden = 0;            ///< residual width
    int64_t intermediate = 0;      ///< a layer MLP's inner width
    int64_t correction = 0;        ///< Hidden-Correction's inner width
    int64_t n_layers = 0;
    int64_t n_head = 0;            ///< query heads, a multiple of n_head_kv
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t n_rot = 0;             ///< rotated values per head, even and at most head_dim
    double rope_base = 0;
    double attn_norm_eps = 0;
    double rms_norm_eps = 0;
    int64_t vocab = 0;
    int64_t block = 0;             ///< rows of a block, at least 2
    int64_t mask_token_id = 0;
    int64_t sel_cap = 0;           ///< most cells a row's selection holds
    std::vector<int> target_layers;   ///< n_layers entries, ascending
};

/// One tensor of the file.
struct DrafterTensor {
    std::string name;
    std::vector<int64_t> shape;    ///< outermost first
    bool matrix = false;           ///< a projection ([out][in]); false for a norm weight, a gate or a bias
};

/// Checks a shape; false with the reason in `err`.
inline bool drafter_check_shape(const DrafterShape& s, std::string& err) {
    if (s.hidden <= 0 || s.intermediate <= 0 || s.correction <= 0 || s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.vocab <= 0 || s.sel_cap <= 0) {
        err = "drafter shape has a dimension that is not positive";
        return false;
    }
    if (s.n_layers <= 0 || s.n_layers > kCheckpointMaxLayers || (int64_t) s.target_layers.size() != s.n_layers) { err = "drafter has " + std::to_string(s.n_layers) + " layers and " + std::to_string(s.target_layers.size()) + " target layers"; return false; }
    for (size_t i = 0; i < s.target_layers.size(); ++i)
        if (s.target_layers[i] < 0 || (i != 0 && s.target_layers[i] <= s.target_layers[i - 1])) { err = "drafter target layers are not ascending"; return false; }
    if (s.n_head % s.n_head_kv != 0) { err = "drafter query heads (" + std::to_string(s.n_head) + ") are not a multiple of its key/value heads (" + std::to_string(s.n_head_kv) + ")"; return false; }
    if (s.n_rot < 0 || s.n_rot > s.head_dim || s.n_rot % 2 != 0) { err = "drafter rotated width " + std::to_string(s.n_rot) + " does not fit a head of " + std::to_string(s.head_dim); return false; }
    if (!(s.rope_base > 1.0) || !(s.attn_norm_eps > 0.0) || !(s.rms_norm_eps > 0.0)) { err = "drafter rope base or norm eps is out of range"; return false; }
    if (s.block < 2 || s.block > 4096) { err = "drafter block size " + std::to_string(s.block) + " is out of range"; return false; }
    if (s.mask_token_id < 0 || s.mask_token_id >= s.vocab) { err = "drafter mask token " + std::to_string(s.mask_token_id) + " is outside the vocabulary of " + std::to_string(s.vocab); return false; }
    return true;
}

/// Every tensor of a drafter of shape `s` (a checked shape), in the file's canonical order.
inline std::vector<DrafterTensor> drafter_tensors(const DrafterShape& s) {
    const int64_t D = s.hidden, I = s.intermediate, C = s.correction, dh = s.head_dim;
    std::vector<DrafterTensor> t;
    for (int64_t i = 0; i < s.n_layers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        t.push_back({p + "input_layernorm.weight", {D}, false});
        t.push_back({p + "self_attn.q_proj.weight", {2 * s.n_head * dh, D}, true});
        t.push_back({p + "self_attn.k_proj.weight", {s.n_head_kv * dh, D}, true});
        t.push_back({p + "self_attn.v_proj.weight", {s.n_head_kv * dh, D}, true});
        t.push_back({p + "self_attn.o_proj.weight", {D, s.n_head * dh}, true});
        t.push_back({p + "self_attn.q_norm.weight", {dh}, false});
        t.push_back({p + "self_attn.k_norm.weight", {dh}, false});
        t.push_back({p + "post_attention_layernorm.weight", {D}, false});
        t.push_back({p + "mlp.gate_proj.weight", {I, D}, true});
        t.push_back({p + "mlp.up_proj.weight", {I, D}, true});
        t.push_back({p + "mlp.down_proj.weight", {D, I}, true});
    }
    t.push_back({"final_norm.weight", {D}, false});
    t.push_back({"mtp_fusion.enorm.weight", {D}, false});
    t.push_back({"mtp_fusion.hnorm.weight", {D}, false});
    t.push_back({"mtp_fusion.eh_proj.weight", {D, 2 * D}, true});
    t.push_back({"mtp_fusion.gate", {D}, false});
    t.push_back({"hidden_correction.hidden_norm.weight", {D}, false});
    t.push_back({"hidden_correction.embed_norm.weight", {D}, false});
    t.push_back({"hidden_correction.gate_proj.weight", {C, 2 * D}, true});
    t.push_back({"hidden_correction.up_proj.weight", {C, 2 * D}, true});
    t.push_back({"hidden_correction.down_proj.weight", {D, C}, true});
    t.push_back({"confidence_head.proj.weight", {1, D}, true});
    t.push_back({"confidence_head.proj.bias", {1}, false});
    return t;
}

namespace detail {

/// printf's %.9g of `v`.
inline std::string real_text(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

/// The whole of `text` as a decimal integer; false when it is not one.
inline bool parse_int(const std::string& text, int64_t& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') return false;
    out = (int64_t) v;
    return true;
}

/// The whole of `text` as a real; false when it is not one.
inline bool parse_real(const std::string& text, double& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    out = std::strtod(text.c_str(), &end);
    return end != nullptr && *end == '\0';
}

}  // namespace detail

/// The shape's metadata keys and the format and version keys. The caller adds dfly2.weight_dtype, dfly2.step and
/// the pack's identity.
inline std::map<std::string, std::string> drafter_metadata(const DrafterShape& s) {
    std::map<std::string, std::string> m;
    m[kKeyFormat] = kCheckpointFormat;
    m[kKeyVersion] = kCheckpointVersion;
    m[kKeyHidden] = std::to_string(s.hidden);
    m[kKeyIntermediate] = std::to_string(s.intermediate);
    m[kKeyCorrection] = std::to_string(s.correction);
    m[kKeyLayers] = std::to_string(s.n_layers);
    m[kKeyHeads] = std::to_string(s.n_head);
    m[kKeyKvHeads] = std::to_string(s.n_head_kv);
    m[kKeyHeadDim] = std::to_string(s.head_dim);
    m[kKeyRopeDims] = std::to_string(s.n_rot);
    m[kKeyRopeBase] = detail::real_text(s.rope_base);
    m[kKeyAttnNormEps] = detail::real_text(s.attn_norm_eps);
    m[kKeyRmsNormEps] = detail::real_text(s.rms_norm_eps);
    m[kKeyVocab] = std::to_string(s.vocab);
    m[kKeyBlock] = std::to_string(s.block);
    m[kKeyMaskToken] = std::to_string(s.mask_token_id);
    m[kKeySelectionWidth] = std::to_string(s.sel_cap);
    std::string layers;
    for (size_t i = 0; i < s.target_layers.size(); ++i) layers += (i != 0 ? "," : "") + std::to_string(s.target_layers[i]);
    m[kKeyTargetLayers] = layers;
    return m;
}

/// Reads a shape back from a file's metadata and checks it; false with `err` for a missing or malformed key, another
/// format or another version.
inline bool drafter_shape_from_metadata(const std::map<std::string, std::string>& m, DrafterShape& out, std::string& err) {
    const auto text = [&](const char* key, std::string& v) {
        const auto it = m.find(key);
        if (it == m.end()) { err = std::string("checkpoint metadata has no ") + key; return false; }
        v = it->second;
        return true;
    };
    const auto integer = [&](const char* key, int64_t& v) {
        std::string t;
        if (!text(key, t)) return false;
        if (!detail::parse_int(t, v)) { err = std::string("checkpoint metadata ") + key + " is not an integer: " + t; return false; }
        return true;
    };
    const auto real = [&](const char* key, double& v) {
        std::string t;
        if (!text(key, t)) return false;
        if (!detail::parse_real(t, v)) { err = std::string("checkpoint metadata ") + key + " is not a number: " + t; return false; }
        return true;
    };
    std::string format, version, layers;
    if (!text(kKeyFormat, format) || !text(kKeyVersion, version)) return false;
    if (format != kCheckpointFormat) { err = "checkpoint format is '" + format + "', not " + kCheckpointFormat; return false; }
    if (version != kCheckpointVersion) { err = "checkpoint version " + version + ", this build reads " + kCheckpointVersion; return false; }
    out = DrafterShape{};
    if (!integer(kKeyHidden, out.hidden) || !integer(kKeyIntermediate, out.intermediate) || !integer(kKeyCorrection, out.correction) ||
        !integer(kKeyLayers, out.n_layers) || !integer(kKeyHeads, out.n_head) || !integer(kKeyKvHeads, out.n_head_kv) ||
        !integer(kKeyHeadDim, out.head_dim) || !integer(kKeyRopeDims, out.n_rot) || !real(kKeyRopeBase, out.rope_base) ||
        !real(kKeyAttnNormEps, out.attn_norm_eps) || !real(kKeyRmsNormEps, out.rms_norm_eps) || !integer(kKeyVocab, out.vocab) ||
        !integer(kKeyBlock, out.block) || !integer(kKeyMaskToken, out.mask_token_id) || !integer(kKeySelectionWidth, out.sel_cap) ||
        !text(kKeyTargetLayers, layers))
        return false;
    // the comma-separated list, each entry a whole integer
    size_t at = 0;
    while (at <= layers.size()) {
        const size_t comma = layers.find(',', at);
        const std::string item = layers.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        int64_t v = 0;
        if (!detail::parse_int(item, v) || v < 0 || v > 1 << 20) { err = std::string("checkpoint metadata ") + kKeyTargetLayers + " is malformed: " + layers; return false; }
        out.target_layers.push_back((int) v);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return drafter_check_shape(out, err);
}

}  // namespace basalt::dfly2
