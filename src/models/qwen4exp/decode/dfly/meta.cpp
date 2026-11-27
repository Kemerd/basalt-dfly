// src/models/qwen4exp/decode/dfly/meta.cpp - the DFly2 drafter's host side that needs no device: the tensors it holds, its
// check of the pack's record against the target, and which starts can run it.
#include "basalt/models/qwen4exp/decode/dfly.hpp"

#include <cmath>

namespace basalt::core {

namespace {

constexpr char kFormat[] = "dfly2-qwen4exp";
constexpr uint64_t kQ8Block = 32;   // values of a Q8_0 block: a projection's input width is whole blocks

}  // namespace

uint64_t DflySpec::elems() const {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

std::vector<DflySpec> dfly_specs(const DflyDims& d) {
    const uint64_t D = (uint64_t) d.n_embd, I = (uint64_t) d.ffn, C = (uint64_t) d.corr_ffn;
    const uint64_t Q = (uint64_t) d.n_head * (uint64_t) d.head_dim, KV = (uint64_t) d.n_head_kv * (uint64_t) d.head_dim;
    const uint64_t dh = (uint64_t) d.head_dim;
    std::vector<DflySpec> v;
    auto vec = [&](std::string n, std::vector<uint64_t> s) { v.push_back({std::move(n), std::move(s), true}); };
    auto q8 = [&](std::string n, std::vector<uint64_t> s) { v.push_back({std::move(n), std::move(s), false}); };
    for (int i = 0; i < d.layers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        vec(p + "input_layernorm.weight", {D});
        // the query rows carry each head's query half, then its gate half
        q8(p + "self_attn.q_proj.weight", {2 * Q, D});
        q8(p + "self_attn.k_proj.weight", {KV, D});
        q8(p + "self_attn.v_proj.weight", {KV, D});
        q8(p + "self_attn.o_proj.weight", {D, Q});
        vec(p + "self_attn.q_norm.weight", {dh});
        vec(p + "self_attn.k_norm.weight", {dh});
        vec(p + "post_attention_layernorm.weight", {D});
        q8(p + "mlp.gate_proj.weight", {I, D});
        q8(p + "mlp.up_proj.weight", {I, D});
        q8(p + "mlp.down_proj.weight", {D, I});
    }
    vec("final_norm.weight", {D});
    vec("mtp_fusion.enorm.weight", {D});
    vec("mtp_fusion.hnorm.weight", {D});
    q8("mtp_fusion.eh_proj.weight", {D, 2 * D});
    vec("mtp_fusion.gate", {D});
    vec("hidden_correction.hidden_norm.weight", {D});
    vec("hidden_correction.embed_norm.weight", {D});
    q8("hidden_correction.gate_proj.weight", {C, 2 * D});
    q8("hidden_correction.up_proj.weight", {C, 2 * D});
    q8("hidden_correction.down_proj.weight", {D, C});
    vec("confidence_head.proj.weight", {1, D});
    vec("confidence_head.proj.bias", {1});
    return v;
}

bool dfly_check_record(const Json& meta, const dfly2::Record& rec, const ModelGeometry& g, const DflyTarget& t, DflyDims& d,
                       std::string& err) {
    d = DflyDims{};
    if (!meta.is_object() || !rec.present) {
        err = "meta.dfly2 is not a drafter's record";
        return false;
    }
    // the record is this model's, written for the forward this drafter runs
    const Json* fmt = meta.get("format");
    if (!fmt || !fmt->is_string() || fmt->s != kFormat) {
        err = std::string("meta.dfly2: the format is not ") + kFormat;
        return false;
    }
    const Json *causal = meta.get("causal_block"), *fusion = meta.get("mtp_fusion");
    if (!causal || causal->type != Json::Type::Bool || !causal->b || !fusion || !fusion->is_string() || fusion->s != "slot0") {
        err = "meta.dfly2: the block is not causal or the fusion is not of the anchor's row";
        return false;
    }
    if (rec.local_window != 0) {
        err = "meta.dfly2: a local window of the context is not run";
        return false;
    }
    auto num = [&](const char* k) {
        const Json* j = meta.get(k);
        return j && j->is_number() ? j->num() : 0.0;
    };
    d.layers = rec.layers;
    d.mask_id = rec.mask_id;
    d.block = rec.block;
    d.n_embd = (int) num("hidden_size");
    d.ffn = (int) num("intermediate_size");
    d.corr_ffn = (int) num("correction_intermediate_size");
    d.n_head = (int) num("num_heads");
    d.n_head_kv = (int) num("num_kv_heads");
    d.head_dim = (int) num("head_dim");
    d.n_rot = (int) num("rope_dims");
    d.sel_width = (int) num("selection_width");
    d.rope_base = (float) num("rope_base");
    d.attn_eps = (float) num("attn_norm_eps");
    d.rms_eps = (float) num("rms_norm_eps");
    d.step = (int64_t) num("step");
    d.vocab = (int) t.n_vocab;
    d.target_layers = rec.kv_layers;
    // each draft layer reads the cache and the selection of one target layer, which must be a full-attention one
    if (rec.kv_layers != rec.idx_layers) {
        err = "meta.dfly2: a draft layer's cache and selection are of different target layers";
        return false;
    }
    for (int l : d.target_layers) {
        if (l < 0 || l >= g.n_layers || !is_qsa_layer(g, l)) {
            err = "meta.dfly2: target layer " + std::to_string(l) + " is not a full-attention layer";
            return false;
        }
    }
    if (d.n_embd != g.n_embd || d.n_head != g.n_head || d.n_head_kv != g.n_head_kv || d.head_dim != g.head_dim ||
        d.n_rot != t.n_rot) {
        err = "meta.dfly2: the width, the head counts, the head width or the rotary dims are not the target's";
        return false;
    }
    if (d.mask_id >= t.n_vocab) {
        err = "meta.dfly2: the mask token " + std::to_string(d.mask_id) + " is outside the vocabulary";
        return false;
    }
    if (d.block < 2 || d.block > kDflyBlockMax) {
        err = "meta.dfly2: a block of " + std::to_string(d.block) + " rows (2.." + std::to_string(kDflyBlockMax) + " are run)";
        return false;
    }
    if (d.sel_width != t.sel_width) {
        err = "meta.dfly2: the selection width " + std::to_string(d.sel_width) + " is not the target's " +
              std::to_string(t.sel_width);
        return false;
    }
    // every projection is read in Q8_0 blocks of its input row
    if (d.ffn <= 0 || d.corr_ffn <= 0 || (uint64_t) d.ffn % kQ8Block != 0 || (uint64_t) d.corr_ffn % kQ8Block != 0 ||
        (uint64_t) d.n_embd % kQ8Block != 0 || ((uint64_t) d.n_head * (uint64_t) d.head_dim) % kQ8Block != 0) {
        err = "meta.dfly2: a projection's input width is not a multiple of " + std::to_string(kQ8Block);
        return false;
    }
    if (!(d.rms_eps > 0.0f) || !(d.attn_eps > 0.0f) || !(d.rope_base > 1.0f) || !std::isfinite(d.rope_base)) {
        err = "meta.dfly2: rms_norm_eps, attn_norm_eps or rope_base are missing";
        return false;
    }
    for (const auto& [name, x] : rec.tensors) {
        if (x.dtype != "BF16") {
            err = "meta.dfly2: " + name + " is not BF16";
            return false;
        }
    }
    for (const DflySpec& s : dfly_specs(d)) {
        const auto it = rec.tensors.find(s.name);
        if (it == rec.tensors.end()) {
            err = "meta.dfly2: no tensor " + s.name;
            return false;
        }
        if (it->second.shape != s.shape || it->second.bytes != s.elems() * 2) {
            err = "meta.dfly2: tensor " + s.name + " has another shape than the drafter's";
            return false;
        }
    }
    return true;
}

bool dfly_read_pack(const ModelFile& file, const ModelGeometry& g, const DflyTarget& t, DflyPack& out, std::string& err) {
    out = DflyPack{};
    const ModelFile::Section* sec = file.find("dfly");
    const Json* meta = file.meta().get("dfly2");
    const bool has_meta = meta != nullptr && meta->is_object();
    // neither: the pack has no drafter
    if (sec == nullptr && !has_meta) return true;
    if (sec == nullptr || !has_meta) {
        err = "the dfly section and meta.dfly2 do not come together";
        return false;
    }
    if (!dfly2::parse_record(*meta, sec->offset, out.rec, err) || !dfly_check_record(*meta, out.rec, g, t, out.dims, err))
        return false;
    // a tensor outside the section would be read from another section's bytes
    for (const auto& [name, x] : out.rec.tensors) {
        if (x.offset < sec->offset || x.bytes > sec->bytes || x.offset - sec->offset > sec->bytes - x.bytes) {
            err = "meta.dfly2: " + name + " lies outside the dfly section";
            return false;
        }
    }
    out.present = true;
    return true;
}

std::string dfly_pass_over(const DflyDims& d, const DflyConfig& c) {
    // the block pass is captured for the persistent engine's windows on one card
    if (!c.serve) return "it drafts only under --serve";
    if (c.layer_split) return "a layer split leaves the paired layers' K/V on another card";
    // the K/V forms its attention does not read
    if (c.kv_q4) return "--kv q4_0 K/V is not read by its attention";
    if (c.kv_hybrid) return "--kv k8v4 K/V is not read by its attention";
    if (c.kv_resident) return "--kv-resident keeps cells out of the pools it gathers from";
    if (c.control_vector) return "a control vector changes the hidden states it was trained on";
    // its own rows are rotated as the cache's cells were
    if (c.rope_scaled) return "the rotary scaling is not none";
    if (c.rope_base != d.rope_base) return "the rotary base differs from the one it was trained with";
    if (!c.head_two_stage) return "the output head's type has no two-stage argmax";
    return {};
}

}  // namespace basalt::core
