// src/dflash/meta.cpp - a block drafter's shape from a pack's metadata, and its tensors by their pack names.
#include "basalt/dflash/weights.hpp"

namespace basalt::dflash {

bool read_meta(const std::map<std::string, MetaValue>& kv, const std::string& arch, int n_embd, Meta& out, std::string& err) {
    const std::string prefix = arch + ".dflash.";
    const auto find = [&](const char* k) -> const MetaValue* {
        const auto it = kv.find(prefix + k);
        return it == kv.end() ? nullptr : &it->second;
    };
    // a key the pack lacks reads as 0, which the checks below refuse where a count must be positive
    const auto u = [&](const char* k) { const MetaValue* v = find(k); return v ? (int) v->u : 0; };
    const auto f = [&](const char* k) { const MetaValue* v = find(k); return v ? (float) v->num() : 0.0f; };
    Meta d;
    d.n_embd = n_embd;
    d.block = u("block_size");
    d.mask_id = u("mask_token_id");
    d.conv_group = u("conv_group");
    d.rank = u("selector_rank");
    d.top_k = u("selector_top_k");
    d.window = u("sliding_window");
    d.n_head = u("attention.head_count");
    d.n_kv = u("attention.head_count_kv");
    d.head_dim = u("attention.key_length");
    d.theta = f("rope.freq_base");
    d.eps = f("attention.layer_norm_rms_epsilon");
    d.n_layers = u("block_count");
    d.n_ff = u("feed_forward_length");
    if (const MetaValue* t = find("target_layers"))
        for (const MetaValue& x : t->items) d.target_layers.push_back((int) x.u);
    // the shapes the kernels run: a block of up to 8 rows, a two-tap convolution over whole groups, 128- or 256-wide heads
    // in even groups, the selector's candidates within kDraftQ
    const bool shape = d.block >= 2 && d.block <= 8 && !d.target_layers.empty() && u("conv_kernel") == 2 && d.conv_group > 0 &&
                       n_embd > 0 && n_embd % d.conv_group == 0 && (d.head_dim == 128 || d.head_dim == 256) && d.n_kv > 0 &&
                       d.n_head > 0 && d.n_head % d.n_kv == 0 && d.top_k > 0 && d.top_k <= kDraftQ;
    // and every count a buffer is sized by
    const bool counts = d.rank > 0 && d.window > 0 && d.n_layers > 0 && d.n_ff > 0 && d.mask_id >= 0;
    if (!shape || !counts) {
        err = arch + ": the pack's block drafter has a shape the engine does not run";
        return false;
    }
    out = std::move(d);
    return true;
}

bool read_weights(BlockDrafter& d, int64_t n_vocab, const Loaders& load) {
    d.fc.clear();
    d.layers.clear();
    d.present = false;
    if (!load.floats || !load.bf16 || !load.linear || d.conv_group <= 0 || n_vocab <= 0) return false;
    const int64_t H = d.n_embd, AQ = (int64_t) d.n_head * d.head_dim, KV = (int64_t) d.n_kv * d.head_dim;
    const int64_t conv = 2 * 2 * (H / d.conv_group);   // two taps before the sublayer and two after, per group
    for (size_t j = 0; j < d.target_layers.size(); ++j) {
        d.fc.emplace_back();
        if (!load.linear("dflash.fc." + std::to_string(j), H, H, d.fc.back())) return false;
    }
    d.hidden_norm = load.floats("dflash.hidden_norm.weight", H);
    d.norm = load.floats("dflash.norm.weight", H);
    if (!d.hidden_norm || !d.norm) return false;
    for (int l = 0; l < d.n_layers; ++l) {
        const std::string p = "dflash.blk." + std::to_string(l) + ".";
        BlockDrafter::Layer L;
        L.attn_norm = load.floats(p + "attn_norm.weight", H);
        L.ffn_norm = load.floats(p + "post_attention_norm.weight", H);
        L.q_norm = load.floats(p + "attn_q_norm.weight", d.head_dim);
        L.k_norm = load.floats(p + "attn_k_norm.weight", d.head_dim);
        L.base_a = load.floats(p + "attn_conv.base", 4 * H);
        L.base_f = load.floats(p + "ffn_conv.base", 4 * H);
        if (!L.attn_norm || !L.ffn_norm || !L.q_norm || !L.k_norm || !L.base_a || !L.base_f ||
            !load.linear(p + "attn_q", AQ, H, L.q) || !load.linear(p + "attn_k", KV, H, L.k) ||
            !load.linear(p + "attn_v", KV, H, L.v) || !load.linear(p + "attn_output", H, AQ, L.o) ||
            !load.linear(p + "ffn_gate", d.n_ff, H, L.gate) || !load.linear(p + "ffn_up", d.n_ff, H, L.up) ||
            !load.linear(p + "ffn_down", H, d.n_ff, L.down) || !load.linear(p + "attn_conv", conv, H, L.conv_a) ||
            !load.linear(p + "ffn_conv", conv, H, L.conv_f))
            return false;
        d.layers.push_back(L);
    }
    d.sel_h = load.bf16("dflash.selector.hidden.weight", (int64_t) d.rank * H);
    d.pred = load.bf16("dflash.selector.pred", (int64_t) d.rank * n_vocab);
    d.succ = load.bf16("dflash.selector.succ", (int64_t) d.rank * n_vocab);
    if (!d.sel_h || !d.pred || !d.succ) return false;
    d.present = true;
    return true;
}

}  // namespace basalt::dflash
