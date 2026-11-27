// tests/dflash/meta_test.cpp - a block drafter's shape from a pack's metadata and the shapes it refuses, and its tensors
// by their pack names and sizes, at the geometry of the DFlash 2 drafter released for Qwen3.8-Flash-Next (a target 2560
// wide, 248320 tokens). Host only.
#include "basalt/dflash/weights.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace basalt;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

constexpr int H = 2560;
constexpr int64_t kVocab = 248320;

MetaValue num(uint64_t v) {
    MetaValue m;
    m.type = MetaType::U32;
    m.u = v;
    return m;
}
MetaValue real(double v) {
    MetaValue m;
    m.type = MetaType::F32;
    m.f = v;
    return m;
}
MetaValue list(std::initializer_list<int> v) {
    MetaValue m;
    m.type = MetaType::ARRAY;
    for (int x : v) m.items.push_back(num((uint64_t) x));
    m.count = m.items.size();
    return m;
}

// the released drafter's settings as a converter writes them, under the architecture `arch`
std::map<std::string, MetaValue> released(const std::string& arch) {
    const std::string p = arch + ".dflash.";
    std::map<std::string, MetaValue> kv;
    kv[p + "block_count"] = num(5);
    kv[p + "block_size"] = num(8);
    kv[p + "mask_token_id"] = num(248077);
    kv[p + "target_layers"] = list({5, 15, 25, 35, 45});
    kv[p + "conv_kernel"] = num(2);
    kv[p + "conv_group"] = num(16);
    kv[p + "selector_rank"] = num(256);
    kv[p + "selector_top_k"] = num(16);
    kv[p + "sliding_window"] = num(2048);
    kv[p + "attention.head_count"] = num(32);
    kv[p + "attention.head_count_kv"] = num(8);
    kv[p + "attention.key_length"] = num(128);
    kv[p + "feed_forward_length"] = num(7680);
    kv[p + "rope.freq_base"] = real(10000.0);
    kv[p + "attention.layer_norm_rms_epsilon"] = real(1e-6);
    return kv;
}

// read_meta of the released settings with one key changed (`value` null: the key left out)
bool reads(const char* key, const MetaValue* value, std::string* why = nullptr) {
    std::map<std::string, MetaValue> kv = released("arch");
    if (value) kv[std::string("arch.dflash.") + key] = *value;
    else kv.erase(std::string("arch.dflash.") + key);
    dflash::Meta m;
    std::string err;
    const bool ok = dflash::read_meta(kv, "arch", H, m, err);
    if (why) *why = err;
    return ok;
}
bool refuses(const char* key, uint64_t value) {
    const MetaValue v = num(value);
    std::string err;
    return !reads(key, &v, &err) && err == "arch: the pack's block drafter has a shape the engine does not run";
}

void test_meta() {
    dflash::Meta m;
    std::string err;
    check(dflash::read_meta(released("arch"), "arch", H, m, err) && err.empty(), "the released settings are read");
    check(m.n_embd == H && m.block == 8 && m.mask_id == 248077 && m.conv_group == 16 && m.rank == 256 && m.top_k == 16 &&
              m.window == 2048,
          "block, mask token, convolution groups, selector and window");
    check(m.n_head == 32 && m.n_kv == 8 && m.head_dim == 128 && m.theta == 10000.0f && m.eps == 1e-6f, "heads, rope base and epsilon");
    check(m.target_layers == (std::vector<int>{5, 15, 25, 35, 45}) && m.n_layers == 5 && m.n_ff == 7680, "target layers, layers and FFN width");
    // the keys are the architecture's own: another architecture's are not read
    dflash::Meta other;
    check(!dflash::read_meta(released("other"), "arch", H, other, err) && other.block == 0, "another architecture's keys are not this one's");

    check(refuses("block_size", 9) && refuses("block_size", 1), "a block of 9 rows or of 1 is refused");
    check(refuses("conv_kernel", 3), "a three-tap convolution is refused");
    check(refuses("attention.key_length", 96), "96-wide heads are refused");
    check(refuses("conv_group", 0), "no convolution groups is refused, not divided by");
    check(refuses("conv_group", 48), "groups that do not divide the width are refused");
    check(refuses("selector_top_k", 21) && refuses("selector_top_k", 0), "more candidates than a position keeps, or none, is refused");
    check(refuses("attention.head_count_kv", 0) && refuses("attention.head_count_kv", 5), "no KV heads, or uneven groups, is refused");
    check(refuses("block_count", 0) && refuses("feed_forward_length", 0) && refuses("selector_rank", 0) && refuses("sliding_window", 0),
          "a count of zero is refused");
    check(!reads("target_layers", nullptr) && !reads("block_size", nullptr), "a missing key is refused");
    // a refused read leaves the caller's shape as it was
    dflash::Meta kept;
    kept.block = 4;
    std::map<std::string, MetaValue> bad = released("arch");
    bad["arch.dflash.block_size"] = num(9);
    check(!dflash::read_meta(bad, "arch", H, kept, err) && kept.block == 4, "a refused read changes nothing");
    check(!dflash::read_meta(released("arch"), "arch", 0, kept, err), "a target of no width is refused");
}

// what a model's loaders were asked for, in order: the name and the sizes
struct Asked {
    std::string name;
    int64_t rows, cols;   // a matrix; cols 0: a plain tensor of `rows` values
};

dflash::Loaders recorder(std::vector<Asked>& asked, const std::string& fail = {}) {
    static const float f32 = 0.0f;
    static const uint16_t b16 = 0;
    dflash::Loaders l;
    l.floats = [&asked, fail](const std::string& name, int64_t elems) -> const float* {
        asked.push_back({name, elems, 0});
        return name == fail ? nullptr : &f32;
    };
    l.bf16 = [&asked, fail](const std::string& name, int64_t elems) -> const uint16_t* {
        asked.push_back({name, elems, 0});
        return name == fail ? nullptr : &b16;
    };
    l.linear = [&asked, fail](const std::string& base, int64_t rows, int64_t cols, model::Linear& out) {
        asked.push_back({base, rows, cols});
        if (base == fail) return false;
        out.type = 8;
        out.w = &b16;
        out.rows = rows;
        out.cols = cols;
        return true;
    };
    return l;
}

const Asked* find(const std::vector<Asked>& asked, const std::string& name) {
    for (const Asked& a : asked)
        if (a.name == name) return &a;
    return nullptr;
}
bool is(const std::vector<Asked>& asked, const std::string& name, int64_t rows, int64_t cols) {
    const Asked* a = find(asked, name);
    return a && a->rows == rows && a->cols == cols;
}

void test_weights() {
    dflash::BlockDrafter d;
    std::string err;
    check(dflash::read_meta(released("arch"), "arch", H, d, err), "the shape reads into a drafter");
    std::vector<Asked> asked;
    check(dflash::read_weights(d, kVocab, recorder(asked)) && d.present, "every tensor is placed");
    // the checkpoint's 81 tensors, its one feature projection as a block per target layer
    check(asked.size() == 85 && d.fc.size() == 5 && d.layers.size() == 5, "85 pack tensors: 5 projection blocks, 2 norms, 15 a layer, 3 of the selector");
    check(asked.front().name == "dflash.fc.0" && asked.back().name == "dflash.selector.succ", "the projection first, the selector last");
    check(is(asked, "dflash.fc.4", H, H) && is(asked, "dflash.hidden_norm.weight", H, 0) && is(asked, "dflash.norm.weight", H, 0),
          "the projection blocks and the two norms");
    const std::string p = "dflash.blk.4.";
    check(is(asked, p + "attn_q", 4096, H) && is(asked, p + "attn_k", 1024, H) && is(asked, p + "attn_v", 1024, H) &&
              is(asked, p + "attn_output", H, 4096),
          "a layer's attention projections");
    check(is(asked, p + "ffn_gate", 7680, H) && is(asked, p + "ffn_up", 7680, H) && is(asked, p + "ffn_down", H, 7680), "its FFN");
    check(is(asked, p + "attn_conv", 640, H) && is(asked, p + "ffn_conv", 640, H) && is(asked, p + "attn_conv.base", 4 * H, 0) &&
              is(asked, p + "ffn_conv.base", 4 * H, 0),
          "its convolutions: 2 x 2 taps a group, and their base kernels");
    check(is(asked, p + "attn_norm.weight", H, 0) && is(asked, p + "post_attention_norm.weight", H, 0) &&
              is(asked, p + "attn_q_norm.weight", 128, 0) && is(asked, p + "attn_k_norm.weight", 128, 0),
          "its norms");
    check(is(asked, "dflash.selector.hidden.weight", 256 * (int64_t) H, 0) && is(asked, "dflash.selector.pred", 256 * kVocab, 0) &&
              is(asked, "dflash.selector.succ", 256 * kVocab, 0),
          "the selector's projection and codebooks");
    check(d.layers[4].q.rows == 4096 && d.layers[4].conv_f.rows == 640 && d.fc[4].cols == H, "the matrices land in their fields");

    // a tensor the pack lacks fails the load and leaves the drafter absent
    for (const char* missing : {"dflash.fc.2", "dflash.norm.weight", "dflash.blk.3.ffn_down", "dflash.selector.pred"}) {
        std::vector<Asked> got;
        dflash::BlockDrafter e;
        static_cast<dflash::Meta&>(e) = d;
        const bool ok = dflash::read_weights(e, kVocab, recorder(got, missing));
        check(!ok && !e.present && find(got, missing) != nullptr, missing);
    }
    dflash::BlockDrafter none;
    static_cast<dflash::Meta&>(none) = d;
    check(!dflash::read_weights(none, kVocab, dflash::Loaders{}) && !none.present, "no loaders: nothing is placed");
    check(!dflash::read_weights(none, 0, recorder(asked)) && !none.present, "no vocabulary: nothing is placed");
}

}  // namespace

int main() {
    test_meta();
    test_weights();
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
