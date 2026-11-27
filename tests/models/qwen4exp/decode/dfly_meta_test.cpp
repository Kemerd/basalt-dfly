// tests/models/qwen4exp/decode/dfly_meta_test.cpp - CPU tests of the DFly2 drafter's host side: the pack's record to dims,
// each record it refuses, the starts that pass it over, and the source row of a block in a solo and a batch window.
#include "basalt/models/qwen4exp/decode/dfly.hpp"

#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::core;

static int g_fail = 0;
#define CHECK(c)                                                                       \
    do {                                                                               \
        if (!(c)) {                                                                    \
            std::fprintf(stderr, "dfly_meta_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                                  \
        }                                                                              \
    } while (0)

namespace {

// a record as the converter writes it; the edits of a case are applied to these fields before it is printed
struct Rec {
    std::string format = "dfly2-qwen4exp", fusion = "slot0", dtype = "BF16";
    bool causal = true;
    int layers = 5, mask = 248319, block = 8, hidden = 2560, inter = 7680, corr = 2560, heads = 24, kv_heads = 2, head_dim = 256;
    int rope_dims = 64, sel = 2051, local_window = 0;
    double rope_base = 1e7, attn_eps = 1e-6, rms_eps = 1e-6;
    std::vector<int> kv = {31, 35, 39, 43, 47}, idx = {31, 35, 39, 43, 47};
    std::string drop;                          // a tensor left out
    std::string reshape;                       // a tensor given one more row
    std::string retype;                        // a tensor stored as F32
};

DflyDims dims_of(const Rec& r) {
    DflyDims d;
    d.layers = r.layers;
    d.n_embd = r.hidden;
    d.ffn = r.inter;
    d.corr_ffn = r.corr;
    d.n_head = r.heads;
    d.n_head_kv = r.kv_heads;
    d.head_dim = r.head_dim;
    return d;
}

std::string list(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

std::string text(const Rec& r) {
    std::string s = "{\"section\":\"dfly\",\"format\":\"" + r.format + "\",\"layers\":" + std::to_string(r.layers) +
                    ",\"mask_token_id\":" + std::to_string(r.mask) + ",\"target_kv_layers\":" + list(r.kv) +
                    ",\"target_index_layers\":" + list(r.idx) + ",\"target_index_topk\":2048,\"block_size\":" +
                    std::to_string(r.block) + ",\"hidden_size\":" + std::to_string(r.hidden) + ",\"intermediate_size\":" +
                    std::to_string(r.inter) + ",\"correction_intermediate_size\":" + std::to_string(r.corr) +
                    ",\"num_heads\":" + std::to_string(r.heads) + ",\"num_kv_heads\":" + std::to_string(r.kv_heads) +
                    ",\"head_dim\":" + std::to_string(r.head_dim) + ",\"rope_dims\":" + std::to_string(r.rope_dims) +
                    ",\"rope_base\":" + std::to_string(r.rope_base) + ",\"attn_norm_eps\":" + std::to_string(r.attn_eps) +
                    ",\"rms_norm_eps\":" + std::to_string(r.rms_eps) + ",\"selection_width\":" + std::to_string(r.sel) +
                    ",\"causal_block\":" + (r.causal ? "true" : "false") + ",\"mtp_fusion\":\"" + r.fusion + "\",\"step\":1200," +
                    (r.local_window ? "\"context_local_window\":" + std::to_string(r.local_window) + "," : std::string()) +
                    "\"tensors\":{";
    bool first = true;
    uint64_t off = 0;
    for (const DflySpec& t : dfly_specs(dims_of(r))) {
        if (t.name == r.drop) continue;
        std::vector<uint64_t> shape = t.shape;
        if (t.name == r.reshape) ++shape[0];
        uint64_t n = 1;
        for (uint64_t x : shape) n *= x;
        s += std::string(first ? "" : ",") + "\"" + t.name + "\":{\"dtype\":\"" + (t.name == r.retype ? "F32" : r.dtype) +
             "\",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
        s += "],\"offset\":" + std::to_string(off) + ",\"bytes\":" + std::to_string(n * 2) + "}";
        off += (n * 2 + 63) / 64 * 64;
        first = false;
    }
    return s + "}}";
}

const DflyTarget kTarget = {248320, 2051, 64};

// the record through the shared reader and this drafter's check
bool read(const Rec& r, DflyDims& d, std::string& err) {
    const std::string t = text(r);
    Json j;
    dfly2::Record rec;
    if (!parse_json(t.data(), t.size(), j, err) || !dfly2::parse_record(j, 1u << 20, rec, err)) return false;
    return dfly_check_record(j, rec, ModelGeometry{}, kTarget, d, err);
}

void test_record() {
    DflyDims d;
    std::string err;
    CHECK(read(Rec{}, d, err));
    CHECK(d.layers == 5 && d.n_embd == 2560 && d.ffn == 7680 && d.corr_ffn == 2560 && d.n_head == 24 && d.n_head_kv == 2);
    CHECK(d.head_dim == 256 && d.n_rot == 64 && d.block == 8 && d.sel_width == 2051 && d.mask_id == 248319 && d.vocab == 248320);
    CHECK(d.rope_base == 1e7f && d.rms_eps > 0.0f && d.attn_eps > 0.0f && d.step == 1200);
    CHECK(d.target_layers == std::vector<int>({31, 35, 39, 43, 47}));
    // the tensor set and the bytes it takes on the card: Q8_0 projections and FP32 vectors
    const std::vector<DflySpec> specs = dfly_specs(d);
    CHECK(specs.size() == 5 * 11 + 12);
    uint64_t q8 = 0, f32 = 0;
    for (const DflySpec& s : specs) (s.f32 ? f32 : q8) += s.elems();
    CHECK(q8 == 589824000ull && f32 == 46081ull);
    CHECK(specs[1].name == "layers.0.self_attn.q_proj.weight" && specs[1].shape == std::vector<uint64_t>({12288, 2560}));
    CHECK(specs[1].bytes() == 12288ull * 2560 / 32 * 34);
}

void test_refusals() {
    struct Case {
        const char* what;
        std::function<void(Rec&)> edit;
    };
    const std::vector<Case> cases = {
        {"another format", [](Rec& r) { r.format = "dfly2-other"; }},
        {"a block that is not causal", [](Rec& r) { r.causal = false; }},
        {"another fusion", [](Rec& r) { r.fusion = "all"; }},
        {"a local window", [](Rec& r) { r.local_window = 128; }},
        {"layer lists that disagree", [](Rec& r) { r.idx[2] = 43; }},
        {"a layer without full attention", [](Rec& r) { r.kv[0] = r.idx[0] = 30; }},
        {"a layer past the last", [](Rec& r) { r.kv[4] = r.idx[4] = 51; }},
        {"another width", [](Rec& r) { r.hidden = 2048; }},
        {"another head count", [](Rec& r) { r.heads = 16; }},
        {"another kv head count", [](Rec& r) { r.kv_heads = 4; }},
        {"another head width", [](Rec& r) { r.head_dim = 128; }},
        {"other rotary dims", [](Rec& r) { r.rope_dims = 128; }},
        {"a mask token at the vocabulary's size", [](Rec& r) { r.mask = 248320; }},
        {"a block above the widest", [](Rec& r) { r.block = 9; }},
        {"a block of one row", [](Rec& r) { r.block = 1; }},
        {"another selection width", [](Rec& r) { r.sel = 2048; }},
        {"an MLP width that is not whole blocks", [](Rec& r) { r.inter = 7696; }},
        {"a correction width that is not whole blocks", [](Rec& r) { r.corr = 2576; }},
        {"no norm epsilon", [](Rec& r) { r.rms_eps = 0; }},
        {"no rotary base", [](Rec& r) { r.rope_base = 0; }},
        {"a tensor that is not BF16", [](Rec& r) { r.retype = "layers.3.mlp.up_proj.weight"; }},
        {"a missing tensor", [](Rec& r) { r.drop = "hidden_correction.down_proj.weight"; }},
        {"a tensor of another shape", [](Rec& r) { r.reshape = "layers.4.self_attn.k_norm.weight"; }},
    };
    for (const Case& c : cases) {
        Rec r;
        c.edit(r);
        DflyDims d;
        std::string err;
        const bool ok = read(r, d, err);
        if (ok || err.empty()) {
            std::fprintf(stderr, "dfly_meta_test: FAILED a record with %s was taken\n", c.what);
            ++g_fail;
        }
    }
    // a record the shared reader refuses never reaches the check: more layers than lists
    Rec r;
    r.layers = 4;
    DflyDims d;
    std::string err;
    CHECK(!read(r, d, err) && !err.empty());
}

void test_pass_over() {
    DflyDims d;
    d.rope_base = 1e7f;
    DflyConfig ok;
    ok.serve = true;
    ok.rope_base = 1e7f;
    ok.head_two_stage = true;
    CHECK(dfly_pass_over(d, ok).empty());
    // each fact alone passes the drafter over, with words for the warning
    const std::vector<std::function<void(DflyConfig&)>> edits = {
        [](DflyConfig& c) { c.serve = false; },        [](DflyConfig& c) { c.layer_split = true; },
        [](DflyConfig& c) { c.kv_q4 = true; },         [](DflyConfig& c) { c.kv_hybrid = true; },
        [](DflyConfig& c) { c.kv_resident = true; },   [](DflyConfig& c) { c.control_vector = true; },
        [](DflyConfig& c) { c.rope_scaled = true; },   [](DflyConfig& c) { c.rope_base = 1e6f; },
        [](DflyConfig& c) { c.head_two_stage = false; },
    };
    std::map<std::string, int> seen;
    for (const auto& e : edits) {
        DflyConfig c = ok;
        e(c);
        const std::string why = dfly_pass_over(d, c);
        CHECK(!why.empty());
        ++seen[why];
    }
    CHECK(seen.size() == edits.size());   // a reason of its own for each
}

void test_source_row() {
    const int layers = 5;
    const unsigned every = (1u << layers) - 1u;
    unsigned sel_ok[dfly2::kTapRows] = {};
    // a solo window of 4 rows from position 100: rows 100..103 are kept
    const int rows = 4;
    for (int r = 0; r < rows; ++r) sel_ok[r] = every;
    const unsigned all = (1u << rows) - 1u;
    CHECK(dfly2::source_row(100, all, sel_ok, layers, 104) == 3);   // every row accepted: the last row is the source
    CHECK(dfly2::source_row(100, all, sel_ok, layers, 102) == 1);   // two rows kept
    CHECK(dfly2::source_row(100, all, sel_ok, layers, 105) == -1);  // the window does not hold row 104
    CHECK(dfly2::source_row(100, all, sel_ok, layers, 100) == -1);  // nor the row before its first
    CHECK(dfly2::source_row(100, all, sel_ok, layers - 1, 104) == -1);   // a layer's selection was not kept
    // a batch window: a slot's segment is rows r0 .. r0 + n of the window, at the slot's own positions
    const int r0 = 3, n = 2;
    const int64_t slot_pos0 = 7000;
    unsigned seg_ok[dfly2::kTapRows] = {};
    for (int r = 0; r < n; ++r) seg_ok[r] = every;
    const int in_seg = dfly2::source_row(slot_pos0, (1u << n) - 1u, seg_ok, layers, slot_pos0 + 2);
    CHECK(in_seg == 1 && r0 + in_seg == 4);   // the window's row 4
    CHECK(dfly2::source_row(slot_pos0, (1u << n) - 1u, seg_ok, layers, slot_pos0 + 3) == -1);   // past the slot's rows
    CHECK(dfly2::source_row(slot_pos0, (1u << n) - 1u, seg_ok, layers, 104) == -1);             // another slot's position
}

}  // namespace

int main() {
    test_record();
    test_refusals();
    test_pass_over();
    test_source_row();
    if (g_fail) {
        std::fprintf(stderr, "dfly_meta_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("dfly_meta_test: ok\n");
    return 0;
}
