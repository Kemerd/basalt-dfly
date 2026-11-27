// src/models/hy4flash/dfly_test.cpp - CPU tests of the DFly2 drafter's host side: the pack record as this drafter reads it,
// the tensors it must hold (the real checkpoint's shapes), and the flags.
#include "models/hy4flash/dfly.hpp"

#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::models::hy4flash;

static int g_fail = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "dfly_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                             \
        }                                                                         \
    } while (0)

namespace {

// tencent/Hy4-flash, model.dfly_layers.* of the last shard (safetensors order), layer 0 and the heads
struct Real {
    const char* name;
    std::vector<uint64_t> shape;
};
const std::vector<Real>& real_layer0() {
    static const std::vector<Real> v = {
        {"input_layernorm.weight", {3072}},
        {"mlp.down_proj.weight", {3072, 9216}},
        {"mlp.gate_proj.weight", {9216, 3072}},
        {"mlp.up_proj.weight", {9216, 3072}},
        {"post_attention_layernorm.weight", {3072}},
        {"self_attn.kv_a_layernorm.weight", {512}},
        {"self_attn.kv_a_proj_with_mqa.weight", {576, 3072}},
        {"self_attn.kv_b_proj.weight", {14336, 512}},
        {"self_attn.learnable_sink_param", {32}},
        {"self_attn.linear_gate.weight", {8192, 3072}},
        {"self_attn.o_proj.weight", {3072, 8192}},
        {"self_attn.post_proj_norm.weight", {3072}},
        {"self_attn.q_a_layernorm.weight", {1024}},
        {"self_attn.q_a_proj.weight", {1024, 3072}},
        {"self_attn.q_b_proj.weight", {8192, 1024}},
    };
    return v;
}
const std::vector<Real>& real_head() {
    static const std::vector<Real> v = {
        {"confidence_head.proj.bias", {1}},
        {"confidence_head.proj.weight", {1, 3072}},
        {"final_norm.weight", {3072}},
        {"hidden_correction.down_proj.weight", {3072, 3072}},
        {"hidden_correction.embed_norm.weight", {3072}},
        {"hidden_correction.gate_proj.weight", {3072, 6144}},
        {"hidden_correction.hidden_norm.weight", {3072}},
        {"hidden_correction.up_proj.weight", {3072, 6144}},
        {"mtp_fusion.eh_proj.weight", {3072, 6144}},
        {"mtp_fusion.enorm.weight", {3072}},
        {"mtp_fusion.gate", {3072}},
        {"mtp_fusion.hnorm.weight", {3072}},
    };
    return v;
}

std::map<std::string, dfly2::Tensor> real_tensors(int layers) {
    std::map<std::string, dfly2::Tensor> m;
    auto add = [&](const std::string& n, const std::vector<uint64_t>& s) {
        uint64_t e = 1;
        for (uint64_t d : s) e *= d;
        m[n] = {0, e * 2, s, "BF16"};
    };
    for (int i = 0; i < layers; ++i)
        for (const Real& r : real_layer0()) add(std::to_string(i) + "." + r.name, r.shape);
    for (const Real& r : real_head()) add(r.name, r.shape);
    return m;
}

std::string meta_text(const std::map<std::string, dfly2::Tensor>& t, const std::string& extra = "") {
    std::string s = "{\"section\":\"dfly\",\"layers\":5,\"mask_token_id\":120009,\"target_kv_layers\":[25,29,33,37,41],"
                    "\"target_index_layers\":[25,29,33,37,41],\"target_index_topk\":2048,\"context_local_window\":0," + extra +
                    "\"tensors\":{";
    bool first = true;
    uint64_t off = 1000;
    for (const auto& [n, x] : t) {
        s += std::string(first ? "" : ",") + "\"" + n + "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t i = 0; i < x.shape.size(); ++i) s += (i ? "," : "") + std::to_string(x.shape[i]);
        s += "],\"offset\":" + std::to_string(off) + ",\"bytes\":" + std::to_string(x.bytes) + "}";
        off += x.bytes;
        first = false;
    }
    return s + "}}";
}

bool parse(const std::string& text, dfly2::Record& out, std::string& err, uint64_t sec = 1u << 21) {
    core::Json j;
    if (!core::parse_json(text.data(), text.size(), j, err)) return false;
    return dfly2::parse_record(j, sec, out, err);
}

void test_pack_record() {
    const auto tensors = real_tensors(5);
    dfly2::Record p;
    DflyDims d;
    std::string err;
    CHECK(parse(meta_text(tensors), p, err, 4096));
    CHECK(p.present);
    CHECK(p.tensors.size() == 87);   // the checkpoint's count: 5 x 15 + 12
    CHECK(p.tensors.begin()->second.offset >= 4096 + 1000);   // offsets are the section's plus the record's
    CHECK(dfly_check_shapes(p, d, err));
    CHECK(d.layers == 5 && d.mask_id == 120009 && d.topk == 2048);
    CHECK(d.kv_layers == (std::vector<int>{25, 29, 33, 37, 41}) && d.idx_layers == d.kv_layers);
    CHECK(d.E == 3072 && d.q_lora == 1024 && d.heads == 32 && d.ffn == 9216 && d.hc_ffn == 3072);
    // the drafter's bytes: 0.85B parameters in BF16 is the 1.69 GB the model card gives
    uint64_t bytes = 0;
    for (const DflySpec& s : dfly_specs(d)) bytes += s.elems() * 2;
    CHECK(dfly_specs(d).size() == 87);
    CHECK(bytes > 1'680'000'000ull && bytes < 1'700'000'000ull);

    // a record without a selection width has the checkpoint's 2048; one that states a width keeps it
    std::string s = meta_text(tensors);
    s.replace(s.find("\"target_index_topk\":2048,"), 25, "");
    CHECK(parse(s, p, err) && p.topk == 0 && dfly_check_shapes(p, d, err) && d.topk == 2048);
    s = meta_text(tensors);
    s.replace(s.find("\"target_index_topk\":2048"), 24, "\"target_index_topk\":1024");
    CHECK(parse(s, p, err) && dfly_check_shapes(p, d, err) && d.topk == 1024);

    // absent, a wrong shape, a missing tensor, a layer count that disagrees: refused or not present
    CHECK(parse("null", p, err) && !p.present);
    auto bad = tensors;
    bad["3.mlp.up_proj.weight"].shape = {9216, 3071};
    CHECK(parse(meta_text(bad), p, err) && !dfly_check_shapes(p, d, err));
    bad = tensors;
    bad.erase("mtp_fusion.gate");
    CHECK(parse(meta_text(bad), p, err) && !dfly_check_shapes(p, d, err));
    s = meta_text(tensors);
    s.replace(s.find("\"layers\":5"), 10, "\"layers\":4");
    CHECK(!parse(s, p, err));
    // a window of the context and a tensor that is not BF16 are records this drafter's own check refuses
    s = meta_text(tensors);
    s.replace(s.find("\"context_local_window\":0"), 24, "\"context_local_window\":64");
    CHECK(parse(s, p, err) && p.local_window == 64 && !dfly_check_shapes(p, d, err));
    CHECK(err == "meta.dfly2: a local window of the context is not run");
    s = meta_text(tensors);
    s.replace(s.find("\"dtype\":\"BF16\""), 14, "\"dtype\":\"F32\"");
    CHECK(parse(s, p, err) && !dfly_check_shapes(p, d, err));
    CHECK(err == "meta.dfly2: " + tensors.begin()->first + " is not BF16");
}

void test_flags() {
    program::CommonOptions o;
    char a0[] = "x", a1[] = "--spec", a2[] = "4", a3[] = "--spec-oracle", a4[] = "t.txt", a5[] = "--draft-log", a6[] = "d.log",
         a7[] = "--diag-teacher", a8[] = "--unknown";
    char* argv[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
    int i = 1;
    model::FlagArgs args(9, argv, i);
    CHECK(dfly_parse_flag("--spec", args, o) && o.spec == 4 && o.spec_given && i == 2);
    i = 3;
    CHECK(dfly_parse_flag("--spec-oracle", args, o) && o.spec_oracle == "t.txt");
    i = 5;
    CHECK(dfly_parse_flag("--draft-log", args, o));
    CHECK(dfly_parse_flag("--diag-teacher", args, o) && o.diag_teacher);
    CHECK(!dfly_parse_flag("--unknown", args, o));
    // no pack: a given --spec without an oracle is dropped, none stays 0
    program::CommonOptions p;
    p.spec = 6;
    p.spec_given = true;
    dfly_finish_flags(p);
    CHECK(p.spec == 0);
    program::CommonOptions q;
    dfly_finish_flags(q);
    CHECK(q.spec == 0);
    program::CommonOptions r;
    r.spec = 6;
    r.spec_given = true;
    r.spec_oracle = "t.txt";
    dfly_finish_flags(r);
    CHECK(r.spec == 6);
}

}  // namespace

int main() {
    test_pack_record();
    test_flags();
    if (g_fail) return 1;
    std::printf("dfly_test: ok\n");
    return 0;
}
