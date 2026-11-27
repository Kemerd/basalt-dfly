// tests/models/qwen4exp/decode/dfly_host_test.cpp - CPU tests of what the DFly2 drafter decides before it touches the card:
// the pools a session of the target hands a sequence and the sessions it refuses, the bytes a start charges for the
// drafter and for its bind, the head's type read from a model file's header, and the calls refused on a drafter that
// is not loaded or not bound. No device is used.
#include "basalt/core/loader.hpp"
#include "basalt/models/qwen4exp/decode/dfly.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::core;
namespace bk = basalt::kernels;

static int g_fail = 0;
#define CHECK(c)                                                                       \
    do {                                                                               \
        if (!(c)) {                                                                    \
            std::fprintf(stderr, "dfly_host_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                                  \
        }                                                                              \
    } while (0)

namespace {

// the drafter trained for the target: five layers paired with its last five full-attention layers
DflyDims real_dims() {
    DflyDims d;
    d.layers = 5;
    d.n_embd = 2560;
    d.ffn = 7680;
    d.corr_ffn = 2560;
    d.n_head = 24;
    d.n_head_kv = 2;
    d.head_dim = 256;
    d.n_rot = 64;
    d.vocab = 248320;
    d.mask_id = 248319;
    d.block = kDflyBlockMax;
    d.sel_width = 2051;
    d.rms_eps = d.attn_eps = 1e-6f;
    d.rope_base = 1e7f;
    d.target_layers = {31, 35, 39, 43, 47};
    return d;
}

// A session whose paired layers hold K/V in the given form. The pools are only named, never read, so each array is a
// distinct address inside one buffer.
struct Session {
    std::vector<QsaState> states;
    std::vector<int32_t> mem;
    SessionState ss;
    Session(const DflyDims& d, bool q8) {
        const ModelGeometry g;
        states.resize((size_t) g.n_qsa_layers());
        mem.resize(states.size() * 8);
        for (int l : d.target_layers) {
            const size_t ord = (size_t) (l / g.qsa_interval);
            int32_t* at = mem.data() + ord * 8;
            QsaState& q = states[ord];
            q.page_table = at;
            q.kv_int8 = q8;
            if (q8) {
                q.k_q = (int8_t*) (at + 1);
                q.v_q = (int8_t*) (at + 2);
                q.k_scale = (uint16_t*) (at + 3);
                q.v_scale = (uint16_t*) (at + 4);
            } else {
                q.k_pool = (uint16_t*) (at + 5);
                q.v_pool = (uint16_t*) (at + 6);
            }
        }
        ss.qsa_states = states.data();
        ss.layer_lo = 0;
        ss.layer_hi = g.n_layers;
        ss.qsa_ord0 = 0;
        ss.qsa_alloc = g.n_qsa_layers();
    }
};

void test_session_pools() {
    const DflyDims d = real_dims();
    const ModelGeometry g;
    for (const bool q8 : {false, true}) {
        Session ses(d, q8);
        std::vector<bk::QsaAttnPools> pools((size_t) d.layers);
        std::string err;
        bool same = dfly_session_pools(ses.ss, g, d, q8, pools.data(), err);
        // each draft layer's pools are its own paired layer's, as that layer stores them
        for (int i = 0; same && i < d.layers; ++i) {
            const QsaState& q = ses.states[(size_t) (d.target_layers[(size_t) i] / g.qsa_interval)];
            const bk::QsaAttnPools& p = pools[(size_t) i];
            same = p.page_table == q.page_table && p.k_q4 == nullptr && p.v_q4 == nullptr &&
                   (q8 ? p.k_q == q.k_q && p.v_q == q.v_q && p.k_scale == q.k_scale && p.v_scale == q.v_scale &&
                             p.k_pool == nullptr && p.v_pool == nullptr
                       : p.k_pool == q.k_pool && p.v_pool == q.v_pool && p.k_q == nullptr && p.v_q == nullptr);
        }
        CHECK(same);
        // each state the gather cannot read, on the last paired layer
        const size_t last = (size_t) (d.target_layers.back() / g.qsa_interval);
        struct Case {
            const char* what;
            std::function<void(Session&)> edit;
            bool ask_q8;
        };
        const std::vector<Case> cases = {
            {"streamed K/V", [&](Session& s) { s.states[last].kv_mode = 1; }, q8},
            {"a ring of K/V", [&](Session& s) { s.states[last].kv_mode = 2; }, q8},
            {"Q4_0 K/V", [&](Session& s) { s.states[last].kv_q4 = true; }, q8},
            {"K8V4 K/V", [&](Session& s) { s.states[last].kv_hybrid = true; }, q8},
            {"no page table", [&](Session& s) { s.states[last].page_table = nullptr; }, q8},
            {"the other K/V form", [](Session&) {}, !q8},
            {"a layer range that ends before a paired layer", [&](Session& s) { s.ss.layer_hi = d.target_layers.back(); }, q8},
            {"a layer range that starts after a paired layer", [&](Session& s) { s.ss.layer_lo = d.target_layers.front() + 1; }, q8},
            {"no state for a paired layer", [&](Session& s) { s.ss.qsa_alloc = (int64_t) last; }, q8},
            {"states that start after a paired layer", [&](Session& s) { s.ss.qsa_ord0 = (int64_t) last; s.ss.qsa_alloc = 1; }, q8},
            {"no states", [](Session& s) { s.ss.qsa_states = nullptr; }, q8},
        };
        for (const Case& c : cases) {
            Session s2(d, q8);
            c.edit(s2);
            std::string e;
            if (dfly_session_pools(s2.ss, g, d, c.ask_q8, pools.data(), e) || e.empty()) {
                std::fprintf(stderr, "dfly_host_test: FAILED a session with %s was taken\n", c.what);
                ++g_fail;
            }
        }
        // a drafter paired with a layer that has no full attention, and no room for the pools
        DflyDims odd = d;
        odd.target_layers[0] = 30;
        std::string e;
        CHECK(!dfly_session_pools(ses.ss, g, odd, q8, pools.data(), e) && !e.empty());
        e.clear();
        CHECK(!dfly_session_pools(ses.ss, g, d, q8, nullptr, e) && !e.empty());
    }
}

void test_bytes() {
    const DflyDims d = real_dims();
    // the weights as they are held: every projection in Q8_0 blocks, every vector in fp32
    uint64_t q8 = 0, f32 = 0;
    for (const DflySpec& s : dfly_specs(d)) (s.f32 ? f32 : q8) += s.bytes();
    CHECK(q8 == 626688000ull && f32 == 184324ull);
    // the plan is the weights and a block's working memory, a few MiB of it
    const uint64_t full = DflyDrafter::plan_bytes(d, kDflyBlockMax, false);
    CHECK(full > q8 + f32 && full < q8 + f32 + (32ull << 20));
    // a narrower block takes less, and an int8 pool about half the fp16 pool's 4,218,880 bytes plus its scales
    CHECK(DflyDrafter::plan_bytes(d, 2, false) < full);
    const uint64_t pool = full - DflyDrafter::plan_bytes(d, kDflyBlockMax, true);
    CHECK(pool > 2043520ull - 4096 && pool < 2043520ull + 4096);
    // the bind's bytes: the 2-bit copy of the whole head, its scratch and its row map, for the types with the argmax
    const uint64_t bind = DflyDrafter::bind_bytes(8, d.vocab);
    CHECK(bind > (180ull << 20) && bind < (215ull << 20));
    CHECK(DflyDrafter::bind_bytes(13, d.vocab) == bind && DflyDrafter::bind_bytes(14, d.vocab) == bind);
    CHECK(DflyDrafter::bind_bytes(12, d.vocab) == 0 && DflyDrafter::bind_bytes(8, 0) == 0);
    CHECK(NativeHead::has_draft_top1(8) && NativeHead::has_draft_top1(13) && NativeHead::has_draft_top1(14) &&
          !NativeHead::has_draft_top1(12) && !NativeHead::has_draft_top1(30));
    std::printf("dfly_host_test: the drafter of %d layers takes %llu bytes with an fp16 pool and %llu with an int8 pool, "
                "its bind %llu\n", d.layers, (unsigned long long) full,
                (unsigned long long) DflyDrafter::plan_bytes(d, kDflyBlockMax, true), (unsigned long long) bind);
}

// The head's type is known from the file's header, before the head is loaded: what a start decides the drafter on.
void test_head_type() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("basalt-dfly-host-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const uint64_t rows = 64;
    // Q8_0, Q5_K and Q6_K have the two-stage argmax; a Q4_K head loads but has none
    struct Case {
        int type;
        uint64_t row_bytes;
        bool two_stage;
    };
    for (const Case& c : {Case{8, 2720, true}, Case{13, 1760, true}, Case{14, 2100, true}, Case{12, 1440, false}}) {
        const std::string path = (dir / ("head-" + std::to_string(c.type) + ".gguf")).string();
        fixture::write(path, {fixture::str("general.architecture", "qwen4exp")}, {{"output.weight", {2560, rows}, (uint32_t) c.type, 1}});
        uint64_t bytes = 0;
        int type = -1;
        std::string err;
        CHECK(NativeHead::plan_bytes({path}, bytes, err, &type) && type == c.type && bytes >= rows * c.row_bytes);
        CHECK(NativeHead::has_draft_top1(type) == c.two_stage);
        // the type is optional: the bytes alone, as before
        uint64_t alone = 0;
        CHECK(NativeHead::plan_bytes({path}, alone, err) && alone == bytes);
    }
    // a file without the head is an error, and the type is left as it was
    const std::string none = (dir / "none.gguf").string();
    fixture::write(none, {fixture::str("general.architecture", "qwen4exp")}, {{"token_embd.weight", {2560, rows}, 8, 1}});
    uint64_t bytes = 0;
    int type = -7;
    std::string err;
    CHECK(!NativeHead::plan_bytes({none}, bytes, err, &type) && !err.empty() && type == -7);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_refusals() {
    // every call on a drafter that holds nothing is an error, before any device call
    DflyDrafter dr;
    std::string err;
    bk::QsaAttnPools pools[dfly2::kMaxLayers];
    CHECK(!dr.loaded() && !dr.bound() && dr.vram_bytes() == 0);
    CHECK(dr.add_sequence(pools, err) == -1 && !err.empty());
    bool queued = true;
    err.clear();
    CHECK(!dr.draft(0, 1, 100, 99, 0, 1, queued, err) && !queued && !err.empty());
    err.clear();
    CHECK(!dr.wait(err) && !err.empty());
    CHECK(dr.raw_conf(0) == nullptr && dr.head_prob(-1) == nullptr);
    // a pack without a drafter, and a block wider or narrower than the drafter's, are not loaded
    DflyPack none;
    LoadPlan reads;
    err.clear();
    CHECK(!dr.load(none, "pack.basalt", kDflyBlockMax, false, reads, err) && !err.empty());
    DflyPack pack;
    pack.present = true;
    pack.dims = real_dims();
    pack.dims.block = 4;
    for (const int rows : {1, 5, kDflyBlockMax + 1}) {
        err.clear();
        CHECK(!dr.load(pack, "pack.basalt", rows, false, reads, err) && !err.empty());
    }
    err.clear();
    CHECK(!dr.load(pack, "", 4, false, reads, err) && !err.empty());
    // nothing binds before the load
    DflyDrafter::Target t;
    const float rows[1] = {};
    const int32_t* sel[dfly2::kMaxLayers] = {};
    err.clear();
    CHECK(!dr.bind(t, rows, sel, 2051, 8, err) && !err.empty());
    CHECK(dr.rounds == 0 && dr.skipped == 0 && dr.ms_draft == 0.0);
}

}  // namespace

int main() {
    test_session_pools();
    test_bytes();
    test_head_type();
    test_refusals();
    if (g_fail) {
        std::fprintf(stderr, "dfly_host_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("dfly_host_test: ok\n");
    return 0;
}
