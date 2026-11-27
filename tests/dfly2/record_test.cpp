// tests/dfly2/record_test.cpp - CPU tests of what the DFly2 drafters share that needs no device: the pack's record and the
// records it refuses, the source row, the draft loop, and the exactness of verification: whatever drafts a window holds,
// the stream equals plain decoding's, greedy or drawn, against a fake target.
#include "basalt/dfly2/record.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace basalt;

static int g_fail = 0;
#define CHECK(c)                                                                    \
    do {                                                                            \
        if (!(c)) {                                                                 \
            std::fprintf(stderr, "record_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                               \
        }                                                                           \
    } while (0)

namespace {

// ---- the record ----------------------------------------------------------------------------------------------------------

// A drafter's record in parts, so a test can leave one out or change it: three draft layers and three tensors, one of
// them not BF16 and one that names no type.
struct Parts {
    std::string layers = "\"layers\":3,";
    std::string mask = "\"mask_token_id\":77,";
    std::string kv = "\"target_kv_layers\":[4,8,12],";
    std::string idx = "\"target_index_layers\":[5,9,13],";
    std::string optional = "\"target_index_topk\":512,\"context_local_window\":128,\"block_size\":6,";
    std::string tensors = "\"tensors\":{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[4,8],\"offset\":64,\"bytes\":64},"
                          "\"b.bias\":{\"dtype\":\"F32\",\"shape\":[8],\"offset\":128,\"bytes\":32},"
                          "\"c.raw\":{\"shape\":[2],\"offset\":192,\"bytes\":2}}";
    std::string text() const {
        // a key the shared parser does not read stands for a model's own settings
        return "{\"section\":\"dfly\",\"format\":\"any\"," + layers + mask + kv + idx + optional + tensors + "}";
    }
};

bool parse(const std::string& text, dfly2::Record& out, std::string& err, uint64_t sec = 1u << 21) {
    core::Json j;
    if (!core::parse_json(text.data(), text.size(), j, err)) return false;
    return dfly2::parse_record(j, sec, out, err);
}

void test_record() {
    dfly2::Record r;
    std::string err;
    CHECK(parse(Parts().text(), r, err, 4096));
    CHECK(r.present && r.layers == 3 && r.mask_id == 77 && r.topk == 512 && r.local_window == 128 && r.block == 6);
    CHECK(r.kv_layers == (std::vector<int>{4, 8, 12}) && r.idx_layers == (std::vector<int>{5, 9, 13}));
    CHECK(r.tensors.size() == 3);
    // offsets are the section's plus the record's; the stored type is kept as named, whatever it is
    const dfly2::Tensor& a = r.tensors.at("a.weight");
    CHECK(a.offset == 4096 + 64 && a.bytes == 64 && a.shape == (std::vector<uint64_t>{4, 8}) && a.dtype == "BF16");
    CHECK(r.tensors.at("b.bias").dtype == "F32" && r.tensors.at("b.bias").offset == 4096 + 128);
    CHECK(r.tensors.at("c.raw").dtype.empty() && r.tensors.at("c.raw").shape == (std::vector<uint64_t>{2}));

    // the optional settings read as 0 when the record leaves them out
    Parts p;
    p.optional.clear();
    CHECK(parse(p.text(), r, err) && r.present && r.topk == 0 && r.local_window == 0 && r.block == 0);

    // no record: not an error, and nothing of an earlier record is left
    CHECK(parse("null", r, err) && !r.present && r.layers == 0 && r.tensors.empty());

    // the layer count, the two lists and the mask token must be there and agree
    const std::string disagree = "meta.dfly2: layers, target_kv_layers, target_index_layers or mask_token_id are missing or disagree";
    p = Parts();
    p.layers = "\"layers\":4,";
    CHECK(!parse(p.text(), r, err) && err == disagree && !r.present);
    p = Parts();
    p.layers.clear();
    CHECK(!parse(p.text(), r, err) && err == disagree);
    p = Parts();
    p.kv = "\"target_kv_layers\":[4,8],";
    CHECK(!parse(p.text(), r, err) && err == disagree);
    p = Parts();
    p.idx = "\"target_index_layers\":5,";
    CHECK(!parse(p.text(), r, err) && err == disagree);
    p = Parts();
    p.mask.clear();
    CHECK(!parse(p.text(), r, err) && err == disagree);
    p = Parts();
    p.mask = "\"mask_token_id\":-1,";
    CHECK(!parse(p.text(), r, err) && err == disagree);
    // more layers than a record may name, even with lists of that length
    p = Parts();
    p.layers = "\"layers\":9,";
    p.kv = "\"target_kv_layers\":[1,2,3,4,5,6,7,8,9],";
    p.idx = "\"target_index_layers\":[1,2,3,4,5,6,7,8,9],";
    CHECK(!parse(p.text(), r, err) && err == disagree);
    p.layers = "\"layers\":8,";
    p.kv = "\"target_kv_layers\":[1,2,3,4,5,6,7,8],";
    p.idx = "\"target_index_layers\":[1,2,3,4,5,6,7,8],";
    CHECK(parse(p.text(), r, err) && r.layers == dfly2::kMaxLayers);

    // the tensors: missing, not an object, or one without its offset, bytes or shape
    p = Parts();
    p.tensors = "\"other\":0";
    CHECK(!parse(p.text(), r, err) && err == "meta.dfly2 has no tensors");
    p.tensors = "\"tensors\":[1,2]";
    CHECK(!parse(p.text(), r, err) && err == "meta.dfly2 has no tensors");
    const std::string lacks = "meta.dfly2: a.weight has no offset, bytes or shape";
    p.tensors = "\"tensors\":{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[4,8],\"bytes\":64}}";
    CHECK(!parse(p.text(), r, err) && err == lacks && !r.present);
    p.tensors = "\"tensors\":{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[4,8],\"offset\":64}}";
    CHECK(!parse(p.text(), r, err) && err == lacks);
    p.tensors = "\"tensors\":{\"a.weight\":{\"dtype\":\"BF16\",\"offset\":64,\"bytes\":64}}";
    CHECK(!parse(p.text(), r, err) && err == lacks);
    p.tensors = "\"tensors\":{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":32,\"offset\":64,\"bytes\":64}}";
    CHECK(!parse(p.text(), r, err) && err == lacks);
}

// ---- the source row and the draft loop -----------------------------------------------------------------------------------

void test_source_row() {
    unsigned sel[dfly2::kTapRows] = {};
    const unsigned all = (1u << 5) - 1;
    // a window began at position 100 and kept rows 0..2 whole, row 3's hidden state only
    for (int r = 0; r < 3; ++r) sel[r] = all;
    sel[3] = all & ~(1u << 2);
    const unsigned hid = 0xFu;
    CHECK(dfly2::source_row(100, hid, sel, 5, 101) == 0);   // the token at 101 was chosen by the row at 100
    CHECK(dfly2::source_row(100, hid, sel, 5, 103) == 2);
    CHECK(dfly2::source_row(100, hid, sel, 5, 104) == -1);  // row 3 lacks a selection
    CHECK(dfly2::source_row(100, hid, sel, 5, 100) == -1);  // the row before the window is not kept
    CHECK(dfly2::source_row(100, hid, sel, 5, 200) == -1);
    CHECK(dfly2::source_row(100, 0x3u, sel, 5, 103) == -1);  // no hidden state for row 2
    CHECK(dfly2::source_row(-1, hid, sel, 5, 1) == -1);      // invalidated
    CHECK(dfly2::source_row(100, hid, nullptr, 5, 101) == -1);   // no selections at all
}

void test_draft_loop() {
    // draft j sees the token before it: the anchor, then each draft
    std::vector<int32_t> seen;
    int32_t ids[5];
    float conf[5];
    int made = -1;
    CHECK(dfly2::draft_loop(
        5, 7,
        [&](int j, int32_t prev, int32_t& tok, float& c, bool&) {
            seen.push_back(prev);
            tok = 10 * (j + 1) + (int32_t) seen.size();
            c = 0.5f + 0.1f * (float) j;
            return true;
        },
        ids, conf, made));
    CHECK(seen == (std::vector<int32_t>{7, 11, 22, 33, 44}) && made == 5);
    CHECK(ids[4] == 55 && std::fabs(conf[3] - 0.8f) < 1e-6f);
    CHECK(!dfly2::draft_loop(3, 0, [](int j, int32_t, int32_t&, float&, bool&) { return j < 1; }, ids, conf, made));
    // a step that ends the block: the drafts before it are kept, no step runs after it
    seen.clear();
    CHECK(dfly2::draft_loop(
        5, 7,
        [&](int j, int32_t prev, int32_t& tok, float&, bool& go) {
            seen.push_back(prev);
            if (j == 2) go = false;
            else tok = 100 + j;
            return true;
        },
        ids, conf, made));
    CHECK(made == 2 && ids[1] == 101 && seen == (std::vector<int32_t>{7, 100, 101}));
    // a block of no drafts runs no step; no step to run, or nowhere to put the drafts, fails with none made
    const auto never = [](int, int32_t, int32_t&, float&, bool&) { return false; };
    made = -1;
    CHECK(dfly2::draft_loop(0, 7, never, ids, conf, made) && made == 0);
    made = -1;
    CHECK(!dfly2::draft_loop(5, 7, nullptr, ids, conf, made) && made == 0);
    made = -1;
    CHECK(!dfly2::draft_loop(5, 7, never, nullptr, conf, made) && made == 0);
    made = -1;
    CHECK(!dfly2::draft_loop(5, 7, never, ids, nullptr, made) && made == 0);
}

// ---- exact verification --------------------------------------------------------------------------------------------------

constexpr int kVocab = 23;

// The fake target: row i's logits are a function of the whole token history up to it, so a wrong draft in a window changes
// every later row, like a real model's. The pick is the argmax, or a draw keyed by the output position (a counter-based
// draw: the token at output index k is drawn from (seed, k) whatever the window it is made in).
struct FakeTarget {
    uint64_t seed;
    bool sampled;
    static uint64_t mix(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
        return x;
    }
    std::vector<double> logits(const std::vector<int32_t>& hist) const {
        uint64_t h = seed;
        for (size_t i = hist.size() > 3 ? hist.size() - 3 : 0; i < hist.size(); ++i) h = mix(h * 31 + (uint64_t) hist[i] + 1);
        std::vector<double> l(kVocab);
        for (int v = 0; v < kVocab; ++v) l[(size_t) v] = (double) (mix(h + (uint64_t) v * 977) % 10000) / 1000.0;
        return l;
    }
    int32_t pick(const std::vector<int32_t>& hist, int64_t out_index) const {
        const std::vector<double> l = logits(hist);
        if (!sampled) return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
        // Gumbel-max with noise from (seed, output index, token): the row's own draw
        int best = 0;
        double bv = -1e300;
        for (int v = 0; v < kVocab; ++v) {
            const double u = ((double) (mix(seed ^ mix((uint64_t) out_index * 131 + (uint64_t) v)) >> 11) + 0.5) / 9007199254740992.0;
            const double sc = l[(size_t) v] - std::log(-std::log(u));
            if (sc > bv) { bv = sc; best = v; }
        }
        return best;
    }
};

// Plain decoding: one row a step.
std::vector<int32_t> plain(const FakeTarget& t, std::vector<int32_t> prompt, int n) {
    std::vector<int32_t> out, hist = prompt;
    for (int k = 0; k < n; ++k) {
        const int32_t tok = t.pick(hist, k);
        out.push_back(tok);
        hist.push_back(tok);
    }
    return out;
}

// The window loop of the engine (src/program/generate/adapter.cpp): the window is the last pick and the drafts; row i's pick
// follows the window's first i + 1 tokens; the longest prefix of drafts the rows confirm is kept, with the row after it.
std::vector<int32_t> drafted(const FakeTarget& t, std::vector<int32_t> prompt, int n, int block, std::mt19937& rng, double quality,
                             long& offered, long& accepted) {
    std::vector<int32_t> out, hist = prompt;
    int32_t next = t.pick(hist, 0);
    out.push_back(next);
    while ((int) out.size() < n) {
        // the drafter: a draft is the true continuation with probability `quality`, else any token; each draft sees the one before
        const int nd = std::uniform_int_distribution<int>(0, block - 1)(rng);
        std::vector<float> c((size_t) block);
        std::vector<int32_t> truth_hist = hist;
        truth_hist.push_back(next);
        std::vector<int32_t> ids((size_t) block);
        int made = 0;
        dfly2::draft_loop(
            nd, next,
            [&](int j, int32_t prev, int32_t& tok, float& chance, bool&) {
                (void) prev;
                const int64_t at = (int64_t) out.size() + j;
                const int32_t truth = t.pick(truth_hist, at);   // what plain decoding picks next
                truth_hist.push_back(truth);
                tok = std::bernoulli_distribution(quality)(rng) ? truth : std::uniform_int_distribution<int32_t>(0, kVocab - 1)(rng);
                chance = 0.5f;
                return true;
            },
            ids.data(), c.data(), made);
        std::vector<int32_t> window = {next};
        window.insert(window.end(), ids.begin(), ids.begin() + nd);
        // row i: the pick after hist + window[0..i], drawn at the output index of the token it picks
        std::vector<int32_t> rows;
        std::vector<int32_t> h = hist;
        for (int i = 0; i <= nd; ++i) {
            h.push_back(window[(size_t) i]);
            rows.push_back(t.pick(h, (int64_t) out.size() + i));
        }
        int a = 0;
        while (a < nd && window[(size_t) a + 1] == rows[(size_t) a]) ++a;
        offered += nd;
        accepted += a;
        hist.push_back(next);
        for (int i = 0; i < a; ++i) hist.push_back(window[(size_t) i + 1]);
        for (int i = 0; i <= a && (int) out.size() < n; ++i) out.push_back(rows[(size_t) i]);
        next = rows[(size_t) a];
    }
    out.resize((size_t) n);
    return out;
}

void test_exact_verify() {
    long offered = 0, accepted = 0;
    for (int sampled = 0; sampled < 2; ++sampled) {
        for (uint64_t seed = 1; seed <= 40; ++seed) {
            const FakeTarget t{seed, sampled != 0};
            std::mt19937 rng((uint32_t) seed);
            std::vector<int32_t> prompt = {3, 1, 4, 1, 5};
            const std::vector<int32_t> want = plain(t, prompt, 60);
            for (double q : {0.0, 0.4, 0.9, 1.0}) {
                for (int block : {2, 4, 6}) {
                    const std::vector<int32_t> got = drafted(t, prompt, 60, block, rng, q, offered, accepted);
                    if (got != want) {
                        std::fprintf(stderr, "record_test: stream differs (sampled %d, seed %llu, quality %.1f, block %d)\n", sampled,
                                     (unsigned long long) seed, q, block);
                        ++g_fail;
                        return;
                    }
                }
            }
        }
    }
    CHECK(offered > 1000 && accepted > 100 && accepted < offered);   // the windows really held drafts, some right, some wrong
}

}  // namespace

int main() {
    test_record();
    test_source_row();
    test_draft_loop();
    test_exact_verify();
    if (g_fail) return 1;
    std::printf("record_test: ok\n");
    return 0;
}
