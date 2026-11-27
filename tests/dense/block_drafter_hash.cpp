// tests/dense/block_drafter_hash.cpp - the dense stack's block drafter on seeded inputs, without a pack: the feature
// projection of a window's rows, the context those rows become (the K/V rings) and the draft pass, at the width and
// vocabulary of the 27B and the shape of its DFlash 2 drafter (5 layers, 32 x 128 query heads on 8 KV heads, an FFN of
// 17408, blocks of 8, a window of 2048).
//
// Prints one line per result with the hash of its bits: the features, each layer's rings after a context that wraps
// them, and the drafts and confidences of a block, a wide block, a block on a short context, two sessions in one pass
// and a drawn block. A change that must not move a bit prints the same lines; two builds are compared by their output.
// Fails when a pass fails or gives a draft outside the scored vocabulary, a confidence outside [0, 1] or a ring value
// that is not finite.
//
// The model is hand-built: one layer that is never run (the drafter's passes read no layer of the model), a seeded head
// and embedding, the seeded drafter. Every input comes from one integer generator and exact conversions, so the bits
// are the same on every platform. Exit 77 without a GPU or without the device memory.
#include "dense/internal.hpp"

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace basalt;
using namespace basalt::dense;

constexpr int H = 5120, FF = 17408, VOCAB = 248320;
constexpr int kLayers = 5, kHeads = 32, kKv = 8, kHeadDim = 128, kBlock = 8, kGroup = 16, kRank = 256, kTop = 16;
constexpr int kTaps = 5, kRingWindow = 2048;
// The embedding rows the passes read are the anchors' and the mask token's: the table holds the first rows only, and
// every id embedded here is below it.
constexpr int kEmbedRows = 4096, kMask = 4000;
constexpr int64_t kContext = 4096;
constexpr int64_t kPositions = 2400;         // the context read into the first session: past its ring's cells
constexpr size_t kNeedBytes = 6ull << 30;    // device memory the weights, the scratch and two sessions take, with room

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
void must(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "block_drafter_hash: %s\n", what.c_str());
        std::exit(1);
    }
}

// splitmix64: integer arithmetic alone
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};
uint64_t g_seed = 1;   // each tensor draws from its own stream, numbered in the order the tensors are made

// n bytes of the generator's output (the hosts are little-endian)
void bytes(Rng& r, uint8_t* p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint64_t z = r.next();
        std::memcpy(p + i, &z, 8);
    }
    if (i < n) {
        const uint64_t z = r.next();
        std::memcpy(p + i, &z, n - i);
    }
}

std::vector<uint8_t> g_host;   // one tensor at a time on the host
uint8_t* host(size_t n) {
    if (g_host.size() < n) g_host.resize(n);
    return g_host.data();
}
void* upload(const void* p, size_t n) {
    void* d = nullptr;
    ck(cudaMalloc(&d, n), "cudaMalloc");
    ck(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice), "upload");
    return d;
}

// A Q8_0 matrix (ggml's 34-byte blocks: an fp16 scale, then 32 codes): codes over [-127, 127], each block's scale within
// a factor of two of the one that takes unit-rms rows to unit-rms outputs.
model::Linear q8_matrix(int64_t rows, int64_t cols) {
    int e = 0;
    while ((1ull << (2 * e)) < 5329ull * (uint64_t) cols) ++e;   // 2^-e: about 1 / (73 sqrt(cols)), 73 the codes' rms
    const size_t blocks = (size_t) rows * (size_t) cols / 32, n = blocks * 34;
    uint8_t* h = host(n);
    Rng r(g_seed++);
    bytes(r, h, n);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = h + b * 34;
        const uint16_t d = kernels::f16_from_f32(std::ldexp((float) (64 + (p[0] & 63)) / 64.0f, -e));
        std::memcpy(p, &d, 2);
        for (int i = 2; i < 34; ++i)
            if (p[i] == 0x80) p[i] = 0x81;
    }
    model::Linear l;
    l.type = 8;
    l.w = upload(h, n);
    l.rows = rows;
    l.cols = cols;
    return l;
}

// BF16 values k 2^-shift, k a code over [-127, 127]: exact in BF16, so its bits are the float's upper half.
const uint16_t* bf16_tensor(size_t n, int shift) {
    uint16_t table[256];
    for (int b = 0; b < 256; ++b) {
        const int k = b == 0x80 ? -127 : (int) (int8_t) b;
        const float f = std::ldexp((float) k, -shift);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        table[b] = (uint16_t) (u >> 16);
    }
    uint8_t* h = host(n * 3);
    uint16_t* v = reinterpret_cast<uint16_t*>(h);
    uint8_t* code = h + n * 2;
    Rng r(g_seed++);
    bytes(r, code, n);
    for (size_t i = 0; i < n; ++i) v[i] = table[code[i]];
    return (const uint16_t*) upload(v, n * 2);
}

// FP32 values base + k / div
const float* f32_tensor(size_t n, float base, float div) {
    uint8_t* h = host(n * 5);
    float* v = reinterpret_cast<float*>(h);
    uint8_t* code = h + n * 4;
    Rng r(g_seed++);
    bytes(r, code, n);
    for (size_t i = 0; i < n; ++i) v[i] = base + (float) (int8_t) code[i] / div;
    return (const float*) upload(v, n * 4);
}

// The head as the 27B pack holds it: E4M3 codes [rows][cols] with an fp32 scale per row; no code is a NaN.
model::Linear f8_head(int64_t rows, int64_t cols) {
    const size_t n = (size_t) rows * (size_t) cols;
    uint8_t* h = host(n);
    Rng r(g_seed++);
    bytes(r, h, n);
    for (size_t i = 0; i < n; ++i)
        if ((h[i] & 0x7F) == 0x7F) h[i] &= 0xFE;
    model::Linear l;
    l.type = 44;
    l.w = upload(h, n);
    l.rows = rows;
    l.cols = cols;
    // a row's scale 2^-12 times [1, 2): logits of a few units against unit-rms rows
    std::vector<float> scales((size_t) rows);
    Rng rs(g_seed++);
    for (float& s : scales) s = std::ldexp((float) (64 + (rs.next() & 63)) / 64.0f, -12);
    l.row_scales = (const float*) upload(scales.data(), scales.size() * 4);
    return l;
}

dflash::BlockDrafter drafter() {
    dflash::BlockDrafter d;
    d.n_embd = H;
    d.block = kBlock;
    d.mask_id = kMask;
    d.conv_group = kGroup;
    d.rank = kRank;
    d.top_k = kTop;
    d.window = kRingWindow;
    d.n_head = kHeads;
    d.n_kv = kKv;
    d.head_dim = kHeadDim;
    d.theta = 1e7f;
    d.eps = 1e-6f;
    d.target_layers = {5, 19, 33, 47, 61};
    d.n_layers = kLayers;
    d.n_ff = FF;
    const int64_t AQ = (int64_t) kHeads * kHeadDim, KV = (int64_t) kKv * kHeadDim, conv = 2 * 2 * (H / kGroup);
    for (int j = 0; j < kTaps; ++j) d.fc.push_back(q8_matrix(H, H));
    d.hidden_norm = f32_tensor(H, 1.0f, 512.0f);
    d.norm = f32_tensor(H, 1.0f, 512.0f);
    for (int l = 0; l < kLayers; ++l) {
        dflash::BlockDrafter::Layer L;
        L.attn_norm = f32_tensor(H, 1.0f, 512.0f);
        L.ffn_norm = f32_tensor(H, 1.0f, 512.0f);
        L.q_norm = f32_tensor(kHeadDim, 1.0f, 512.0f);
        L.k_norm = f32_tensor(kHeadDim, 1.0f, 512.0f);
        L.base_a = f32_tensor(4 * (size_t) H, 0.0f, 128.0f);
        L.base_f = f32_tensor(4 * (size_t) H, 0.0f, 128.0f);
        L.q = q8_matrix(AQ, H);
        L.k = q8_matrix(KV, H);
        L.v = q8_matrix(KV, H);
        L.o = q8_matrix(H, AQ);
        L.gate = q8_matrix(FF, H);
        L.up = q8_matrix(FF, H);
        L.down = q8_matrix(H, FF);
        L.conv_a = q8_matrix(conv, H);
        L.conv_f = q8_matrix(conv, H);
        d.layers.push_back(L);
    }
    d.sel_h = bf16_tensor((size_t) kRank * H, 13);
    d.pred = bf16_tensor((size_t) VOCAB * kRank, 7);
    d.succ = bf16_tensor((size_t) VOCAB * kRank, 7);
    d.present = true;
    return d;
}

// FNV-1a over 32-bit words
struct Fnv {
    uint64_t h = 1469598103934665603ull;
    void add(const void* p, size_t bytes) {
        const uint8_t* b = (const uint8_t*) p;
        for (size_t i = 0; i + 4 <= bytes; i += 4) {
            uint32_t u;
            std::memcpy(&u, b + i, 4);
            h = (h ^ u) * 1099511628211ull;
        }
    }
};

// ---- where a session keeps the drafter's state
float* features(SessionState& st) { return st.seq.acc; }
size_t ring_layers(const SessionState& st) { return st.seq.k.size(); }
uint16_t* ring_k(SessionState& st, size_t l) { return st.seq.k[l]; }
uint16_t* ring_v(SessionState& st, size_t l) { return st.seq.v[l]; }
int64_t ring_cells(const SessionState& st) { return st.seq.cap; }
int32_t* draft_ids(SessionState& st) { return st.seq.ids; }
float* draft_conf(SessionState& st) { return st.seq.conf; }
int32_t* drawn_ids(SessionState& st) { return st.seq.q_ids; }
float* drawn_p(SessionState& st) { return st.seq.q_p; }
void set_draw(SessionState& st, float temperature, uint64_t seed) {
    st.seq.temperature = temperature;
    st.seq.seed = seed;
}
// the feature projection j of a window's rows into the session's features
void tap(Model& m, SessionState& st, size_t j, const uint16_t* rows16, int n) {
    dflash::tap(m.bd_dec, m.bs, m.ws.w8, j, rows16, n, features(st), st.stream);
}

int g_bad = 0;

// A session's rings zeroed: its arena comes as the device left it, and a context shorter than a ring writes only the
// cells of its own positions, while every cell is hashed.
void clear_rings(SessionState& st) {
    const size_t bytes = (size_t) ring_cells(st) * kKv * kHeadDim * 2;
    for (size_t l = 0; l < ring_layers(st); ++l) {
        must(ring_k(st, l) && ring_v(st, l), "a session without a ring");
        ck(cudaMemset(ring_k(st, l), 0, bytes), "clearing a K ring");
        ck(cudaMemset(ring_v(st, l), 0, bytes), "clearing a V ring");
    }
}

// A session's rings, each layer's K then V: one line a ring.
void print_rings(const char* name, SessionState& st) {
    ck(cudaStreamSynchronize(st.stream), "sync");
    const size_t n = (size_t) ring_cells(st) * kKv * kHeadDim;
    std::vector<uint16_t> h(n);
    for (size_t l = 0; l < ring_layers(st); ++l)
        for (int kv = 0; kv < 2; ++kv) {
            ck(cudaMemcpy(h.data(), kv ? ring_v(st, l) : ring_k(st, l), n * 2, cudaMemcpyDeviceToHost), "ring");
            Fnv f;
            f.add(h.data(), n * 2);
            size_t bad = 0;
            for (uint16_t x : h) bad += (x & 0x7C00u) == 0x7C00u;   // fp16 inf or NaN
            if (bad) ++g_bad;
            std::printf("%s ring layer %zu %c %016llx%s\n", name, l, kv ? 'V' : 'K', (unsigned long long) f.h, bad ? " NOT FINITE" : "");
        }
}

// A session's last drafts: n ids and confidences, and for a drawn block the candidates and their distribution.
void print_drafts(const char* name, SessionState& st, int n, bool drawn = false) {
    ck(cudaStreamSynchronize(st.stream), "sync");
    std::vector<int32_t> ids((size_t) n);
    std::vector<float> conf((size_t) n);
    ck(cudaMemcpy(ids.data(), draft_ids(st), (size_t) n * 4, cudaMemcpyDeviceToHost), "ids");
    ck(cudaMemcpy(conf.data(), draft_conf(st), (size_t) n * 4, cudaMemcpyDeviceToHost), "conf");
    Fnv fi, fc;
    fi.add(ids.data(), (size_t) n * 4);
    fc.add(conf.data(), (size_t) n * 4);
    bool bad = false;
    for (int i = 0; i < n; ++i) bad = bad || ids[(size_t) i] < 0 || ids[(size_t) i] >= kDraftVocab || !(conf[(size_t) i] >= 0.0f && conf[(size_t) i] <= 1.0f);
    if (bad) ++g_bad;
    std::printf("%s ids %016llx conf %016llx%s\n ", name, (unsigned long long) fi.h, (unsigned long long) fc.h, bad ? " OUT OF RANGE" : "");
    for (int i = 0; i < n; ++i) std::printf(" %d:%.6f", ids[(size_t) i], (double) conf[(size_t) i]);
    std::printf("\n");
    if (!drawn) return;
    std::vector<int32_t> q_ids((size_t) n * kTop);
    std::vector<float> q_p((size_t) n * kTop);
    ck(cudaMemcpy(q_ids.data(), drawn_ids(st), q_ids.size() * 4, cudaMemcpyDeviceToHost), "q_ids");
    ck(cudaMemcpy(q_p.data(), drawn_p(st), q_p.size() * 4, cudaMemcpyDeviceToHost), "q_p");
    Fnv fq, fp;
    fq.add(q_ids.data(), q_ids.size() * 4);
    fp.add(q_p.data(), q_p.size() * 4);
    std::printf("%s candidates %016llx distribution %016llx\n", name, (unsigned long long) fq.h, (unsigned long long) fp.h);
}

// Windows of seeded rows through the feature projection into the session's context, from position 0 up to `upto` or past
// it; the hash of every window's features. Returns the position after the last row.
int64_t read_context(Model& m, SessionState& st, int64_t upto, uint64_t seed, Fnv& hash) {
    static const int rows[8] = {16, 16, 5, 16, 1, 16, 8, 16};   // the windows' rows, in turn
    static uint16_t* d_rows[kTaps] = {};
    for (uint16_t*& p : d_rows)
        if (!p) ck(cudaMalloc((void**) &p, (size_t) kWindow * H * 2), "cudaMalloc");
    // a target layer's rows: BF16 codes over [-127, 127] times 2^-6
    uint16_t table[256];
    for (int b = 0; b < 256; ++b) {
        const float f = std::ldexp((float) (b == 0x80 ? -127 : (int) (int8_t) b), -6);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        table[b] = (uint16_t) (u >> 16);
    }
    std::vector<uint8_t> code((size_t) kWindow * H);
    std::vector<uint16_t> h_rows((size_t) kWindow * H);
    std::vector<float> h_acc((size_t) kWindow * H);
    Rng r(seed);
    std::string err;
    int64_t pos = 0;
    for (int w = 0; pos < upto; ++w) {
        const int n = rows[w % 8];
        const size_t vals = (size_t) n * H;
        for (int j = 0; j < kTaps; ++j) {
            bytes(r, code.data(), vals);
            for (size_t i = 0; i < vals; ++i) h_rows[i] = table[code[i]];
            ck(cudaMemcpyAsync(d_rows[j], h_rows.data(), vals * 2, cudaMemcpyHostToDevice, st.stream), "rows");
        }
        {
            kernels::PdlScope pdl(true, true);   // after the uploads, as a window's layers run the projections
            for (int j = 0; j < kTaps; ++j) tap(m, st, (size_t) j, d_rows[j], n);
        }
        must(block_ctx_window(m, st, n, pos, err), err);
        ck(cudaMemcpyAsync(h_acc.data(), features(st), vals * 4, cudaMemcpyDeviceToHost, st.stream), "features");
        ck(cudaStreamSynchronize(st.stream), "sync");
        hash.add(h_acc.data(), vals * 4);
        pos += n;
    }
    return pos;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < kNeedBytes) {
        std::printf("needs %zu MiB of device memory, %zu MiB free\n", kNeedBytes >> 20, free_b >> 20);
        return 77;
    }

    Geometry g;
    g.n_layers = 1;
    g.n_embd = H;
    g.n_ff = FF;
    g.n_vocab = VOCAB;
    g.n_head = 24;
    g.n_kv = 4;
    g.head_dim = 256;
    g.rope.rot = 64;
    g.rope.theta = 1e7f;
    g.kinds = {Mixer::Attention};
    g.context_length = kContext;
    Weights w;
    w.layers.resize(1);
    w.layers[0].kind = Mixer::Attention;
    w.embd = bf16_tensor((size_t) kEmbedRows * H, 6);
    w.head = f8_head(VOCAB, H);
    w.dflash = drafter();
    Options o;
    o.max_context = kContext;
    o.sessions = 2;
    o.spec = kWindow;
    o.draft = true;
    o.drafter = Drafter::Block;
    o.arch = "dense";
    std::string err;
    ModelPtr m = make_model(g, w, o, err);
    must(m != nullptr, err);
    must(attach_drafter(*m, err), err);
    SessionState a, b;
    must(init_state(*m, a, err), err);
    must(init_state(*m, b, err), err);
    clear_rings(a);
    clear_rings(b);

    // the first session: a context that wraps its rings; the second: a short one
    Fnv fa, fb;
    const int64_t pa = read_context(*m, a, kPositions, 1001, fa);
    const int64_t pb = read_context(*m, b, 40, 1002, fb);
    std::printf("positions %lld and %lld, rings of %lld cells\n", (long long) pa, (long long) pb, (long long) ring_cells(a));
    std::printf("a features %016llx\nb features %016llx\n", (unsigned long long) fa.h, (unsigned long long) fb.h);
    print_rings("a context", a);
    print_rings("b context", b);

    // a block, a wide block, a block on the short context
    must(block_draft(*m, a, 1234, pa, kBlock, err), err);
    print_drafts("a block", a, kBlock - 1);
    must(block_draft(*m, a, 77, pa, kWindow, err), err);
    print_drafts("a wide block", a, kWindow - 1);
    must(block_draft(*m, b, 9, pb, kBlock, err), err);
    print_drafts("b block", b, kBlock - 1);

    // both sessions in one pass (on the first one's stream), blocks alike and unlike
    SessionState* both[2] = {&a, &b};
    const int32_t anchors[2] = {1234, 9};
    const int64_t pos[2] = {pa, pb};
    const int same[2] = {kBlock, kBlock}, unlike[2] = {kWindow, kBlock};
    must(block_draft_batch(*m, both, anchors, pos, same, 2, err), err);
    ck(cudaStreamSynchronize(a.stream), "sync");
    print_drafts("a of two", a, kBlock - 1);
    print_drafts("b of two", b, kBlock - 1);
    must(block_draft_batch(*m, both, anchors, pos, unlike, 2, err), err);
    ck(cudaStreamSynchronize(a.stream), "sync");
    print_drafts("a wide of two", a, kWindow - 1);
    print_drafts("b beside a wide", b, kBlock - 1);

    // a drawn block: each draft from the selector's distribution over its candidates
    set_draw(a, 0.8f, 42);
    must(block_draft(*m, a, 1234, pa, kBlock, err), err);
    print_drafts("a drawn block", a, kBlock - 1, true);

    // the rings as the passes leave them (a block's own rows land in its cells)
    print_rings("a drafted", a);
    print_rings("b drafted", b);

    free_state(a);
    free_state(b);
    std::printf("%s\n", g_bad ? "FAILED" : "done");
    return g_bad ? 1 : 0;
}
