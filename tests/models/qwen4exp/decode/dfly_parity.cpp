// tests/models/qwen4exp/decode/dfly_parity.cpp - the DFly2 block pass against a CPU double reference of its equations.
//
// Seeded weights of the real widths (BF16 in a temporary file, rounded as the loader rounds them), seeded cache cells,
// selections and head rows handed to bind as plain buffers, and a synthetic head and embedding table. The reference reads
// the dequantised Q8_0 weights with the q8_1 rounding of their inputs and stores a block's own K/V as the cache stores a
// cell. Checked: the rows after the final norm, the corrected rows and the confidence logits; that a shorter block gives the
// same leading rows; that two sequences queued back to back each get their own result; fp16 and int8 pools; a context of
// the widest selection; a round's calls (a window without the source row, a slot's rows inside a window, the counters);
// the bytes the plan charges. Then the drafter bound to a native head and embedding table read from a model file, for a
// head of each type with the two-stage argmax: the bytes the bind takes, the block against the same reference, and each
// pick against the argmax of the whole head's exact logits.
// Arguments: the draft layers (default 2; the real drafter has 5) and the vocabulary's rows (default 4096; the target's
// 248320 takes several GiB of RAM and VRAM). GPU, synthetic, no model.
#include "basalt/core/device_copy.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/models/qwen4exp/decode/dfly.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "gguf_fixture.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/q5k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q80_proxy.hpp"
#include "basalt/models/qwen4exp/kernels/q80_vhead.hpp"
#include "basalt/models/qwen4exp/kernels/rope_scaling.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace basalt::core {

// a weight table row by name, for the head's row the bind reads
struct WeightTableFixture {
    static void put(WeightTable& t, const std::string& name, int64_t ne0, int64_t ne1) {
        WeightRef r;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.elements = ne0 * ne1;
        t.table_[name] = r;
    }
};

}  // namespace basalt::core

using namespace basalt;
namespace bk = basalt::kernels;

namespace {

int g_fail = 0;
std::mt19937 rng(8675309);
float gauss(float mean, float sd) { return std::normal_distribution<float>(mean, sd)(rng); }

uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dfly_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <class T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(h.size() * sizeof(T), 256)), "an allocation");
    ck(core::device_copy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "an upload");
    return d;
}
std::vector<float> download(const float* d, size_t n) {
    std::vector<float> h(n);
    ck(core::device_copy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "a download");
    return h;
}

// ---- the weights as the engine holds them: a projection's Q8_0 blocks, a vector's fp32 values
struct Weights {
    std::map<std::string, std::vector<uint8_t>> q8;     // [rows][cols / 32] blocks of 34 bytes
    std::map<std::string, std::vector<float>> f32;
    std::map<std::string, std::vector<uint64_t>> shape;
};

// W x as the engine's product makes it: x in q8_1 blocks (d = amax / 127 kept as fp16, q = round(x / d)), an integer dot
// per block, scaled by both blocks' fp16 d
std::vector<double> mvq(const Weights& w, const std::string& name, const std::vector<double>& x) {
    const std::vector<uint64_t>& sh = w.shape.at(name);
    const uint64_t rows = sh[0], cols = sh[1], nb = cols / 32;
    const std::vector<uint8_t>& blocks = w.q8.at(name);
    std::vector<int> qx(cols);
    std::vector<double> dx(nb);
    for (uint64_t b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs((float) x[b * 32 + i]));
        const float d = amax / 127.f;
        for (int i = 0; i < 32; ++i) qx[b * 32 + i] = amax == 0.f ? 0 : (int) std::round((float) x[b * 32 + i] / d);
        dx[b] = bk::f32_from_f16(bk::f16_from_f32(d));
    }
    std::vector<double> y(rows);
    for (uint64_t r = 0; r < rows; ++r) {
        double s = 0;
        for (uint64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = blocks.data() + (r * nb + b) * 34;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            int dot = 0;
            for (int i = 0; i < 32; ++i) dot += (int) (int8_t) blk[2 + i] * qx[b * 32 + i];
            s += (double) bk::f32_from_f16(dh) * dx[b] * dot;
        }
        y[r] = s;
    }
    return y;
}

// x * rsqrt(mean(x^2) + eps) * w over x[0, n)
std::vector<double> rms(const double* x, size_t n, const std::vector<float>& w, double eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const double inv = 1.0 / std::sqrt(ss / (double) n + eps);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * (double) w[i];
    return out;
}

// the rotation of a head's row at `pos`: pairs (i, i + n_rot / 2), angle pos * base^(-2 i / n_rot)
void rope(double* x, int n_rot, double base, int64_t pos) {
    for (int i = 0; i < n_rot / 2; ++i) {
        const double th = (double) pos * std::pow(base, -2.0 * i / n_rot);
        const double c = std::cos(th), s = std::sin(th), a = x[i], b = x[i + n_rot / 2];
        x[i] = a * c - b * s;
        x[i + n_rot / 2] = a * s + b * c;
    }
}

// ---- a cache layer's cells in the stored form, with the values a reader sees
struct Cache {
    std::vector<uint8_t> k, v, ks, vs;   // fp16 rows, or int8 codes with fp16 scales
    uint8_t *d_k = nullptr, *d_v = nullptr, *d_ks = nullptr, *d_vs = nullptr;
    int32_t* d_table = nullptr;
};

struct Setup {
    core::DflyDims d;
    core::DflyPack pack;
    Weights w;
    std::filesystem::path file;
    int vocab = 4096, cells = 4096, win_rows = 8;
    int64_t cap = 0;
    std::vector<float> table, head_rows;            // [vocab][N] embeddings, [win_rows][N] head inputs
    std::vector<uint8_t> embd_q8, head_q8;          // the table and the head as a model file stores them: Q8_0 rows
    std::vector<std::vector<int32_t>> sel;          // per layer [win_rows][cap]
    float *d_table = nullptr, *d_head_rows = nullptr;
    std::vector<int32_t*> d_sel;
    void *d_head = nullptr, *d_proxy = nullptr, *d_scratch = nullptr;
    int32_t* d_map = nullptr;
};

// the row of KV head h of cell c under the identity table
size_t prow(const bk::QsaShapes& s, int64_t c, int64_t h) { return (size_t) (((c / s.page_size) * s.n_head_kv + h) * s.page_size + c % s.page_size); }

Cache make_cache(const Setup& st, bool q8) {
    const bk::QsaShapes s = bk::qsa_real_shapes();
    const size_t rows = (size_t) st.cells * (size_t) s.n_head_kv, HD = (size_t) s.head_dim, nsc = HD / bk::KV_Q8_GROUP;
    Cache c;
    if (q8) {
        c.k.resize(rows * HD); c.v.resize(rows * HD); c.ks.resize(rows * nsc * 2); c.vs.resize(rows * nsc * 2);
        for (auto* a : {&c.k, &c.v}) for (uint8_t& b : *a) b = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
        for (auto* a : {&c.ks, &c.vs})
            for (size_t i = 0; i < a->size() / 2; ++i) {
                const uint16_t h = bk::f16_from_f32(0.002f + 0.004f * (float) (rng() % 1000) / 1000.f);
                std::memcpy(a->data() + i * 2, &h, 2);
            }
    } else {
        c.k.resize(rows * HD * 2); c.v.resize(rows * HD * 2);
        for (size_t i = 0; i < rows * HD; ++i) {
            const uint16_t hk = bk::f16_from_f32(gauss(0.f, 0.25f)), hv = bk::f16_from_f32(gauss(0.f, 1.f));
            std::memcpy(c.k.data() + i * 2, &hk, 2);
            std::memcpy(c.v.data() + i * 2, &hv, 2);
        }
    }
    c.d_k = upload(c.k); c.d_v = upload(c.v);
    if (q8) { c.d_ks = upload(c.ks); c.d_vs = upload(c.vs); }
    std::vector<int32_t> table((size_t) (st.cells / s.page_size));
    std::iota(table.begin(), table.end(), 0);
    c.d_table = upload(table);
    return c;
}
// value e of the stored K or V row `row`
double cell_value(const Cache& c, bool q8, bool is_v, size_t row, size_t e, size_t HD) {
    const std::vector<uint8_t>& a = is_v ? c.v : c.k;
    if (!q8) {
        uint16_t h;
        std::memcpy(&h, a.data() + (row * HD + e) * 2, 2);
        return bk::f32_from_f16(h);
    }
    const std::vector<uint8_t>& sc = is_v ? c.vs : c.ks;
    uint16_t h;
    std::memcpy(&h, sc.data() + (row * (HD / bk::KV_Q8_GROUP) + e / bk::KV_Q8_GROUP) * 2, 2);
    return (double) (int8_t) a[row * HD + e] * (double) bk::f32_from_f16(h);
}
// a block row's own K or V as the cache would store and read back a cell
void store_round(std::vector<double>& x, bool q8) {
    if (!q8) {
        for (double& v : x) v = bk::f32_from_f16(bk::f16_from_f32((float) v));
        return;
    }
    for (size_t g = 0; g < x.size(); g += bk::KV_Q8_GROUP) {
        float amax = 0.f;
        for (int i = 0; i < bk::KV_Q8_GROUP; ++i) amax = std::max(amax, std::fabs((float) x[g + i]));
        const float sf = bk::f32_from_f16(bk::f16_from_f32(amax / 127.0f));
        for (int i = 0; i < bk::KV_Q8_GROUP; ++i) {
            int q = sf > 0.f ? (int) std::nearbyint((float) x[g + i] / sf) : 0;
            q = std::max(-127, std::min(127, q));
            x[g + i] = (double) q * (double) sf;
        }
    }
}

void build(Setup& st, int layers) {
    core::DflyDims& d = st.d;
    d.layers = layers; d.n_embd = 2560; d.ffn = 7680; d.corr_ffn = 2560;
    d.n_head = 24; d.n_head_kv = 2; d.head_dim = 256; d.n_rot = 64;
    d.vocab = st.vocab; d.mask_id = st.vocab - 1; d.block = core::kDflyBlockMax; d.sel_width = 2051;
    d.rms_eps = 1e-6f; d.attn_eps = 1e-6f; d.rope_base = (float) bk::rope_scaling().freq_base;
    for (int i = 0; i < layers; ++i) d.target_layers.push_back(31 + 4 * i);
    // ---- the section's image: every tensor BF16, 64-byte aligned, and the same values as the engine will hold them
    st.file = std::filesystem::temp_directory_path() / "qwen4exp_dfly_parity.bin";
    std::ofstream f(st.file, std::ios::binary | std::ios::trunc);
    uint64_t off = 0;
    for (const core::DflySpec& s : core::dfly_specs(d)) {
        const uint64_t n = s.elems();
        std::vector<uint16_t> bits(n);
        const bool norm = s.f32 && s.name.find("confidence") == std::string::npos && s.name != "mtp_fusion.gate";
        const float sd = s.f32 ? 0.05f : 1.0f / std::sqrt((float) s.shape.back());
        for (uint16_t& b : bits) b = to_bf16(gauss(norm ? 1.0f : 0.0f, sd) + (s.name == "mtp_fusion.gate" ? 0.5f : 0.0f));
        f.write((const char*) bits.data(), (std::streamsize) (n * 2));
        const uint64_t pad = (64 - (n * 2) % 64) % 64;
        const char zero[64] = {};
        f.write(zero, (std::streamsize) pad);
        st.pack.rec.tensors[s.name] = {off, n * 2, s.shape, "BF16"};
        off += n * 2 + pad;
        st.w.shape[s.name] = s.shape;
        if (s.f32) {
            std::vector<float>& v = st.w.f32[s.name];
            v.resize(n);
            for (uint64_t i = 0; i < n; ++i) v[i] = from_bf16(bits[i]);
        } else {
            std::vector<uint8_t>& q = st.w.q8[s.name];
            q.resize(n / 32 * 34);
            core::quantize_bf16_q8_0(bits.data(), n, q.data());
        }
    }
    f.close();
    st.pack.present = true;
    st.pack.dims = d;
    st.pack.rec.present = true;
    // ---- the target's side: an embedding table, the last window's head rows, each layer's kept selections
    const bk::QsaShapes s = bk::qsa_real_shapes();
    st.cap = bk::qsa_selection_width(bk::kTopkMaxCells, s);
    // the table is a model file's Q8_0 rows, and the rows the reference embeds are their values
    {
        std::vector<uint16_t> bits((size_t) st.vocab * d.n_embd);
        for (uint16_t& b : bits) b = to_bf16(gauss(0.f, 0.05f));
        st.embd_q8.resize(bits.size() / 32 * 34);
        core::quantize_bf16_q8_0(bits.data(), bits.size(), st.embd_q8.data());
        st.table.resize(bits.size());
        for (size_t b = 0; b < bits.size() / 32; ++b) {
            const uint8_t* blk = st.embd_q8.data() + b * 34;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float dq = bk::f32_from_f16(dh);
            for (int i = 0; i < 32; ++i) st.table[b * 32 + (size_t) i] = dq * (float) (int8_t) blk[2 + i];
        }
    }
    st.head_rows.resize((size_t) st.win_rows * d.n_embd);
    for (float& x : st.head_rows) x = gauss(0.f, 1.5f);
    st.d_table = upload(st.table);
    st.d_head_rows = upload(st.head_rows);
    // a synthetic Q8_0 head in bit planes with its 2-bit copy: the pick the real head's type would make
    {
        std::vector<uint16_t> bits((size_t) st.vocab * d.n_embd);
        for (uint16_t& b : bits) b = to_bf16(gauss(0.f, 0.02f));
        std::vector<uint8_t> q(bits.size() / 32 * 34);
        core::quantize_bf16_q8_0(bits.data(), bits.size(), q.data());
        st.head_q8 = q;
        st.d_head = upload(q);
        bk::q80p_pack(st.d_head, st.vocab, nullptr);
        std::vector<int32_t> map((size_t) st.vocab);
        std::iota(map.begin(), map.end(), 0);
        st.d_map = upload(map);
        const size_t pb = bk::q5k_proxy_bytes(st.vocab), sb = bk::q5k_proxy_scratch_bytes(st.vocab);
        ck(cudaMalloc(&st.d_proxy, pb), "the head copy");
        ck(cudaMalloc(&st.d_scratch, sb), "the head scratch");
        ck(cudaMemset(st.d_proxy, 0, pb), "the head copy");
        ck(cudaMemset(st.d_scratch, 0, sb), "the head scratch");
        bk::q80_proxy_pack(st.d_head, 0, st.vocab, st.d_proxy, st.vocab, nullptr);
        ck(cudaDeviceSynchronize(), "packing the head");
    }
}

// each layer's selections: row r of the window, `n` distinct ascending cells
void make_selections(Setup& st, int n) {
    for (int32_t* p : st.d_sel) cudaFree(p);
    st.d_sel.clear();
    st.sel.assign((size_t) st.d.layers, std::vector<int32_t>((size_t) (st.win_rows * st.cap), -1));
    std::vector<int32_t> all((size_t) st.cells);
    for (auto& layer : st.sel) {
        for (int r = 0; r < st.win_rows; ++r) {
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            std::sort(all.begin(), all.begin() + n);
            std::copy(all.begin(), all.begin() + n, layer.begin() + (size_t) (r * st.cap));
        }
        st.d_sel.push_back(upload(layer));
    }
}

struct Ref {
    std::vector<double> z, hp, craw;   // [R][N], [R - 1][N], [R - 1]
};

// The block of R rows by the contract's equations, the drafts' tokens taken from `ids` (the engine's picks).
Ref reference(const Setup& st, const std::vector<Cache>& caches, bool q8, int R, int32_t anchor, int row, int64_t pos,
              const std::vector<int32_t>& ids) {
    const core::DflyDims& d = st.d;
    const bk::QsaShapes s = bk::qsa_real_shapes();
    const size_t N = (size_t) d.n_embd, NH = (size_t) d.n_head, NKV = (size_t) d.n_head_kv, HD = (size_t) d.head_dim;
    const int nctx = (int) bk::qsa_selection_width(pos, s);
    const double base = bk::rope_scaling().freq_base;
    auto emb = [&](int32_t t) { return std::vector<double>(st.table.begin() + (size_t) t * N, st.table.begin() + (size_t) (t + 1) * N); };
    auto f = [&](const std::string& n) -> const std::vector<float>& { return st.w.f32.at(n); };
    // ---- rows: the fused anchor, then the mask rows
    std::vector<std::vector<double>> x((size_t) R);
    {
        const std::vector<double> e = emb(anchor);
        const std::vector<double> h(st.head_rows.begin() + (size_t) row * N, st.head_rows.begin() + (size_t) (row + 1) * N);
        std::vector<double> cat = rms(e.data(), N, f("mtp_fusion.enorm.weight"), d.rms_eps);
        const std::vector<double> hn = rms(h.data(), N, f("mtp_fusion.hnorm.weight"), d.rms_eps);
        cat.insert(cat.end(), hn.begin(), hn.end());
        const std::vector<double> fused = mvq(st.w, "mtp_fusion.eh_proj.weight", cat);
        x[0].resize(N);
        for (size_t i = 0; i < N; ++i) x[0][i] = (double) (float) e[i] + (double) f("mtp_fusion.gate")[i] * fused[i];
        for (int r = 1; r < R; ++r) x[(size_t) r] = emb(d.mask_id);
    }
    for (int l = 0; l < d.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        const Cache& c = caches[(size_t) l];
        std::vector<std::vector<double>> q((size_t) R), gate((size_t) R), kown((size_t) R), vown((size_t) R);
        for (int r = 0; r < R; ++r) {
            const std::vector<double> y = rms(x[(size_t) r].data(), N, f(p + "input_layernorm.weight"), d.rms_eps);
            const std::vector<double> qf = mvq(st.w, p + "self_attn.q_proj.weight", y);
            kown[(size_t) r] = mvq(st.w, p + "self_attn.k_proj.weight", y);
            vown[(size_t) r] = mvq(st.w, p + "self_attn.v_proj.weight", y);
            q[(size_t) r].resize(NH * HD);
            gate[(size_t) r].resize(NH * HD);
            for (size_t h = 0; h < NH; ++h) {
                // a head's row of the projection: the query half, then the gate half
                std::vector<double> qh = rms(qf.data() + h * 2 * HD, HD, f(p + "self_attn.q_norm.weight"), d.attn_eps);
                rope(qh.data(), d.n_rot, base, pos + r);
                std::copy(qh.begin(), qh.end(), q[(size_t) r].begin() + h * HD);
                std::copy(qf.begin() + h * 2 * HD + HD, qf.begin() + (h + 1) * 2 * HD, gate[(size_t) r].begin() + h * HD);
            }
            for (size_t h = 0; h < NKV; ++h) {
                std::vector<double> kh = rms(kown[(size_t) r].data() + h * HD, HD, f(p + "self_attn.k_norm.weight"), d.attn_eps);
                rope(kh.data(), d.n_rot, base, pos + r);
                std::copy(kh.begin(), kh.end(), kown[(size_t) r].begin() + h * HD);
            }
            store_round(kown[(size_t) r], q8);
            store_round(vown[(size_t) r], q8);
        }
        for (int r = 0; r < R; ++r) {
            std::vector<double> attn(NH * HD, 0.0);
            for (size_t h = 0; h < NH; ++h) {
                const size_t kvh = h / (NH / NKV);
                const double* qh = q[(size_t) r].data() + h * HD;
                // one softmax over the selected cells and the block's rows 0..r
                const int n = nctx + r + 1;
                std::vector<double> sc((size_t) n);
                auto key = [&](int j, size_t e) {
                    return j < nctx ? cell_value(c, q8, false, prow(s, st.sel[(size_t) l][(size_t) (row * st.cap + j)], (int64_t) kvh), e, HD)
                                    : kown[(size_t) (j - nctx)][kvh * HD + e];
                };
                auto val = [&](int j, size_t e) {
                    return j < nctx ? cell_value(c, q8, true, prow(s, st.sel[(size_t) l][(size_t) (row * st.cap + j)], (int64_t) kvh), e, HD)
                                    : vown[(size_t) (j - nctx)][kvh * HD + e];
                };
                double mx = -1e300;
                for (int j = 0; j < n; ++j) {
                    double dot = 0;
                    for (size_t e = 0; e < HD; ++e) dot += qh[e] * key(j, e);
                    sc[(size_t) j] = dot / std::sqrt((double) HD);
                    mx = std::max(mx, sc[(size_t) j]);
                }
                double sum = 0;
                for (double& v : sc) sum += (v = std::exp(v - mx));
                for (int j = 0; j < n; ++j) {
                    const double wgt = sc[(size_t) j] / sum;
                    if (wgt < 1e-14) continue;
                    for (size_t e = 0; e < HD; ++e) attn[h * HD + e] += wgt * val(j, e);
                }
                for (size_t e = 0; e < HD; ++e) attn[h * HD + e] *= 1.0 / (1.0 + std::exp(-gate[(size_t) r][h * HD + e]));
            }
            const std::vector<double> o = mvq(st.w, p + "self_attn.o_proj.weight", attn);
            for (size_t i = 0; i < N; ++i) x[(size_t) r][i] += o[i];
            const std::vector<double> u = rms(x[(size_t) r].data(), N, f(p + "post_attention_layernorm.weight"), d.rms_eps);
            const std::vector<double> gg = mvq(st.w, p + "mlp.gate_proj.weight", u), uu = mvq(st.w, p + "mlp.up_proj.weight", u);
            std::vector<double> hh(gg.size());
            for (size_t i = 0; i < gg.size(); ++i) hh[i] = gg[i] / (1.0 + std::exp(-gg[i])) * uu[i];
            const std::vector<double> dn = mvq(st.w, p + "mlp.down_proj.weight", hh);
            for (size_t i = 0; i < N; ++i) x[(size_t) r][i] += dn[i];
        }
    }
    Ref ref;
    for (int r = 0; r < R; ++r) {
        const std::vector<double> z = rms(x[(size_t) r].data(), N, f("final_norm.weight"), d.rms_eps);
        ref.z.insert(ref.z.end(), z.begin(), z.end());
    }
    for (int j = 0; j + 1 < R; ++j) {
        const double* z = ref.z.data() + (size_t) (j + 1) * N;
        const std::vector<double> pe = emb(j == 0 ? anchor : ids[(size_t) j - 1]);
        std::vector<double> cat = rms(z, N, f("hidden_correction.hidden_norm.weight"), d.rms_eps);
        const std::vector<double> en = rms(pe.data(), N, f("hidden_correction.embed_norm.weight"), d.rms_eps);
        cat.insert(cat.end(), en.begin(), en.end());
        const std::vector<double> gg = mvq(st.w, "hidden_correction.gate_proj.weight", cat),
                                  uu = mvq(st.w, "hidden_correction.up_proj.weight", cat);
        std::vector<double> hh(gg.size());
        for (size_t i = 0; i < gg.size(); ++i) hh[i] = gg[i] / (1.0 + std::exp(-gg[i])) * uu[i];
        const std::vector<double> dn = mvq(st.w, "hidden_correction.down_proj.weight", hh);
        double raw = 0;
        for (size_t i = 0; i < N; ++i) {
            ref.hp.push_back(z[i] + dn[i]);
            raw += ref.hp.back() * (double) f("confidence_head.proj.weight")[i];
        }
        ref.craw.push_back(raw);
    }
    return ref;
}

// the largest difference over the reference's largest magnitude
double rel_err(const std::vector<float>& got, const std::vector<double>& want) {
    double scale = 0, e = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        scale = std::max(scale, std::fabs(want[i]));
        e = std::max(e, std::fabs((double) got[i] - want[i]));
    }
    return e / (scale + 1e-12);
}

struct Block {
    std::vector<float> z, hp, craw;
    std::vector<int32_t> ids;
};

// A session of the target whose paired layers' caches are the setup's: only the K/V state the drafter reads is set.
struct Session {
    std::vector<core::QsaState> states;
    core::SessionState ss;
    Session(const Setup& st, const std::vector<Cache>& caches, bool q8) {
        const core::ModelGeometry g;
        states.resize((size_t) g.n_qsa_layers());
        for (int i = 0; i < st.d.layers; ++i) {
            const Cache& c = caches[(size_t) i];
            core::QsaState& q = states[(size_t) (st.d.target_layers[(size_t) i] / g.qsa_interval)];
            q.page_table = c.d_table;
            q.kv_int8 = q8;
            if (q8) { q.k_q = (int8_t*) c.d_k; q.v_q = (int8_t*) c.d_v; q.k_scale = (uint16_t*) c.d_ks; q.v_scale = (uint16_t*) c.d_vs; }
            else { q.k_pool = (uint16_t*) c.d_k; q.v_pool = (uint16_t*) c.d_v; }
        }
        ss.qsa_states = states.data();
        ss.layer_lo = 0;
        ss.layer_hi = g.n_layers;
        ss.qsa_ord0 = 0;
        ss.qsa_alloc = g.n_qsa_layers();
    }
};

// One drafter of `R` rows over pools of the form, bound to the setup's buffers.
struct Rig {
    core::DflyDrafter drafter;
    std::vector<bk::QsaAttnPools> pools;
    int seq_a = -1, seq_b = -1, win_rows = 0;
    bool make(Setup& st, const std::vector<Cache>& caches, bool q8, int R) {
        std::string err;
        core::LoadPlan plan;
        core::Loader loader;
        if (!drafter.load(st.pack, st.file.string(), R, q8, plan, err) || !loader.run(plan, err)) {
            std::fprintf(stderr, "dfly_parity: load: %s\n", err.c_str());
            return false;
        }
        core::DflyDrafter::Target t;
        t.n_vocab = st.vocab;
        const float* table = st.d_table;
        const int64_t row_bytes = (int64_t) st.d.n_embd * 4;
        t.embed = [table, row_bytes](const int32_t* tokens, int64_t n, float* out, void* stream) {
            bk::gather_rows((const uint8_t*) table, row_bytes, tokens, n, (uint8_t*) out, stream);
        };
        const Setup* sp = &st;
        t.pick = [sp](const void* x, int* out_id, float* out_prob, void* stream) {
            return bk::q80_proxy_top1(sp->d_proxy, sp->d_head, x, sp->vocab, sp->vocab, sp->d_map, out_id, out_prob, sp->d_scratch,
                                      stream, false);
        };
        std::vector<const int32_t*> sel(st.d_sel.begin(), st.d_sel.end());
        if (!drafter.bind(t, st.d_head_rows, sel.data(), st.cap, st.win_rows, err)) {
            std::fprintf(stderr, "dfly_parity: bind: %s\n", err.c_str());
            return false;
        }
        // the load took what plan_bytes says, and binding the target's calls as given takes nothing more
        if (drafter.vram_bytes() != core::DflyDrafter::plan_bytes(st.d, R, q8)) {
            std::fprintf(stderr, "dfly_parity: the drafter holds %llu bytes, its plan says %llu\n",
                         (unsigned long long) drafter.vram_bytes(), (unsigned long long) core::DflyDrafter::plan_bytes(st.d, R, q8));
            return false;
        }
        // the pools as a session of the target holds them
        Session ses(st, caches, q8);
        pools.resize((size_t) st.d.layers);
        if (!core::dfly_session_pools(ses.ss, core::ModelGeometry{}, st.d, q8, pools.data(), err)) {
            std::fprintf(stderr, "dfly_parity: the session's pools: %s\n", err.c_str());
            return false;
        }
        win_rows = st.win_rows;
        seq_a = drafter.add_sequence(pools.data(), err);
        seq_b = drafter.add_sequence(pools.data(), err);
        return seq_a >= 0 && seq_b >= 0;
    }
    // a block whose source row is `row` of the setup's window: the window began `row` + 1 positions before the anchor
    bool queue(int seq, int32_t anchor, int row, int64_t pos, std::string& err) {
        bool queued = false;
        if (!drafter.draft(seq, anchor, pos, pos - 1 - row, 0, win_rows, queued, err)) return false;
        if (!queued) err = "the window does not hold the block's source row";
        return queued;
    }
    // the device rows are the last block's, so they are read for a block queued alone
    bool run(int seq, int32_t anchor, int row, int64_t pos, Block& out) {
        std::string err;
        if (!queue(seq, anchor, row, pos, err) || !drafter.wait(err)) {
            std::fprintf(stderr, "dfly_parity: block: %s\n", err.c_str());
            return false;
        }
        const size_t N = (size_t) drafter.dims().n_embd;
        out.z = download(drafter.block_rows(), (size_t) drafter.rows() * N);
        out.hp = download(drafter.corrected_rows(), (size_t) drafter.drafts() * N);
        out.ids.resize((size_t) drafter.drafts());
        std::vector<float> conf((size_t) drafter.drafts());
        drafter.read(seq, out.ids.data(), conf.data());
        out.craw.assign(drafter.raw_conf(seq), drafter.raw_conf(seq) + drafter.drafts());
        return true;
    }
};

constexpr double kTol = 2e-2;   // the rounding of fp32 against double ahead of each q8_1 and cache rounding

void check(bool cond, const char* what) {
    if (cond) return;
    std::fprintf(stderr, "dfly_parity: FAILED %s\n", what);
    ++g_fail;
}

// A round's calls: a window that lacks the source row queues nothing, a slot's rows inside a wider window name the same
// row, a sequence drafts once a round, and the counters follow.
void check_rounds(const Setup& st, Rig& rig, int32_t anchor, int row, int64_t pos, const Block& want) {
    core::DflyDrafter& d = rig.drafter;
    std::string err;
    const int64_t rounds0 = d.rounds, skipped0 = d.skipped;
    bool queued = true;
    // the window ended a row early, or began after the source row: no block, and no wait is owed
    bool ok = d.draft(rig.seq_a, anchor, pos, pos - 1 - row, 0, row, queued, err) && !queued;
    queued = true;
    ok = ok && d.draft(rig.seq_a, anchor, pos, pos, 0, rig.win_rows, queued, err) && !queued && d.wait(err);
    check(ok && d.skipped == skipped0 + 2 && d.rounds == rounds0, "a window without the source row queues no block");
    // rows outside the bound window, and an unknown sequence, are errors
    err.clear();
    check(!d.draft(rig.seq_a, anchor, pos, pos - 1 - row, 1, rig.win_rows, queued, err) && !err.empty(), "rows past the bound window are refused");
    err.clear();
    check(!d.draft(99, anchor, pos, pos - 1 - row, 0, rig.win_rows, queued, err) && !err.empty(), "an unknown sequence is refused");
    // a slot's two rows from the window's row `row - 1`: the block reads the same row of the window
    err.clear();
    ok = d.draft(rig.seq_a, anchor, pos, pos - 2, row - 1, 2, queued, err) && queued;
    // the sequence's record is queued: a second block of it waits for the round's end
    std::string again;
    bool q2 = false;
    const bool twice = d.draft(rig.seq_a, anchor, pos, pos - 2, row - 1, 2, q2, again);
    ok = ok && d.wait(err);
    std::vector<int32_t> ids((size_t) d.drafts());
    std::vector<float> conf(ids.size());
    d.read(rig.seq_a, ids.data(), conf.data());
    check(ok && ids == want.ids && std::equal(want.craw.begin(), want.craw.end(), d.raw_conf(rig.seq_a)),
          "a slot's rows inside a window name the same source row");
    check(!twice && !again.empty(), "a sequence drafts once a round");
    check(d.rounds == rounds0 + 1 && d.ms_draft > 0.0, "a round that drafted is counted and timed");
    // the confidence is the sigmoid of the logit and the checkpoint's bias, a stated chance
    const double bias = (double) st.w.f32.at("confidence_head.proj.bias")[0];
    bool chance = true;
    for (size_t j = 0; j < conf.size(); ++j) {
        const double p = 1.0 / (1.0 + std::exp(-((double) d.raw_conf(rig.seq_a)[j] + bias)));
        chance = chance && conf[j] > 0.0f && conf[j] < 1.0f && std::fabs((double) conf[j] - p) < 1e-5;
    }
    check(chance, "each draft's confidence is the chance its logit and the bias state");
}

void run_form(Setup& st, bool q8) {
    const char* form = q8 ? "int8" : "fp16";
    std::vector<Cache> caches;
    for (int l = 0; l < st.d.layers; ++l) caches.push_back(make_cache(st, q8));
    // a short context, where the selection is every cell, and one of the widest selection
    for (const int64_t pos : {(int64_t) 300, (int64_t) 3000}) {
        const int nctx = (int) bk::qsa_selection_width(pos, bk::qsa_real_shapes());
        make_selections(st, nctx);
        const int R = core::kDflyBlockMax;
        const int32_t anchor = 17;
        const int row = 5;
        Rig full;
        Block a;
        if (!full.make(st, caches, q8, R) || !full.run(full.seq_a, anchor, row, pos, a)) { ++g_fail; return; }
        const Ref ref = reference(st, caches, q8, R, anchor, row, pos, a.ids);
        const double ez = rel_err(a.z, ref.z), eh = rel_err(a.hp, ref.hp), ec = rel_err(a.craw, ref.craw);
        std::printf("dfly_parity: %s, %d context cells, %d rows: rel err z %.3g, corrected %.3g, confidence logit %.3g\n", form,
                    nctx, R, ez, eh, ec);
        check(ez < kTol && eh < kTol && ec < kTol, "the block against the reference");
        check_rounds(st, full, anchor, row, pos, a);
        // ---- a shorter block gives the same leading rows
        for (const int r2 : {2, 5}) {
            Rig part;
            Block b;
            if (!part.make(st, caches, q8, r2) || !part.run(part.seq_a, anchor, row, pos, b)) { ++g_fail; return; }
            const bool same = std::equal(b.z.begin(), b.z.end(), a.z.begin()) && std::equal(b.hp.begin(), b.hp.end(), a.hp.begin()) &&
                              std::equal(b.ids.begin(), b.ids.end(), a.ids.begin());
            check(same, "a shorter block's rows are the full block's leading rows");
        }
        // ---- two sequences queued back to back each get their own record's result
        {
            std::string err;
            Block alone_b;
            const int32_t anchor_b = 911;
            const int row_b = 2;
            if (!full.run(full.seq_b, anchor_b, row_b, pos - 7, alone_b)) { ++g_fail; return; }
            const bool queued = full.queue(full.seq_a, anchor, row, pos, err) &&
                                full.queue(full.seq_b, anchor_b, row_b, pos - 7, err) && full.drafter.wait(err);
            std::vector<int32_t> ia((size_t) full.drafter.drafts()), ib(ia.size());
            std::vector<float> ca(ia.size()), cb(ia.size());
            full.drafter.read(full.seq_a, ia.data(), ca.data());
            full.drafter.read(full.seq_b, ib.data(), cb.data());
            const bool own = queued && ia == a.ids && ib == alone_b.ids &&
                             std::equal(a.craw.begin(), a.craw.end(), full.drafter.raw_conf(full.seq_a)) &&
                             std::equal(alone_b.craw.begin(), alone_b.craw.end(), full.drafter.raw_conf(full.seq_b));
            check(own, "two sequences queued back to back each get their own drafts");
            // the timing of a block, queued alone
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 20;
            cudaEventRecord(e0, nullptr);
            for (int i = 0; i < reps && queued; ++i)
                if (!full.queue(full.seq_a, anchor, row, pos, err) || !full.drafter.wait(err)) break;
            cudaEventRecord(e1, nullptr);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("dfly_parity: %s, %d context cells: %.3f ms per block of %d rows\n", form, nctx, ms / reps, R);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    }
}

// A head's rows as a model file of type 13 (Q5_K) or 14 (Q6_K) stores them: seeded codes in whole blocks, their scales
// set so the weights are of a head's size. The picks are checked against the head itself, so no value is kept.
std::vector<uint8_t> kquant_rows(int type, int vocab, int n_embd) {
    const size_t blk = type == 13 ? 176 : 210, blocks = (size_t) vocab * (size_t) (n_embd / 256);
    std::vector<uint8_t> rows(blocks * blk);
    for (size_t i = 0; i + 4 <= rows.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(rows.data() + i, &r, 4);
    }
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = rows.data() + b * blk;
        if (type == 13) {
            // d, dmin, twelve bytes of 6-bit scales and mins, the codes' high bits, their low four. A group of 32 codes
            // q is d * scale * q - dmin * min: with dmin = 15.5 d and min = scale its weights are centred on zero.
            const uint16_t d = bk::f16_from_f32(3.8e-4f), dmin = bk::f16_from_f32(15.5f * 3.8e-4f);
            std::memcpy(p, &d, 2);
            std::memcpy(p + 2, &dmin, 2);
            for (int i = 0; i < 4; ++i) {
                // groups i and i + 4: scales below 8, so the two high bits kept in the first eight bytes are zero
                const uint8_t lo = (uint8_t) (1 + (p[4 + i] & 3)), hi = (uint8_t) (1 + (p[8 + i] & 3));
                p[4 + i] = lo;
                p[8 + i] = lo;
                p[12 + i] = (uint8_t) (hi | (hi << 4));
            }
        } else {
            // the codes' low four bits and high two, sixteen 8-bit scales (kept small), d
            for (int i = 0; i < 16; ++i) p[192 + i] = (uint8_t) (int8_t) ((int) (p[192 + i] & 15) - 8);
            const uint16_t d = bk::f16_from_f32(2.3e-4f);
            std::memcpy(p + 208, &d, 2);
        }
    }
    return rows;
}

// The drafter bound to the native head and embedding table of a model file whose head is of ggml type `type`: the bytes
// the bind takes, the block against the reference, and each pick against the whole head's exact argmax.
void run_native(Setup& st, int type) {
    const core::DflyDims& d = st.d;
    const int N = d.n_embd, R = core::kDflyBlockMax;
    const int64_t pos = 300;
    const int32_t anchor = 17;
    const int row = 5;
    std::string err;
    // ---- the model file: the head's rows in its type, and the embedding table the reference reads
    const std::filesystem::path gguf =
        std::filesystem::temp_directory_path() / ("qwen4exp_dfly_parity_" + std::to_string(type) + ".gguf");
    {
        const std::vector<uint8_t> rows = type == 8 ? st.head_q8 : kquant_rows(type, st.vocab, N);
        auto bytes_of = [](const std::vector<uint8_t>* v) {
            return [v](uint8_t* dst, uint64_t n) { std::memcpy(dst, v->data(), (size_t) std::min<uint64_t>(n, v->size())); };
        };
        const std::vector<fixture::Kv> keys = {
            fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
        fixture::write(gguf, keys,
                       {{"output.weight", {(uint64_t) N, (uint64_t) st.vocab}, (uint32_t) type, 1, bytes_of(&rows)},
                        {"token_embd.weight", {(uint64_t) N, (uint64_t) st.vocab}, 8, 2, bytes_of(&st.embd_q8)}});
    }
    const std::vector<std::string> shards = {gguf.string()};
    // the head's type before anything is loaded: what a start decides the drafter on
    uint64_t planned = 0;
    int planned_type = -1;
    check(core::NativeHead::plan_bytes(shards, planned, err, &planned_type) && planned_type == type &&
              core::NativeHead::has_draft_top1(planned_type),
          "the head's type is read from its header");
    std::vector<Cache> caches;
    for (int l = 0; l < d.layers; ++l) caches.push_back(make_cache(st, false));
    make_selections(st, (int) bk::qsa_selection_width(pos, bk::qsa_real_shapes()));
    {
        core::NativeHead head;
        core::NativeEmbed embed;
        core::DflyDrafter drafter;
        core::LoadPlan plan;
        core::Loader loader;
        if (!head.load(shards, N, st.vocab, plan, err) || !embed.load(shards, N, st.vocab, plan, err) ||
            !drafter.load(st.pack, st.file.string(), R, false, plan, err) || !loader.run(plan, err) || !head.finish(err)) {
            std::fprintf(stderr, "dfly_parity: head type %d: load: %s\n", type, err.c_str());
            ++g_fail;
            return;
        }
        check(planned >= head.weight_bytes() && head.type() == type, "the head loads in the planned type");
        core::WeightTable wt, other;
        core::WeightTableFixture::put(wt, "output.weight", N, st.vocab);
        core::WeightTableFixture::put(other, "output.weight", N, st.vocab + 1);
        std::vector<const int32_t*> sel(st.d_sel.begin(), st.d_sel.end());
        // a bind without the embedding table, or over a head of another vocabulary, is refused and takes nothing
        const uint64_t before = drafter.vram_bytes();
        core::set_native_embed(nullptr);
        std::string refused;
        check(!drafter.bind(wt, &head, st.d_head_rows, sel.data(), st.cap, st.win_rows, refused) && !refused.empty(),
              "a bind without the embedding table is refused");
        core::set_native_embed(&embed);
        refused.clear();
        check(!drafter.bind(other, &head, st.d_head_rows, sel.data(), st.cap, st.win_rows, refused) && !refused.empty() &&
                  drafter.vram_bytes() == before,
              "a bind over a head of another vocabulary is refused");
        if (!drafter.bind(wt, &head, st.d_head_rows, sel.data(), st.cap, st.win_rows, err)) {
            std::fprintf(stderr, "dfly_parity: head type %d: bind: %s\n", type, err.c_str());
            core::set_native_embed(nullptr);
            ++g_fail;
            return;
        }
        check(drafter.vram_bytes() - before == core::DflyDrafter::bind_bytes(type, st.vocab) &&
                  core::DflyDrafter::bind_bytes(type, st.vocab) > 0,
              "the bind takes the bytes bind_bytes says");
        // ---- one block through the head and the table
        Session ses(st, caches, false);
        std::vector<bk::QsaAttnPools> pools((size_t) d.layers);
        bool queued = false;
        int seq = -1;
        if (!core::dfly_session_pools(ses.ss, core::ModelGeometry{}, d, false, pools.data(), err) ||
            (seq = drafter.add_sequence(pools.data(), err)) < 0 ||
            !drafter.draft(seq, anchor, pos, pos - 1 - row, 0, st.win_rows, queued, err) || !queued || !drafter.wait(err)) {
            std::fprintf(stderr, "dfly_parity: head type %d: block: %s\n", type, err.c_str());
            core::set_native_embed(nullptr);
            ++g_fail;
            return;
        }
        Block a;
        a.z = download(drafter.block_rows(), (size_t) R * (size_t) N);
        a.hp = download(drafter.corrected_rows(), (size_t) (R - 1) * (size_t) N);
        a.ids.resize((size_t) (R - 1));
        std::vector<float> conf((size_t) (R - 1));
        drafter.read(seq, a.ids.data(), conf.data());
        a.craw.assign(drafter.raw_conf(seq), drafter.raw_conf(seq) + (R - 1));
        bool in_vocab = true;
        for (int32_t id : a.ids) in_vocab = in_vocab && id >= 0 && id < st.vocab;
        check(in_vocab, "every draft is a token of the vocabulary");
        if (in_vocab) {
            const Ref ref = reference(st, caches, false, R, anchor, row, pos, a.ids);
            const double ez = rel_err(a.z, ref.z), eh = rel_err(a.hp, ref.hp), ec = rel_err(a.craw, ref.craw);
            std::printf("dfly_parity: native head type %d, %d rows: rel err z %.3g, corrected %.3g, confidence logit %.3g\n", type,
                        st.vocab, ez, eh, ec);
            check(ez < kTol && eh < kTol && ec < kTol, "the block through the native head and table against the reference");
        }
        // ---- each pick against the whole head: the exact logits of its corrected row over every row of the head
        float* d_logits = nullptr;
        void* d_xq = nullptr;
        ck(cudaMalloc((void**) &d_logits, (size_t) st.vocab * 4), "the logits");
        ck(cudaMalloc(&d_xq, bk::native_q8_1_bytes(N, 1)), "the q8_1 row");
        int agree = 0, worst_rank = 0;
        double worst_p = 0;
        try {
            for (int j = 0; in_vocab && j + 1 < R; ++j) {
                bk::native_quantize_q8_1(drafter.corrected_rows() + (size_t) j * (size_t) N, d_xq, N, 1, nullptr);
                head.score_rows(head.weights(), d_xq, d_logits, st.vocab, 1, nullptr);
                const std::vector<float> logits = download(d_logits, (size_t) st.vocab);
                const float mine = logits[(size_t) a.ids[(size_t) j]];
                // the pick's place among the exact logits, and its exact softmax probability
                int above = 0;
                double sum = 0;
                for (float v : logits) {
                    above += v > mine;
                    sum += std::exp((double) v - (double) mine);
                }
                agree += above == 0;
                worst_rank = std::max(worst_rank, above);
                worst_p = std::max(worst_p, std::fabs(1.0 / sum - (double) drafter.head_prob(seq)[j]));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "dfly_parity: head type %d: scoring the head: %s\n", type, e.what());
            ++g_fail;
        }
        std::printf("dfly_parity: native head type %d: %d of %d picks are the whole head's argmax, the worst has %d rows "
                    "above it, the pick's probability within %.3g of the exact one\n", type, agree, R - 1, worst_rank, worst_p);
        // The head is noise, where the 2-bit copy can leave the best row unscored; the copy's own test plants the
        // logits. A pick outside the head's best few rows means the copy, the map or the rows are not this head's.
        check(in_vocab && worst_rank < 8, "the picks are among the whole head's best rows");
        cudaFree(d_logits);
        cudaFree(d_xq);
        core::set_native_embed(nullptr);
    }
    for (Cache& c : caches) {
        cudaFree(c.d_k);
        cudaFree(c.d_v);
        cudaFree(c.d_table);
    }
    std::error_code ec;
    std::filesystem::remove(gguf, ec);
}

}  // namespace

int main(int argc, char** argv) {
    basalt::models::register_builtin_models();
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("dfly_parity: no CUDA device, skipped\n");
        return 77;
    }
    const int layers = argc > 1 ? std::atoi(argv[1]) : 2;
    const int vocab = argc > 2 ? std::atoi(argv[2]) : 4096;
    if (layers < 1 || layers > dfly2::kMaxLayers || vocab < 1024 || vocab > 1 << 20) {
        std::fprintf(stderr, "dfly_parity: 1..%d layers, 1024..%d vocabulary rows\n", dfly2::kMaxLayers, 1 << 20);
        return 2;
    }
    Setup st;
    st.vocab = vocab;
    build(st, layers);
    for (const bool q8 : {false, true}) run_form(st, q8);
    // the head's types with the two-stage argmax: Q8_0, Q5_K, Q6_K
    for (const int type : {8, 13, 14}) run_native(st, type);
    std::error_code ec;
    std::filesystem::remove(st.file, ec);
    if (g_fail) {
        std::fprintf(stderr, "dfly_parity: %d failed\n", g_fail);
        return 1;
    }
    std::printf("dfly_parity: ok\n");
    return 0;
}
