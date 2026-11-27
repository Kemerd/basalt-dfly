// tests/dfly2/ops_parity.cpp - GPU parity of the steps the DFly2 drafters share (basalt/dfly2/ops.hpp) against a
// double-precision CPU reference: the gated add, the Gated Fusion and the Hidden-Correction pass, on random inputs at a
// small width and at a drafter's real width, with a microbench line for each. The reference dequantises the Q8_0
// projections and rounds their inputs as the q8_1 activations are rounded. Synthetic: no model.
#include "basalt/core/device_copy.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/dfly2/ops.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace basalt;

static int g_fail = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "ops_parity: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                                \
        }                                                                            \
    } while (0)

namespace {

std::mt19937 rng(4321);
float gauss(float sd) { return std::normal_distribution<float>(0.f, sd)(rng); }

uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

std::vector<float> rand_vec(size_t n, float mean, float sd) {
    std::vector<float> v(n);
    for (float& x : v) x = mean + gauss(sd);
    return v;
}

/// A projection [rows][cols] as the loader rounds it: BF16 values, then Q8_0 blocks.
struct Q8 {
    uint64_t rows = 0, cols = 0;
    std::vector<uint8_t> blocks;
};
Q8 rand_q8(uint64_t rows, uint64_t cols, float sd) {
    Q8 m;
    m.rows = rows;
    m.cols = cols;
    std::vector<uint16_t> bits(rows * cols);
    for (uint16_t& b : bits) b = to_bf16(gauss(sd));
    m.blocks.resize(rows * cols / 32 * 34);
    core::quantize_bf16_q8_0(bits.data(), bits.size(), m.blocks.data());
    return m;
}

// W x as MMVQ makes it: x in q8_1 blocks (q = round(x / d), d = amax / 127 kept as fp16), an integer dot per block,
// scaled by both blocks' fp16 d
std::vector<double> mvq(const Q8& w, const std::vector<double>& x) {
    const uint64_t nb = w.cols / 32;
    std::vector<int> qx(w.cols);
    std::vector<double> dx(nb);
    for (uint64_t b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs((float) x[b * 32 + i]));
        const float d = amax / 127.f;
        for (int i = 0; i < 32; ++i) qx[b * 32 + i] = amax == 0.f ? 0 : (int) std::round((float) x[b * 32 + i] / d);
        dx[b] = kernels::f32_from_f16(kernels::f16_from_f32(d));
    }
    std::vector<double> y(w.rows);
    for (uint64_t r = 0; r < w.rows; ++r) {
        double s = 0;
        for (uint64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = w.blocks.data() + (r * nb + b) * 34;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            int dot = 0;
            for (int i = 0; i < 32; ++i) dot += (int) (int8_t) blk[2 + i] * qx[b * 32 + i];
            s += (double) kernels::f32_from_f16(dh) * dx[b] * dot;
        }
        y[r] = s;
    }
    return y;
}

// x * rsqrt(mean(x^2) + eps) * w, appended to `out`
void rms(const std::vector<float>& x, const std::vector<float>& w, double eps, std::vector<double>& out) {
    double ss = 0;
    for (float v : x) ss += (double) v * v;
    const double inv = 1.0 / std::sqrt(ss / (double) x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) out.push_back((double) x[i] * inv * w[i]);
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size() * sizeof(T), 16));
    core::device_copy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
float* scratch(size_t floats) {
    float* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(floats * 4, 16));
    return d;
}
std::vector<float> download(const float* d, size_t n) {
    std::vector<float> h(n);
    core::device_copy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

struct Err {
    double abs = 0, rel = 0;
};
Err compare(const std::vector<float>& got, const std::vector<double>& want) {
    Err e;
    double scale = 0;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    for (size_t i = 0; i < want.size(); ++i) e.abs = std::max(e.abs, std::fabs((double) got[i] - want[i]));
    e.rel = e.abs / (scale + 1e-12);
    return e;
}

// microseconds a call of `f` queued `calls` times on `st`
template <class F>
double time_us(cudaStream_t st, int calls, F&& f) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    f();
    cudaEventRecord(e0, st);
    for (int i = 0; i < calls; ++i) f();
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1e3 * ms / calls;
}

void test_gate_add(cudaStream_t st) {
    for (int n : {1, 300, 2560}) {
        const std::vector<float> a = rand_vec((size_t) n, 0, 1), b = rand_vec((size_t) n, 0, 1), g = rand_vec((size_t) n, 0, 1);
        float *da = upload(a), *db = upload(b), *dg = upload(g), *dr = scratch((size_t) n + 1);
        const std::vector<float> guard = {123.f};
        core::device_copy(dr + n, guard.data(), 4, cudaMemcpyHostToDevice);
        dfly2::gate_add(da, db, dg, n, dr, st);
        cudaStreamSynchronize(st);
        std::vector<double> want((size_t) n);
        for (int i = 0; i < n; ++i) want[(size_t) i] = (double) a[(size_t) i] + (double) g[(size_t) i] * b[(size_t) i];
        const std::vector<float> got = download(dr, (size_t) n + 1);
        const Err e = compare(std::vector<float>(got.begin(), got.end() - 1), want);
        std::printf("gate_add n %d: max abs %.2e rel %.2e\n", n, e.abs, e.rel);
        CHECK(e.abs < 1e-6 && got.back() == 123.f);   // one fp32 multiply-add, and nothing past n is written
        if (n == 2560) {
            const double us = time_us(st, 500, [&] { dfly2::gate_add(da, db, dg, n, dr, st); });
            std::printf("gate_add n %d: %.2f us a call, %.2f GB/s of the four rows\n", n, us, 4.0 * n * 4 / us / 1e3);
        }
        cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dr);
    }
    bool threw = false;
    try { dfly2::gate_add(nullptr, nullptr, nullptr, 4, nullptr, st); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_fusion(cudaStream_t st) {
    const float eps = 1e-5f;
    for (int E : {64, 2560}) {
        const Q8 proj = rand_q8((uint64_t) E, (uint64_t) 2 * E, 0.02f);
        const std::vector<float> emb = rand_vec((size_t) E, 0, 0.05f), hid = rand_vec((size_t) E, 0, 3.f),
                                 enorm = rand_vec((size_t) E, 1, 0.2f), hnorm = rand_vec((size_t) E, 1, 0.2f),
                                 gate = rand_vec((size_t) E, 0, 0.5f);
        void* dproj = upload(proj.blocks);
        float *demb = upload(emb), *dhid = upload(hid), *den = upload(enorm), *dhn = upload(hnorm), *dg = upload(gate);
        float *cat = scratch((size_t) 2 * E), *fused = scratch((size_t) E), *out = scratch((size_t) E);
        void* cat_q8 = scratch(kernels::native_q8_1_bytes(2 * E) / 4 + 1);
        const dfly2::FusionWeights w{dproj, den, dhn, dg};
        dfly2::gated_fusion(w, demb, dhid, E, eps, cat, cat_q8, fused, out, st);
        cudaStreamSynchronize(st);
        std::vector<double> c;
        rms(emb, enorm, eps, c);
        rms(hid, hnorm, eps, c);
        const std::vector<double> f = mvq(proj, c);
        std::vector<double> want((size_t) E);
        for (int i = 0; i < E; ++i) want[(size_t) i] = (double) emb[(size_t) i] + (double) gate[(size_t) i] * f[(size_t) i];
        const Err e = compare(download(out, (size_t) E), want);
        std::printf("gated_fusion E %d: max abs %.2e rel %.2e\n", E, e.abs, e.rel);
        CHECK(e.rel < 2e-4);
        if (E == 2560) {
            const double us = time_us(st, 200, [&] { dfly2::gated_fusion(w, demb, dhid, E, eps, cat, cat_q8, fused, out, st); });
            std::printf("gated_fusion E %d: %.1f us a call, %.1f GB/s of projection bytes\n", E, us, (double) proj.blocks.size() / us / 1e3);
        }
        cudaFree(dproj); cudaFree(demb); cudaFree(dhid); cudaFree(den); cudaFree(dhn); cudaFree(dg);
        cudaFree(cat); cudaFree(fused); cudaFree(out); cudaFree(cat_q8);
    }
    bool threw = false;
    try {
        dfly2::gated_fusion(dfly2::FusionWeights{}, nullptr, nullptr, 64, eps, nullptr, nullptr, nullptr, nullptr, st);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_correction(cudaStream_t st) {
    const float eps = 1e-5f;
    for (int E : {64, 2560})
        for (int F : {32, E}) {
            const Q8 gate = rand_q8((uint64_t) F, (uint64_t) 2 * E, 0.02f), up = rand_q8((uint64_t) F, (uint64_t) 2 * E, 0.02f),
                     down = rand_q8((uint64_t) E, (uint64_t) F, 0.02f);
            const std::vector<float> z = rand_vec((size_t) E, 0, 1.f), pe = rand_vec((size_t) E, 0, 0.05f),
                                     hn = rand_vec((size_t) E, 1, 0.2f), en = rand_vec((size_t) E, 1, 0.2f);
            void *dgate = upload(gate.blocks), *dup = upload(up.blocks), *ddown = upload(down.blocks);
            float *dz = upload(z), *dpe = upload(pe), *dhn = upload(hn), *den = upload(en);
            float *cat = scratch((size_t) 2 * E), *g = scratch((size_t) F), *u = scratch((size_t) F), *h = scratch((size_t) F),
                  *delta = scratch((size_t) E);
            void *cat_q8 = scratch(kernels::native_q8_1_bytes(2 * E) / 4 + 1), *h_q8 = scratch(kernels::native_q8_1_bytes(F) / 4 + 1);
            const dfly2::CorrectionWeights w{dgate, dup, ddown, dhn, den, F};
            dfly2::correction_mlp(w, dz, dpe, E, eps, cat, cat_q8, g, u, h, h_q8, delta, st);
            cudaStreamSynchronize(st);
            std::vector<double> c;
            rms(z, hn, eps, c);
            rms(pe, en, eps, c);
            const std::vector<double> gv = mvq(gate, c), uv = mvq(up, c);
            std::vector<double> hv((size_t) F);
            for (int i = 0; i < F; ++i) hv[(size_t) i] = gv[(size_t) i] / (1.0 + std::exp(-gv[(size_t) i])) * uv[(size_t) i];
            const Err e = compare(download(delta, (size_t) E), mvq(down, hv));
            std::printf("correction_mlp E %d ffn %d: max abs %.2e rel %.2e\n", E, F, e.abs, e.rel);
            // a value that lies at a rounding edge of the inner row's q8_1 image may fall to the other side in fp32
            CHECK(e.rel < 2e-3);
            if (E == 2560 && F == E) {
                const double us = time_us(st, 200, [&] { dfly2::correction_mlp(w, dz, dpe, E, eps, cat, cat_q8, g, u, h, h_q8, delta, st); });
                const double bytes = (double) (gate.blocks.size() + up.blocks.size() + down.blocks.size());
                std::printf("correction_mlp E %d ffn %d: %.1f us a call, %.1f GB/s of projection bytes\n", E, F, us, bytes / us / 1e3);
            }
            cudaFree(dgate); cudaFree(dup); cudaFree(ddown); cudaFree(dz); cudaFree(dpe); cudaFree(dhn); cudaFree(den);
            cudaFree(cat); cudaFree(g); cudaFree(u); cudaFree(h); cudaFree(delta); cudaFree(cat_q8); cudaFree(h_q8);
        }
    bool threw = false;
    try {
        dfly2::correction_mlp(dfly2::CorrectionWeights{}, nullptr, nullptr, 64, eps, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, st);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

}  // namespace

int main() {
    cudaStream_t st = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess) {
        std::fprintf(stderr, "ops_parity: no CUDA stream\n");
        return 1;
    }
    test_gate_add(st);
    test_fusion(st);
    test_correction(st);
    cudaStreamDestroy(st);
    if (g_fail) {
        std::fprintf(stderr, "ops_parity: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("ops_parity: ok\n");
    return 0;
}
