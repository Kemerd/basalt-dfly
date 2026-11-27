// tests/models/qwen4exp/kernels/stream_mean_parity.cpp - the stream-mean kernel (stream_mean.hpp) at the real geometry, for a
// verify window's 1 to 8 rows and a prompt batch's row count:
//   1. no pending write: bitwise the mean of R's streams summed in order, and against double precision.
//   2. a pending write: bitwise the mean of the residual the fused read's folded write leaves (fused_gr_read with
//      apply), R untouched; against gr_write followed by the mean and against double precision.
//   3. a row's bits depend neither on the row count nor on the row's place in the call.
//   4. row strides wider than the rows give the same bits, and bad arguments are refused before a launch.
//   5. inside a PdlScope the kernel reads the output of the kernel before it.
//
//     build/stream_mean_parity [--bench]     --bench adds us per call and GB/s over the bytes a call moves

#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"
#include "basalt/models/qwen4exp/kernels/gr.hpp"
#include "basalt/models/qwen4exp/kernels/stream_mean.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = basalt::kernels;

namespace {

constexpr int64_t N = 2560, HC = 4, LR = 320, D = HC * N;
constexpr int64_t kWin = k::kVerifyMaxT;      // a verify window's most rows
constexpr int64_t kMax = k::wgemm::kRows;     // a prompt batch's rows

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

template <typename T>
void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// the kernel's mean on the host: T stacks' streams summed in fp32 in order from stream 0, then the division
std::vector<float> host_mean(const std::vector<float>& R, int64_t T) {
    std::vector<float> m((size_t) (T * N));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < N; ++d) {
            float s = 0.0f;
            for (int64_t c = 0; c < HC; ++c) s += R[(size_t) (t * D + c * N + d)];
            m[(size_t) (t * N + d)] = s / (float) HC;
        }
    return m;
}

// the operation in double precision: the stream mean of R, or of R + bo * 2 sigmoid(inj / hc)
std::vector<double> double_mean(const std::vector<float>& R, const std::vector<float>& bo, const std::vector<float>& inj,
                                int64_t T, bool write) {
    std::vector<double> m((size_t) (T * N));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < N; ++d) {
            double s = 0.0;
            for (int64_t c = 0; c < HC; ++c) {
                double v = R[(size_t) (t * D + c * N + d)];
                if (write) {
                    const double w = 2.0 / (1.0 + std::exp(-(double) inj[(size_t) (t * HC + c)] / (double) HC));
                    v += (double) bo[(size_t) (t * N + d)] * w;
                }
                s += v;
            }
            m[(size_t) (t * N + d)] = s / (double) HC;
        }
    return m;
}

// the worst |got - want|, printed with its share of the largest |want|; true within `rel` of that magnitude
template <typename Ref>
bool within(const std::vector<float>& got, const std::vector<Ref>& want, double rel, const char* what) {
    double worst = 0.0, mag = 1e-30;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::fmax(worst, std::fabs((double) got[i] - (double) want[i]));
        mag = std::fmax(mag, std::fabs((double) want[i]));
    }
    std::printf("    %s: max |err| %.3g, relative to max |ref| %.3g\n", what, worst, worst / mag);
    return worst <= rel * mag;
}

template <typename F>
bool refuses(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(20251);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // kMax tokens: residual stacks whose streams differ in scale, a block output and gate inputs on both sides of 0
    std::vector<float> R((size_t) (kMax * D)), bo((size_t) (kMax * N)), inj((size_t) (kMax * HC));
    for (int64_t t = 0; t < kMax; ++t)
        for (int64_t c = 0; c < HC; ++c)
            for (int64_t d = 0; d < N; ++d) R[(size_t) (t * D + c * N + d)] = nd(rng) * (float) (1 << c);
    for (auto& x : bo) x = nd(rng);
    for (auto& x : inj) x = nd(rng) * 3.0f;
    float* dR = dalloc<float>(R.size());
    float* dbo = dalloc<float>(bo.size());
    float* dinj = dalloc<float>(inj.size());
    float* dout = dalloc<float>((size_t) (kMax * N));
    up(dR, R);
    up(dbo, bo);
    up(dinj, inj);
    const float poison = std::nanf("");
    const std::vector<float> blank((size_t) (kMax * N), poison);

    // rows [r0, r0 + T) of the inputs in one call, with or without the pending write
    auto run = [&](int64_t r0, int64_t T, bool write) {
        up(dout, blank);
        k::stream_mean(dR + r0 * D, T, D, N, HC, write ? dbo + r0 * N : nullptr, N, write ? dinj + r0 * HC : nullptr, HC,
                       dout, nullptr);
        ck(cudaDeviceSynchronize(), "stream_mean");
        return down(dout, (size_t) (T * N));
    };

    // ---- 1. no pending write
    std::printf("no pending write\n");
    const std::vector<float> plain = run(0, kMax, false);
    check(same_bits(plain.data(), host_mean(R, kMax).data(), plain.size()), "bitwise the streams' sum in order over hc");
    check(within(plain, double_mean(R, bo, inj, kMax, false), 1e-5, "against double precision"),
          "the stream mean in double precision");

    // ---- 2. a pending write
    std::printf("a pending write\n");
    const std::vector<float> pending = run(0, kMax, true);
    check(same_bits(down(dR, R.size()).data(), R.data(), R.size()), "R is left as it was");
    {
        // the write the next layer's first read folds, one token at a time on random bf16 weights
        std::vector<uint16_t> wd((size_t) (LR * D)), wu((size_t) (D * LR)), wi((size_t) (HC * D));
        std::vector<float> wn((size_t) D);
        auto bf = [&](float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) (u >> 16); };
        for (auto& x : wd) x = bf(nd(rng) * 0.02f);
        for (auto& x : wu) x = bf(nd(rng) * 0.02f);
        for (auto& x : wi) x = bf(nd(rng) * 0.02f);
        for (auto& x : wn) x = 1.0f + 0.1f * nd(rng);
        uint16_t* dwd = dalloc<uint16_t>(wd.size());
        uint16_t* dwu = dalloc<uint16_t>(wu.size());
        uint16_t* dwi = dalloc<uint16_t>(wi.size());
        float* dwn = dalloc<float>(wn.size());
        up(dwd, wd); up(dwu, wu); up(dwi, wi); up(dwn, wn);
        float* lo = dalloc<float>(LR);
        float* rs = dalloc<float>(HC);
        float* inj_out = dalloc<float>(HC);
        float* mixed = dalloc<float>(N);
        float* dF = dalloc<float>(R.size());
        up(dF, R);
        for (int64_t t = 0; t < kMax; ++t) {
            k::FusedGrArgs a;
            a.R = dF + t * D; a.R_out = dF + t * D; a.apply = true;
            a.bo_prev = dbo + t * N; a.inj_prev = dinj + t * HC;
            a.w_norm = dwn; a.w_down = dwd; a.w_up = dwu; a.w_inject = dwi;
            a.lo = lo; a.rs = rs; a.inject_out = inj_out; a.mixed = mixed;
            k::fused_gr_read(a, nullptr);
        }
        ck(cudaDeviceSynchronize(), "fused read");
        check(same_bits(pending.data(), host_mean(down(dF, R.size()), kMax).data(), pending.size()),
              "bitwise the mean of the residual the fused read's folded write leaves");
        for (void* p : {(void*) dwd, (void*) dwu, (void*) dwi, (void*) dwn, (void*) lo, (void*) rs, (void*) inj_out,
                        (void*) mixed, (void*) dF})
            ck(cudaFree(p), "free");
    }
    {
        // the unfused write, then the mean
        float* dG = dalloc<float>(R.size());
        up(dG, R);
        const k::GrShapes gs{N, HC, LR};
        for (int64_t t = 0; t < kMax; ++t) k::gr_write(dG + t * D, dbo + t * N, dinj + t * HC, gs, dG + t * D, nullptr);
        check(within(pending, host_mean(down(dG, R.size()), kMax), 1e-5, "against gr_write, then the mean"),
              "gr_write followed by the stream mean");
        ck(cudaFree(dG), "free");
    }
    check(within(pending, double_mean(R, bo, inj, kMax, true), 1e-5, "against double precision"),
          "the write and the stream mean in double precision");

    // ---- 3. the row count and the row's place
    std::printf("rows\n");
    for (int w = 0; w < 2; ++w) {
        const std::vector<float>& full = w ? pending : plain;
        bool first = true, last = true;
        for (int64_t T = 1; T <= kWin; ++T) {
            first = first && same_bits(run(0, T, w != 0).data(), full.data(), (size_t) (T * N));
            last = last && same_bits(run(kMax - T, T, w != 0).data(), full.data() + (kMax - T) * N, (size_t) (T * N));
        }
        check(first, w ? "pending write: 1 to 8 rows are the batch's first rows, bitwise"
                       : "no pending write: 1 to 8 rows are the batch's first rows, bitwise");
        check(last, w ? "pending write: a row's bits do not depend on its place in the call"
                      : "no pending write: a row's bits do not depend on its place in the call");
    }

    // ---- 4. strides and arguments
    std::printf("strides and arguments\n");
    {
        // every row followed by values the kernel must not read
        constexpr int64_t r_ld = D + 64, bo_ld = N + 32, inj_ld = HC + 4;
        std::vector<float> Rw((size_t) (kWin * r_ld), 1e30f), bw((size_t) (kWin * bo_ld), 1e30f),
            iw((size_t) (kWin * inj_ld), 1e30f);
        for (int64_t t = 0; t < kWin; ++t) {
            std::memcpy(Rw.data() + t * r_ld, R.data() + t * D, (size_t) D * sizeof(float));
            std::memcpy(bw.data() + t * bo_ld, bo.data() + t * N, (size_t) N * sizeof(float));
            std::memcpy(iw.data() + t * inj_ld, inj.data() + t * HC, (size_t) HC * sizeof(float));
        }
        float* dRw = dalloc<float>(Rw.size());
        float* dbw = dalloc<float>(bw.size());
        float* diw = dalloc<float>(iw.size());
        up(dRw, Rw); up(dbw, bw); up(diw, iw);
        k::stream_mean(dRw, kWin, r_ld, N, HC, dbw, bo_ld, diw, inj_ld, dout, nullptr);
        ck(cudaDeviceSynchronize(), "strides");
        check(same_bits(down(dout, (size_t) (kWin * N)).data(), pending.data(), (size_t) (kWin * N)),
              "wider row strides give the same bits");
        ck(cudaFree(dRw), "free"); ck(cudaFree(dbw), "free"); ck(cudaFree(diw), "free");
    }
    check(refuses([&] { k::stream_mean(nullptr, 1, D, N, HC, nullptr, 0, nullptr, 0, dout, nullptr); }) &&
          refuses([&] { k::stream_mean(dR, 1, D, N, HC, nullptr, 0, nullptr, 0, nullptr, nullptr); }) &&
          refuses([&] { k::stream_mean(dR, 1, D - 1, N, HC, nullptr, 0, nullptr, 0, dout, nullptr); }) &&
          refuses([&] { k::stream_mean(dR, 1, D, N, HC, dbo, N, nullptr, HC, dout, nullptr); }) &&
          refuses([&] { k::stream_mean(dR, 1, D, N, HC, dbo, N - 1, dinj, HC, dout, nullptr); }),
          "a null buffer, a short stride or a write without its gates is refused");

    // ---- 5. a programmatic dependent: the second launch takes the first one's rows as one-stream stacks
    std::printf("programmatic launch\n");
    {
        cudaStream_t st = nullptr;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        float* dmid = dalloc<float>((size_t) (kMax * N));
        bool same = true;
        for (int it = 0; it < 64 && same; ++it) {
            ck(cudaMemcpyAsync(dmid, blank.data(), blank.size() * sizeof(float), cudaMemcpyHostToDevice, st), "blank");
            {
                const k::PdlScope scope(true, true);   // the first launch follows the copy
                k::stream_mean(dR, kMax, D, N, HC, dbo, N, dinj, HC, dmid, st);
                k::stream_mean(dmid, kMax, N, N, 1, nullptr, 0, nullptr, 0, dout, st);
            }
            ck(cudaStreamSynchronize(st), "programmatic launch");
            same = same_bits(down(dout, (size_t) (kMax * N)).data(), pending.data(), pending.size());
        }
        check(same, "behind a kernel in a PdlScope it reads that kernel's finished rows");
        ck(cudaFree(dmid), "free");
        ck(cudaStreamDestroy(st), "stream");
    }

    if (bench) {
        std::printf("bench\n");
        cudaStream_t st = nullptr;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        constexpr int kIters = 2000;
        for (const int64_t T : {kWin, kMax})
            for (int w = 0; w < 2; ++w) {
                auto call = [&] {
                    k::stream_mean(dR, T, D, N, HC, w ? dbo : nullptr, N, w ? dinj : nullptr, HC, dout, st);
                };
                for (int i = 0; i < 20; ++i) call();
                ck(cudaStreamSynchronize(st), "warm");
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kIters; ++i) call();
                ck(cudaStreamSynchronize(st), "bench");
                const std::chrono::duration<double, std::micro> took = std::chrono::steady_clock::now() - t0;
                const double us = took.count() / kIters;
                // the stacks read, the rows written, and with a pending write its block output and gate inputs
                const double bytes = (double) (T * (D + N) + (w ? T * (N + HC) : 0)) * sizeof(float);
                std::printf("  %2lld rows, %s: %.2f us per call, %.1f GB/s over %.0f bytes\n", (long long) T,
                            w ? "pending write   " : "no pending write", us, bytes / us * 1e-3, bytes);
            }
        ck(cudaStreamDestroy(st), "stream");
    }

    for (void* p : {(void*) dR, (void*) dbo, (void*) dinj, (void*) dout}) ck(cudaFree(p), "free");
    std::printf(g_fail ? "stream_mean_parity: %d FAILED\n" : "stream_mean_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
