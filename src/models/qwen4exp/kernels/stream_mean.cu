// src/models/qwen4exp/kernels/stream_mean.cu - see include/basalt/models/qwen4exp/kernels/stream_mean.hpp.
#include "basalt/models/qwen4exp/kernels/stream_mean.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int THREADS = 256;

struct StreamMean {
    const float* R;     // T stacks at r_ld, stream c at c * n
    const float* bo;    // the pending write's block output, rows at bo_ld; null: R is complete
    const float* inj;   // its gate inputs, hc per row at inj_ld
    float* out;         // T rows of n
    int64_t T, r_ld, bo_ld, inj_ld;
    int n, hc;
};

// the fused hyper-connection read's gate (fused_gr.cu), so a write added here is bitwise the one that read folds
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// output i = t * n + d: column d of token t's streams in order, each with its pending write, then the mean
__device__ __forceinline__ void stream_mean_at(const StreamMean& a, int64_t i) {
    const int64_t t = i / a.n;
    const int d = (int) (i - t * a.n);
    const float* r = a.R + t * a.r_ld + d;
    const bool write = a.bo != nullptr;
    const float b = write ? a.bo[t * a.bo_ld + d] : 0.0f;
    float s = 0.0f;
    for (int c = 0; c < a.hc; ++c) {
        float v = r[(int64_t) c * a.n];
        if (write) v = fmaf(b, 2.0f * sigmoidf_(a.inj[t * a.inj_ld + c] / (float) a.hc), v);
        s += v;
    }
    a.out[i] = s / (float) a.hc;
}

// one thread per output value; the inputs are earlier kernels' outputs, read after the wait
__global__ void __launch_bounds__(THREADS) stream_mean_kernel(const StreamMean a) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i < a.T * a.n) stream_mean_at(a, i);
}

}  // namespace

void stream_mean(const float* R, int64_t T, int64_t r_ld, int64_t n_embd, int64_t hc, const float* bo, int64_t bo_ld,
                 const float* inj, int64_t inj_ld, float* out, void* stream) {
    if (T < 1) return;
    if (R == nullptr || out == nullptr || n_embd < 1 || hc < 1 || n_embd > INT_MAX / hc || r_ld < n_embd * hc)
        throw std::invalid_argument("stream_mean: a null buffer or a bad residual shape");
    if (bo != nullptr && (inj == nullptr || bo_ld < n_embd || inj_ld < hc))
        throw std::invalid_argument("stream_mean: the pending write needs its gate inputs and both row strides");
    // one launch's grid holds INT_MAX blocks
    if (T > (int64_t) INT_MAX * THREADS / n_embd) throw std::invalid_argument("stream_mean: too many rows for one launch");
    StreamMean a;
    a.R = R; a.bo = bo; a.inj = inj; a.out = out;
    a.T = T; a.r_ld = r_ld; a.bo_ld = bo_ld; a.inj_ld = inj_ld;
    a.n = (int) n_embd; a.hc = (int) hc;
    const unsigned blocks = (unsigned) ((T * n_embd + THREADS - 1) / THREADS);
    launch_pdl(stream_mean_kernel, dim3(blocks), dim3(THREADS), 0, (cudaStream_t) stream, a);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("stream_mean: launch failed");
}

}  // namespace basalt::kernels
