// src/dfly2/kernels.cu - the one kernel the shared DFly2 steps add to the shared ones (ops.hpp): the fusion's gated add.
#include "basalt/dfly2/ops.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::dfly2 {
namespace {

constexpr int kThreads = 256;

// element i of the gated residual
__device__ __forceinline__ void gate_add_stage(int i, const float* __restrict__ base, const float* __restrict__ fused,
                                               const float* __restrict__ gate, int n, float* __restrict__ out) {
    if (i < n) out[i] = base[i] + gate[i] * fused[i];
}

__global__ void __launch_bounds__(kThreads) gate_add_kernel(const float* __restrict__ base, const float* __restrict__ fused,
                                                            const float* __restrict__ gate, int n, float* __restrict__ out) {
    gate_add_stage(blockIdx.x * blockDim.x + threadIdx.x, base, fused, gate, n, out);
}

}  // namespace

void gate_add(const float* base, const float* fused, const float* gate, int n, float* out, void* stream) {
    if (!base || !fused || !gate || !out) throw std::invalid_argument("dfly2::gate_add: a null buffer");
    if (n <= 0) throw std::invalid_argument("dfly2::gate_add: n");
    gate_add_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(base, fused, gate, n, out);
}

}  // namespace basalt::dfly2
