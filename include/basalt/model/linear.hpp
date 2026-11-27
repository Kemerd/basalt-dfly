// include/basalt/model/linear.hpp - one weight matrix as the shared stacks hold it on the device, and its product on the
// window gemm.
#pragma once

#include "basalt/kernels/window_gemm.hpp"

#include <cstdint>

namespace basalt::model {

constexpr int kTypeF8 = 44;            // F8_E4M3 projections (Linear::type)
constexpr int kTypeNvfp4Planes = 45;   // NVFP4_PLANES matrices (Linear::type, nvfp4_planes.hpp)
constexpr int kTypeQ8P = 108;          // Q8_0 in planes (window_gemm Format::Q8), made at load for the decode window

/// One projection: `rows` outputs of `cols` inputs, ggml blocks of `type` (Q8_0 8, NVFP4 40, F8_E4M3 44: one E4M3 code
/// per value, rows of `cols` bytes) on the device; `scale` an NVFP4 or F8 matrix's fp32 tensor scale (device), null for 1.
/// An NVFP4 product leaves the scale to the step after it; an F8 product applies it itself.  The head alone may be F8
/// with a scale per row (`row_scales`, fp32 [rows] on the device, `scale` then null): it runs on the window gemm only.
struct Linear {
    int type = -1;
    const void* w = nullptr;
    int64_t rows = 0, cols = 0;
    const float* scale = nullptr;
    const float* row_scales = nullptr;
};

/// Whether products of a matrix of `type` run on the window gemm (which applies the tensor scale itself).
inline bool on_wg(int type) { return type == kTypeF8 || type == kTypeNvfp4Planes || type == kTypeQ8P; }

/// A window gemm call for a matrix of `l`'s format and width.
inline kernels::wgemm::Call wcall(const Linear& l, kernels::wgemm::Epilogue epi = kernels::wgemm::Epilogue::Store) {
    namespace wg = kernels::wgemm;
    wg::Call c;
    c.fmt = l.type == kTypeF8 ? wg::Format::F8 : l.type == kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
    c.K = l.cols;
    c.epi = epi;
    return c;
}
/// `l` as one matrix of such a call, its output rows in `y`.
inline kernels::wgemm::Matrix wmat(const Linear& l, float* y) {
    kernels::wgemm::Matrix w{(const uint8_t*) l.w, l.rows, l.scale, y, l.rows};
    w.row_scales = l.row_scales;
    return w;
}

}  // namespace basalt::model
