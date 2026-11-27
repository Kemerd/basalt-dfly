// include/basalt/model/linear.hpp - one weight matrix as the shared stacks hold it on the device.
#pragma once

#include <cstdint>

namespace basalt::model {

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

}  // namespace basalt::model
