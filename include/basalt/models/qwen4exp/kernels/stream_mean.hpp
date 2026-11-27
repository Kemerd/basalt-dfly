// include/basalt/models/qwen4exp/kernels/stream_mean.hpp - the mean over the residual's hc streams, one n_embd row per
// token: the layer output a reader outside the hyper-connection takes.
//
//   out[t][d] = (1 / hc) * sum over c of R'[t][c][d]
//
// R' is the residual after the layer's FFN write.  Where that write is still pending (a verify window folds it into
// the next layer's first read) the kernel adds it per stream without writing R:
//
//   R'[t][c][d] = R[t][c][d] + bo[t][d] * 2 sigmoid(inj[t][c] / hc)
//
// with the fused read's own arithmetic (fused_gr.hpp, hc_fused.hpp), so the mean is of the bits that read goes on to
// write.  The streams are summed in order from stream 0.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// The stream mean of T tokens' residual stacks (`R + t * r_ld`, hc streams of n_embd, stream c at c * n_embd) into
/// `out + t * n_embd`.  With `bo`, the pending FFN write (`bo + t * bo_ld`, n_embd values, gated per stream by
/// `inj + t * inj_ld`, hc values) is added first and R is left as it is; with a null `bo`, R must already hold the
/// layer's output and `inj` is not read.  `out` overlaps none of the inputs.  A row's values do not depend on T or on
/// the row's place in the call.  Launched through launch_pdl (pdl.hpp): inside a PdlScope it follows a kernel directly.
/// Throws std::invalid_argument on a bad shape or pointer, std::runtime_error when the launch fails.
void stream_mean(const float* R, int64_t T, int64_t r_ld, int64_t n_embd, int64_t hc, const float* bo, int64_t bo_ld,
                 const float* inj, int64_t inj_ld, float* out, void* stream);

}  // namespace basalt::kernels
