#include "state.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::head_input(const float* R_rows, int64_t n, uint16_t* out, std::string& err, const float** rows) {
    Impl& m = *impl_;
    if (R_rows == nullptr || out == nullptr || m.wt == nullptr || n < 1 || n > m.T) {
        err = "prefill: the head's input needs the rows of one chunk";
        return false;
    }
    // the output hyper-connection, in the forms a layer's read takes its own
    const core::WeightRef *wn = m.wt->find("output_hc_norm.weight"), *wd = m.wt->find("output_hc_down.weight"),
                          *wu = m.wt->find("output_hc_up.weight");
    if (!wn || !wd || !wu) {
        err = "prefill: an output_hc_* weight is missing";
        return false;
    }
    if (wn->kind != core::WeightKind::F32 || !wn->data || wu->kind != core::WeightKind::Bf16InF32 || !wu->data ||
        wu->ne1 != D || wu->ne0 != LR || wd->ne0 != D || wd->ne1 != LR) {
        err = "prefill: the output_hc_* weights are not resident tensors of the hc shape";
        return false;
    }
    // a layer half's read on the chunk's own buffers, which are idle once its last layer has written R
    gr_norm_rs(R_rows, (const float*) wn->data, EPS, m.grs, m.xn16, n, m.cs);
    if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, n, "output_hc_down.weight", err)) return false;
    gr_silu(m.lo, m.lo16, n, m.cs);
    up_mix(m.gemm, (const uint16_t*) wu->data, m.lo16, m.gated, R_rows, m.grs, (const float*) wn->data, m.mixed, nullptr,
           m.mixed_h, n, m.cs);
    // the fp16 rows to the host, and the caller reads them when this returns
    if (cudaMemcpyAsync(out, m.mixed_h, (size_t) n * (size_t) N * sizeof(uint16_t), cudaMemcpyDeviceToHost, m.cs) !=
            cudaSuccess ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: the head's input: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (rows != nullptr) *rows = m.mixed;
    return true;
}

}  // namespace basalt::prefill
