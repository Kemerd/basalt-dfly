// src/dfly2/ops.cpp - the Gated Fusion and the Hidden-Correction pass as chains of the shared kernels (ops.hpp).
#include "basalt/dfly2/ops.hpp"

#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/native_mmvq.hpp"

#include <stdexcept>

namespace basalt::dfly2 {

namespace {

namespace bk = basalt::kernels;
constexpr int kQ8_0 = 8;   // ggml's type id of the projections

}  // namespace

void gated_fusion(const FusionWeights& w, const float* emb, const float* hidden, int n_embd, float eps, float* cat,
                  void* cat_q8, float* fused, float* out, void* stream) {
    if (!w.eh_proj || !w.enorm || !w.hnorm || !w.gate || !emb || !hidden || !cat || !cat_q8 || !fused || !out || !stream)
        throw std::invalid_argument("dfly2::gated_fusion: a null buffer");
    // the joined row is read in blocks of 32 values
    if (n_embd <= 0 || n_embd % 16 != 0) throw std::invalid_argument("dfly2::gated_fusion: n_embd");
    const int two = 2 * n_embd;
    // both halves are normed into the joined row
    bk::rms_norm_rows(emb, w.enorm, cat, 1, n_embd, eps, stream, two);
    bk::rms_norm_rows(hidden, w.hnorm, cat + n_embd, 1, n_embd, eps, stream, two);
    bk::native_quantize_q8_1(cat, cat_q8, two, 1, stream);
    bk::native_mmvq(kQ8_0, w.eh_proj, cat_q8, fused, two, n_embd, 1, stream);
    gate_add(emb, fused, w.gate, n_embd, out, stream);
}

void correction_mlp(const CorrectionWeights& w, const float* z, const float* prev_emb, int n_embd, float eps, float* cat,
                    void* cat_q8, float* g, float* u, float* h, void* h_q8, float* delta, void* stream) {
    if (!w.gate || !w.up || !w.down || !w.hidden_norm || !w.embed_norm || !z || !prev_emb || !cat || !cat_q8 || !g || !u ||
        !h || !h_q8 || !delta || !stream)
        throw std::invalid_argument("dfly2::correction_mlp: a null buffer");
    if (n_embd <= 0 || n_embd % 16 != 0 || w.ffn <= 0 || w.ffn % 32 != 0)
        throw std::invalid_argument("dfly2::correction_mlp: n_embd or ffn");
    const int two = 2 * n_embd;
    bk::rms_norm_rows(z, w.hidden_norm, cat, 1, n_embd, eps, stream, two);
    bk::rms_norm_rows(prev_emb, w.embed_norm, cat + n_embd, 1, n_embd, eps, stream, two);
    // the gate and the up projection read one q8_1 image of the joined row
    bk::native_quantize_q8_1(cat, cat_q8, two, 1, stream);
    bk::native_mmvq(kQ8_0, w.gate, cat_q8, g, two, w.ffn, 1, stream);
    bk::native_mmvq(kQ8_0, w.up, cat_q8, u, two, w.ffn, 1, stream);
    bk::swiglu_scaled(g, u, nullptr, nullptr, h, 1, w.ffn, w.ffn, stream);
    bk::native_quantize_q8_1(h, h_q8, w.ffn, 1, stream);
    bk::native_mmvq(kQ8_0, w.down, h_q8, delta, w.ffn, n_embd, 1, stream);
}

}  // namespace basalt::dfly2
