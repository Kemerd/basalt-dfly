// include/basalt/dfly2/ops.hpp - the device steps every DFly2 drafter runs the same way: the Gated Fusion of the anchor's
// embedding with the target's hidden state, and the Hidden-Correction of a draft row up to its delta.
//
// Rows are fp32 on the device and projections Q8_0 (ggml's blocks, [out][in]), read through q8_1 activations. Every call
// only enqueues on `stream` (not null) and takes no per-call value by argument but pointers, so a captured graph may hold
// it. A model owns the buffers and the layer pass between the two steps.
#pragma once

namespace basalt::dfly2 {

/// out[i] = base[i] + gate[i] * fused[i] for i < n (the Gated Fusion's residual).
void gate_add(const float* base, const float* fused, const float* gate, int n, float* out, void* stream);

/// The fusion's weights: eh_proj Q8_0 [n_embd][2 * n_embd]; enorm, hnorm and gate fp32 [n_embd].
struct FusionWeights {
    const void* eh_proj = nullptr;
    const float *enorm = nullptr, *hnorm = nullptr, *gate = nullptr;
};

/// out = emb + gate * eh_proj(rms(emb, enorm) :: rms(hidden, hnorm)): the anchor's row of a block, from its embedding
/// `emb` and the target's hidden state `hidden` at the row before it (n_embd floats each; n_embd a multiple of 16).
/// Scratch: `cat` 2 * n_embd floats, `cat_q8` its q8_1 image (native_q8_1_bytes(2 * n_embd)), `fused` n_embd floats.
void gated_fusion(const FusionWeights& w, const float* emb, const float* hidden, int n_embd, float eps, float* cat,
                  void* cat_q8, float* fused, float* out, void* stream);

/// The correction's weights: gate and up Q8_0 [ffn][2 * n_embd], down Q8_0 [n_embd][ffn]; the norms fp32 [n_embd].
struct CorrectionWeights {
    const void *gate = nullptr, *up = nullptr, *down = nullptr;
    const float *hidden_norm = nullptr, *embed_norm = nullptr;
    int ffn = 0;   ///< the correction's inner width, a multiple of 32
};

/// delta = down(silu(gate(c)) * up(c)), c = rms(z, hidden_norm) :: rms(prev_emb, embed_norm): a draft row's correction
/// from its output `z` and the embedding `prev_emb` of the token before it. The model adds delta to z.
/// Scratch: `cat` 2 * n_embd floats and `cat_q8` its q8_1 image; `g`, `u`, `h` ffn floats each and `h_q8` h's q8_1 image.
void correction_mlp(const CorrectionWeights& w, const float* z, const float* prev_emb, int n_embd, float eps, float* cat,
                    void* cat_q8, float* g, float* u, float* h, void* h_q8, float* delta, void* stream);

}  // namespace basalt::dfly2
