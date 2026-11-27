// tools/dfly2-train/src/drafter.hpp - the DFly2 drafter as LibTorch modules: the dense layers that read the target's
// cached cells, the Gated Fusion of the anchor with the target's last hidden state, the Hidden-Correction and the
// confidence head. The forward and the parameter names are the checkpoint contract's
// (basalt/artifact/dfly2_checkpoint.hpp); the target's embedding and output head are handed in, never owned.
#pragma once

#include "basalt/artifact/dfly2_checkpoint.hpp"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace basalt::dfly2 {

/// FP64 for an FP64 tensor, FP32 for every other: the type norms, rotations and the softmax compute in.
inline torch::Dtype compute_dtype(const torch::Tensor& x) {
    return x.scalar_type() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
}

/// x * rsqrt(mean(x^2) + eps) * weight over the last dimension, computed and returned in compute_dtype(x).
struct RmsNormImpl : torch::nn::Module {
    RmsNormImpl(int64_t dim, double eps);
    torch::Tensor forward(const torch::Tensor& x);
    torch::Tensor weight;   ///< [dim], ones at construction
    double eps;
};
TORCH_MODULE(RmsNorm);

/// One sequence's frozen teacher data on the training device.
struct SequenceContext {
    torch::Tensor tokens;             ///< [L] int64
    torch::Tensor hidden;             ///< [L][hidden]: the target's head input per row
    std::vector<torch::Tensor> k;     ///< per draft layer [L][n_head_kv][head_dim]: the paired layer's cached keys
    std::vector<torch::Tensor> v;     ///< per draft layer, the cached values
    int64_t pos0 = 0;                 ///< rotary position of row 0
};

/// The blocks of one forward: their anchors and, per draft layer, the cells each block reads.
struct BlockBatch {
    torch::Tensor anchors;            ///< [N] int64 rows of the sequence
    std::vector<torch::Tensor> sel;   ///< per draft layer [N][W] int64 cell ids, -1 padded
};

/// The attention of one draft layer (step 3 of the contract's forward).
struct DraftAttentionImpl : torch::nn::Module {
    explicit DraftAttentionImpl(const DrafterShape& s);
    /// `y` [N][B][hidden] normed layer input, `pos` [N][B] int64 rotary positions; returns the layer's attention
    /// output [N][B][hidden] in y's type.
    torch::Tensor forward(const torch::Tensor& y, const torch::Tensor& pos, const torch::Tensor& k_ctx, const torch::Tensor& v_ctx,
                          const torch::Tensor& idx, int64_t chunk_blocks);
    /// Turns the first n_rot values of each head of `x` [N][B][heads][head_dim] by the rows' positions.
    torch::Tensor rope(const torch::Tensor& x, const torch::Tensor& pos) const;

    DrafterShape shape;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};
    RmsNorm q_norm{nullptr}, k_norm{nullptr};
};
TORCH_MODULE(DraftAttention);

/// down_proj(silu(gate_proj(x)) * up_proj(x)), no biases.
struct SwiGluImpl : torch::nn::Module {
    SwiGluImpl(int64_t in, int64_t inner, int64_t out);
    torch::Tensor forward(const torch::Tensor& x);
    torch::nn::Linear gate_proj{nullptr}, up_proj{nullptr}, down_proj{nullptr};
};
TORCH_MODULE(SwiGlu);

/// A pre-norm draft layer on one residual stream.
struct DraftLayerImpl : torch::nn::Module {
    explicit DraftLayerImpl(const DrafterShape& s);
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& pos, const torch::Tensor& k_ctx, const torch::Tensor& v_ctx,
                          const torch::Tensor& idx, int64_t chunk_blocks);
    RmsNorm input_layernorm{nullptr}, post_attention_layernorm{nullptr};
    DraftAttention self_attn{nullptr};
    SwiGlu mlp{nullptr};
};
TORCH_MODULE(DraftLayer);

/// Gated Fusion: what is added to the anchor row's embedding. The gate starts at zero, so the sum starts as the
/// embedding alone.
struct FusionImpl : torch::nn::Module {
    FusionImpl(int64_t hidden, double eps);
    /// `anchor` [N][hidden] anchor embeddings, `prev_hidden` [N][hidden] the target's head input at the row before;
    /// returns gate * eh_proj(enorm(anchor) :: hnorm(prev_hidden)).
    torch::Tensor forward(const torch::Tensor& anchor, const torch::Tensor& prev_hidden);
    RmsNorm enorm{nullptr}, hnorm{nullptr};
    torch::nn::Linear eh_proj{nullptr};
    torch::Tensor gate;   ///< [hidden], zeros at construction
};
TORCH_MODULE(Fusion);

/// Hidden-Correction: h + down_proj(silu(gate_proj(c)) * up_proj(c)), c = hidden_norm(h) :: embed_norm(e(prev)).
/// down_proj starts at zero, so the correction starts as the identity.
struct HiddenCorrectionImpl : torch::nn::Module {
    HiddenCorrectionImpl(int64_t hidden, int64_t inner, double eps);
    torch::Tensor forward(const torch::Tensor& h, const torch::Tensor& prev_embed);
    RmsNorm hidden_norm{nullptr}, embed_norm{nullptr};
    torch::nn::Linear gate_proj{nullptr}, up_proj{nullptr}, down_proj{nullptr};
};
TORCH_MODULE(HiddenCorrection);

/// One logit per row: the drafter's estimate of the chance the target accepts the row's draft.
struct ConfidenceHeadImpl : torch::nn::Module {
    explicit ConfidenceHeadImpl(int64_t hidden);
    torch::Tensor forward(const torch::Tensor& h);
    torch::nn::Linear proj{nullptr};
};
TORCH_MODULE(ConfidenceHead);

/// The target weights one draft layer's attention starts from: the paired layer's projections and norms,
/// dequantised, shaped as the drafter's own.
struct TargetAttention {
    torch::Tensor q, k, v, o;          ///< [2*n_head*head_dim][hidden], [n_head_kv*head_dim][hidden] x2, [hidden][n_head*head_dim]
    torch::Tensor q_norm, k_norm;      ///< [head_dim]
};

struct DrafterImpl : torch::nn::Module {
    /// Builds a drafter of shape `s` with the default initialisation: projections uniform in +-1/sqrt(fan_in), norm
    /// weights one, the fusion gate and the correction's down projection zero. Throws std::invalid_argument for a
    /// shape drafter_check_shape refuses.
    explicit DrafterImpl(const DrafterShape& s);

    /// Copies a paired target layer's attention into draft layer `layer`, so the block rows' keys and values land
    /// in the space of the cells that layer cached. False with `err` when a tensor is missing or misshapen.
    bool init_attention(int64_t layer, const TargetAttention& t, std::string& err);

    /// The block inputs [N][block][hidden]: the anchor's embedding with the fusion, then the mask token's.
    /// `embedding` is the target's [vocab][hidden] table.
    torch::Tensor block_inputs(const torch::Tensor& embedding, const SequenceContext& seq, const BlockBatch& blocks);

    /// Steps 1 to 3 of the forward: the block inputs through every layer and the final norm, [N][block][hidden].
    torch::Tensor backbone(const torch::Tensor& embedding, const SequenceContext& seq, const BlockBatch& blocks,
                           int64_t chunk_blocks);

    /// The type the weights are held in.
    torch::Dtype weight_dtype() const;

    DrafterShape shape;
    torch::nn::ModuleList layers{nullptr};
    RmsNorm final_norm{nullptr};
    Fusion mtp_fusion{nullptr};
    HiddenCorrection hidden_correction{nullptr};
    ConfidenceHead confidence_head{nullptr};
};
TORCH_MODULE(Drafter);

}  // namespace basalt::dfly2
