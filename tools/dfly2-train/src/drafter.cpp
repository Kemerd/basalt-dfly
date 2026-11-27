// tools/dfly2-train/src/drafter.cpp - the drafter's modules (drafter.hpp).
#include "drafter.hpp"

#include "sparse_attention.hpp"

#include <stdexcept>

namespace basalt::dfly2 {

namespace {

/// A projection without a bias, at LibTorch's default initialisation (uniform in +-1/sqrt(fan_in)).
torch::nn::Linear linear(int64_t in, int64_t out) {
    return torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false));
}

/// Copies `src` into the parameter `dst` when their shapes agree.
bool copy_weight(torch::Tensor& dst, const torch::Tensor& src, const char* name, std::string& err) {
    if (!src.defined() || !(src.sizes() == dst.sizes())) {
        err = std::string("target ") + name + " is missing or does not have the drafter's shape";
        return false;
    }
    torch::NoGradGuard no_grad;
    dst.copy_(src.to(dst.device(), dst.scalar_type()));
    return true;
}

}  // namespace

// ---------------------------------------------------------------- norm

RmsNormImpl::RmsNormImpl(int64_t dim, double eps_) : eps(eps_) {
    weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RmsNormImpl::forward(const torch::Tensor& x) {
    const torch::Dtype ct = compute_dtype(x);
    const torch::Tensor xf = x.to(ct);
    return xf * torch::rsqrt(xf.pow(2).mean(-1, true) + eps) * weight.to(ct);
}

// ---------------------------------------------------------------- attention

DraftAttentionImpl::DraftAttentionImpl(const DrafterShape& s) : shape(s) {
    const int64_t D = s.hidden, dh = s.head_dim;
    q_proj = register_module("q_proj", linear(D, 2 * s.n_head * dh));
    k_proj = register_module("k_proj", linear(D, s.n_head_kv * dh));
    v_proj = register_module("v_proj", linear(D, s.n_head_kv * dh));
    o_proj = register_module("o_proj", linear(s.n_head * dh, D));
    q_norm = register_module("q_norm", RmsNorm(dh, s.attn_norm_eps));
    k_norm = register_module("k_norm", RmsNorm(dh, s.attn_norm_eps));
}

torch::Tensor DraftAttentionImpl::rope(const torch::Tensor& x, const torch::Tensor& pos) const {
    const int64_t half = shape.n_rot / 2;
    if (half == 0) return x;
    // the angles in FP64, as the target builds its table: position * base^(-2p / n_rot)
    const auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(x.device());
    const torch::Tensor inv_freq = torch::pow(shape.rope_base, torch::arange(half, f64) * (-2.0 / (double) shape.n_rot));
    const torch::Tensor theta = pos.to(torch::kFloat64).unsqueeze(-1) * inv_freq;        // [N][B][half]
    const torch::Tensor c = torch::cos(theta).to(x.scalar_type()).unsqueeze(2);          // one angle for every head of a row
    const torch::Tensor s = torch::sin(theta).to(x.scalar_type()).unsqueeze(2);
    // pair p is (x[p], x[p + half]); the values past n_rot pass through
    const torch::Tensor a = x.slice(3, 0, half), b = x.slice(3, half, 2 * half);
    return torch::cat({a * c - b * s, a * s + b * c, x.slice(3, 2 * half, x.size(3))}, 3);
}

torch::Tensor DraftAttentionImpl::forward(const torch::Tensor& y, const torch::Tensor& pos, const torch::Tensor& k_ctx,
                                          const torch::Tensor& v_ctx, const torch::Tensor& idx, int64_t chunk_blocks) {
    const int64_t N = y.size(0), B = y.size(1), H = shape.n_head, G = shape.n_head_kv, dh = shape.head_dim;
    const torch::Dtype ct = compute_dtype(y);
    // each head's projection is its query half, then its gate half
    const torch::Tensor qg = q_proj->forward(y).view({N, B, H, 2, dh});
    const torch::Tensor q = rope(q_norm->forward(qg.select(3, 0)), pos);
    const torch::Tensor gate = qg.select(3, 1).to(ct);
    // the block rows' own cells, in the space of the cached ones
    const torch::Tensor k_blk = rope(k_norm->forward(k_proj->forward(y).view({N, B, G, dh})), pos);
    const torch::Tensor v_blk = v_proj->forward(y).view({N, B, G, dh}).to(ct);
    const torch::Tensor attn = sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx, chunk_blocks);
    return o_proj->forward((attn * torch::sigmoid(gate)).to(y.scalar_type()).reshape({N, B, H * dh}));
}

// ---------------------------------------------------------------- MLP and layer

SwiGluImpl::SwiGluImpl(int64_t in, int64_t inner, int64_t out) {
    gate_proj = register_module("gate_proj", linear(in, inner));
    up_proj = register_module("up_proj", linear(in, inner));
    down_proj = register_module("down_proj", linear(inner, out));
}

torch::Tensor SwiGluImpl::forward(const torch::Tensor& x) {
    return down_proj->forward(torch::silu(gate_proj->forward(x)) * up_proj->forward(x));
}

DraftLayerImpl::DraftLayerImpl(const DrafterShape& s) {
    input_layernorm = register_module("input_layernorm", RmsNorm(s.hidden, s.rms_norm_eps));
    self_attn = register_module("self_attn", DraftAttention(s));
    post_attention_layernorm = register_module("post_attention_layernorm", RmsNorm(s.hidden, s.rms_norm_eps));
    mlp = register_module("mlp", SwiGlu(s.hidden, s.intermediate, s.hidden));
}

torch::Tensor DraftLayerImpl::forward(const torch::Tensor& x, const torch::Tensor& pos, const torch::Tensor& k_ctx,
                                      const torch::Tensor& v_ctx, const torch::Tensor& idx, int64_t chunk_blocks) {
    const torch::Dtype dt = x.scalar_type();
    const torch::Tensor h = x + self_attn->forward(input_layernorm->forward(x).to(dt), pos, k_ctx, v_ctx, idx, chunk_blocks);
    return h + mlp->forward(post_attention_layernorm->forward(h).to(dt));
}

// ---------------------------------------------------------------- fusion, correction, confidence

FusionImpl::FusionImpl(int64_t hidden, double eps) {
    enorm = register_module("enorm", RmsNorm(hidden, eps));
    hnorm = register_module("hnorm", RmsNorm(hidden, eps));
    eh_proj = register_module("eh_proj", linear(2 * hidden, hidden));
    gate = register_parameter("gate", torch::zeros({hidden}));
}

torch::Tensor FusionImpl::forward(const torch::Tensor& anchor, const torch::Tensor& prev_hidden) {
    const torch::Dtype dt = anchor.scalar_type();
    const torch::Tensor joined = torch::cat({enorm->forward(anchor).to(dt), hnorm->forward(prev_hidden).to(dt)}, -1);
    return gate * eh_proj->forward(joined);
}

HiddenCorrectionImpl::HiddenCorrectionImpl(int64_t hidden, int64_t inner, double eps) {
    hidden_norm = register_module("hidden_norm", RmsNorm(hidden, eps));
    embed_norm = register_module("embed_norm", RmsNorm(hidden, eps));
    gate_proj = register_module("gate_proj", linear(2 * hidden, inner));
    up_proj = register_module("up_proj", linear(2 * hidden, inner));
    down_proj = register_module("down_proj", linear(inner, hidden));
    torch::NoGradGuard no_grad;
    down_proj->weight.zero_();
}

torch::Tensor HiddenCorrectionImpl::forward(const torch::Tensor& h, const torch::Tensor& prev_embed) {
    const torch::Dtype dt = h.scalar_type();
    const torch::Tensor c = torch::cat({hidden_norm->forward(h).to(dt), embed_norm->forward(prev_embed).to(dt)}, -1);
    return h + down_proj->forward(torch::silu(gate_proj->forward(c)) * up_proj->forward(c));
}

ConfidenceHeadImpl::ConfidenceHeadImpl(int64_t hidden) {
    proj = register_module("proj", torch::nn::Linear(hidden, 1));
}

torch::Tensor ConfidenceHeadImpl::forward(const torch::Tensor& h) { return proj->forward(h).squeeze(-1); }

// ---------------------------------------------------------------- the drafter

DrafterImpl::DrafterImpl(const DrafterShape& s) : shape(s) {
    std::string err;
    if (!drafter_check_shape(s, err)) throw std::invalid_argument(err);
    layers = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < s.n_layers; ++i) layers->push_back(DraftLayer(s));
    final_norm = register_module("final_norm", RmsNorm(s.hidden, s.rms_norm_eps));
    mtp_fusion = register_module("mtp_fusion", Fusion(s.hidden, s.rms_norm_eps));
    hidden_correction = register_module("hidden_correction", HiddenCorrection(s.hidden, s.correction, s.rms_norm_eps));
    confidence_head = register_module("confidence_head", ConfidenceHead(s.hidden));
}

torch::Dtype DrafterImpl::weight_dtype() const { return final_norm->weight.scalar_type(); }

bool DrafterImpl::init_attention(int64_t layer, const TargetAttention& t, std::string& err) {
    if (layer < 0 || layer >= shape.n_layers) { err = "no draft layer " + std::to_string(layer); return false; }
    DraftAttentionImpl& a = *layers->ptr<DraftLayerImpl>((size_t) layer)->self_attn;
    return copy_weight(a.q_proj->weight, t.q, "attn_q", err) && copy_weight(a.k_proj->weight, t.k, "attn_k", err) &&
           copy_weight(a.v_proj->weight, t.v, "attn_v", err) && copy_weight(a.o_proj->weight, t.o, "attn_output", err) &&
           copy_weight(a.q_norm->weight, t.q_norm, "attn_q_norm", err) && copy_weight(a.k_norm->weight, t.k_norm, "attn_k_norm", err);
}

torch::Tensor DrafterImpl::block_inputs(const torch::Tensor& embedding, const SequenceContext& seq, const BlockBatch& blocks) {
    const int64_t D = shape.hidden;
    if (!embedding.defined() || embedding.dim() != 2 || embedding.size(0) != shape.vocab || embedding.size(1) != D)
        throw std::invalid_argument("drafter: the embedding is not [vocab][hidden]");
    if (!seq.tokens.defined() || seq.tokens.dim() != 1 || seq.tokens.scalar_type() != torch::kLong || !seq.hidden.defined() ||
        seq.hidden.dim() != 2 || seq.hidden.size(0) != seq.tokens.size(0) || seq.hidden.size(1) != D)
        throw std::invalid_argument("drafter: the sequence's tokens or hidden rows are misshapen");
    if (!blocks.anchors.defined() || blocks.anchors.dim() != 1 || blocks.anchors.scalar_type() != torch::kLong || blocks.anchors.size(0) == 0)
        throw std::invalid_argument("drafter: the anchors are not a list of int64 rows");
    if (blocks.anchors.min().item<int64_t>() < 0 || blocks.anchors.max().item<int64_t>() >= seq.tokens.size(0))
        throw std::invalid_argument("drafter: an anchor is outside the sequence");
    const torch::Dtype dt = weight_dtype();
    const int64_t N = blocks.anchors.size(0);
    const torch::Tensor anchor_embed = embedding.index_select(0, seq.tokens.index_select(0, blocks.anchors)).to(dt);
    // the target's head input at the row before the anchor; an anchor at row 0 has none and gets no fusion
    const torch::Tensor prev_hidden = seq.hidden.index_select(0, (blocks.anchors - 1).clamp_min(0));
    const torch::Tensor has_prev = blocks.anchors.gt(0).to(dt).unsqueeze(1);
    const torch::Tensor row0 = anchor_embed + mtp_fusion->forward(anchor_embed, prev_hidden) * has_prev;
    const torch::Tensor mask = embedding.select(0, shape.mask_token_id).to(dt).expand({N, shape.block - 1, D});
    return torch::cat({row0.unsqueeze(1), mask}, 1);
}

torch::Tensor DrafterImpl::backbone(const torch::Tensor& embedding, const SequenceContext& seq, const BlockBatch& blocks,
                                    int64_t chunk_blocks) {
    if ((int64_t) seq.k.size() != shape.n_layers || (int64_t) seq.v.size() != shape.n_layers || (int64_t) blocks.sel.size() != shape.n_layers)
        throw std::invalid_argument("drafter: the sequence or the blocks do not carry one cache and one selection per layer");
    torch::Tensor x = block_inputs(embedding, seq, blocks);
    // row r of a block sits r positions after its anchor
    const torch::Tensor pos = blocks.anchors.unsqueeze(1) + torch::arange(shape.block, torch::TensorOptions().dtype(torch::kLong).device(x.device())) + seq.pos0;
    for (int64_t i = 0; i < shape.n_layers; ++i)
        x = layers->ptr<DraftLayerImpl>((size_t) i)->forward(x, pos, seq.k[(size_t) i], seq.v[(size_t) i], blocks.sel[(size_t) i], chunk_blocks);
    return final_norm->forward(x).to(x.scalar_type());
}

}  // namespace basalt::dfly2
