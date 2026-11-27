// tools/dfly2-train/src/loss.cpp - the chunked loss and the accepted-length metrics (loss.hpp).
#include "loss.hpp"

#include <algorithm>
#include <stdexcept>

namespace basalt::dfly2 {

namespace {

namespace F = torch::nn::functional;

/// Checks what both entry points take and returns the block count.
int64_t check_inputs(const Drafter& drafter, const torch::Tensor& z, const torch::Tensor& embedding, const torch::Tensor& head,
                     const SequenceContext& seq, const torch::Tensor& anchors) {
    const DrafterShape& s = drafter->shape;
    if (!z.defined() || z.dim() != 3 || z.size(1) != s.block || z.size(2) != s.hidden) throw std::invalid_argument("loss: z is not [blocks][block][hidden]");
    if (!anchors.defined() || anchors.dim() != 1 || anchors.scalar_type() != torch::kLong || anchors.size(0) != z.size(0) || anchors.size(0) == 0)
        throw std::invalid_argument("loss: the anchors do not match z");
    if (!embedding.defined() || embedding.dim() != 2 || embedding.size(0) != s.vocab || embedding.size(1) != s.hidden || !head.defined() ||
        head.dim() != 2 || head.size(0) != s.vocab || head.size(1) != s.hidden)
        throw std::invalid_argument("loss: the embedding or the head is not [vocab][hidden]");
    if (!seq.tokens.defined() || seq.tokens.dim() != 1 || seq.tokens.scalar_type() != torch::kLong || !seq.hidden.defined() || seq.hidden.dim() != 2 ||
        seq.hidden.size(0) != seq.tokens.size(0) || seq.hidden.size(1) != s.hidden)
        throw std::invalid_argument("loss: the sequence's tokens or hidden rows are misshapen");
    // the last drafted row's label is the token at anchor + block - 1
    if (anchors.min().item<int64_t>() < 0 || anchors.max().item<int64_t>() + s.block > seq.tokens.size(0))
        throw std::invalid_argument("loss: a block runs past the end of its sequence");
    return z.size(0);
}

/// The rows [anchor + first, anchor + first + count) of each block in `anchors`, as [blocks][count] indices.
torch::Tensor block_rows(const torch::Tensor& anchors, int64_t first, int64_t count) {
    return anchors.unsqueeze(1) + torch::arange(first, first + count, torch::TensorOptions().dtype(torch::kLong).device(anchors.device()));
}

/// The target's logits at the rows `rows` ([blocks][count]): the head over the exported head input.
torch::Tensor teacher_logits(const SequenceContext& seq, const torch::Tensor& head, const torch::Tensor& rows, torch::Dtype ct) {
    const torch::Tensor hidden = seq.hidden.index_select(0, rows.reshape({-1})).to(head.scalar_type());
    return F::linear(hidden, head).to(ct).reshape({rows.size(0), rows.size(1), head.size(0)});
}

}  // namespace

torch::Tensor dpace_weights(const torch::Tensor& confidences, double alpha) {
    if (!(alpha >= 0.0 && alpha <= 1.0)) throw std::invalid_argument("dpace: alpha is outside [0, 1]");
    torch::NoGradGuard no_grad;
    const torch::Tensor smoothed = confidences * (1.0 - alpha) + alpha;
    const torch::Tensor prefix = torch::cumprod(smoothed, -1);
    // the sum from each row to the last
    return torch::flip(torch::cumsum(torch::flip(prefix, {-1}), -1), {-1});
}

torch::Tensor e2e_tv_per_block(const torch::Tensor& accept) {
    return 1.0 - torch::cumprod(accept, -1).sum(-1) / (double) accept.size(-1);
}

double simulated_accept_length(const std::vector<double>& hits, double blocks) {
    if (blocks <= 0) return 0.0;
    double prefix = 1.0, total = 0.0;
    for (const double h : hits) {
        prefix *= h / blocks;
        total += prefix;
    }
    return total;
}

LossTerms chunked_loss(Drafter& drafter, const torch::Tensor& z_in, const torch::Tensor& embedding, const torch::Tensor& head,
                       const SequenceContext& seq, const torch::Tensor& anchors, const torch::Tensor& block_weight,
                       const LossConfig& cfg, double scale, torch::Tensor* grad_z) {
    const int64_t N = check_inputs(drafter, z_in, embedding, head, seq, anchors);
    if (block_weight.defined() && (block_weight.dim() != 1 || block_weight.size(0) != N || !block_weight.is_floating_point() ||
                                   block_weight.min().item<double>() < 0.0))
        throw std::invalid_argument("loss: the block weights are not N weights of at least zero");
    const DrafterShape& s = drafter->shape;
    const int64_t B = s.block, rows = B - 1, topk = std::min<int64_t>(std::max<int64_t>(cfg.kl_topk, 1), s.vocab);
    const int64_t step = std::max<int64_t>(1, cfg.chunk_blocks);
    const bool want_grad = grad_z != nullptr;
    const torch::Tensor z = z_in.detach();
    const torch::Dtype dt = z.scalar_type(), ct = compute_dtype(z);

    // the parameters after the backbone, and one accumulator pair per parameter and for z
    std::vector<torch::Tensor> params = drafter->hidden_correction->parameters();
    for (const torch::Tensor& p : drafter->confidence_head->parameters()) params.push_back(p);
    std::vector<torch::Tensor> acc_w(params.size()), acc_e(params.size());
    std::vector<torch::Tensor> gz_w, gz_e;

    LossTerms out;
    out.blocks = (double) N;
    out.hits_corpus.assign((size_t) rows, 0.0);
    out.hits_teacher.assign((size_t) rows, 0.0);
    double sum_w = 0, sum_kl = 0, sum_bce = 0, sum_e2e = 0, sum_mae = 0, sum_b = 0;
    const torch::Tensor weights = block_weight.defined() ? block_weight.to(z.device(), ct)
                                                         : torch::ones({N}, torch::TensorOptions().dtype(ct).device(z.device()));

    for (int64_t c0 = 0; c0 < N; c0 += step) {
        const int64_t c1 = std::min(N, c0 + step);
        const torch::Tensor a = anchors.slice(0, c0, c1);
        // row k (1..B-1) drafts the token at anchor + k from the token and the target state at anchor + k - 1
        const torch::Tensor prev_rows = block_rows(a, 0, rows), label_rows = block_rows(a, 1, rows);
        const torch::Tensor labels = seq.tokens.index_select(0, label_rows.reshape({-1})).reshape({c1 - c0, rows});
        torch::Tensor t_logp, t_logits;
        {
            torch::NoGradGuard no_grad;
            t_logits = teacher_logits(seq, head, prev_rows, ct);
            t_logp = torch::log_softmax(t_logits, -1);
        }

        torch::AutoGradMode grad_mode(want_grad);
        torch::Tensor zc = z.slice(0, c0, c1).slice(1, 1, B).clone();
        if (want_grad) zc.requires_grad_(true);
        const torch::Tensor prev_embed = embedding.index_select(0, seq.tokens.index_select(0, prev_rows.reshape({-1}))).reshape({c1 - c0, rows, s.hidden}).to(dt);
        const torch::Tensor corrected = drafter->hidden_correction->forward(zc, prev_embed);
        const torch::Tensor logits = F::linear(corrected.to(head.scalar_type()), head).to(ct);
        const torch::Tensor conf = drafter->confidence_head->forward(corrected).to(ct);
        const torch::Tensor logq = torch::log_softmax(logits, -1);

        // KL over the teacher's likeliest tokens, both distributions normalised over the whole vocabulary
        const torch::Tensor top = std::get<1>(t_logp.topk(topk, -1));
        const torch::Tensor t_top = t_logp.gather(-1, top);
        const torch::Tensor kl = (t_top.exp() * (t_top - logq.gather(-1, top))).sum(-1);
        // the accept chance of each draft; the gradient runs through the student's side of the minimum
        const torch::Tensor accept = torch::minimum(t_logp.exp(), logq.exp()).sum(-1);
        const torch::Tensor target = accept.detach().clamp(0.0, 1.0);
        const torch::Tensor bce = F::binary_cross_entropy_with_logits(conf, target, F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
        // D-PACE: a row counts for the rows after it that its draft keeps alive; a block for its own weight
        const torch::Tensor b = weights.slice(0, c0, c1);
        const torch::Tensor w = dpace_weights(logq.detach().gather(-1, labels.unsqueeze(-1)).squeeze(-1).exp(), cfg.dpace_alpha) * b.unsqueeze(1);
        const torch::Tensor weighted = (w * (kl + cfg.confidence_weight * bce)).sum();
        const torch::Tensor e2e = (e2e_tv_per_block(accept) * b).sum();

        if (want_grad) {
            std::vector<torch::Tensor> inputs = {zc};
            inputs.insert(inputs.end(), params.begin(), params.end());
            const bool with_e2e = cfg.e2e_weight != 0.0;
            const auto gw = torch::autograd::grad({weighted}, inputs, {}, with_e2e, false, true);
            gz_w.push_back(gw[0].to(ct));
            for (size_t i = 0; i < params.size(); ++i)
                if (gw[i + 1].defined()) acc_w[i] = acc_w[i].defined() ? acc_w[i] + gw[i + 1].to(ct) : gw[i + 1].to(ct);
            if (with_e2e) {
                const auto ge = torch::autograd::grad({e2e}, inputs, {}, false, false, true);
                gz_e.push_back(ge[0].to(ct));
                for (size_t i = 0; i < params.size(); ++i)
                    if (ge[i + 1].defined()) acc_e[i] = acc_e[i].defined() ? acc_e[i] + ge[i + 1].to(ct) : ge[i + 1].to(ct);
            }
        }

        // what the chunk adds to the sums and the metrics
        torch::NoGradGuard no_grad;
        sum_w += w.sum().item<double>();
        sum_kl += (w * kl).sum().item<double>();
        sum_bce += (w * bce).sum().item<double>();
        sum_e2e += e2e.item<double>();
        sum_b += b.sum().item<double>();
        sum_mae += (w * (torch::sigmoid(conf) - target).abs()).sum().item<double>();
        out.tv_accept += torch::cumprod(accept, -1).sum().item<double>();
        const torch::Tensor pick = logits.argmax(-1);
        const torch::Tensor hit_c = pick.eq(labels).sum(0).to(torch::kFloat64).cpu(), hit_t = pick.eq(t_logits.argmax(-1)).sum(0).to(torch::kFloat64).cpu();
        for (int64_t k = 0; k < rows; ++k) {
            out.hits_corpus[(size_t) k] += hit_c[k].item<double>();
            out.hits_teacher[(size_t) k] += hit_t[k].item<double>();
        }
    }

    const double denom = std::max(sum_w, 1e-6), blocks = sum_b > 0.0 ? sum_b : 1.0;
    out.kl = sum_kl / denom;
    out.confidence = sum_bce / denom;
    out.confidence_mae = sum_mae / denom;
    out.e2e = sum_e2e / blocks;
    out.loss = out.kl + cfg.confidence_weight * out.confidence + cfg.e2e_weight * out.e2e;

    if (want_grad) {
        torch::NoGradGuard no_grad;
        // join the two sums now that both denominators are known; row 0 drafts nothing and gets no gradient
        const double fw = scale / denom, fe = scale * cfg.e2e_weight / blocks;
        torch::Tensor g = torch::cat(gz_w, 0) * fw;
        if (!gz_e.empty()) g = g + torch::cat(gz_e, 0) * fe;
        *grad_z = torch::cat({torch::zeros({N, 1, s.hidden}, g.options()), g}, 1).to(dt);
        for (size_t i = 0; i < params.size(); ++i) {
            torch::Tensor total;
            if (acc_w[i].defined()) total = acc_w[i] * fw;
            if (acc_e[i].defined()) total = total.defined() ? total + acc_e[i] * fe : acc_e[i] * fe;
            if (!total.defined()) continue;
            total = total.to(params[i].scalar_type());
            if (params[i].grad().defined()) params[i].mutable_grad() += total;
            else params[i].mutable_grad() = total;
        }
    }
    return out;
}

ServedChain served_chain(Drafter& drafter, const torch::Tensor& z_in, const torch::Tensor& embedding, const torch::Tensor& head,
                         const torch::Tensor& anchor_tokens) {
    const DrafterShape& s = drafter->shape;
    if (!z_in.defined() || z_in.dim() != 3 || z_in.size(1) != s.block || z_in.size(2) != s.hidden) throw std::invalid_argument("served chain: z is not [blocks][block][hidden]");
    if (!anchor_tokens.defined() || anchor_tokens.dim() != 1 || anchor_tokens.scalar_type() != torch::kLong || anchor_tokens.size(0) != z_in.size(0))
        throw std::invalid_argument("served chain: the anchor tokens do not match z");
    if (!embedding.defined() || embedding.dim() != 2 || embedding.size(0) != s.vocab || !head.defined() || head.dim() != 2 || head.size(0) != s.vocab)
        throw std::invalid_argument("served chain: the embedding or the head is not [vocab][hidden]");
    const int64_t rows = s.block - 1;
    torch::NoGradGuard no_grad;
    const torch::Tensor z = z_in.detach();
    const torch::Dtype dt = z.scalar_type(), ct = compute_dtype(z);
    std::vector<torch::Tensor> corrected, confidence, drafts;
    torch::Tensor prev = anchor_tokens;
    for (int64_t k = 0; k < rows; ++k) {
        // row k + 1 from the token the row before it drafted
        const torch::Tensor h = drafter->hidden_correction->forward(z.select(1, k + 1), embedding.index_select(0, prev).to(dt));
        const torch::Tensor draft = F::linear(h.to(head.scalar_type()), head).to(ct).argmax(-1);
        corrected.push_back(h.to(dt));
        confidence.push_back(drafter->confidence_head->forward(h).to(ct));
        drafts.push_back(draft);
        prev = draft;
    }
    return {torch::stack(corrected, 1), torch::stack(confidence, 1), torch::stack(drafts, 1)};
}

double served_chain_accept(Drafter& drafter, const torch::Tensor& z_in, const torch::Tensor& embedding, const torch::Tensor& head,
                           const SequenceContext& seq, const torch::Tensor& anchors, int64_t chunk_blocks) {
    const int64_t N = check_inputs(drafter, z_in, embedding, head, seq, anchors);
    const DrafterShape& s = drafter->shape;
    const int64_t B = s.block, rows = B - 1, step = std::max<int64_t>(1, chunk_blocks);
    torch::NoGradGuard no_grad;
    const torch::Tensor z = z_in.detach();
    const torch::Dtype ct = compute_dtype(z);
    double accepted = 0;
    for (int64_t c0 = 0; c0 < N; c0 += step) {
        const int64_t c1 = std::min(N, c0 + step);
        const torch::Tensor a = anchors.slice(0, c0, c1);
        const torch::Tensor teacher = teacher_logits(seq, head, block_rows(a, 0, rows), ct).argmax(-1);   // [C][rows]
        const ServedChain chain = served_chain(drafter, z.slice(0, c0, c1), embedding, head, seq.tokens.index_select(0, a));
        // a block's drafts count up to its first miss
        const torch::Tensor alive = chain.drafts.eq(teacher).to(torch::kLong).cumprod(1);
        accepted += alive.sum().item<double>();
    }
    return accepted;
}

}  // namespace basalt::dfly2
