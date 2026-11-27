// tools/dfly2-train/tests/drafter_test.cpp - the drafter module: its parameters are the checkpoint contract's, a fresh
// drafter's fusion and correction are the identity, the attention's rotation and norm follow the target's formulas,
// and target weights copy into a layer. CPU only.
#include "drafter.hpp"

#include "test_util.hpp"
#include "tiny.hpp"

#include <cmath>
#include <map>

using namespace basalt::dfly2;
using dfly2_test::check;
using dfly2_test::max_abs_diff;

int main() {
    return dfly2_test::run("drafter_test", []() -> int {
    torch::manual_seed(5);
    const DrafterShape s = dfly2_test::tiny_shape();
    Drafter drafter(s);
    drafter->to(torch::kFloat64);
    const torch::Tensor embedding = torch::randn({s.vocab, s.hidden}, torch::kFloat64);
    const SequenceContext seq = dfly2_test::random_sequence(s, 30, torch::kFloat64, 100);
    const BlockBatch blocks = dfly2_test::blocks_at(s, {0, 1, 5, 12, 26});

    std::printf("1. the parameters are the checkpoint contract's\n");
    {
        std::map<std::string, std::vector<int64_t>> have;
        for (const auto& p : drafter->named_parameters()) have[p.key()] = std::vector<int64_t>(p.value().sizes().begin(), p.value().sizes().end());
        const std::vector<DrafterTensor> want = drafter_tensors(s);
        bool same = have.size() == want.size();
        for (const DrafterTensor& t : want) {
            const auto it = have.find(t.name);
            if (it == have.end() || it->second != t.shape) {
                std::printf("    missing or misshapen: %s\n", t.name.c_str());
                same = false;
            }
        }
        check(same, "every contract tensor is a parameter of that name and shape, and there is no other");
        std::string err;
        DrafterShape bad = s;
        bad.n_head = 3;
        bool threw = false;
        try { Drafter d(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw && !drafter_check_shape(bad, err), "a shape the contract refuses does not build: " + err);
    }

    std::printf("2. a fresh drafter: identity fusion and identity correction\n");
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor x = drafter->block_inputs(embedding, seq, blocks);
        const torch::Tensor anchor_embed = embedding.index_select(0, seq.tokens.index_select(0, blocks.anchors));
        check(x.size(0) == 5 && x.size(1) == s.block && x.size(2) == s.hidden, "block inputs are [blocks][block][hidden]");
        check(max_abs_diff(x.select(1, 0), anchor_embed) == 0.0, "row 0 is the anchor's embedding: the fusion adds nothing");
        check(max_abs_diff(x.slice(1, 1, s.block), embedding.select(0, s.mask_token_id).expand({5, s.block - 1, s.hidden})) == 0.0, "the other rows are the mask token's embedding");
        check(drafter->mtp_fusion->gate.abs().max().item<double>() == 0.0, "the fusion gate is zero");
        const torch::Tensor h = torch::randn({7, s.hidden}, torch::kFloat64), e = torch::randn({7, s.hidden}, torch::kFloat64);
        check(max_abs_diff(drafter->hidden_correction->forward(h, e), h) == 0.0, "the correction returns its input");
        check(drafter->hidden_correction->down_proj->weight.abs().max().item<double>() == 0.0, "the correction's down projection is zero");
        bool ones = true;
        for (const auto& p : drafter->named_parameters())
            if (p.value().dim() == 1 && p.key().find("norm") != std::string::npos) ones = ones && max_abs_diff(p.value(), torch::ones_like(p.value())) == 0.0;
        check(ones, "every norm weight is one");
        // with the gate opened the fusion reaches row 0, except for an anchor at row 0, which has no row before it
        drafter->mtp_fusion->gate.fill_(0.5);
        const torch::Tensor opened = drafter->block_inputs(embedding, seq, blocks);
        check(max_abs_diff(opened.select(1, 0).select(0, 0), anchor_embed.select(0, 0)) == 0.0, "an anchor at row 0 gets no fusion");
        check(max_abs_diff(opened.select(1, 0).slice(0, 1, 5), anchor_embed.slice(0, 1, 5)) > 1e-3, "an opened gate changes the other anchors' rows");
        const torch::Tensor want = anchor_embed + 0.5 * drafter->mtp_fusion->eh_proj->forward(torch::cat({drafter->mtp_fusion->enorm->forward(anchor_embed),
                                                                                                           drafter->mtp_fusion->hnorm->forward(seq.hidden.index_select(0, (blocks.anchors - 1).clamp_min(0)))}, -1));
        check(max_abs_diff(opened.select(1, 0).slice(0, 1, 5), want.slice(0, 1, 5)) < 1e-12, "the fused row is e(a) + gate * eh_proj(enorm(e(a)) :: hnorm(h))");
        drafter->mtp_fusion->gate.zero_();
    }

    std::printf("3. the norm and the rotation follow the target's formulas\n");
    {
        torch::NoGradGuard no_grad;
        DraftAttentionImpl& attn = *drafter->layers->ptr<DraftLayerImpl>(0)->self_attn;
        attn.k_norm->weight.copy_(torch::rand({s.head_dim}, torch::kFloat64) + 0.5);
        // rms: s = rsqrt(mean(x^2) + eps), x * s * w
        const torch::Tensor x = torch::randn({s.head_dim}, torch::kFloat64);
        double ss = 0;
        for (int64_t d = 0; d < s.head_dim; ++d) ss += x[d].item<double>() * x[d].item<double>();
        const double scale = 1.0 / std::sqrt(ss / (double) s.head_dim + s.attn_norm_eps);
        const torch::Tensor normed = attn.k_norm->forward(x);
        double worst = 0;
        for (int64_t d = 0; d < s.head_dim; ++d) worst = std::max(worst, std::abs(normed[d].item<double>() - x[d].item<double>() * scale * attn.k_norm->weight[d].item<double>()));
        check(worst < 1e-14, "rms(x, w, eps) = x * rsqrt(mean(x^2) + eps) * w");
        // rope: pair p is (u[p], u[p + n_rot/2]), angle = position * base^(-2p / n_rot); the rest passes through
        const torch::Tensor u = torch::randn({1, 2, s.n_head_kv, s.head_dim}, torch::kFloat64);
        const torch::Tensor pos = torch::tensor(std::vector<int64_t>{0, 1234}, torch::kLong).view({1, 2});
        const torch::Tensor turned = attn.rope(u, pos);
        check(max_abs_diff(turned.select(1, 0), u.select(1, 0)) == 0.0, "position 0 turns nothing");
        worst = 0;
        const int64_t half = s.n_rot / 2;
        for (int64_t h = 0; h < s.n_head_kv; ++h)
            for (int64_t d = 0; d < s.head_dim; ++d) {
                double want = u[0][1][h][d].item<double>();
                if (d < s.n_rot) {
                    const int64_t p = d % half;
                    const double theta = 1234.0 * std::pow(s.rope_base, -2.0 * (double) p / (double) s.n_rot);
                    const double a = u[0][1][h][p].item<double>(), b = u[0][1][h][p + half].item<double>();
                    want = d < half ? a * std::cos(theta) - b * std::sin(theta) : a * std::sin(theta) + b * std::cos(theta);
                }
                worst = std::max(worst, std::abs(turned[0][1][h][d].item<double>() - want));
            }
        check(worst < 1e-13, "position 1234 turns each pair by its angle and leaves the other values");
        attn.k_norm->weight.fill_(1.0);
    }

    std::printf("4. attention weights from a target layer\n");
    {
        std::string err;
        TargetAttention t;
        t.q = torch::randn({2 * s.n_head * s.head_dim, s.hidden});
        t.k = torch::randn({s.n_head_kv * s.head_dim, s.hidden});
        t.v = torch::randn({s.n_head_kv * s.head_dim, s.hidden});
        t.o = torch::randn({s.hidden, s.n_head * s.head_dim});
        t.q_norm = torch::rand({s.head_dim}) + 0.5;
        t.k_norm = torch::rand({s.head_dim}) + 0.5;
        check(drafter->init_attention(1, t, err), "a paired layer's weights copy in: " + err);
        DraftAttentionImpl& a = *drafter->layers->ptr<DraftLayerImpl>(1)->self_attn;
        check(max_abs_diff(a.q_proj->weight, t.q) == 0.0 && max_abs_diff(a.k_proj->weight, t.k) == 0.0 && max_abs_diff(a.v_proj->weight, t.v) == 0.0 &&
                  max_abs_diff(a.o_proj->weight, t.o) == 0.0 && max_abs_diff(a.q_norm->weight, t.q_norm) == 0.0 && max_abs_diff(a.k_norm->weight, t.k_norm) == 0.0,
              "the six tensors are the target's");
        check(a.q_proj->weight.scalar_type() == torch::kFloat64 && a.q_proj->weight.requires_grad(), "they keep the drafter's type and stay trainable");
        // the block rows' own keys are then what the target would cache for the same input: rope(rms(attn_k y))
        {
            torch::NoGradGuard no_grad;
            const torch::Tensor y = torch::randn({1, 1, s.hidden}, torch::kFloat64);
            const torch::Tensor pos = torch::full({1, 1}, 77, torch::kLong);
            const torch::Tensor raw = torch::matmul(t.k.to(torch::kFloat64), y.view({s.hidden})).view({s.n_head_kv, s.head_dim});
            const torch::Tensor cached = a.rope((raw * torch::rsqrt(raw.pow(2).mean(-1, true) + s.attn_norm_eps) * t.k_norm.to(torch::kFloat64)).view({1, 1, s.n_head_kv, s.head_dim}), pos);
            const torch::Tensor own = a.rope(a.k_norm->forward(a.k_proj->forward(y).view({1, 1, s.n_head_kv, s.head_dim})), pos);
            check(max_abs_diff(own, cached) < 1e-12, "a block row's key is the cell the paired layer would cache for that input");
        }
        t.o = torch::randn({s.hidden, s.n_head * s.head_dim + 1});
        check(!drafter->init_attention(0, t, err), "a misshapen target tensor is refused: " + err);
        check(!drafter->init_attention(9, t, err), "a layer the drafter does not have is refused: " + err);
    }

    std::printf("5. the backbone\n");
    {
        const torch::Tensor z = drafter->backbone(embedding, seq, blocks, 2);
        check(z.size(0) == 5 && z.size(1) == s.block && z.size(2) == s.hidden && z.scalar_type() == torch::kFloat64, "the output is [blocks][block][hidden] in the weights' type");
        check(torch::isfinite(z).all().item<bool>(), "every value is finite");
        check(max_abs_diff(drafter->backbone(embedding, seq, blocks, 64), z) < 1e-12, "the attention's chunk size does not change it");
        z.pow(2).sum().backward();
        bool grads = true;
        for (const auto& p : drafter->named_parameters()) {
            const bool head = p.key().rfind("hidden_correction", 0) == 0 || p.key().rfind("confidence_head", 0) == 0;
            // everything the backbone runs through has a gradient; the correction and the confidence head come after it
            if (!head && !p.value().grad().defined()) {
                std::printf("    no gradient: %s\n", p.key().c_str());
                grads = false;
            }
        }
        check(grads, "the backward reaches every backbone and fusion parameter");
        check(drafter->layers->ptr<DraftLayerImpl>(0)->self_attn->k_proj->weight.grad().abs().max().item<double>() > 0, "the block keys' projection has a nonzero gradient");
        // a block's rows move with the sequence's first position, through the rotation
        torch::NoGradGuard no_grad;
        SequenceContext moved = seq;
        moved.pos0 = 0;
        check(max_abs_diff(drafter->backbone(embedding, moved, blocks, 64), z) > 1e-6, "pos0 reaches the rotation");
        bool threw = false;
        BlockBatch outside = blocks;
        outside.anchors = torch::tensor(std::vector<int64_t>{30}, torch::kLong);
        try { drafter->backbone(embedding, seq, outside, 64); } catch (const std::exception&) { threw = true; }
        check(threw, "an anchor outside the sequence is refused");
    }

    std::printf("6. single precision weights over a half precision cache\n");
    {
        torch::NoGradGuard no_grad;
        Drafter f32(s);
        const SequenceContext seq16 = dfly2_test::random_sequence(s, 30, torch::kFloat16);
        const torch::Tensor z = f32->backbone(embedding.to(torch::kFloat32), seq16, blocks, 64);
        check(z.scalar_type() == torch::kFloat32 && torch::isfinite(z).all().item<bool>(), "the forward runs and stays finite in FP32");
    }
    return dfly2_test::finish("drafter_test");
    });
}
