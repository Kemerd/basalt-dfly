// tools/dfly2-train/tests/loss_test.cpp - the training loss: the D-PACE weights and the accepted-length term on cases
// worked by hand, the whole loss on a block worked by hand, the chunked two-stage loss against one written plainly
// over every block at once (value and gradients), and the accepted-length metrics. CPU only.
#include "loss.hpp"

#include "test_util.hpp"
#include "tiny.hpp"

#include <cmath>

using namespace basalt::dfly2;
using dfly2_test::check;
using dfly2_test::max_abs_diff;

namespace {

/// The loss of every block at once, written row by row with plain operations. `z` may carry a graph.
torch::Tensor reference_loss(Drafter& d, const torch::Tensor& z, const torch::Tensor& embedding, const torch::Tensor& head,
                             const SequenceContext& seq, const torch::Tensor& anchors, const LossConfig& cfg) {
    const int64_t N = z.size(0), B = z.size(1), rows = B - 1;
    const torch::Tensor offsets = torch::arange(rows, torch::kLong);
    const torch::Tensor prev_rows = anchors.unsqueeze(1) + offsets;
    const torch::Tensor prev_tokens = seq.tokens.index_select(0, prev_rows.reshape({-1})).reshape({N, rows});
    const torch::Tensor corrected = d->hidden_correction->forward(z.slice(1, 1, B), embedding.index_select(0, prev_tokens.reshape({-1})).reshape({N, rows, z.size(2)}));
    const torch::Tensor q = torch::softmax(torch::matmul(corrected, head.t()), -1);
    const torch::Tensor conf = d->confidence_head->forward(corrected);
    const torch::Tensor p = torch::softmax(torch::matmul(seq.hidden.index_select(0, prev_rows.reshape({-1})).reshape({N, rows, z.size(2)}), head.t()), -1).detach();

    torch::Tensor numerator = torch::zeros({}, torch::kFloat64), e2e = torch::zeros({}, torch::kFloat64);
    double denominator = 0;
    for (int64_t n = 0; n < N; ++n) {
        // D-PACE from the probability each row gives its corpus token
        std::vector<double> prefix((size_t) rows);
        double running = 1;
        for (int64_t k = 0; k < rows; ++k) {
            const int64_t label = seq.tokens[anchors[n].item<int64_t>() + k + 1].item<int64_t>();
            running *= (1.0 - cfg.dpace_alpha) * q[n][k][label].item<double>() + cfg.dpace_alpha;
            prefix[(size_t) k] = running;
        }
        torch::Tensor alive = torch::ones({}, torch::kFloat64), accepted = torch::zeros({}, torch::kFloat64);
        for (int64_t k = 0; k < rows; ++k) {
            double w = 0;
            for (int64_t m = k; m < rows; ++m) w += prefix[(size_t) m];
            const torch::Tensor pk = p[n][k], qk = q[n][k];
            const torch::Tensor top = std::get<1>(pk.topk(cfg.kl_topk));
            torch::Tensor kl = torch::zeros({}, torch::kFloat64);
            for (int64_t i = 0; i < top.size(0); ++i) {
                const int64_t v = top[i].item<int64_t>();
                kl = kl + pk[v] * (pk[v].log() - qk[v].log());
            }
            const torch::Tensor a = torch::minimum(pk, qk).sum();
            const double target = std::min(1.0, std::max(0.0, a.item<double>()));
            const torch::Tensor sig = torch::sigmoid(conf[n][k]);
            const torch::Tensor bce = -(target * sig.log() + (1.0 - target) * (1.0 - sig).log());
            numerator = numerator + w * (kl + cfg.confidence_weight * bce);
            denominator += w;
            alive = alive * a;
            accepted = accepted + alive;
        }
        e2e = e2e + (1.0 - accepted / (double) rows);
    }
    return numerator / std::max(denominator, 1e-6) + cfg.e2e_weight * e2e / (double) N;
}

/// The gradients of the parameters after the backbone, cloned, and then cleared.
std::vector<torch::Tensor> take_head_grads(Drafter& d) {
    std::vector<torch::Tensor> out;
    std::vector<torch::Tensor> params = d->hidden_correction->parameters();
    for (const torch::Tensor& p : d->confidence_head->parameters()) params.push_back(p);
    for (torch::Tensor& p : params) {
        out.push_back(p.grad().defined() ? p.grad().clone() : torch::zeros_like(p));
        if (p.grad().defined()) p.mutable_grad().zero_();
    }
    return out;
}

double worst(const std::vector<torch::Tensor>& a, const std::vector<torch::Tensor>& b) {
    double w = a.size() == b.size() ? 0.0 : 1e300;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) w = std::max(w, max_abs_diff(a[i], b[i]));
    return w;
}

}  // namespace

int main() {
    return dfly2_test::run("loss_test", []() -> int {
    torch::manual_seed(23);
    const auto f64 = torch::TensorOptions().dtype(torch::kFloat64);

    std::printf("1. D-PACE weights by hand\n");
    {
        // c = [0.5, 1, 0], alpha 0.5: s = [0.75, 1, 0.5], prefix products [0.75, 0.75, 0.375], sums to the end [1.875, 1.125, 0.375]
        const torch::Tensor c = torch::tensor(std::vector<double>{0.5, 1.0, 0.0}, f64);
        check(max_abs_diff(dpace_weights(c, 0.5), torch::tensor(std::vector<double>{1.875, 1.125, 0.375}, f64)) < 1e-15, "alpha 0.5");
        check(max_abs_diff(dpace_weights(c, 0.0), torch::tensor(std::vector<double>{1.0, 0.5, 0.0}, f64)) < 1e-15, "alpha 0: the raw confidences");
        check(max_abs_diff(dpace_weights(c, 1.0), torch::tensor(std::vector<double>{3.0, 2.0, 1.0}, f64)) < 1e-15, "alpha 1: the rows left to draft");
        check(max_abs_diff(dpace_weights(torch::stack({c, c * 0.5}), 0.5).select(0, 0), dpace_weights(c, 0.5)) < 1e-15, "blocks are weighted independently");
        bool threw = false;
        try { dpace_weights(c, 1.5); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "alpha outside [0, 1] is refused");
    }

    std::printf("2. the accepted-length term by hand\n");
    {
        // block 0 accepts everything; block 1: prefixes [0.5, 0.25, 0.25]; block 2 loses its first draft
        const torch::Tensor accept = torch::tensor(std::vector<double>{1, 1, 1, 0.5, 0.5, 1, 0, 1, 1}, f64).view({3, 3});
        check(max_abs_diff(e2e_tv_per_block(accept), torch::tensor(std::vector<double>{0.0, 1.0 - 1.0 / 3.0, 1.0}, f64)) < 1e-15, "1 - (1/rows) * sum of prefix products");
        check(std::abs(simulated_accept_length({8, 4, 2}, 8) - 1.625) < 1e-15, "simulated accepted length: 1 + 0.5 + 0.125");
        check(simulated_accept_length({1, 1}, 0) == 0.0, "no blocks, no length");
    }

    std::printf("3. a whole block by hand\n");
    {
        // hidden = vocab = 4 with an identity head, so a row's logits are the vector itself
        DrafterShape s;
        s.hidden = s.vocab = s.intermediate = s.correction = 4;
        s.n_layers = 1;
        s.n_head = 2;
        s.n_head_kv = 1;
        s.head_dim = 2;
        s.n_rot = 2;
        s.rope_base = 1e7;
        s.attn_norm_eps = 1e-6;
        s.rms_norm_eps = 1e-5;
        s.block = 3;
        s.mask_token_id = 3;
        s.sel_cap = 1;
        s.target_layers = {0};
        Drafter d(s);
        d->to(torch::kFloat64);
        {
            torch::NoGradGuard no_grad;
            d->confidence_head->proj->weight.zero_();
            d->confidence_head->proj->bias.zero_();
        }
        const torch::Tensor eye = torch::eye(4, f64);
        const std::vector<double> p1 = {0.5, 0.25, 0.125, 0.125}, q1 = {0.25, 0.5, 0.125, 0.125}, p2 = {0.7, 0.1, 0.1, 0.1};
        SequenceContext seq;
        seq.tokens = torch::tensor(std::vector<int64_t>{2, 2, 0, 1}, torch::kLong);   // anchor at row 1; its drafts' corpus tokens are 0 and 1
        seq.hidden = torch::zeros({4, 4}, f64);
        seq.hidden[1] = torch::tensor(p1, f64).log();
        seq.hidden[2] = torch::tensor(p2, f64).log();
        torch::Tensor z = torch::zeros({1, 3, 4}, f64);
        z[0][1] = torch::tensor(q1, f64).log();
        z[0][2] = torch::tensor(p2, f64).log();
        LossConfig cfg;
        cfg.kl_topk = 2;
        const LossTerms t = chunked_loss(d, z, eye, eye, seq, torch::tensor(std::vector<int64_t>{1}, torch::kLong), torch::Tensor(), cfg, 1.0, nullptr);
        // row 1: KL over the teacher's two likeliest = 0.5 ln 2 - 0.25 ln 2; accept 0.75. Row 2: equal distributions.
        // confidences 0.25 and 0.1 -> s = [0.625, 0.55], weights [0.96875, 0.34375]
        const double ln2 = std::log(2.0), kl = 0.96875 * 0.25 * ln2 / 1.3125;
        std::printf("  kl %.9f (hand %.9f), confidence %.9f (hand %.9f), e2e %.9f (hand 0.25), loss %.9f (hand %.9f)\n", t.kl, kl, t.confidence, ln2, t.e2e, t.loss, kl + ln2 + 0.25);
        check(std::abs(t.kl - kl) < 1e-12, "the D-PACE-weighted top-2 KL");
        check(std::abs(t.confidence - ln2) < 1e-12, "the confidence term of a head at zero is ln 2");
        check(std::abs(t.e2e - 0.25) < 1e-12, "the accepted-length term: 1 - (0.75 + 0.75) / 2");
        check(std::abs(t.loss - (kl + ln2 + 0.25)) < 1e-12, "the loss is their sum");
        check(std::abs(t.tv_accept - 1.5) < 1e-12 && t.blocks == 1, "the expected accepted drafts are 1.5");
        check(t.hits_teacher == std::vector<double>({0.0, 1.0}) && t.hits_corpus == std::vector<double>({0.0, 0.0}), "row 1 misses the teacher's pick, row 2 hits it; neither is the corpus token");
        check(std::abs(t.confidence_mae - (0.96875 * 0.25 + 0.34375 * 0.5) / 1.3125) < 1e-12, "the confidence error against the accept chances");
    }

    std::printf("4. chunked against the plain loss over every block\n");
    const DrafterShape s = dfly2_test::tiny_shape();
    Drafter drafter(s);
    drafter->to(torch::kFloat64);
    {
        torch::NoGradGuard no_grad;
        drafter->hidden_correction->down_proj->weight.copy_(torch::randn({s.hidden, s.correction}, f64) * 0.2);
        drafter->mtp_fusion->gate.fill_(0.3);
    }
    const torch::Tensor embedding = torch::randn({s.vocab, s.hidden}, f64), head = torch::randn({s.vocab, s.hidden}, f64) * 0.5;
    const SequenceContext seq = dfly2_test::random_sequence(s, 40, torch::kFloat64);
    const std::vector<int64_t> anchor_rows = {0, 3, 9, 17, 36};
    const BlockBatch blocks = dfly2_test::blocks_at(s, anchor_rows);
    LossConfig cfg;
    cfg.kl_topk = 5;
    cfg.e2e_weight = 0.7;
    cfg.confidence_weight = 0.3;
    const torch::Tensor z = torch::randn({5, s.block, s.hidden}, f64);
    {
        torch::Tensor leaf = z.clone().requires_grad_(true);
        const torch::Tensor ref = reference_loss(drafter, leaf, embedding, head, seq, blocks.anchors, cfg);
        ref.backward();
        const torch::Tensor ref_gz = leaf.grad().clone();
        const std::vector<torch::Tensor> ref_gp = take_head_grads(drafter);
        for (const int64_t chunk : {1, 2, 64}) {
            cfg.chunk_blocks = chunk;
            torch::Tensor gz;
            const LossTerms t = chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, &gz);
            const double dl = std::abs(t.loss - ref.item<double>()), dz = max_abs_diff(gz, ref_gz), dp = worst(take_head_grads(drafter), ref_gp);
            std::printf("  %lld blocks at a time: |loss - plain| %.3e, max |dz - plain| %.3e, max |dparams - plain| %.3e\n", (long long) chunk, dl, dz, dp);
            check(dl < 1e-12 && dz < 1e-12 && dp < 1e-12, "chunks of " + std::to_string(chunk) + ": the same value and gradients");
            check(gz.select(1, 0).abs().max().item<double>() == 0.0, "row 0 gets no gradient");
        }
        // the scale multiplies the gradients and they add to what is already there
        torch::Tensor gz1, gz3;
        chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, &gz1);
        chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 3.0, &gz3);
        std::vector<torch::Tensor> four = ref_gp;
        for (torch::Tensor& g : four) g = g * 4.0;
        check(max_abs_diff(gz3, gz1 * 3.0) < 1e-12 && worst(take_head_grads(drafter), four) < 1e-11, "a scale of 3 after a scale of 1 leaves four times the gradient");
        const LossTerms quiet = chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, nullptr);
        check(std::abs(quiet.loss - ref.item<double>()) < 1e-12, "without a gradient request the value is the same");
        bool clean = true;
        for (const torch::Tensor& g : take_head_grads(drafter)) clean = clean && g.abs().max().item<double>() == 0.0;
        check(clean, "and no gradient is left behind");

        // block weights: zero leaves a block out, the same weight everywhere changes nothing
        cfg.chunk_blocks = 2;
        const torch::Tensor keep = torch::tensor(std::vector<int64_t>{0, 2, 4}, torch::kLong);
        torch::Tensor gz_sub, gz_w, gz_half, gz_one;
        const LossTerms sub = chunked_loss(drafter, z.index_select(0, keep), embedding, head, seq, blocks.anchors.index_select(0, keep),
                                           torch::Tensor(), cfg, 1.0, &gz_sub);
        const std::vector<torch::Tensor> sub_gp = take_head_grads(drafter);
        const torch::Tensor w01 = torch::tensor(std::vector<double>{1, 0, 1, 0, 1}, f64);
        const LossTerms weighted = chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, w01, cfg, 1.0, &gz_w);
        const double dw = std::abs(weighted.loss - sub.loss), dzw = max_abs_diff(gz_w.index_select(0, keep), gz_sub);
        std::printf("  weights 1,0,1,0,1 against the three blocks alone: |loss| %.3e, max |dz| %.3e\n", dw, dzw);
        check(dw < 1e-12 && dzw < 1e-12 && worst(take_head_grads(drafter), sub_gp) < 1e-12 &&
                  gz_w.index_select(0, torch::tensor(std::vector<int64_t>{1, 3}, torch::kLong)).abs().max().item<double>() == 0.0,
              "a block of weight 0 is left out, its rows get no gradient");
        const LossTerms one = chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, &gz_one);
        const std::vector<torch::Tensor> one_gp = take_head_grads(drafter);
        const LossTerms half = chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::full({5}, 0.5, f64), cfg, 1.0, &gz_half);
        check(std::abs(half.loss - one.loss) < 1e-12 && max_abs_diff(gz_half, gz_one) < 1e-12 && worst(take_head_grads(drafter), one_gp) < 1e-12 &&
                  half.blocks == 5.0 && half.hits_teacher == one.hits_teacher,
              "the same weight on every block is the unweighted loss, and the counts are not weighted");
        bool threw = false;
        try { chunked_loss(drafter, z, embedding, head, seq, blocks.anchors, torch::full({5}, -1.0, f64), cfg, 1.0, nullptr); } catch (const std::exception&) { threw = true; }
        check(threw, "a negative weight is refused");
        cfg.chunk_blocks = 64;
    }

    std::printf("5. two stages against one graph through the backbone\n");
    {
        cfg.chunk_blocks = 2;
        drafter->zero_grad();
        reference_loss(drafter, drafter->backbone(embedding, seq, blocks, 64), embedding, head, seq, blocks.anchors, cfg).backward();
        std::vector<torch::Tensor> ref;
        for (const auto& p : drafter->named_parameters()) ref.push_back(p.value().grad().defined() ? p.value().grad().clone() : torch::zeros_like(p.value()));
        drafter->zero_grad();
        const torch::Tensor zb = drafter->backbone(embedding, seq, blocks, 64);
        torch::Tensor gz;
        chunked_loss(drafter, zb, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, &gz);
        zb.backward(gz);
        std::vector<torch::Tensor> two;
        for (const auto& p : drafter->named_parameters()) two.push_back(p.value().grad().defined() ? p.value().grad().clone() : torch::zeros_like(p.value()));
        const double d = worst(two, ref);
        double largest = 0;
        for (const torch::Tensor& g : ref) largest = std::max(largest, g.abs().max().item<double>());
        std::printf("  max |two-stage - one graph| over every parameter %.3e (largest gradient %.3e)\n", d, largest);
        check(d < 1e-11 && largest > 1e-4, "every parameter's gradient is the one-graph gradient");
    }

    std::printf("6. the serving chain\n");
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor zb = drafter->backbone(embedding, seq, blocks, 64);
        // by hand: each row's correction reads the draft before it, and a block scores up to its first miss
        double expect = 0;
        const int64_t rows = s.block - 1;
        for (size_t n = 0; n < anchor_rows.size(); ++n) {
            int64_t prev = seq.tokens[anchor_rows[n]].item<int64_t>();
            bool alive = true;
            for (int64_t k = 0; k < rows; ++k) {
                const torch::Tensor h = drafter->hidden_correction->forward(zb[(int64_t) n][k + 1].unsqueeze(0), embedding[prev].unsqueeze(0));
                const int64_t draft = torch::matmul(h, head.t()).argmax(-1).item<int64_t>();
                const int64_t teacher = torch::matmul(seq.hidden[anchor_rows[n] + k], head.t()).argmax(-1).item<int64_t>();
                alive = alive && draft == teacher;
                expect += alive ? 1.0 : 0.0;
                prev = draft;
            }
        }
        check(served_chain_accept(drafter, zb, embedding, head, seq, blocks.anchors, 2) == expect, "the accepted drafts equal a row-by-row walk (" + std::to_string(expect) + " over 5 blocks)");
        // a drafter that reproduces the teacher: z is the teacher's head input, the correction is off
        drafter->hidden_correction->down_proj->weight.zero_();
        torch::Tensor perfect = torch::zeros({5, s.block, s.hidden}, f64);
        for (size_t n = 0; n < anchor_rows.size(); ++n)
            for (int64_t k = 0; k < rows; ++k) perfect[(int64_t) n][k + 1] = seq.hidden[anchor_rows[n] + k];
        check(served_chain_accept(drafter, perfect, embedding, head, seq, blocks.anchors, 64) == 5.0 * (double) rows, "a drafter equal to the teacher is accepted everywhere");
        const LossTerms t = chunked_loss(drafter, perfect, embedding, head, seq, blocks.anchors, torch::Tensor(), cfg, 1.0, nullptr);
        check(std::abs(t.kl) < 1e-12 && std::abs(t.e2e) < 1e-12 && std::abs(t.tv_accept - 5.0 * (double) rows) < 1e-9 && simulated_accept_length(t.hits_teacher, t.blocks) == (double) rows,
              "and its KL and accepted-length terms are zero");
        bool threw = false;
        try { chunked_loss(drafter, perfect, embedding, head, seq, torch::tensor(std::vector<int64_t>{0, 3, 9, 17, 37}, torch::kLong), torch::Tensor(), cfg, 1.0, nullptr); } catch (const std::exception&) { threw = true; }
        check(threw, "a block that runs past its sequence is refused");
    }
    return dfly2_test::finish("loss_test");
    });
}
