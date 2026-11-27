// tools/dfly2-train/tests/optimizer_test.cpp - the optimiser: the schedule by hand, Newton-Schulz's output, a Muon
// step against one written out here, AdamW against LibTorch's, the joint clip, accumulation over micro-batches, FP32
// masters behind BF16 weights, and a resumed run. CPU only.
#include "optimizer.hpp"

#include "test_util.hpp"
#include "tiny.hpp"

#include <cmath>

using namespace basalt::dfly2;
using dfly2_test::check;
using dfly2_test::max_abs_diff;
using dfly2_test::refused;

namespace {

using Named = std::vector<std::pair<std::string, torch::Tensor>>;
const auto f64 = torch::TensorOptions().dtype(torch::kFloat64);

/// Newton-Schulz on one matrix, written out: the reference the optimiser's batched form is checked against.
torch::Tensor ns_by_hand(const torch::Tensor& g, int steps) {
    const bool tall = g.size(0) > g.size(1);
    torch::Tensor x = tall ? g.t() : g;
    double fro = 0;
    for (int64_t i = 0; i < x.size(0); ++i)
        for (int64_t j = 0; j < x.size(1); ++j) fro += x[i][j].item<double>() * x[i][j].item<double>();
    x = x / (std::sqrt(fro) + 1e-7);
    for (int s = 0; s < steps; ++s) {
        const torch::Tensor A = torch::mm(x, x.t());
        x = 3.4445 * x + torch::mm(-4.7750 * A + 2.0315 * torch::mm(A, A), x);
    }
    return tall ? x.t() : x;
}

/// Sets a parameter's gradient.
void set_grad(torch::Tensor& p, const torch::Tensor& g) { p.mutable_grad() = g.clone(); }

}  // namespace

int main() {
    return dfly2_test::run("optimizer_test", []() -> int {
    torch::manual_seed(31);

    std::printf("1. the schedule by hand\n");
    {
        OptimConfig c;
        c.lr = 1e-3;
        c.min_lr = 1e-4;
        c.total_steps = 100;
        c.warmup_ratio = 0.04;   // four warmup steps
        check(scheduled_lr(c, 0) == 0.0 && std::abs(scheduled_lr(c, 2) - 5e-4) < 1e-18 && std::abs(scheduled_lr(c, 4) - 1e-3) < 1e-18, "linear from zero to the peak at step 4");
        check(std::abs(scheduled_lr(c, 52) - (1e-4 + 0.5 * 9e-4)) < 1e-15, "half way down the cosine at step 52");
        check(std::abs(scheduled_lr(c, 28) - (1e-4 + 0.5 * (1.0 + std::cos(3.14159265358979323846 * 0.25)) * 9e-4)) < 1e-15, "a quarter of the way at step 28");
        check(std::abs(scheduled_lr(c, 100) - 1e-4) < 1e-15 && scheduled_lr(c, 500) == 1e-4, "the floor at the last step and after it");
        c.total_steps = 10;      // int(0.4) = 0: no warmup
        check(scheduled_lr(c, 0) == 1e-3, "a run too short for a warmup step starts at the peak");
    }

    std::printf("2. Newton-Schulz\n");
    {
        for (const auto& shape : std::vector<std::pair<int64_t, int64_t>>{{8, 20}, {20, 8}}) {
            const torch::Tensor g = torch::randn({shape.first, shape.second}, f64);
            const torch::Tensor x = newton_schulz(g, 5);
            const torch::Tensor sv = torch::linalg_svdvals(x);
            const double lo = sv.min().item<double>(), hi = sv.max().item<double>();
            std::printf("  %lldx%lld: singular values of the output in [%.4f, %.4f]\n", (long long) shape.first, (long long) shape.second, lo, hi);
            check(lo > 0.6 && hi < 1.25, "every singular value is near 1");
            // the output keeps the input's singular vectors: U^T X V is diagonal
            const auto usv = torch::linalg_svd(g, false);
            const torch::Tensor core = torch::mm(torch::mm(std::get<0>(usv).t(), x), std::get<2>(usv).t());
            const torch::Tensor off = core - torch::diag(torch::diag(core));
            check(off.abs().max().item<double>() < 1e-10, "and it has the input's singular vectors");
            check(max_abs_diff(x, ns_by_hand(g, 5)) < 1e-12, "it equals the iteration written out");
        }
        const torch::Tensor heads = torch::randn({3, 4, 9}, f64);
        const torch::Tensor batched = newton_schulz(heads, 5);
        double worst = 0;
        for (int64_t h = 0; h < 3; ++h) worst = std::max(worst, max_abs_diff(batched[h], ns_by_hand(heads[h], 5)));
        check(worst < 1e-12, "a stack of matrices is orthogonalised one by one");
        check(max_abs_diff(newton_schulz(heads[0], 0), heads[0] / (heads[0].norm() + 1e-7)) < 1e-15, "no steps: the normalised input");
    }

    std::printf("3. a Muon step against the update written out\n");
    {
        OptimConfig c;
        c.lr = 2e-2;
        c.min_lr = 1e-3;
        c.total_steps = 10;        // no warmup; steps 0 and 1 of the cosine
        c.max_grad_norm = 1.0;
        c.query_head_rows = 2;
        torch::Tensor wide = torch::randn({6, 10}, f64).requires_grad_(true);
        torch::Tensor tall = torch::randn({9, 4}, f64).requires_grad_(true);
        torch::Tensor query = torch::randn({8, 5}, f64).requires_grad_(true);   // four heads of two rows
        torch::Tensor norm = torch::ones({10}, f64).requires_grad_(true);
        const Named named = {{"layers.0.mlp.up_proj.weight", wide}, {"layers.0.mlp.down_proj.weight", tall}, {"layers.0.self_attn.q_proj.weight", query}, {"final_norm.weight", norm}};
        DrafterOptimizer opt(named, c);
        // the reference's state
        std::vector<torch::Tensor> w = {wide.detach().clone(), tall.detach().clone(), query.detach().clone()};
        std::vector<torch::Tensor> buf = {torch::zeros_like(w[0]), torch::zeros_like(w[1]), torch::zeros_like(w[2])};
        bool same = true;
        for (int step = 0; step < 2; ++step) {
            // step 0's gradient is large enough to be clipped, step 1's is not
            const double size = step == 0 ? 1.0 : 0.01;
            const std::vector<torch::Tensor> g = {torch::randn({6, 10}, f64) * size, torch::randn({9, 4}, f64) * size, torch::randn({8, 5}, f64) * size, torch::randn({10}, f64) * size};
            set_grad(wide, g[0]);
            set_grad(tall, g[1]);
            set_grad(query, g[2]);
            set_grad(norm, g[3]);
            opt.accumulate(1.0);
            const double reported = opt.step();
            double sq = 0;
            for (const torch::Tensor& x : g) sq += x.pow(2).sum().item<double>();
            const double total = std::sqrt(sq), clip = std::min(1.0, 1.0 / (total + 1e-6)), lr = scheduled_lr(c, step);
            same = same && std::abs(reported - total) < 1e-12;
            for (int i = 0; i < 3; ++i) {
                const torch::Tensor gc = g[(size_t) i] * clip;
                buf[(size_t) i] = 0.95 * buf[(size_t) i] + gc;
                const torch::Tensor update = gc + 0.95 * buf[(size_t) i];
                torch::Tensor u;
                double rows = (double) update.size(0);
                if (i == 2) {
                    // head by head: rows 2h and 2h + 1
                    std::vector<torch::Tensor> parts;
                    for (int64_t h = 0; h < 4; ++h) parts.push_back(ns_by_hand(update.slice(0, 2 * h, 2 * h + 2), 5));
                    u = torch::cat(parts, 0);
                    rows = 2;
                } else {
                    u = ns_by_hand(update, 5);
                }
                const double adjusted = lr * std::sqrt(std::max(rows, (double) update.size(1))) * 0.2;
                w[(size_t) i] = w[(size_t) i] * (1.0 - lr * 0.1) - adjusted * u;
            }
            const double d = std::max({max_abs_diff(wide, w[0]), max_abs_diff(tall, w[1]), max_abs_diff(query, w[2])});
            std::printf("  step %d: gradient norm %.4f, clip %.4f, max |weight - written out| %.3e\n", step, total, clip, d);
            same = same && d < 1e-12;
        }
        check(same, "two steps (the first clipped) match for a wide, a tall and a head-wise matrix");
        check(opt.steps() == 2 && opt.adam_steps() == 2, "the counters advance");
    }

    std::printf("4. AdamW against LibTorch's\n");
    {
        OptimConfig c;
        c.lr = 3e-3;
        c.min_lr = 1e-4;
        c.total_steps = 10;
        c.max_grad_norm = 1e9;     // no clipping
        torch::Tensor a = torch::randn({7}, f64).requires_grad_(true), b = torch::randn({1, 7}, f64).requires_grad_(true);
        torch::Tensor ra = a.detach().clone().requires_grad_(true), rb = b.detach().clone().requires_grad_(true);
        DrafterOptimizer opt({{"final_norm.weight", a}, {"confidence_head.proj.weight", b}}, c);
        torch::optim::AdamW ref(std::vector<torch::Tensor>{ra, rb}, torch::optim::AdamWOptions(c.lr).betas({c.beta1, c.beta2}).eps(c.adam_eps).weight_decay(c.weight_decay));
        double worst = 0;
        for (int step = 0; step < 4; ++step) {
            const torch::Tensor ga = torch::randn({7}, f64), gb = torch::randn({1, 7}, f64);
            set_grad(a, ga);
            set_grad(b, gb);
            set_grad(ra, ga);
            set_grad(rb, gb);
            static_cast<torch::optim::AdamWOptions&>(ref.param_groups()[0].options()).lr(scheduled_lr(c, step));
            opt.accumulate(1.0);
            opt.step();
            ref.step();
            worst = std::max({worst, max_abs_diff(a, ra), max_abs_diff(b, rb)});
        }
        std::printf("  max |ours - LibTorch AdamW| over four steps %.3e\n", worst);
        check(worst < 1e-14, "a vector and the confidence head's matrix follow LibTorch's AdamW on the schedule");
    }

    std::printf("5. the clip, a zero gradient, and accumulation\n");
    {
        OptimConfig c;
        c.total_steps = 10;
        const torch::Tensor w0 = torch::randn({4, 6}, f64), n0 = torch::randn({6}, f64), g = torch::randn({4, 6}, f64), gn = torch::randn({6}, f64);
        const auto run = [&](const std::vector<std::pair<torch::Tensor, torch::Tensor>>& grads, const std::vector<double>& weights, double& norm) {
            torch::Tensor w = w0.clone().requires_grad_(true), n = n0.clone().requires_grad_(true);
            DrafterOptimizer opt({{"layers.0.mlp.up_proj.weight", w}, {"final_norm.weight", n}}, c);
            for (size_t i = 0; i < grads.size(); ++i) {
                set_grad(w, grads[i].first);
                set_grad(n, grads[i].second);
                opt.accumulate(weights[i]);
            }
            norm = opt.step();
            return std::make_pair(w.detach().clone(), n.detach().clone());
        };
        double norm_big = 0, norm_unit = 0, norm_zero = 0, norm_acc = 0, norm_avg = 0;
        const double total = std::sqrt(g.pow(2).sum().item<double>() + gn.pow(2).sum().item<double>());
        const auto big = run({{g * 50.0, gn * 50.0}}, {1.0}, norm_big);
        const double clipped = 1.0 / (50.0 * total + 1e-6);
        const auto unit = run({{g * 50.0 * clipped, gn * 50.0 * clipped}}, {1.0}, norm_unit);
        check(std::abs(norm_big - 50.0 * total) < 1e-9 && max_abs_diff(big.first, unit.first) < 1e-12 && max_abs_diff(big.second, unit.second) < 1e-12,
              "a gradient of norm 50x is stepped as the same gradient scaled to norm 1");
        const auto zero = run({{g * 0.0, gn * 0.0}}, {1.0}, norm_zero);
        check(norm_zero == 0.0 && max_abs_diff(zero.first, w0) == 0.0 && max_abs_diff(zero.second, n0) == 0.0, "a zero gradient changes nothing");
        // two micro-batches carrying 2 and 6 units: the step uses their sum over 8
        const torch::Tensor g2 = torch::randn({4, 6}, f64) * 0.01, gn2 = torch::randn({6}, f64) * 0.01;
        const auto acc = run({{g * 0.01, gn * 0.01}, {g2, gn2}}, {2.0, 6.0}, norm_acc);
        const auto avg = run({{(g * 0.01 + g2) / 8.0, (gn * 0.01 + gn2) / 8.0}}, {1.0}, norm_avg);
        check(std::abs(norm_acc - norm_avg) < 1e-15 && max_abs_diff(acc.first, avg.first) < 1e-13 && max_abs_diff(acc.second, avg.second) < 1e-13,
              "accumulated micro-batches step as their weighted mean");
        torch::Tensor w = w0.clone().requires_grad_(true);
        DrafterOptimizer opt({{"layers.0.mlp.up_proj.weight", w}}, c);
        set_grad(w, g);
        opt.accumulate(1.0);
        check(w.grad().abs().max().item<double>() == 0.0 && opt.accumulated_weight() == 1.0, "accumulating clears the model's gradient");
        opt.discard();
        check(opt.step() == 0.0 && max_abs_diff(w, w0) == 0.0 && opt.steps() == 1 && opt.adam_steps() == 0, "a discarded sum steps nothing but the schedule");
    }

    std::printf("6. FP32 masters behind BF16 weights\n");
    {
        OptimConfig c;
        c.lr = 1e-5;               // steps far below a BF16 weight's resolution
        c.min_lr = 1e-5;
        c.total_steps = 100;
        c.warmup_ratio = 0.0;
        c.muon_weight_decay = 0.0;
        c.weight_decay = 0.0;
        torch::Tensor w = torch::ones({4, 6}, torch::kBFloat16).requires_grad_(true), n = torch::ones({6}, torch::kBFloat16).requires_grad_(true);
        DrafterOptimizer opt({{"layers.0.mlp.up_proj.weight", w}, {"final_norm.weight", n}}, c);
        const torch::Tensor g = torch::randn({4, 6}), gn = torch::randn({6});
        for (int i = 0; i < 20; ++i) {
            set_grad(w, g.to(torch::kBFloat16));
            set_grad(n, gn.to(torch::kBFloat16));
            opt.accumulate(1.0);
            opt.step();
        }
        const auto masters = opt.masters();
        const torch::Tensor mw = masters[0].second, mn = masters[1].second;
        check(mw.scalar_type() == torch::kFloat32 && mn.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kBFloat16, "the masters are FP32, the model stays BF16");
        check(max_abs_diff(w, mw.to(torch::kBFloat16)) == 0.0 && max_abs_diff(n, mn.to(torch::kBFloat16)) == 0.0, "the model is the masters rounded to BF16");
        const double moved = (mn - 1.0).abs().max().item<double>();
        std::printf("  after 20 steps of 1e-5 the master moved by up to %.3e; a BF16 weight at 1 resolves 7.8e-03\n", moved);
        check(moved > 1.5e-4 && moved < 3e-4 && max_abs_diff(n, torch::ones({6})) == 0.0, "steps too small for BF16 add up in the master and not yet in the model");
    }

    std::printf("7. a resumed run\n");
    {
        OptimConfig c;
        c.total_steps = 20;
        c.query_head_rows = 2;
        const auto make = [&](torch::Tensor& w, torch::Tensor& q, torch::Tensor& n) {
            return DrafterOptimizer({{"layers.0.mlp.up_proj.weight", w}, {"layers.0.self_attn.q_proj.weight", q}, {"final_norm.weight", n}}, c);
        };
        torch::manual_seed(77);
        torch::Tensor w = torch::randn({4, 6}).to(torch::kBFloat16).requires_grad_(true), q = torch::randn({4, 5}).to(torch::kBFloat16).requires_grad_(true),
                      n = torch::ones({6}, torch::kBFloat16).requires_grad_(true);
        DrafterOptimizer first = make(w, q, n);
        std::vector<std::vector<torch::Tensor>> grads;
        for (int i = 0; i < 3; ++i) grads.push_back({torch::randn({4, 6}).to(torch::kBFloat16), torch::randn({4, 5}).to(torch::kBFloat16), torch::randn({6}).to(torch::kBFloat16)});
        const auto feed = [&](DrafterOptimizer& o, torch::Tensor& a, torch::Tensor& b, torch::Tensor& d, int i) {
            set_grad(a, grads[(size_t) i][0]);
            set_grad(b, grads[(size_t) i][1]);
            set_grad(d, grads[(size_t) i][2]);
            o.accumulate(1.0);
            o.step();
        };
        feed(first, w, q, n, 0);
        feed(first, w, q, n, 1);
        // what a checkpoint holds: the masters, the buffers and the counters
        std::map<std::string, torch::Tensor> masters, buffers;
        for (const auto& [name, t] : first.masters()) masters[name] = t.clone();
        for (const auto& [name, t] : first.buffers()) buffers[name] = t.clone();
        torch::Tensor w2 = torch::zeros({4, 6}, torch::kBFloat16).requires_grad_(true), q2 = torch::zeros({4, 5}, torch::kBFloat16).requires_grad_(true),
                      n2 = torch::zeros({6}, torch::kBFloat16).requires_grad_(true);
        DrafterOptimizer second = make(w2, q2, n2);
        std::string err;
        check(second.restore(masters, buffers, first.steps(), first.adam_steps(), err), "the state restores", err);
        check(max_abs_diff(w2, w) == 0.0 && max_abs_diff(q2, q) == 0.0 && max_abs_diff(n2, n) == 0.0 && second.steps() == 2 && second.next_lr() == first.next_lr(),
              "the model and the schedule are where the first run left them");
        feed(first, w, q, n, 2);
        feed(second, w2, q2, n2, 2);
        const auto ma = first.masters(), mb = second.masters();
        check(max_abs_diff(ma[0].second, mb[0].second) == 0.0 && max_abs_diff(ma[1].second, mb[1].second) == 0.0 && max_abs_diff(ma[2].second, mb[2].second) == 0.0,
              "the next step is bit for bit the uninterrupted run's");
        buffers.erase("muon.layers.0.mlp.up_proj.weight");
        refused(!second.restore(masters, buffers, 2, 2, err), "a state missing a buffer", err);
        check(second.steps() == 3, "and a refused state changes nothing");
        bool threw = false;
        try { DrafterOptimizer none({}, c); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "an optimiser with no parameters is refused");
    }
    return dfly2_test::finish("optimizer_test");
    });
}
