// tools/dfly2-train/tests/attention_test.cpp - the drafter's attention function against a plain dense-masked attention
// written with ordinary operations: the same output, the same gradients, and gradients that agree with central
// differences in FP64. CPU only.
#include "sparse_attention.hpp"

#include "test_util.hpp"
#include "tiny.hpp"

#include <cmath>
#include <functional>

using basalt::dfly2::sparse_context_attention;
using dfly2_test::check;
using dfly2_test::max_abs_diff;

namespace {

/// The attention over every cached cell and every block row at once, with a mask saying what a row may read.
torch::Tensor dense_attention(const torch::Tensor& q, const torch::Tensor& k_blk, const torch::Tensor& v_blk, const torch::Tensor& k_ctx,
                              const torch::Tensor& v_ctx, const torch::Tensor& idx) {
    const int64_t N = q.size(0), B = q.size(1), H = q.size(2), dh = q.size(3), G = k_blk.size(2), L = k_ctx.size(0), R = H / G;
    // which key/value head each query head reads: h / R
    const torch::Tensor head = torch::floor_divide(torch::arange(H, torch::kLong), R);
    const torch::Tensor keys = torch::cat({k_ctx.to(q.scalar_type()).unsqueeze(0).expand({N, L, G, dh}), k_blk}, 1).index_select(2, head);
    const torch::Tensor values = torch::cat({v_ctx.to(q.scalar_type()).unsqueeze(0).expand({N, L, G, dh}), v_blk}, 1).index_select(2, head);
    // a row may read its block's selected cells and the block's rows up to itself
    torch::Tensor allowed = torch::zeros({N, B, L + B}, torch::kBool);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t r = 0; r < B; ++r) {
            for (int64_t j = 0; j < idx.size(1); ++j) {
                const int64_t cell = idx[n][j].item<int64_t>();
                if (cell >= 0) allowed[n][r][cell] = true;
            }
            for (int64_t b = 0; b <= r; ++b) allowed[n][r][L + b] = true;
        }
    torch::Tensor scores = torch::einsum("nrhd,njhd->nrhj", {q, keys}) / std::sqrt((double) dh);
    scores = scores.masked_fill(allowed.logical_not().unsqueeze(2), -std::numeric_limits<double>::infinity());
    return torch::einsum("nrhj,njhd->nrhd", {torch::softmax(scores, 3), values});
}

/// d(sum(f() * weight)) / d(x) by central differences.
torch::Tensor numeric_grad(const std::function<torch::Tensor()>& f, torch::Tensor& x, const torch::Tensor& weight, double eps) {
    torch::NoGradGuard no_grad;
    torch::Tensor grad = torch::zeros_like(x);
    torch::Tensor flat = x.view({-1}), gflat = grad.view({-1});
    for (int64_t i = 0; i < flat.size(0); ++i) {
        const double keep = flat[i].item<double>();
        flat[i] = keep + eps;
        const double up = (f() * weight).sum().item<double>();
        flat[i] = keep - eps;
        const double down = (f() * weight).sum().item<double>();
        flat[i] = keep;
        gflat[i] = (up - down) / (2 * eps);
    }
    return grad;
}

}  // namespace

int main() {
    return dfly2_test::run("attention_test", []() -> int {
    torch::manual_seed(11);
    const int64_t N = 3, B = 4, H = 4, G = 2, dh = 6, L = 10;
    const auto f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor q = torch::randn({N, B, H, dh}, f64).requires_grad_(true);
    torch::Tensor k_blk = torch::randn({N, B, G, dh}, f64).requires_grad_(true);
    torch::Tensor v_blk = torch::randn({N, B, G, dh}, f64).requires_grad_(true);
    const torch::Tensor k_ctx = torch::randn({L, G, dh}, f64), v_ctx = torch::randn({L, G, dh}, f64);
    // block 0 reads five cells, block 1 two and padding, block 2 nothing but its own rows
    const torch::Tensor idx = torch::tensor(std::vector<int64_t>{0, 2, 3, 7, 9, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1}, torch::kLong).view({N, 5});
    const torch::Tensor weight = torch::randn({N, B, H, dh}, f64);

    std::printf("1. forward against the dense-masked attention\n");
    const torch::Tensor ours = sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx);
    const torch::Tensor dense = dense_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx);
    const double fwd = max_abs_diff(ours, dense);
    std::printf("  max |ours - dense| = %.3e\n", fwd);
    check(fwd < 1e-12, "the output equals the dense-masked attention's");
    check(max_abs_diff(sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx, 1), ours) < 1e-13, "one block at a time gives the same output");
    check(max_abs_diff(sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx, 2), ours) < 1e-13, "two blocks at a time give the same output");

    std::printf("2. gradients against the dense-masked attention's autograd\n");
    const auto ga = torch::autograd::grad({(ours * weight).sum()}, {q, k_blk, v_blk});
    const auto gd = torch::autograd::grad({(dense * weight).sum()}, {q, k_blk, v_blk});
    const double dq = max_abs_diff(ga[0], gd[0]), dk = max_abs_diff(ga[1], gd[1]), dv = max_abs_diff(ga[2], gd[2]);
    std::printf("  max |grad - dense grad|: q %.3e, block k %.3e, block v %.3e\n", dq, dk, dv);
    check(dq < 1e-12 && dk < 1e-12 && dv < 1e-12, "the three gradients equal the dense attention's");
    const auto g1 = torch::autograd::grad({(sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx, 1) * weight).sum()}, {q, k_blk, v_blk});
    check(max_abs_diff(g1[0], ga[0]) < 1e-13 && max_abs_diff(g1[1], ga[1]) < 1e-13 && max_abs_diff(g1[2], ga[2]) < 1e-13,
          "one block at a time gives the same gradients");

    std::printf("3. gradients against central differences (FP64)\n");
    const auto f = [&]() { return sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx); };
    const double nq = max_abs_diff(numeric_grad(f, q, weight, 1e-6), ga[0]);
    const double nk = max_abs_diff(numeric_grad(f, k_blk, weight, 1e-6), ga[1]);
    const double nv = max_abs_diff(numeric_grad(f, v_blk, weight, 1e-6), ga[2]);
    std::printf("  max |grad - numeric|: q %.3e, block k %.3e, block v %.3e\n", nq, nk, nv);
    check(nq < 1e-7 && nk < 1e-7 && nv < 1e-7, "the three gradients pass the numerical check");

    std::printf("4. what a row reads\n");
    {
        torch::NoGradGuard no_grad;
        // a later row's key and value do not reach an earlier row
        torch::Tensor k2 = k_blk.clone(), v2 = v_blk.clone();
        k2.select(1, 2) += 3.0;
        v2.select(1, 2) -= 2.0;
        const torch::Tensor moved = sparse_context_attention(q, k2, v2, k_ctx, v_ctx, idx);
        check(max_abs_diff(moved.slice(1, 0, 2), ours.slice(1, 0, 2)) == 0.0, "rows 0 and 1 do not read row 2");
        check(max_abs_diff(moved.slice(1, 2, 4), ours.slice(1, 2, 4)) > 1e-3, "rows 2 and 3 do");
        // a cell outside a block's list does not reach it, a cell inside does
        torch::Tensor kc = k_ctx.clone(), vc = v_ctx.clone();
        kc[5] += 4.0;
        vc[5] += 4.0;
        check(max_abs_diff(sparse_context_attention(q, k_blk, v_blk, kc, vc, idx), ours) == 0.0, "an unselected cell changes nothing");
        vc[4] += 4.0;
        const torch::Tensor sel = sparse_context_attention(q, k_blk, v_blk, k_ctx, vc, idx);
        check(max_abs_diff(sel.select(0, 1), ours.select(0, 1)) > 1e-3 && max_abs_diff(sel.select(0, 0), ours.select(0, 0)) == 0.0,
              "a selected cell reaches its block and no other");
        // the query head to key/value head map is h / (n_head / n_head_kv)
        torch::Tensor vh = v_ctx.clone();
        vh.select(1, 1) += 5.0;
        const torch::Tensor heads = sparse_context_attention(q, k_blk, v_blk, k_ctx, vh, idx);
        check(max_abs_diff(heads.slice(2, 0, 2), ours.slice(2, 0, 2)) == 0.0 && max_abs_diff(heads.slice(2, 2, 4).select(0, 0), ours.slice(2, 2, 4).select(0, 0)) > 1e-3,
              "query heads 0 and 1 read key/value head 0, heads 2 and 3 read head 1");
    }

    std::printf("5. a half-precision cache under single-precision queries\n");
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor k16 = k_ctx.to(torch::kFloat16), v16 = v_ctx.to(torch::kFloat16);
        const torch::Tensor out32 = sparse_context_attention(q.to(torch::kFloat32), k_blk.to(torch::kFloat32), v_blk.to(torch::kFloat32), k16, v16, idx);
        const torch::Tensor ref = sparse_context_attention(q, k_blk, v_blk, k16.to(torch::kFloat64), v16.to(torch::kFloat64), idx);
        const double d = max_abs_diff(out32, ref);
        std::printf("  max |fp32 - fp64 on the same cells| = %.3e\n", d);
        check(out32.scalar_type() == torch::kFloat32 && d < 1e-5, "the cells are read as stored and the output is FP32");
    }

    std::printf("6. refusals\n");
    const auto throws = [](const std::function<void()>& g) {
        try { g(); } catch (const std::exception&) { return true; }
        return false;
    };
    check(throws([&] { sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx.to(torch::kInt32)); }), "cell ids that are not int64");
    check(throws([&] { sparse_context_attention(q, k_blk, v_blk, k_ctx, v_ctx, idx + 20); }), "a cell id outside the cache");
    check(throws([&] { sparse_context_attention(q, k_blk.slice(1, 0, 3), v_blk, k_ctx, v_ctx, idx); }), "block keys of another shape");
    check(throws([&] { sparse_context_attention(q.slice(2, 0, 3), k_blk, v_blk, k_ctx, v_ctx, idx); }), "query heads that are not a multiple of the key/value heads");
    check(throws([&] { sparse_context_attention(q, k_blk, v_blk, torch::Tensor(), v_ctx, idx); }), "an undefined cache");
    return dfly2_test::finish("attention_test");
    });
}
