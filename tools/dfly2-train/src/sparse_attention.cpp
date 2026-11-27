// tools/dfly2-train/src/sparse_attention.cpp - the attention function of sparse_attention.hpp.
#include "sparse_attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace basalt::dfly2 {

namespace {

/// What one chunk of blocks needs in either pass: everything shaped [C * n_head_kv][rows][...], rows = B * ratio
/// (the query heads of one key/value head, row-major by block row).
struct Chunk {
    torch::Tensor q;          ///< [C*G][B*R][dh]
    torch::Tensor k_ctx;      ///< [C*G][W][dh]
    torch::Tensor v_ctx;      ///< [C*G][W][dh]
    torch::Tensor k_blk;      ///< [C*G][B][dh]
    torch::Tensor v_blk;      ///< [C*G][B][dh]
    torch::Tensor probs;      ///< [C*G][B*R][W + B]
    torch::Tensor out;        ///< [C*G][B*R][dh]
};

/// The sizes every helper shares.
struct Dims {
    int64_t N = 0, B = 0, H = 0, G = 0, R = 0, dh = 0, W = 0;
    double scale = 1.0;
    torch::Dtype type = torch::kFloat32;
};

/// [C][B][G][R][dh] query rows as [C*G][B*R][dh].
torch::Tensor heads_to_rows(const torch::Tensor& x, const Dims& d) {
    const int64_t C = x.size(0);
    return x.reshape({C, d.B, d.G, d.R, d.dh}).permute({0, 2, 1, 3, 4}).reshape({C * d.G, d.B * d.R, d.dh});
}

/// The inverse of heads_to_rows: [C*G][B*R][dh] as [C][B][G*R][dh].
torch::Tensor rows_to_heads(const torch::Tensor& x, int64_t C, const Dims& d) {
    return x.reshape({C, d.G, d.B, d.R, d.dh}).permute({0, 2, 1, 3, 4}).reshape({C, d.B, d.H, d.dh});
}

/// [C][B][G][dh] block keys or values as [C*G][B][dh].
torch::Tensor block_to_groups(const torch::Tensor& x, const Dims& d) {
    const int64_t C = x.size(0);
    return x.permute({0, 2, 1, 3}).reshape({C * d.G, d.B, d.dh});
}

/// Gathered cells [C][W][G][dh], the blocks `int8` ([C] bool, or empty for none) marks read as an INT8 cache returns
/// them.
torch::Tensor as_cached(const torch::Tensor& cells, const torch::Tensor& int8, const Dims& d) {
    if (int8.numel() == 0) return cells.to(d.type);
    return torch::where(int8.view({-1, 1, 1, 1}), kv_int8_round(cells).to(d.type), cells.to(d.type));
}

/// Gathers and scores blocks [n0, n1): the probabilities and the output of the chunk.
Chunk run_chunk(const torch::Tensor& q, const torch::Tensor& k_blk, const torch::Tensor& v_blk, const torch::Tensor& k_ctx,
                const torch::Tensor& v_ctx, const torch::Tensor& idx, const torch::Tensor& int8, int64_t n0, int64_t n1, const Dims& d) {
    const int64_t C = n1 - n0;
    Chunk c;
    c.q = heads_to_rows(q.slice(0, n0, n1).to(d.type), d);
    c.k_blk = block_to_groups(k_blk.slice(0, n0, n1).to(d.type), d);
    c.v_blk = block_to_groups(v_blk.slice(0, n0, n1).to(d.type), d);

    // the selected cells once per block: every row of the block reads the same ones
    const torch::Tensor ids = idx.slice(0, n0, n1);
    const torch::Tensor valid = ids.ge(0);
    const torch::Tensor flat = ids.clamp_min(0).reshape({-1});
    const torch::Tensor marks = int8.numel() > 0 ? int8.slice(0, n0, n1) : int8;
    c.k_ctx = as_cached(k_ctx.index_select(0, flat).reshape({C, d.W, d.G, d.dh}), marks, d).permute({0, 2, 1, 3}).reshape({C * d.G, d.W, d.dh});
    c.v_ctx = as_cached(v_ctx.index_select(0, flat).reshape({C, d.W, d.G, d.dh}), marks, d).permute({0, 2, 1, 3}).reshape({C * d.G, d.W, d.dh});

    const double ninf = -std::numeric_limits<double>::infinity();
    // context scores: padding cells are out
    torch::Tensor s_ctx = torch::bmm(c.q, c.k_ctx.transpose(1, 2)) * d.scale;
    const torch::Tensor pad = valid.logical_not().unsqueeze(1).expand({C, d.G, d.W}).reshape({C * d.G, 1, d.W});
    s_ctx = s_ctx.masked_fill(pad, ninf);
    // block scores: row r reads the block's rows 0..r
    torch::Tensor s_blk = torch::bmm(c.q, c.k_blk.transpose(1, 2)) * d.scale;
    const torch::Tensor slot = torch::arange(d.B, torch::TensorOptions().dtype(torch::kLong).device(q.device()));
    const torch::Tensor query_slot = slot.repeat_interleave(d.R);
    const torch::Tensor later = slot.unsqueeze(0).gt(query_slot.unsqueeze(1));   // [B*R][B]: the key is after the query
    s_blk = s_blk.masked_fill(later.unsqueeze(0), ninf);

    c.probs = torch::softmax(torch::cat({s_ctx, s_blk}, 2), 2);
    c.out = torch::bmm(c.probs.slice(2, 0, d.W), c.v_ctx) + torch::bmm(c.probs.slice(2, d.W, d.W + d.B), c.v_blk);
    return c;
}

/// Checks the arguments and fills the sizes.
Dims check_dims(const torch::Tensor& q, const torch::Tensor& k_blk, const torch::Tensor& v_blk, const torch::Tensor& k_ctx,
                const torch::Tensor& v_ctx, const torch::Tensor& idx, const torch::Tensor& int8) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("sparse_context_attention: " + what); };
    if (!q.defined() || !k_blk.defined() || !v_blk.defined() || !k_ctx.defined() || !v_ctx.defined() || !idx.defined()) fail("an argument is undefined");
    if (q.dim() != 4 || k_blk.dim() != 4 || v_blk.dim() != 4 || k_ctx.dim() != 3 || v_ctx.dim() != 3 || idx.dim() != 2) fail("an argument has the wrong rank");
    Dims d;
    d.N = q.size(0);
    d.B = q.size(1);
    d.H = q.size(2);
    d.dh = q.size(3);
    d.G = k_blk.size(2);
    d.W = idx.size(1);
    if (d.N <= 0 || d.B <= 0 || d.H <= 0 || d.dh <= 0 || d.G <= 0) fail("an empty dimension");
    if (d.H % d.G != 0) fail("query heads are not a multiple of key/value heads");
    d.R = d.H / d.G;
    if (k_blk.size(0) != d.N || k_blk.size(1) != d.B || k_blk.size(3) != d.dh || !(v_blk.sizes() == k_blk.sizes())) fail("block keys or values do not match the queries");
    if (k_ctx.size(1) != d.G || k_ctx.size(2) != d.dh || !(v_ctx.sizes() == k_ctx.sizes())) fail("cached keys or values do not match the block's");
    if (idx.size(0) != d.N || idx.scalar_type() != torch::kLong) fail("cell ids are not [blocks][width] int64");
    if (!int8.defined() || int8.dim() != 1 || int8.scalar_type() != torch::kBool || (int8.numel() != 0 && int8.size(0) != d.N))
        fail("the INT8 marks are not [blocks] bool");
    if (int8.numel() != 0 && d.dh % kKvInt8Group != 0) fail("an INT8 cache needs a head width that is a multiple of 64");
    if (!q.is_floating_point() || !k_blk.is_floating_point() || !v_blk.is_floating_point() || !k_ctx.is_floating_point() || !v_ctx.is_floating_point()) fail("a tensor is not floating point");
    if (d.W > 0 && k_ctx.size(0) == 0) fail("cell ids without a cache");
    if (d.W > 0) {
        const int64_t hi = idx.max().item<int64_t>(), lo = idx.min().item<int64_t>();
        if (hi >= k_ctx.size(0) || lo < -1) fail("a cell id is outside the cache");
    }
    d.scale = 1.0 / std::sqrt((double) d.dh);
    d.type = q.scalar_type() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
    return d;
}

struct SparseContextAttention : public torch::autograd::Function<SparseContextAttention> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor q, torch::Tensor k_blk, torch::Tensor v_blk,
                                 torch::Tensor k_ctx, torch::Tensor v_ctx, torch::Tensor idx, torch::Tensor int8, int64_t chunk_blocks) {
        const Dims d = check_dims(q, k_blk, v_blk, k_ctx, v_ctx, idx, int8);
        const int64_t step = std::max<int64_t>(1, chunk_blocks);
        std::vector<torch::Tensor> outs;
        for (int64_t n0 = 0; n0 < d.N; n0 += step) {
            const int64_t n1 = std::min(d.N, n0 + step);
            outs.push_back(rows_to_heads(run_chunk(q, k_blk, v_blk, k_ctx, v_ctx, idx, int8, n0, n1, d).out, n1 - n0, d));
        }
        // the inputs only: the gathered cells and the scores are made again in backward
        ctx->save_for_backward({q, k_blk, v_blk, k_ctx, v_ctx, idx});
        ctx->saved_data["chunk"] = step;
        ctx->saved_data["marks"] = int8;
        return torch::cat(outs, 0);
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const torch::Tensor &q = saved[0], &k_blk = saved[1], &v_blk = saved[2], &k_ctx = saved[3], &v_ctx = saved[4], &idx = saved[5];
        const int64_t step = ctx->saved_data["chunk"].toInt();
        const torch::Tensor int8 = ctx->saved_data["marks"].toTensor();
        const torch::Tensor& grad = grad_outputs[0];
        // one slot per forward argument, the marks and the chunk size included; only the first three carry a gradient
        torch::autograd::tensor_list grads(8);
        if (!grad.defined()) return grads;
        torch::NoGradGuard no_grad;
        const Dims d = check_dims(q, k_blk, v_blk, k_ctx, v_ctx, idx, int8);
        std::vector<torch::Tensor> dq, dk, dv;
        for (int64_t n0 = 0; n0 < d.N; n0 += step) {
            const int64_t n1 = std::min(d.N, n0 + step), C = n1 - n0;
            const Chunk c = run_chunk(q, k_blk, v_blk, k_ctx, v_ctx, idx, int8, n0, n1, d);
            const torch::Tensor g = heads_to_rows(grad.slice(0, n0, n1).to(d.type), d);
            const torch::Tensor p_ctx = c.probs.slice(2, 0, d.W), p_blk = c.probs.slice(2, d.W, d.W + d.B);
            // through the softmax: ds = p * (g . v - g . out)
            const torch::Tensor centre = (g * c.out).sum(2, true);
            const torch::Tensor ds_ctx = p_ctx * (torch::bmm(g, c.v_ctx.transpose(1, 2)) - centre);
            const torch::Tensor ds_blk = p_blk * (torch::bmm(g, c.v_blk.transpose(1, 2)) - centre);
            dq.push_back(rows_to_heads((torch::bmm(ds_ctx, c.k_ctx) + torch::bmm(ds_blk, c.k_blk)) * d.scale, C, d));
            // a block key or value collects from every query row of its key/value head
            dk.push_back((torch::bmm(ds_blk.transpose(1, 2), c.q) * d.scale).reshape({C, d.G, d.B, d.dh}).permute({0, 2, 1, 3}));
            dv.push_back(torch::bmm(p_blk.transpose(1, 2), g).reshape({C, d.G, d.B, d.dh}).permute({0, 2, 1, 3}));
        }
        grads[0] = torch::cat(dq, 0).to(q.scalar_type());
        grads[1] = torch::cat(dk, 0).to(k_blk.scalar_type());
        grads[2] = torch::cat(dv, 0).to(v_blk.scalar_type());
        return grads;
    }
};

}  // namespace

torch::Tensor kv_int8_round(const torch::Tensor& x) {
    if (!x.defined() || x.dim() < 1 || x.size(-1) % kKvInt8Group != 0 || !x.is_floating_point())
        throw std::invalid_argument("kv_int8_round: the last dimension is not whole groups of 64 values");
    torch::NoGradGuard no_grad;
    std::vector<int64_t> groups(x.sizes().begin(), x.sizes().end() - 1);
    groups.push_back(x.size(-1) / kKvInt8Group);
    groups.push_back(kKvInt8Group);
    const torch::Tensor g = x.to(torch::kFloat32).reshape(groups);
    // the divisors as tensors, so every device divides exactly rather than multiplying by a reciprocal
    const torch::Tensor amax = g.abs().amax(-1, true);
    const torch::Tensor scale = torch::div(amax, torch::full_like(amax, 127.0f)).to(torch::kFloat16).to(torch::kFloat32);
    // the code against the stored scale, rounded half to even as the kernel's __float2int_rn; a zero scale codes 0
    const torch::Tensor safe = torch::where(scale > 0, scale, torch::ones_like(scale));
    const torch::Tensor code = torch::where(scale > 0, torch::round(torch::div(g, safe)).clamp(-127, 127), torch::zeros_like(g));
    return (code * scale).to(torch::kFloat16).to(torch::kFloat32).reshape(x.sizes());
}

torch::Tensor sparse_context_attention(const torch::Tensor& q, const torch::Tensor& k_blk, const torch::Tensor& v_blk,
                                       const torch::Tensor& k_ctx, const torch::Tensor& v_ctx, const torch::Tensor& idx,
                                       int64_t chunk_blocks, const torch::Tensor& int8) {
    // autograd takes no undefined argument: no marks are an empty list
    const torch::Tensor marks = int8.defined() ? int8 : torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(q.device()));
    return SparseContextAttention::apply(q, k_blk, v_blk, k_ctx, v_ctx, idx, marks, chunk_blocks);
}

}  // namespace basalt::dfly2
