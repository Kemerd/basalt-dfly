// tools/dfly2-train/tests/tiny.hpp - a drafter small enough for the CPU tests, and random teacher data for it.
#pragma once

#include "drafter.hpp"

#include <algorithm>
#include <vector>

namespace dfly2_test {

/// Two layers, four query heads over two key/value heads of 8, a vocabulary of 40.
inline basalt::dfly2::DrafterShape tiny_shape() {
    basalt::dfly2::DrafterShape s;
    s.hidden = 16;
    s.intermediate = 48;
    s.correction = 16;
    s.n_layers = 2;
    s.n_head = 4;
    s.n_head_kv = 2;
    s.head_dim = 8;
    s.n_rot = 4;
    s.rope_base = 1e7;
    s.attn_norm_eps = 1e-6;
    s.rms_norm_eps = 1e-5;
    s.vocab = 40;
    s.block = 4;
    s.mask_token_id = 39;
    s.sel_cap = 7;
    s.target_layers = {3, 7};
    return s;
}

/// A sequence of `rows` random tokens, hidden rows and cached cells in `dtype`.
inline basalt::dfly2::SequenceContext random_sequence(const basalt::dfly2::DrafterShape& s, int64_t rows, torch::Dtype dtype, int64_t pos0 = 0) {
    basalt::dfly2::SequenceContext seq;
    seq.pos0 = pos0;
    seq.tokens = torch::randint(0, s.vocab - 1, {rows}, torch::kLong);   // the mask token never appears in text
    seq.hidden = torch::randn({rows, s.hidden}, torch::kFloat64).to(dtype);
    for (int64_t l = 0; l < s.n_layers; ++l) {
        seq.k.push_back(torch::randn({rows, s.n_head_kv, s.head_dim}, torch::kFloat64).to(dtype));
        seq.v.push_back(torch::randn({rows, s.n_head_kv, s.head_dim}, torch::kFloat64).to(dtype));
    }
    return seq;
}

/// Blocks at `anchors` whose layers read the cells just before the anchor: every cell in layer 0, every other cell
/// in the layers after it, at most sel_cap of them, padded with -1 to the widest row.
inline basalt::dfly2::BlockBatch blocks_at(const basalt::dfly2::DrafterShape& s, const std::vector<int64_t>& anchors) {
    basalt::dfly2::BlockBatch b;
    b.anchors = torch::tensor(anchors, torch::kLong);
    for (int64_t l = 0; l < s.n_layers; ++l) {
        std::vector<std::vector<int64_t>> rows;
        size_t width = 0;
        for (const int64_t a : anchors) {
            std::vector<int64_t> ids;
            for (int64_t c = a - 1; c >= 0 && (int64_t) ids.size() < s.sel_cap; c -= (l == 0 ? 1 : 2)) ids.insert(ids.begin(), c);
            width = std::max(width, ids.size());
            rows.push_back(std::move(ids));
        }
        torch::Tensor sel = torch::full({(int64_t) anchors.size(), (int64_t) width}, -1, torch::kLong);
        for (size_t n = 0; n < rows.size(); ++n)
            for (size_t j = 0; j < rows[n].size(); ++j) sel[(int64_t) n][(int64_t) j] = rows[n][j];
        b.sel.push_back(sel);
    }
    return b;
}

/// The largest absolute difference of two tensors, in FP64.
inline double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
    if (!(a.sizes() == b.sizes())) return 1e300;
    if (a.numel() == 0) return 0.0;
    return (a.to(torch::kFloat64) - b.to(torch::kFloat64)).abs().max().item<double>();
}

}  // namespace dfly2_test
