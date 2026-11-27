// tools/dfly2-train/src/sparse_attention.hpp - the drafter's attention over the target's cached cells and a block's own
// rows, as one autograd function that keeps nothing of the gathered cells for its backward.
//
// Every row of a block reads the same selected cells of a frozen cache plus the block's rows up to itself. A plain
// graph would hold the gathered keys and values and the score-sized tensors of every layer until the backward; this
// function saves only its inputs (the queries, the block's own keys and values, the cell ids and the cache it was
// handed) and gathers and scores again when the gradient arrives. Both passes run a few blocks at a time.
#pragma once

#include <torch/torch.h>

namespace basalt::dfly2 {

/// Blocks scored at once by the attention when the caller names no count.
constexpr int64_t kAttentionChunkBlocks = 64;

/// Grouped-query attention of N blocks of B rows.
///
///   q      [N][B][n_head][head_dim]      queries, normed and rotated
///   k_blk  [N][B][n_head_kv][head_dim]   the block rows' own keys, normed and rotated
///   v_blk  [N][B][n_head_kv][head_dim]   the block rows' own values
///   k_ctx  [L][n_head_kv][head_dim]      the cache's keys, frozen (any float type)
///   v_ctx  [L][n_head_kv][head_dim]      the cache's values, frozen
///   idx    [N][W] int64                  each block's selected cells, -1 for none
///
/// Row r of a block attends its block's cells of `idx` and the block's rows 0..r. Query head h reads key/value head
/// h / (n_head / n_head_kv); a score is q . k / sqrt(head_dim); one softmax runs over both sets. Computed in FP32
/// (FP64 when `q` is), returned as [N][B][n_head][head_dim] in that type. Gradients reach q, k_blk and v_blk only.
/// Throws std::invalid_argument when the shapes do not fit.
torch::Tensor sparse_context_attention(const torch::Tensor& q, const torch::Tensor& k_blk, const torch::Tensor& v_blk,
                                       const torch::Tensor& k_ctx, const torch::Tensor& v_ctx, const torch::Tensor& idx,
                                       int64_t chunk_blocks = kAttentionChunkBlocks);

}  // namespace basalt::dfly2
