// tools/dfly2-train/src/loss.hpp - the drafter's training loss and its offline accepted-length metrics.
//
// A block's rows 1..B-1 each draft one token; row k's teacher is the target's own next-token distribution at that
// position, computed here from the exported head input and the served head. The loss is
//
//   L = sum(w * (KL_top + c * BCE)) / max(sum(w), 1e-6)  +  e * mean over blocks(1 - (1/(B-1)) * sum_k prod_{j<=k} a_j)
//
//   KL_top  sum over the teacher's `kl_topk` likeliest tokens of p * (log p - log q), both softmaxes over the whole
//           vocabulary
//   a_k     sum_v min(p_v, q_v): the chance a sampling verifier accepts draft k
//   BCE     the confidence head's logit against a_k (a_k held constant)
//   w_k     D-PACE: with s_j = (1 - alpha) * q_j[corpus token] + alpha, w_k = sum_{m>=k} prod_{j<=m} s_j, held constant
//
// Logits of every row of every block do not fit a card at once, so the loss runs a few whole blocks at a time
// (the weights and the accepted-length product couple the rows of one block only) and backpropagates each chunk to
// the backbone's output and to the parameters after it. The sum(w) denominator is only known at the end, so the
// chunks' gradients are kept in two sums, one per denominator, and joined once it is.
#pragma once

#include "drafter.hpp"

#include <vector>

namespace basalt::dfly2 {

struct LossConfig {
    int64_t kl_topk = 10;             ///< teacher tokens the KL term sums over
    double dpace_alpha = 0.5;         ///< smoothing of the D-PACE confidences
    double e2e_weight = 1.0;          ///< weight of the accepted-length term
    double confidence_weight = 1.0;   ///< weight of the confidence head's term
    int64_t chunk_blocks = 32;        ///< blocks whose logits are alive at once
};

/// What one call measured. Sums over blocks unless said otherwise, so calls add up.
struct LossTerms {
    double loss = 0;                  ///< the loss of the call's blocks
    double kl = 0, e2e = 0, confidence = 0;   ///< its three terms, before their weights
    double confidence_mae = 0;        ///< D-PACE-weighted mean |sigmoid(confidence) - a|
    double blocks = 0;                ///< blocks in the call
    double tv_accept = 0;             ///< sum over blocks of sum_k prod_{j<=k} a_j
    std::vector<double> hits_corpus;  ///< per drafted row: blocks whose teacher-forced draft is the corpus token
    std::vector<double> hits_teacher; ///< per drafted row: blocks whose teacher-forced draft is the teacher's pick
};

/// D-PACE weights of rows whose drafts hit the corpus token with probability `confidences` ([..][rows]):
/// w_k = sum_{m>=k} prod_{j<=m} ((1 - alpha) * c_j + alpha). No gradient.
torch::Tensor dpace_weights(const torch::Tensor& confidences, double alpha);

/// Per block, 1 - (1/rows) * sum_k prod_{j<=k} accept[k], for `accept` [blocks][rows].
torch::Tensor e2e_tv_per_block(const torch::Tensor& accept);

/// sum_k prod_{j<=k} (hits[j] / blocks): the accepted drafts per block of a verifier that accepts row k as often as
/// `hits` says, independently per row.
double simulated_accept_length(const std::vector<double>& hits, double blocks);

/// The loss of the blocks at `anchors` given the backbone's output `z` ([N][block][hidden], used detached).
///
/// With `grad_z` set, the gradient times `scale` is left in two places: `*grad_z` receives d(loss)/d(z) for the
/// caller to send back through the backbone, and the gradients of the correction's and the confidence head's
/// parameters are added to their .grad. With `grad_z` null nothing is differentiated.
/// `embedding` and `head` are the target's [vocab][hidden] tables; every anchor's block must lie inside the sequence.
LossTerms chunked_loss(Drafter& drafter, const torch::Tensor& z, const torch::Tensor& embedding, const torch::Tensor& head,
                       const SequenceContext& seq, const torch::Tensor& anchors, const LossConfig& cfg, double scale,
                       torch::Tensor* grad_z);

/// The serving chain's accepted drafts, summed over the blocks: each row's Hidden-Correction reads the token the row
/// before it drafted (the anchor for row 1), and a block scores the rows before its first draft that is not the
/// teacher's pick. No gradient.
double served_chain_accept(Drafter& drafter, const torch::Tensor& z, const torch::Tensor& embedding, const torch::Tensor& head,
                           const SequenceContext& seq, const torch::Tensor& anchors, int64_t chunk_blocks);

}  // namespace basalt::dfly2
