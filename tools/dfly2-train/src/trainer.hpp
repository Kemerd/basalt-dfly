// tools/dfly2-train/src/trainer.hpp - what basalt-dfly2-train's commands do: make a drafter for a target, train
// it on teacher shards, measure a trained one, and write the blocks a runtime's forward is checked against.
#pragma once

#include "loss.hpp"
#include "optimizer.hpp"
#include "sparse_attention.hpp"

#include <functional>
#include <string>
#include <vector>

namespace basalt::dfly2 {

/// `init`: a new drafter for the target in a pack.
struct InitOptions {
    std::string model;                 ///< the target's .basalt pack
    std::string out;                   ///< the drafter file to write
    std::vector<int> layers;           ///< target layers to pair with; empty: the last `draft_layers` sparse ones
    int64_t draft_layers = 5;
    int64_t block = 8;                 ///< rows of a block: the anchor and block - 1 drafts
    int64_t mask_token_id = 248077;
    double rms_norm_eps = 1e-5;        ///< eps of the drafter's own norms
    uint64_t seed = 42;                ///< of the random initialisation
};

/// Writes the drafter: attention copied from the paired target layers, the rest at its initialisation.
bool init_drafter(const InitOptions& o, std::string& err);

/// What `train` and `eval` share.
struct RunOptions {
    std::string model;                 ///< the target's .basalt pack
    std::string shards;                ///< directory of teacher shards
    std::string device = "cuda";       ///< cpu, cuda or cuda:N
    std::string head_dtype = "f32";    ///< the frozen output head on the device: f32 or bf16
    int64_t blocks = 256;              ///< blocks of one forward and backward; the loss's row weights are normalised over them
    int64_t attention_blocks = kAttentionChunkBlocks;   ///< blocks whose attention scores are alive at once
    uint64_t queue_bytes = 4ull << 30; ///< host memory of the sequences read ahead
    uint64_t seed = 42;                ///< of the data order
    double prefix_weight = 1.0;        ///< loss weight of a block whose anchor follows a prompt-path row; 0 takes none
    double kv_int8_share = 0.0;        ///< share of the blocks that read the cache as a target serving from INT8 does
    LossConfig loss;                   ///< its chunk_blocks is the blocks whose logits are alive at once
};

struct TrainOptions : RunOptions {
    std::string run;                   ///< the run directory: checkpoints and train.log
    std::string init;                  ///< the drafter file a new run starts from
    std::string dtype = "bf16";        ///< the drafter's weights in the forward: bf16 or f32 (masters are FP32)
    int64_t steps = 0;                 ///< optimiser steps of the run; 0: from `epochs`
    double epochs = 1.0;
    int64_t blocks_per_step = 8192;
    int64_t save_every = 250;          ///< steps between checkpoints
    int64_t keep = 3;                  ///< checkpoints kept
    int64_t log_every = 10;            ///< steps between progress lines
    int64_t muon_head_rows = -1;       ///< rows of a query projection orthogonalised together; -1: head_dim, 0: all
    OptimConfig optim;                 ///< total_steps and query_head_rows are set by the run
};

struct TrainResult {
    int64_t steps = 0;                 ///< optimiser steps behind the last checkpoint
    int64_t total_steps = 0;
    double first_loss = 0, last_loss = 0;   ///< mean loss of the first and the last step this call took
    bool stopped = false;              ///< ended by `stop` before the last step
    std::string checkpoint;            ///< the newest drafter file of the run
};

/// Trains until the run's last step or until `stop` returns true (asked between forwards). A run directory that
/// already holds a checkpoint is resumed from its newest one. Every completed step is in a checkpoint on return.
bool train(const TrainOptions& o, const std::function<bool()>& stop, TrainResult& result, std::string& err);

struct EvalOptions : RunOptions {
    std::string checkpoint;            ///< the drafter file to measure
    int64_t max_blocks = 0;            ///< stop after this many blocks; 0: every block
};

struct EvalResult {
    int64_t blocks = 0, drafts = 0;    ///< blocks measured, and the rows each drafts (block - 1)
    int64_t step = 0;                  ///< optimiser steps behind the drafter
    double loss = 0, kl = 0, e2e = 0, confidence = 0, confidence_mae = 0;
    std::vector<double> accuracy_corpus;    ///< per drafted row, teacher-forced: share equal to the corpus token
    std::vector<double> accuracy_teacher;   ///< per drafted row, teacher-forced: share equal to the teacher's pick
    double accept_corpus = 0;          ///< sum over rows of the product of accuracy_corpus up to the row
    double accept_teacher = 0;         ///< the same against the teacher's pick: greedy verification, teacher-forced
    double accept_sampling = 0;        ///< expected accepted drafts per block of a sampling verifier, teacher-forced
    double accept_served = 0;          ///< accepted drafts per block of the serving chain against the teacher's pick
};

/// Measures a drafter over the shards without changing it.
bool evaluate(const EvalOptions& o, EvalResult& result, std::string& err);

struct FixtureOptions : RunOptions {
    std::string checkpoint;            ///< the drafter file whose forward is written
    std::string out;                   ///< the safetensors file to write
    int64_t count = 1;                 ///< blocks to write
};

/// `fixture`: blocks of the shards, in eval's order, as one safetensors file a runtime's block pass is checked
/// against. Per block i, under "block.<i>.": what the pass reads (`anchor` the anchor's token, `pos` its position,
/// `h_s` the head input of the row before it, `emb_anchor`, `emb_mask`, `emb_prev` the embeddings of each drafted
/// row's previous token, and per draft layer l `layer.<l>.sel`, `.k_ctx`, `.v_ctx`: the selected cells' ids and
/// their cached keys and values in that order) and what the drafter computed from it (`z` after the final norm,
/// `h_corrected`, `conf_logit` with the bias, and `draft_ids`, each drafted row reading the draft before it).
bool write_fixture(const FixtureOptions& o, std::string& err);

}  // namespace basalt::dfly2
