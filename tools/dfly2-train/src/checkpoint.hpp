// tools/dfly2-train/src/checkpoint.hpp - the files of a training run.
//
// A run directory holds, per saved step N (nine digits):
//   step-N.safetensors          the drafter, in the checkpoint contract (basalt/artifact/dfly2_checkpoint.hpp): what
//                               a runtime or a converter reads, and what `eval` and a new run's --init take
//   step-N.state.safetensors    what else a resumed run needs: the optimiser's momentum and moment buffers, its
//                               counters, and the run's place in the data
// The state is written first and the drafter last, each through a temporary file, so a step whose drafter file exists
// is whole.
#pragma once

#include "target_weights.hpp"

#include <map>
#include <string>
#include <vector>

namespace basalt::dfly2 {

/// A drafter file: its shape, its metadata and its tensors.
struct DrafterFile {
    DrafterShape shape;
    std::string weight_dtype = "F32";              ///< BF16 or F32: what the trained forward rounded to
    int64_t step = 0;                              ///< optimiser steps behind the weights
    std::string pack_name;                         ///< the pack the drafter was built on
    uint64_t pack_bytes = 0;
    std::string pack_meta_sha256;                  ///< 64 hex digits
    std::map<std::string, torch::Tensor> tensors;  ///< FP32 on the CPU, exactly drafter_tensors(shape)
};

/// Writes `file` to `path`; false with `err` when a tensor is missing, misshapen or not FP32.
bool write_drafter_file(const std::string& path, const DrafterFile& file, std::string& err);
/// Reads and checks a drafter file: the metadata, and every tensor of its shape present with the right shape.
bool read_drafter_file(const std::string& path, DrafterFile& out, std::string& err);
/// Whether `file` was built on `pack`; false with the reason in `err`.
bool drafter_file_matches(const DrafterFile& file, const PackIdentity& pack, std::string& err);

/// What a resumed run needs beside the drafter file.
struct TrainState {
    int64_t step = 0, adam_steps = 0;              ///< the optimiser's counters
    int64_t epoch = 0;                             ///< the place in the data the next block comes from
    uint64_t position = 0, first = 0;
    double elapsed = 0;                            ///< seconds trained so far
    uint64_t seed = 0;                             ///< settings a resumed run must repeat
    int64_t blocks_per_step = 0, total_steps = 0;
    uint64_t dataset_blocks = 0, dataset_sequences = 0;
    double prefix_weight = 1.0;
    std::map<std::string, torch::Tensor> buffers;  ///< FP32 on the CPU, by the optimiser's names
};

bool write_train_state(const std::string& path, const TrainState& state, std::string& err);
bool read_train_state(const std::string& path, TrainState& out, std::string& err);

/// The two files of step `step` in run directory `run`.
std::string drafter_path(const std::string& run, int64_t step);
std::string state_path(const std::string& run, int64_t step);
/// The steps of `run` that have both files, ascending. False when the directory cannot be read.
bool list_checkpoints(const std::string& run, std::vector<int64_t>& steps, std::string& err);
/// Removes all but the newest `keep` checkpoints of `run`.
void prune_checkpoints(const std::string& run, int64_t keep);

}  // namespace basalt::dfly2
