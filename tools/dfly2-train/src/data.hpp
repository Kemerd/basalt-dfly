// tools/dfly2-train/src/data.hpp - the trainer's teacher data: the shards of a directory, the order sequences and
// their blocks are taken in, and a thread that reads sequences ahead of the training loop.
//
// The order is a function of the seed alone: epoch e takes the sequences in a shuffle seeded by (seed, e), and a
// sequence's blocks in a shuffle seeded by (seed, e, sequence). A place in a run is therefore three numbers (epoch,
// position in the epoch's order, blocks already taken from that sequence), which is what a checkpoint stores.
//
// A block is trained only when it lies inside its sequence (anchor + block <= rows), so every drafted row has a
// corpus token and a teacher row and the loss needs no mask.
//
// The export reads a sequence's first rows by the target's batched prompt path and its last `window_rows` (its
// export.json) through verify windows, as served rows are read. An anchor whose row before is a prompt-path row (a
// prefix anchor) carries that path's hidden row and selection, which serving never hands a drafter; its blocks weigh
// `prefix_weight` in the loss, and none is taken at weight 0.
//
// A share `kv_int8_share` of the blocks is read as a target serving from an INT8 cache would hand it to the drafter
// (contract item 14 of the export); which ones is a function of the run's seed, the epoch, the sequence and the
// anchor.
#pragma once

#include "target_weights.hpp"

#include "basalt/artifact/dfly2_shard.hpp"

#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

namespace basalt::dfly2 {

/// How a shard set's blocks are taken.
struct DataOptions {
    double prefix_weight = 1.0;   ///< loss weight of a prefix anchor's block, in [0, 1]; 0 takes none
    double kv_int8_share = 0.0;   ///< share of the blocks, in [0, 1], that read the cache as INT8 serving does
};

/// A sequence of the set that has at least one trainable block.
struct SequenceRef {
    uint32_t shard = 0;       ///< which file
    uint64_t index = 0;       ///< which sequence of the file
    uint64_t tokens = 0;      ///< its rows
    uint64_t blocks = 0;      ///< its anchors whose block lies inside it and that are taken
    uint64_t window_from = 0; ///< its first row read through a verify window
};

/// One sequence read from a shard, on the CPU, with the blocks still to take from it.
struct LoadedSequence {
    SequenceContext ctx;                 ///< tokens int64; hidden in the shard's type; cells FP16
    torch::Tensor anchors;               ///< [n] int64, in the order the blocks are taken
    torch::Tensor selection;             ///< [n][layers][sel_cap] int32, -1 padded
    torch::Tensor sel_count;             ///< [n][layers] int32: ids before the padding
    torch::Tensor weight;                ///< [n] float32: each block's loss weight
    torch::Tensor kv_int8;               ///< [n] bool: blocks read as from an INT8 cache; undefined at a share of 0
    int64_t epoch = 0;
    uint64_t position = 0;               ///< place in the epoch's order
    uint64_t first = 0;                  ///< blocks of the sequence that came before anchors[0]
    uint64_t bytes = 0;                  ///< host memory the tensors hold
};

/// The `n` numbers 0..n-1 in an order that depends on `seed` alone (the same on every platform).
std::vector<uint64_t> shuffled_order(uint64_t n, uint64_t seed);

/// The seed of epoch `epoch`'s sequence order, and of sequence `sequence`'s block order in that epoch.
uint64_t epoch_seed(uint64_t seed, int64_t epoch);
uint64_t sequence_seed(uint64_t seed, int64_t epoch, uint64_t sequence);

/// The shards of one directory, checked against the drafter they are for.
class ShardSet {
public:
    /// Opens every file of `dir` whose name ends with kShardExtension and counts its trainable blocks, with the
    /// window rows of the export.json beside them. Each must carry `shape`'s geometry and paired layers and name the
    /// pack `pack`. False with `err` when the directory has no usable shard, no export.json, or a shard does not fit.
    bool open(const std::string& dir, const DrafterShape& shape, const PackIdentity& pack, const DataOptions& options,
              std::string& err);

    const std::vector<SequenceRef>& sequences() const { return refs_; }
    uint64_t blocks() const { return blocks_; }
    uint64_t prefix_blocks() const { return prefix_blocks_; }   ///< of blocks(): those after a prompt-path row
    int64_t window_rows() const { return window_rows_; }
    uint64_t tokens() const { return tokens_; }
    size_t shards() const { return paths_.size(); }

    /// Reads sequence `ref` with its blocks in the order `seed` gives, from block `first` of that order on.
    /// One thread at a time.
    bool load(size_t ref, uint64_t seed, uint64_t first, LoadedSequence& out, std::string& err);

private:
    /// The reader of shard `shard`, opened on demand; the least recently used one is closed past kOpenShards.
    ShardReader* reader(uint32_t shard, std::string& err);

    static constexpr size_t kOpenShards = 32;
    DrafterShape shape_;
    DataOptions options_;
    int64_t window_rows_ = 0;
    std::vector<std::string> paths_;
    std::vector<SequenceRef> refs_;
    uint64_t blocks_ = 0, tokens_ = 0, prefix_blocks_ = 0;
    std::list<std::pair<uint32_t, std::unique_ptr<ShardReader>>> open_;   ///< most recently used first
};

/// Reads the sequences of a ShardSet in training order on its own thread, into a queue bounded by the host memory
/// the waiting sequences hold, so the training loop never waits on a disk it could have read from earlier.
class SequenceLoader {
public:
    /// Starts at (`epoch`, `position`, `first`) and stops before epoch `end_epoch` (never, when negative).
    /// The queue takes another sequence while it holds less than `queue_bytes` (always at least one).
    SequenceLoader(ShardSet& set, uint64_t seed, int64_t epoch, uint64_t position, uint64_t first, int64_t end_epoch, uint64_t queue_bytes);
    SequenceLoader(const SequenceLoader&) = delete;
    SequenceLoader& operator=(const SequenceLoader&) = delete;
    /// Stops the thread; what it had read ahead is dropped.
    ~SequenceLoader();

    /// The next sequence, waiting for the thread when the queue is empty. False at the end of the order (`err`
    /// empty) or when a read failed (`err` says why).
    bool next(LoadedSequence& out, std::string& err);

private:
    void run(int64_t epoch, uint64_t position, uint64_t first, int64_t end_epoch);

    ShardSet& set_;
    uint64_t seed_, queue_bytes_;
    std::mutex mutex_;
    std::condition_variable can_push_, can_pop_;
    std::deque<LoadedSequence> queue_;
    uint64_t queued_bytes_ = 0;
    bool stop_ = false, done_ = false;
    std::string error_;
    std::thread thread_;
};

}  // namespace basalt::dfly2
