// src/models/qwen4exp/load/dfly2_shard_queue.hpp - the teacher export's writer (basalt/artifact/dfly2_shard.hpp): the
// compute thread hands over sequences and their pieces in row order, one thread repacks and writes them. The piece
// buffers are the only bound: a compute thread that outruns the disk waits for one. And the pass that reads an
// export's shards back. Host only.
#pragma once

#include "basalt/artifact/dfly2_shard.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::qwen4exp::generate {

/// The export's writer: sequences and their pieces handed over in row order, repacked and written on one thread.
class ShardQueue {
public:
    /// Where the piece buffers come from: the export passes page-locked device-host memory, a test plain memory.
    struct Memory {
        void* (*take)(size_t bytes) = nullptr;   ///< nullptr when there is none
        void (*give)(void* p) = nullptr;
    };
    /// Rows and anchors of one piece, the cells of a cache page, pieces in flight, and the tokens after which a shard
    /// is closed.
    struct Limits {
        int64_t rows = 0, anchors = 0, page = 0;
        int pieces = 0;
        uint64_t shard_tokens = 0;
    };
    /// Rows [row0, row0 + rows) of the open sequence, as the target left them.
    struct Piece {
        int64_t row0 = 0, rows = 0;
        int64_t page0 = 0, pages = 0;              ///< the cache pages `k` and `v` hold: every one a row of the piece is in
        uint16_t* hidden = nullptr;                ///< [rows][hidden] f16
        uint16_t* k[dfly2::kShardMaxLayers] = {};  ///< per paired layer [pages][n_head_kv][page][head_dim] f16, as cached
        uint16_t* v[dfly2::kShardMaxLayers] = {};
        int n_sel = 0;                             ///< anchors whose row before lies in the piece
        int32_t* sel_anchor = nullptr;             ///< [n_sel]: each one's index in the sequence's anchors
        int32_t* sel = nullptr;                    ///< [n_sel][layers][sel_cap]: its first min(anchor, sel_cap) ids are read
    };

    ShardQueue() = default;
    ShardQueue(const ShardQueue&) = delete;
    ShardQueue& operator=(const ShardQueue&) = delete;
    ~ShardQueue();

    /// Opens `dir`: counts the sequences of its closed shards (a restart continues after them) and refuses a shard
    /// whose geometry or pack identity differs from `geometry`. An unfinished shard is deleted.
    bool open(const std::string& dir, const dfly2::ShardHeader& geometry, const Limits& limits, const Memory& memory,
              std::string& err);
    /// Sequences the closed shards held when `open` read them.
    uint64_t sequences_done() const { return done_; }
    /// The most cache pages a piece holds per paired layer.
    int64_t piece_pages() const { return pages_cap_; }

    /// A free piece; blocks while all are in flight. nullptr once the writer has failed or before `open`.
    Piece* take();
    /// Starts a sequence with its tokens and its anchors (ascending).
    bool begin(std::vector<int32_t> tokens, std::vector<int32_t> anchors, std::string& err);
    /// Queues a piece from `take`; the writer gives it back when it is written. Pieces arrive in row order.
    bool put(Piece* p, std::string& err);
    /// Gives a piece from `take` back unwritten.
    void give_back(Piece* p);
    /// Ends the sequence: every row, cell and selection must have been put.
    bool end(std::string& err);
    /// Drains the queue, closes the last shard and joins the writer. False with the first failure.
    bool close(std::string& err);

    /// A shard's file name: shard-NNNNN and the format's extension.
    static std::string shard_name(uint64_t index);
    /// Paired layer `l`'s cells of the piece's own rows as a shard holds them, a cell's heads together
    /// ([rows][heads][dim]), from the piece's cache pages of `page` cells.
    static void piece_cells(const Piece& p, uint32_t l, uint64_t page, uint64_t heads, uint64_t dim, uint16_t* k_cells,
                            uint16_t* v_cells);

private:
    struct Item {
        enum Kind { Begin, Put, End, Stop } kind = Stop;
        std::vector<int32_t> tokens, anchors;
        Piece* piece = nullptr;
    };
    void run();
    bool push(Item item, std::string& err);
    bool write(Item& item, std::string& err);
    bool write_piece(const Piece& p, std::string& err);
    bool finish_shard(std::string& err);
    void fail(const std::string& why);
    void release();

    std::string dir_;
    dfly2::ShardHeader geometry_{};
    Limits limits_;
    Memory memory_;
    uint64_t done_ = 0;
    int64_t pages_cap_ = 0;
    std::vector<Piece> pieces_;
    std::vector<void*> blocks_;                    ///< each piece's memory, from `memory_`

    std::mutex mu_;
    std::condition_variable cv_items_, cv_free_;
    std::deque<Item> items_;
    std::vector<Piece*> free_;
    bool failed_ = false;
    std::string error_;
    std::thread writer_thread_;

    // the writer thread's own (and close()'s once it has joined)
    dfly2::ShardWriter writer_;
    bool shard_open_ = false, in_sequence_ = false;
    uint64_t shard_index_ = 0, shard_tokens_ = 0;
    std::vector<int32_t> anchors_;
    std::vector<uint16_t> k_cells_, v_cells_;      ///< a piece's cells of one layer, cell-major
};

/// The export's settings file in `dir`: written when absent, else compared key by key; false names the key that differs.
/// A directory that holds a closed shard but no settings file is refused.
bool manifest_check(const std::string& dir, const std::map<std::string, std::string>& settings, std::string& err);

/// Whether `ids` (`count` cells) can be what the target's index selects for a row that reads among `n_kv` cells,
/// when `block` cells share an index key and at most `cap` are selected: every cell while they fit. Past that,
/// `cap` ascending cells, of each block its lowest ones, the unfinished last block whole and at most one other
/// block in part. The scores decide which blocks, so they are not needed here.
bool selection_form(const int32_t* ids, uint64_t count, int64_t n_kv, int64_t block, int64_t cap);

/// What reading an export's shards back found.
struct ReadBack {
    uint64_t shards = 0, sequences = 0, tokens = 0, anchors = 0;
    uint64_t selections = 0, off_form = 0;   ///< selections read, and those `selection_form` refuses
    std::string first_off_form;              ///< where the first of those is
};
/// Sequence `seq` as an export hands it to the writer: its tokens and its anchors. False with `err`: it has none.
using SequenceSource = std::function<bool(uint64_t seq, std::vector<int32_t>& tokens, std::vector<int32_t>& anchors,
                                          std::string& err)>;
/// A sequence that read back with its tokens and anchors.
using SequenceSink = std::function<void(uint64_t seq, const dfly2::SequenceData& data)>;

/// Reads the closed shards of `dir` back through the format's reader, in the export's order: every shard is of
/// `geometry`, they hold `sequences` sequences, each starts at position 0 with the tokens and anchors `expected`
/// gives for it, and every array passes the reader's checks. Selections that do not have the form of the
/// target's index (`index_block` cells a key) are counted, not refused. `seen`, when given, gets each sequence.
/// False with `err`, which names the shard and the sequence.
bool shards_read_back(const std::string& dir, const dfly2::ShardHeader& geometry, int64_t index_block, uint64_t sequences,
                      const SequenceSource& expected, const SequenceSink& seen, ReadBack& found, std::string& err);

}  // namespace basalt::models::qwen4exp::generate
