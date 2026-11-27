// include/basalt/artifact/dfly2_shard.hpp - the teacher data a DFly2 drafter is trained on: what the target computed
// for a sequence, written by the engine's export and read by the trainer (tools/dfly2-train). Header-only, so the
// exporter and the trainer share one definition and one writer; it needs the platform file calls and nothing else.
//
// A shard is one file of whole sequences. A sequence is one document: its rows are never packed with another's, so a
// cell's rotary position is `pos0` plus its index.
//
// Layout (little-endian; every array starts at a multiple of kShardAlign from the start of the file, the bytes
// between arrays are zero):
//   ShardHeader                       geometry, the pack the data came from, and the index's place (written last)
//   n_sequences x sequence:
//     SequenceHeader                  n_tokens, n_anchors, pos0, bytes to the next sequence
//     tokens      int32   [n_tokens]                        token ids, each below `vocab`
//     hidden      hidden_dtype [n_tokens][hidden]           the vector that enters the output head's product for the
//                                                           row (after the output mix), so head(hidden[t]) scores
//                                                           token t + 1
//     per paired layer, in `layers` order:
//       K         f16     [n_tokens][n_head_kv][head_dim]   the layer's cached key cells: normed and rotated, as the
//                                                           attention of a served FP16 cache reads them
//       V         f16     [n_tokens][n_head_kv][head_dim]   its cached value cells
//     anchors     int32   [n_anchors]                       strictly ascending positions the exporter sampled
//     selection   int32   [n_anchors][n_layers][sel_cap]    the cells the layer's attention selected for the row
//                                                           before the anchor (row anchor - 1): ascending ids below
//                                                           the anchor, then -1 to the end. Anchor 0 has none.
//   index         uint64  [n_sequences]                     file offset of each SequenceHeader
//
// A writer that stops before close() leaves n_sequences and index_offset zero, which a reader refuses.
#pragma once

#include "basalt/platform/file.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace basalt::dfly2 {

static_assert(std::endian::native == std::endian::little, "the shard layout is little-endian");

constexpr char kShardMagic[8] = {'B', 'S', 'L', 'T', 'D', 'F', '2', 'S'};
constexpr uint32_t kShardVersion = 1;
constexpr const char* kShardExtension = ".dfly2";   ///< a shard's file name ends with this
constexpr uint32_t kShardMaxLayers = 16;       ///< paired layers a shard can carry
constexpr uint64_t kShardAlign = 64;           ///< every array's file offset is a multiple of this
constexpr uint32_t kSequenceMagic = 0x51455332;   ///< "2SEQ"
constexpr uint64_t kShardMaxTokens = 1ull << 30;  ///< rows of one sequence (ids and positions are int32)

/// Element type of an array whose precision the exporter chooses.
enum class ShardDtype : uint32_t { f16 = 1, f32 = 2 };

/// Bytes of one element; 0 for an unknown code.
inline uint64_t shard_dtype_bytes(uint32_t dtype) {
    return dtype == (uint32_t) ShardDtype::f16 ? 2 : dtype == (uint32_t) ShardDtype::f32 ? 4 : 0;
}

/// The first record of a shard. `n_sequences`, `index_offset` and `file_bytes` are written by ShardWriter::close.
struct ShardHeader {
    char magic[8];                       ///< kShardMagic
    uint32_t version;                    ///< kShardVersion
    uint32_t header_bytes;               ///< sizeof(ShardHeader)
    uint32_t hidden;                     ///< width of a hidden row and of the model's residual
    uint32_t vocab;                      ///< token ids are below this
    uint32_t n_head_kv;                  ///< key/value heads of a cached cell
    uint32_t head_dim;                   ///< values per head
    uint32_t n_layers;                   ///< paired target layers (1..kShardMaxLayers)
    uint32_t sel_cap;                    ///< widest selection of a row
    uint32_t n_rot;                      ///< rotated values at the start of each head (pairs d, d + n_rot / 2)
    uint32_t hidden_dtype;               ///< ShardDtype of `hidden`
    uint32_t kv_dtype;                   ///< ShardDtype of K and V: f16, what a served FP16 cache holds
    uint32_t reserved0;
    double rope_base;                    ///< frequency base of the rotation the cached keys carry
    uint32_t layers[kShardMaxLayers];    ///< the target layer of each pair, ascending
    uint64_t pack_bytes;                 ///< size of the pack the target ran from
    uint8_t pack_meta_sha256[32];        ///< the digest the pack's section table records for its `meta` section
    char pack_name[128];                 ///< the pack's file name, UTF-8, NUL-padded
    uint64_t n_sequences;
    uint64_t index_offset;               ///< where the index starts
    uint64_t file_bytes;                 ///< the whole file, for a truncation check
    uint8_t reserved[192];
};
static_assert(sizeof(ShardHeader) == 512);

/// The first record of a sequence.
struct SequenceHeader {
    uint32_t magic;                      ///< kSequenceMagic
    uint32_t reserved0;
    uint64_t n_tokens;
    uint64_t n_anchors;
    int64_t pos0;                        ///< rotary position of row 0
    uint64_t bytes;                      ///< this header and the sequence's arrays, a multiple of kShardAlign
    uint8_t reserved[24];
};
static_assert(sizeof(SequenceHeader) == 64);

/// Where a sequence's arrays start, relative to its header, and where the sequence ends.
struct SequenceLayout {
    uint64_t tokens = 0, hidden = 0, anchors = 0, selection = 0, bytes = 0;
    uint64_t k[kShardMaxLayers] = {}, v[kShardMaxLayers] = {};
    uint64_t hidden_row_bytes = 0;       ///< bytes of one hidden row
    uint64_t cell_bytes = 0;             ///< bytes of one cell's K (or V) over every head
};

namespace detail {

/// a * b, false on overflow.
inline bool mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (b != 0 && a > (std::numeric_limits<uint64_t>::max)() / b) return false;
    out = a * b;
    return true;
}

/// `at` rounded up to the array alignment.
inline uint64_t align_up(uint64_t at) { return (at + kShardAlign - 1) / kShardAlign * kShardAlign; }

/// Places an array of `bytes` at the next aligned offset; false on overflow.
inline bool place(uint64_t& cursor, uint64_t bytes, uint64_t& at) {
    at = align_up(cursor);
    if (at < cursor || bytes > (std::numeric_limits<uint64_t>::max)() - at) return false;
    cursor = at + bytes;
    return true;
}

}  // namespace detail

/// Checks the geometry fields of a header (not the index); false with the reason in `err`.
inline bool shard_check_geometry(const ShardHeader& h, std::string& err) {
    if (std::memcmp(h.magic, kShardMagic, sizeof h.magic) != 0) { err = "not a DFly2 teacher shard (bad magic)"; return false; }
    if (h.version != kShardVersion) { err = "shard version " + std::to_string(h.version) + ", this build reads " + std::to_string(kShardVersion); return false; }
    if (h.header_bytes != sizeof(ShardHeader)) { err = "shard header size " + std::to_string(h.header_bytes); return false; }
    if (h.hidden == 0 || h.vocab == 0 || h.n_head_kv == 0 || h.head_dim == 0 || h.sel_cap == 0) { err = "shard geometry has a zero dimension"; return false; }
    if (h.hidden > (1u << 20) || h.n_head_kv > 1024 || h.head_dim > (1u << 16) || h.sel_cap > (1u << 24) ||
        h.vocab > (uint32_t) (std::numeric_limits<int32_t>::max)()) {
        err = "shard geometry is out of range";
        return false;
    }
    if (h.n_layers == 0 || h.n_layers > kShardMaxLayers) { err = "shard has " + std::to_string(h.n_layers) + " paired layers (1.." + std::to_string(kShardMaxLayers) + ")"; return false; }
    for (uint32_t i = 1; i < h.n_layers; ++i)
        if (h.layers[i] <= h.layers[i - 1]) { err = "shard paired layers are not ascending"; return false; }
    if (h.n_rot > h.head_dim || h.n_rot % 2 != 0) { err = "shard rotated width " + std::to_string(h.n_rot) + " does not fit a head of " + std::to_string(h.head_dim); return false; }
    if (!(h.rope_base > 1.0)) { err = "shard rope base is not above 1"; return false; }
    if (shard_dtype_bytes(h.hidden_dtype) == 0) { err = "shard hidden dtype " + std::to_string(h.hidden_dtype) + " is unknown"; return false; }
    if (h.kv_dtype != (uint32_t) ShardDtype::f16) { err = "shard K/V dtype " + std::to_string(h.kv_dtype) + " is not f16"; return false; }
    return true;
}

/// The layout of a sequence of `n_tokens` rows and `n_anchors` anchors under `h` (a checked header). False with
/// `err` when the counts are out of range.
inline bool shard_sequence_layout(const ShardHeader& h, uint64_t n_tokens, uint64_t n_anchors, SequenceLayout& out, std::string& err) {
    if (n_tokens == 0 || n_tokens > kShardMaxTokens) { err = "sequence has " + std::to_string(n_tokens) + " tokens (1.." + std::to_string(kShardMaxTokens) + ")"; return false; }
    if (n_anchors > n_tokens) { err = "sequence has more anchors (" + std::to_string(n_anchors) + ") than tokens"; return false; }
    out = SequenceLayout{};
    out.hidden_row_bytes = (uint64_t) h.hidden * shard_dtype_bytes(h.hidden_dtype);
    out.cell_bytes = (uint64_t) h.n_head_kv * h.head_dim * shard_dtype_bytes(h.kv_dtype);
    uint64_t cursor = sizeof(SequenceHeader), bytes = 0;
    bool ok = detail::place(cursor, n_tokens * sizeof(int32_t), out.tokens);
    ok = ok && detail::mul_u64(n_tokens, out.hidden_row_bytes, bytes) && detail::place(cursor, bytes, out.hidden);
    for (uint32_t l = 0; ok && l < h.n_layers; ++l) {
        ok = detail::mul_u64(n_tokens, out.cell_bytes, bytes) && detail::place(cursor, bytes, out.k[l]) &&
             detail::place(cursor, bytes, out.v[l]);
    }
    ok = ok && detail::place(cursor, n_anchors * sizeof(int32_t), out.anchors);
    uint64_t sel_row = 0;
    ok = ok && detail::mul_u64((uint64_t) h.n_layers * h.sel_cap, sizeof(int32_t), sel_row) &&
         detail::mul_u64(n_anchors, sel_row, bytes) && detail::place(cursor, bytes, out.selection);
    if (!ok) { err = "sequence size overflows"; return false; }
    out.bytes = detail::align_up(cursor);
    return true;
}

/// Checks one row's selection: `count` strictly ascending ids below `anchor`, at most `sel_cap` of them.
inline bool shard_check_selection(const int32_t* ids, uint64_t count, uint32_t sel_cap, int64_t anchor, std::string& err) {
    if (count > sel_cap) { err = "selection of " + std::to_string(count) + " cells is wider than " + std::to_string(sel_cap); return false; }
    if (count != 0 && ids == nullptr) { err = "selection ids are missing"; return false; }
    int64_t prev = -1;
    for (uint64_t i = 0; i < count; ++i) {
        if (ids[i] <= prev || ids[i] >= anchor) { err = "selection id " + std::to_string(ids[i]) + " is not ascending below anchor " + std::to_string(anchor); return false; }
        prev = ids[i];
    }
    return true;
}

/// Checks a sequence's tokens and anchors against the geometry.
inline bool shard_check_rows(const ShardHeader& h, const int32_t* tokens, uint64_t n_tokens, const int32_t* anchors, uint64_t n_anchors, std::string& err) {
    if (tokens == nullptr || (n_anchors != 0 && anchors == nullptr)) { err = "sequence tokens or anchors are missing"; return false; }
    for (uint64_t t = 0; t < n_tokens; ++t)
        if (tokens[t] < 0 || (uint32_t) tokens[t] >= h.vocab) { err = "token " + std::to_string(tokens[t]) + " at row " + std::to_string(t) + " is outside the vocabulary"; return false; }
    int64_t prev = -1;
    for (uint64_t a = 0; a < n_anchors; ++a) {
        if (anchors[a] <= prev || (uint64_t) anchors[a] >= n_tokens) { err = "anchor " + std::to_string(anchors[a]) + " is not ascending inside the sequence"; return false; }
        prev = anchors[a];
    }
    return true;
}

/// Writes a shard. A sequence is opened with its tokens and anchors, filled in any order of rows-ascending pieces
/// (so an exporter can hand over a prompt chunk at a time), and closed once every row, cell and selection is in.
class ShardWriter {
public:
    ShardWriter() = default;
    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;
    ~ShardWriter() { if (file_ != nullptr) std::fclose(file_); }

    /// Creates `path` (UTF-8) with `header`'s geometry and pack identity; its index fields are ignored.
    bool open(const std::string& path, const ShardHeader& header, std::string& err) {
        if (file_ != nullptr) { err = "shard writer is already open"; return false; }
        header_ = header;
        std::memcpy(header_.magic, kShardMagic, sizeof header_.magic);
        header_.version = kShardVersion;
        header_.header_bytes = sizeof(ShardHeader);
        header_.n_sequences = header_.index_offset = header_.file_bytes = 0;
        if (!shard_check_geometry(header_, err)) return false;
        file_ = platform::open_file_utf8(path, "wb");
        if (file_ == nullptr) { err = "cannot create " + path; return false; }
        path_ = path;
        cursor_ = sizeof(ShardHeader);
        index_.clear();
        return write_at(0, &header_, sizeof header_, err);
    }

    /// Starts a sequence: `n_tokens` rows at rotary positions pos0.., and the anchors its selections are for.
    bool begin_sequence(const int32_t* tokens, uint64_t n_tokens, const int32_t* anchors, uint64_t n_anchors, int64_t pos0, std::string& err) {
        if (file_ == nullptr || in_sequence_) { err = "shard writer cannot begin a sequence now"; return false; }
        if (pos0 < 0 || pos0 > (std::numeric_limits<int32_t>::max)() - (int64_t) kShardMaxTokens) { err = "sequence pos0 " + std::to_string(pos0) + " is out of range"; return false; }
        if (!shard_sequence_layout(header_, n_tokens, n_anchors, layout_, err)) return false;
        if (!shard_check_rows(header_, tokens, n_tokens, anchors, n_anchors, err)) return false;
        seq_ = SequenceHeader{};
        seq_.magic = kSequenceMagic;
        seq_.n_tokens = n_tokens;
        seq_.n_anchors = n_anchors;
        seq_.pos0 = pos0;
        seq_.bytes = layout_.bytes;
        seq_at_ = cursor_;
        anchors_.assign(anchors, anchors + n_anchors);
        hidden_rows_ = 0;
        kv_cells_.assign(header_.n_layers, 0);
        selected_.assign((size_t) (n_anchors * header_.n_layers), 0);
        selections_ = 0;
        in_sequence_ = true;
        // the last byte first, so the file has the sequence's size and unwritten padding reads as zero
        const uint8_t zero = 0;
        return write_at(seq_at_ + layout_.bytes - 1, &zero, 1, err) && write_at(seq_at_, &seq_, sizeof seq_, err) &&
               write_at(seq_at_ + layout_.tokens, tokens, n_tokens * sizeof(int32_t), err) &&
               write_at(seq_at_ + layout_.anchors, anchors, n_anchors * sizeof(int32_t), err);
    }

    /// The hidden rows [row0, row0 + rows) in the header's hidden dtype; pieces arrive in row order.
    bool put_hidden(uint64_t row0, uint64_t rows, const void* hidden, std::string& err) {
        if (!in_sequence_ || hidden == nullptr) { err = "shard writer has no open sequence or no hidden rows"; return false; }
        if (row0 != hidden_rows_ || rows > seq_.n_tokens - row0) { err = "hidden rows " + std::to_string(row0) + "+" + std::to_string(rows) + " are out of order or past the sequence"; return false; }
        hidden_rows_ += rows;
        return write_at(seq_at_ + layout_.hidden + row0 * layout_.hidden_row_bytes, hidden, rows * layout_.hidden_row_bytes, err);
    }

    /// Paired layer `layer`'s cells [cell0, cell0 + cells), each [n_head_kv][head_dim] f16; pieces arrive in cell order.
    bool put_kv(uint32_t layer, uint64_t cell0, uint64_t cells, const uint16_t* k, const uint16_t* v, std::string& err) {
        if (!in_sequence_ || k == nullptr || v == nullptr || layer >= header_.n_layers) { err = "shard writer has no open sequence, no cells or no such paired layer"; return false; }
        if (cell0 != kv_cells_[layer] || cells > seq_.n_tokens - cell0) { err = "cells " + std::to_string(cell0) + "+" + std::to_string(cells) + " of paired layer " + std::to_string(layer) + " are out of order or past the sequence"; return false; }
        kv_cells_[layer] += cells;
        return write_at(seq_at_ + layout_.k[layer] + cell0 * layout_.cell_bytes, k, cells * layout_.cell_bytes, err) &&
               write_at(seq_at_ + layout_.v[layer] + cell0 * layout_.cell_bytes, v, cells * layout_.cell_bytes, err);
    }

    /// The `count` cells paired layer `layer` selected for the row before anchor number `anchor_index`.
    bool put_selection(uint64_t anchor_index, uint32_t layer, const int32_t* ids, uint32_t count, std::string& err) {
        if (!in_sequence_ || layer >= header_.n_layers || anchor_index >= seq_.n_anchors) { err = "shard writer has no open sequence, no such anchor or no such paired layer"; return false; }
        const size_t slot = (size_t) (anchor_index * header_.n_layers + layer);
        if (selected_[slot]) { err = "selection of anchor " + std::to_string(anchor_index) + ", paired layer " + std::to_string(layer) + " was already written"; return false; }
        if (!shard_check_selection(ids, count, header_.sel_cap, anchors_[(size_t) anchor_index], err)) return false;
        row_.assign(header_.sel_cap, -1);
        if (count != 0) std::memcpy(row_.data(), ids, (size_t) count * sizeof(int32_t));
        selected_[slot] = 1;
        ++selections_;
        return write_at(seq_at_ + layout_.selection + (uint64_t) slot * header_.sel_cap * sizeof(int32_t), row_.data(), row_.size() * sizeof(int32_t), err);
    }

    /// Closes the sequence; false when a row, a cell or a selection was never written.
    bool end_sequence(std::string& err) {
        if (!in_sequence_) { err = "shard writer has no open sequence"; return false; }
        if (hidden_rows_ != seq_.n_tokens) { err = "sequence is missing hidden rows"; return false; }
        for (uint32_t l = 0; l < header_.n_layers; ++l)
            if (kv_cells_[l] != seq_.n_tokens) { err = "sequence is missing cells of paired layer " + std::to_string(l); return false; }
        if (selections_ != selected_.size()) { err = "sequence is missing selections"; return false; }
        index_.push_back(seq_at_);
        cursor_ = seq_at_ + layout_.bytes;
        in_sequence_ = false;
        return true;
    }

    /// Writes the index and the final header. The file is not a readable shard before this returns true.
    bool close(std::string& err) {
        if (file_ == nullptr || in_sequence_) { err = "shard writer cannot close now"; return false; }
        header_.n_sequences = index_.size();
        header_.index_offset = detail::align_up(cursor_);
        header_.file_bytes = header_.index_offset + index_.size() * sizeof(uint64_t);
        bool ok = write_at(header_.index_offset, index_.data(), index_.size() * sizeof(uint64_t), err) &&
                  write_at(0, &header_, sizeof header_, err);
        if (std::fclose(file_) != 0 && ok) { err = "cannot finish " + path_; ok = false; }
        file_ = nullptr;
        return ok;
    }

    uint64_t sequences() const { return index_.size(); }

private:
    bool write_at(uint64_t off, const void* src, uint64_t n, std::string& err) {
        if (n == 0) return true;
        if (platform::fseek64(file_, (int64_t) off, SEEK_SET) != 0 || std::fwrite(src, 1, (size_t) n, file_) != (size_t) n) {
            err = "cannot write " + path_;
            return false;
        }
        return true;
    }

    std::FILE* file_ = nullptr;
    std::string path_;
    ShardHeader header_{};
    SequenceHeader seq_{};
    SequenceLayout layout_{};
    uint64_t cursor_ = 0, seq_at_ = 0, hidden_rows_ = 0, selections_ = 0;
    bool in_sequence_ = false;
    std::vector<uint64_t> index_, kv_cells_;
    std::vector<int32_t> anchors_, row_;
    std::vector<uint8_t> selected_;
};

/// Where a reader puts a sequence's arrays; each buffer has the size the sequence's layout gives it.
struct SequenceSpans {
    int32_t* tokens = nullptr;                 ///< [n_tokens]
    void* hidden = nullptr;                    ///< [n_tokens][hidden] in the header's hidden dtype
    uint16_t* k[kShardMaxLayers] = {};         ///< per paired layer [n_tokens][n_head_kv][head_dim]
    uint16_t* v[kShardMaxLayers] = {};
    int32_t* anchors = nullptr;                ///< [n_anchors]
    int32_t* selection = nullptr;              ///< [n_anchors][n_layers][sel_cap], -1 padded
    int32_t* sel_count = nullptr;              ///< [n_anchors][n_layers]: ids before the padding (derived)
};

/// A sequence in buffers the reader owns.
struct SequenceData {
    SequenceHeader info{};
    std::vector<int32_t> tokens, anchors, selection, sel_count;
    std::vector<uint8_t> hidden;
    std::vector<std::vector<uint16_t>> k, v;
};

/// Reads a shard. open() checks the header, the index and every sequence header against the file's size; read()
/// checks the arrays it returns (token range, anchor order, selection order and padding).
class ShardReader {
public:
    ShardReader() = default;
    ShardReader(const ShardReader&) = delete;
    ShardReader& operator=(const ShardReader&) = delete;
    ~ShardReader() { close(); }

    bool open(const std::string& path, std::string& err) {
        close();
        uint64_t size = 0;
        if (!platform::file_size(path, size)) { err = "cannot read " + path; return false; }
        if (size < sizeof(ShardHeader)) { err = path + ": too small for a shard header"; return false; }
        file_ = platform::open_file_utf8(path, "rb");
        if (file_ == nullptr) { err = "cannot open " + path; return false; }
        path_ = path;
        if (!read_at(0, &header_, sizeof header_, err) || !check_table(size, err)) {
            err = path + ": " + err;
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (file_ != nullptr) std::fclose(file_);
        file_ = nullptr;
        infos_.clear();
        offsets_.clear();
    }

    const ShardHeader& header() const { return header_; }
    const std::string& path() const { return path_; }
    uint64_t sequences() const { return infos_.size(); }
    /// Sequence `i`'s header, or nullptr.
    const SequenceHeader* info(uint64_t i) const { return i < infos_.size() ? &infos_[(size_t) i] : nullptr; }

    /// Reads only sequence `i`'s anchors ([n_anchors], checked for order and range), for a pass that counts blocks.
    bool read_anchors(uint64_t i, int32_t* anchors, std::string& err) {
        if (file_ == nullptr || i >= infos_.size()) { err = "no such sequence " + std::to_string(i); return false; }
        const SequenceHeader& s = infos_[(size_t) i];
        SequenceLayout lay;
        if (!shard_sequence_layout(header_, s.n_tokens, s.n_anchors, lay, err)) return false;
        if (s.n_anchors == 0) return true;
        if (!read_at(offsets_[(size_t) i] + lay.anchors, anchors, s.n_anchors * sizeof(int32_t), err)) return false;
        int64_t prev = -1;
        for (uint64_t a = 0; a < s.n_anchors; ++a) {
            if (anchors[a] <= prev || (uint64_t) anchors[a] >= s.n_tokens) { err = "anchor " + std::to_string(anchors[a]) + " is not ascending inside the sequence"; return false; }
            prev = anchors[a];
        }
        return true;
    }

    /// Reads sequence `i` into `dst`, whose buffers are sized from info(i) and the header.
    bool read(uint64_t i, const SequenceSpans& dst, std::string& err) {
        if (file_ == nullptr || i >= infos_.size()) { err = "no such sequence " + std::to_string(i); return false; }
        const SequenceHeader& s = infos_[(size_t) i];
        SequenceLayout lay;
        if (!shard_sequence_layout(header_, s.n_tokens, s.n_anchors, lay, err)) return false;
        if (dst.tokens == nullptr || dst.hidden == nullptr || (s.n_anchors != 0 && (dst.anchors == nullptr || dst.selection == nullptr || dst.sel_count == nullptr))) {
            err = "sequence buffers are missing";
            return false;
        }
        const uint64_t at = offsets_[(size_t) i];
        if (!read_at(at + lay.tokens, dst.tokens, s.n_tokens * sizeof(int32_t), err) ||
            !read_at(at + lay.hidden, dst.hidden, s.n_tokens * lay.hidden_row_bytes, err) ||
            !read_at(at + lay.anchors, dst.anchors, s.n_anchors * sizeof(int32_t), err) ||
            !read_at(at + lay.selection, dst.selection, s.n_anchors * header_.n_layers * header_.sel_cap * sizeof(int32_t), err))
            return false;
        for (uint32_t l = 0; l < header_.n_layers; ++l) {
            if (dst.k[l] == nullptr || dst.v[l] == nullptr) { err = "sequence buffers are missing"; return false; }
            if (!read_at(at + lay.k[l], dst.k[l], s.n_tokens * lay.cell_bytes, err) ||
                !read_at(at + lay.v[l], dst.v[l], s.n_tokens * lay.cell_bytes, err))
                return false;
        }
        if (!shard_check_rows(header_, dst.tokens, s.n_tokens, dst.anchors, s.n_anchors, err)) return false;
        // each row: ascending ids, then nothing but padding
        for (uint64_t a = 0; a < s.n_anchors; ++a)
            for (uint32_t l = 0; l < header_.n_layers; ++l) {
                const size_t slot = (size_t) (a * header_.n_layers + l);
                const int32_t* row = dst.selection + slot * header_.sel_cap;
                uint32_t count = 0;
                while (count < header_.sel_cap && row[count] >= 0) ++count;
                for (uint32_t j = count; j < header_.sel_cap; ++j)
                    if (row[j] != -1) { err = "selection of anchor " + std::to_string(a) + " has ids after its padding"; return false; }
                if (!shard_check_selection(row, count, header_.sel_cap, dst.anchors[a], err)) return false;
                dst.sel_count[slot] = (int32_t) count;
            }
        return true;
    }

    /// Reads sequence `i` into buffers `out` owns.
    bool read(uint64_t i, SequenceData& out, std::string& err) {
        if (file_ == nullptr || i >= infos_.size()) { err = "no such sequence " + std::to_string(i); return false; }
        const SequenceHeader& s = infos_[(size_t) i];
        SequenceLayout lay;
        if (!shard_sequence_layout(header_, s.n_tokens, s.n_anchors, lay, err)) return false;
        out.info = s;
        out.tokens.resize((size_t) s.n_tokens);
        out.hidden.resize((size_t) (s.n_tokens * lay.hidden_row_bytes));
        out.anchors.resize((size_t) s.n_anchors);
        out.selection.resize((size_t) (s.n_anchors * header_.n_layers * header_.sel_cap));
        out.sel_count.resize((size_t) (s.n_anchors * header_.n_layers));
        out.k.resize(header_.n_layers);
        out.v.resize(header_.n_layers);
        SequenceSpans d;
        d.tokens = out.tokens.data();
        d.hidden = out.hidden.data();
        d.anchors = out.anchors.data();
        d.selection = out.selection.data();
        d.sel_count = out.sel_count.data();
        for (uint32_t l = 0; l < header_.n_layers; ++l) {
            out.k[l].resize((size_t) (s.n_tokens * lay.cell_bytes / sizeof(uint16_t)));
            out.v[l].resize(out.k[l].size());
            d.k[l] = out.k[l].data();
            d.v[l] = out.v[l].data();
        }
        return read(i, d, err);
    }

private:
    /// The header, the index and every sequence header against the file's `size`.
    bool check_table(uint64_t size, std::string& err) {
        if (!shard_check_geometry(header_, err)) return false;
        if (header_.n_sequences == 0 || header_.index_offset == 0) { err = "shard holds no sequence or was not closed by its writer"; return false; }
        if (header_.file_bytes != size) { err = "truncated (" + std::to_string(size) + " of " + std::to_string(header_.file_bytes) + " bytes)"; return false; }
        uint64_t index_bytes = 0;
        if (!detail::mul_u64(header_.n_sequences, sizeof(uint64_t), index_bytes) || header_.index_offset > size ||
            index_bytes != size - header_.index_offset) {
            err = "shard index does not fit the file";
            return false;
        }
        offsets_.resize((size_t) header_.n_sequences);
        if (!read_at(header_.index_offset, offsets_.data(), index_bytes, err)) return false;
        infos_.resize(offsets_.size());
        uint64_t expect = sizeof(ShardHeader);
        for (size_t i = 0; i < offsets_.size(); ++i) {
            // sequences are back to back from the header to the index
            if (offsets_[i] != expect || offsets_[i] > header_.index_offset || sizeof(SequenceHeader) > header_.index_offset - offsets_[i]) {
                err = "sequence " + std::to_string(i) + " is misplaced";
                return false;
            }
            if (!read_at(offsets_[i], &infos_[i], sizeof(SequenceHeader), err)) return false;
            SequenceLayout lay;
            if (infos_[i].magic != kSequenceMagic) { err = "sequence " + std::to_string(i) + " has a bad magic"; return false; }
            if (!shard_sequence_layout(header_, infos_[i].n_tokens, infos_[i].n_anchors, lay, err)) return false;
            if (infos_[i].pos0 < 0 || infos_[i].pos0 > (std::numeric_limits<int32_t>::max)() - (int64_t) kShardMaxTokens) { err = "sequence " + std::to_string(i) + " has pos0 out of range"; return false; }
            if (infos_[i].bytes != lay.bytes || lay.bytes > header_.index_offset - offsets_[i]) { err = "sequence " + std::to_string(i) + " does not fit the file"; return false; }
            expect = offsets_[i] + lay.bytes;
        }
        if (detail::align_up(expect) != header_.index_offset) { err = "shard index does not follow the last sequence"; return false; }
        return true;
    }

    bool read_at(uint64_t off, void* dst, uint64_t n, std::string& err) {
        if (n == 0) return true;
        if (dst == nullptr || platform::fseek64(file_, (int64_t) off, SEEK_SET) != 0 || std::fread(dst, 1, (size_t) n, file_) != (size_t) n) {
            err = "cannot read " + path_;
            return false;
        }
        return true;
    }

    std::FILE* file_ = nullptr;
    std::string path_;
    ShardHeader header_{};
    std::vector<uint64_t> offsets_;
    std::vector<SequenceHeader> infos_;
};

}  // namespace basalt::dfly2
