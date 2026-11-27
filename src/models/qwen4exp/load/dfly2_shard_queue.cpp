// src/models/qwen4exp/load/dfly2_shard_queue.cpp - see dfly2_shard_queue.hpp.
#include "dfly2_shard_queue.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/program/community_row.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace basalt::models::qwen4exp::generate {
using namespace basalt::dfly2;

namespace {
constexpr const char* kPart = ".part";              // the suffix of a file that is still being written
constexpr const char* kManifest = "export.json";
constexpr uint64_t kPieceAlign = 256;               // every array of a piece starts at a multiple of this
constexpr uint64_t kLastShard = 99999;              // the highest number a shard's name holds
constexpr uint64_t kManifestMaxBytes = 1u << 20;

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

uint64_t piece_align(uint64_t n) { return (n + kPieceAlign - 1) / kPieceAlign * kPieceAlign; }

/// The first geometry or pack identity field of `a` that is not `b`'s, or nullptr.
const char* differs(const ShardHeader& a, const ShardHeader& b) {
    if (a.hidden != b.hidden) return "hidden width";
    if (a.vocab != b.vocab) return "vocabulary";
    if (a.n_head_kv != b.n_head_kv || a.head_dim != b.head_dim) return "cell shape";
    if (a.n_layers != b.n_layers || std::memcmp(a.layers, b.layers, sizeof a.layers) != 0) return "paired layers";
    if (a.sel_cap != b.sel_cap) return "selection width";
    if (a.n_rot != b.n_rot || a.rope_base != b.rope_base) return "rotation";
    if (a.hidden_dtype != b.hidden_dtype || a.kv_dtype != b.kv_dtype) return "element type";
    if (a.pack_bytes != b.pack_bytes || std::memcmp(a.pack_meta_sha256, b.pack_meta_sha256, sizeof a.pack_meta_sha256) != 0 ||
        std::memcmp(a.pack_name, b.pack_name, sizeof a.pack_name) != 0)
        return "pack";
    return nullptr;
}
}  // namespace

std::string ShardQueue::shard_name(uint64_t index) {
    char name[64];
    std::snprintf(name, sizeof name, "shard-%05llu", (unsigned long long) index);
    return std::string(name) + kShardExtension;
}

ShardQueue::~ShardQueue() {
    if (writer_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            items_.push_back(Item{});
        }
        cv_items_.notify_one();
        writer_thread_.join();
    }
    release();
}

void ShardQueue::release() {
    for (void* block : blocks_)
        if (block != nullptr && memory_.give != nullptr) memory_.give(block);
    blocks_.clear();
    pieces_.clear();
    free_.clear();
}

bool ShardQueue::open(const std::string& dir, const ShardHeader& geometry, const Limits& limits, const Memory& memory,
                      std::string& err) {
    if (writer_thread_.joinable() || !pieces_.empty()) { err = "shard queue is already open"; return false; }
    if (dir.empty()) { err = "shard queue needs a directory"; return false; }
    if (limits.rows <= 0 || limits.anchors <= 0 || limits.page <= 0 || limits.pieces <= 0 || limits.shard_tokens == 0 ||
        (uint64_t) limits.rows > kShardMaxTokens || limits.anchors > limits.rows + 1) {
        err = "shard queue limits are out of range (rows " + std::to_string(limits.rows) + ", anchors " +
              std::to_string(limits.anchors) + ", page " + std::to_string(limits.page) + ", pieces " +
              std::to_string(limits.pieces) + ", shard tokens " + std::to_string(limits.shard_tokens) + ")";
        return false;
    }
    if (memory.take == nullptr || memory.give == nullptr) { err = "shard queue has no memory for its pieces"; return false; }
    geometry_ = geometry;
    std::memcpy(geometry_.magic, kShardMagic, sizeof geometry_.magic);
    geometry_.version = kShardVersion;
    geometry_.header_bytes = sizeof(ShardHeader);
    geometry_.n_sequences = geometry_.index_offset = geometry_.file_bytes = 0;
    if (!shard_check_geometry(geometry_, err)) return false;
    if (geometry_.hidden_dtype != (uint32_t) ShardDtype::f16) { err = "shard queue writes f16 hidden rows"; return false; }
    if (!platform::make_dirs(dir)) { err = "cannot create " + dir; return false; }

    // the closed shards, by name: numbered from 0 without a gap, each of this export's geometry and pack
    std::vector<platform::DirFile> files;
    if (!platform::list_files(dir, files)) { err = "cannot read " + dir; return false; }
    std::sort(files.begin(), files.end(), [](const platform::DirFile& a, const platform::DirFile& b) { return a.name < b.name; });
    uint64_t shards = 0, done = 0;
    for (const platform::DirFile& f : files) {
        const std::string path = dir + "/" + f.name;
        if (ends_with(f.name, std::string(kShardExtension) + kPart)) {
            if (!platform::remove_file(path)) { err = "cannot delete the unfinished " + path; return false; }
            continue;
        }
        if (!ends_with(f.name, kShardExtension)) continue;
        if (f.name != shard_name(shards)) { err = path + ": expected " + shard_name(shards) + " (an export numbers its shards without a gap)"; return false; }
        ShardReader reader;
        if (!reader.open(path, err)) return false;
        if (const char* field = differs(reader.header(), geometry_)) { err = path + ": its " + field + " is not this export's"; return false; }
        done += reader.sequences();
        ++shards;
    }

    // the pieces: one block each, carved once
    const uint64_t layers = geometry_.n_layers;
    const uint64_t page_values = (uint64_t) geometry_.n_head_kv * (uint64_t) limits.page * geometry_.head_dim;
    const uint64_t pages_cap = ((uint64_t) limits.rows + (uint64_t) limits.page - 1) / (uint64_t) limits.page + 1;
    const uint64_t hidden_bytes = piece_align((uint64_t) limits.rows * geometry_.hidden * sizeof(uint16_t));
    const uint64_t kv_bytes = piece_align(pages_cap * page_values * sizeof(uint16_t));
    const uint64_t anchor_bytes = piece_align((uint64_t) limits.anchors * sizeof(int32_t));
    const uint64_t sel_bytes = piece_align((uint64_t) limits.anchors * layers * geometry_.sel_cap * sizeof(int32_t));
    const uint64_t block_bytes = hidden_bytes + 2 * layers * kv_bytes + anchor_bytes + sel_bytes;
    dir_ = dir;
    limits_ = limits;
    memory_ = memory;
    pages_cap_ = (int64_t) pages_cap;
    pieces_.assign((size_t) limits.pieces, Piece{});
    for (Piece& p : pieces_) {
        uint8_t* at = (uint8_t*) memory_.take((size_t) block_bytes);
        if (at == nullptr) {
            release();
            err = "no memory for the export's piece buffers (" + std::to_string(limits.pieces) + " of " +
                  std::to_string(block_bytes >> 20) + " MiB)";
            return false;
        }
        blocks_.push_back(at);
        p.hidden = (uint16_t*) at;
        at += hidden_bytes;
        for (uint64_t l = 0; l < layers; ++l) {
            p.k[l] = (uint16_t*) at;
            at += kv_bytes;
            p.v[l] = (uint16_t*) at;
            at += kv_bytes;
        }
        p.sel_anchor = (int32_t*) at;
        at += anchor_bytes;
        p.sel = (int32_t*) at;
    }
    for (Piece& p : pieces_) free_.push_back(&p);
    k_cells_.assign((size_t) ((uint64_t) limits.rows * geometry_.n_head_kv * geometry_.head_dim), 0);
    v_cells_.assign(k_cells_.size(), 0);
    done_ = done;
    shard_index_ = shards;
    shard_tokens_ = 0;
    shard_open_ = in_sequence_ = false;
    failed_ = false;
    error_.clear();
    items_.clear();
    writer_thread_ = std::thread(&ShardQueue::run, this);
    return true;
}

ShardQueue::Piece* ShardQueue::take() {
    std::unique_lock<std::mutex> lock(mu_);
    if (!writer_thread_.joinable()) return nullptr;
    cv_free_.wait(lock, [&] { return failed_ || !free_.empty(); });
    if (failed_) return nullptr;
    Piece* p = free_.back();
    free_.pop_back();
    p->row0 = p->rows = p->page0 = p->pages = 0;
    p->n_sel = 0;
    return p;
}

bool ShardQueue::push(Item item, std::string& err) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!writer_thread_.joinable()) { err = "shard queue is not open"; return false; }
        if (failed_) { err = error_; return false; }
        items_.push_back(std::move(item));
    }
    cv_items_.notify_one();
    return true;
}

bool ShardQueue::begin(std::vector<int32_t> tokens, std::vector<int32_t> anchors, std::string& err) {
    if (tokens.empty()) { err = "a sequence needs tokens"; return false; }
    Item item;
    item.kind = Item::Begin;
    item.tokens = std::move(tokens);
    item.anchors = std::move(anchors);
    return push(std::move(item), err);
}

bool ShardQueue::put(Piece* p, std::string& err) {
    if (p == nullptr || pieces_.empty() || p < pieces_.data() || p >= pieces_.data() + pieces_.size()) {
        err = "not a piece of this shard queue";
        return false;
    }
    // every row's page must be one the piece holds
    const bool fits = p->rows > 0 && p->rows <= limits_.rows && p->row0 >= 0 && p->n_sel >= 0 && p->n_sel <= limits_.anchors &&
                      p->pages > 0 && p->pages <= pages_cap_ && p->page0 >= 0 && p->page0 <= p->row0 / limits_.page &&
                      (p->row0 + p->rows - 1) / limits_.page < p->page0 + p->pages;
    Item item;
    item.kind = Item::Put;
    item.piece = p;
    if (!fits) {
        err = "piece of " + std::to_string(p->rows) + " rows from " + std::to_string(p->row0) + ", " +
              std::to_string(p->pages) + " pages from " + std::to_string(p->page0) + " and " + std::to_string(p->n_sel) +
              " anchors is past the queue's limits (" + std::to_string(limits_.rows) + " rows, " +
              std::to_string(limits_.anchors) + " anchors)";
    } else if (push(std::move(item), err)) {
        return true;
    }
    // not queued: the piece is free again
    {
        std::lock_guard<std::mutex> lock(mu_);
        free_.push_back(p);
    }
    cv_free_.notify_one();
    return false;
}

void ShardQueue::give_back(Piece* p) {
    if (p == nullptr || pieces_.empty() || p < pieces_.data() || p >= pieces_.data() + pieces_.size()) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        free_.push_back(p);
    }
    cv_free_.notify_one();
}

bool ShardQueue::end(std::string& err) {
    Item item;
    item.kind = Item::End;
    return push(std::move(item), err);
}

void ShardQueue::fail(const std::string& why) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!failed_) {
            failed_ = true;
            error_ = why;
        }
    }
    cv_free_.notify_all();
}

void ShardQueue::run() {
    for (;;) {
        Item item;
        bool skip = false;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_items_.wait(lock, [&] { return !items_.empty(); });
            item = std::move(items_.front());
            items_.pop_front();
            skip = failed_;   // after a failure the queue only drains, so no piece stays in flight
        }
        if (item.kind == Item::Stop) return;
        std::string why;
        const bool ok = skip || write(item, why);
        if (item.kind == Item::Put) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                free_.push_back(item.piece);
            }
            cv_free_.notify_one();
        }
        if (!ok) fail(why);
    }
}

bool ShardQueue::write(Item& item, std::string& err) {
    if (item.kind == Item::Begin) {
        if (in_sequence_) { err = "a sequence began before the last one ended"; return false; }
        if (!shard_open_) {
            if (shard_index_ > kLastShard) { err = "the export holds more shards than their names number"; return false; }
            if (!writer_.open(dir_ + "/" + shard_name(shard_index_) + kPart, geometry_, err)) return false;
            shard_open_ = true;
            shard_tokens_ = 0;
        }
        if (!writer_.begin_sequence(item.tokens.data(), item.tokens.size(), item.anchors.data(), item.anchors.size(), 0, err))
            return false;
        in_sequence_ = true;
        shard_tokens_ += item.tokens.size();
        anchors_ = std::move(item.anchors);
        // anchor 0 has no row before it: its selections are empty
        if (!anchors_.empty() && anchors_[0] == 0)
            for (uint32_t l = 0; l < geometry_.n_layers; ++l)
                if (!writer_.put_selection(0, l, nullptr, 0, err)) return false;
        return true;
    }
    if (!in_sequence_) { err = "a piece or a sequence's end arrived outside a sequence"; return false; }
    if (item.kind == Item::Put) return write_piece(*item.piece, err);
    if (!writer_.end_sequence(err)) return false;
    in_sequence_ = false;
    return shard_tokens_ < limits_.shard_tokens || finish_shard(err);
}

bool ShardQueue::write_piece(const Piece& p, std::string& err) {
    const uint64_t page = (uint64_t) limits_.page, heads = geometry_.n_head_kv, dim = geometry_.head_dim;
    if (!writer_.put_hidden((uint64_t) p.row0, (uint64_t) p.rows, p.hidden, err)) return false;
    // the cache keeps a page head-major; the shard keeps a cell's heads together, and only the piece's own cells
    for (uint32_t l = 0; l < geometry_.n_layers; ++l) {
        for (uint64_t r = 0; r < (uint64_t) p.rows; ++r) {
            const uint64_t cell = (uint64_t) p.row0 + r, in_page = cell % page, pg = cell / page - (uint64_t) p.page0;
            for (uint64_t h = 0; h < heads; ++h) {
                const size_t src = (size_t) (((pg * heads + h) * page + in_page) * dim), dst = (size_t) ((r * heads + h) * dim);
                std::memcpy(k_cells_.data() + dst, p.k[l] + src, (size_t) dim * sizeof(uint16_t));
                std::memcpy(v_cells_.data() + dst, p.v[l] + src, (size_t) dim * sizeof(uint16_t));
            }
        }
        if (!writer_.put_kv(l, (uint64_t) p.row0, (uint64_t) p.rows, k_cells_.data(), v_cells_.data(), err)) return false;
    }
    for (int i = 0; i < p.n_sel; ++i) {
        const int32_t a = p.sel_anchor[i];
        if (a < 0 || (size_t) a >= anchors_.size()) { err = "a piece names anchor number " + std::to_string(a) + ", which the sequence does not have"; return false; }
        const int64_t anchor = anchors_[(size_t) a];
        if (anchor - 1 < p.row0 || anchor - 1 >= p.row0 + p.rows) {
            err = "the row before anchor " + std::to_string(anchor) + " is not in the piece that carries its selection";
            return false;
        }
        // row anchor - 1 selects among `anchor` cells: all of them until they outnumber the selection
        const uint32_t count = (uint32_t) std::min<int64_t>(anchor, geometry_.sel_cap);
        for (uint32_t l = 0; l < geometry_.n_layers; ++l)
            if (!writer_.put_selection((uint64_t) a, l, p.sel + ((size_t) i * geometry_.n_layers + l) * geometry_.sel_cap, count, err))
                return false;
    }
    return true;
}

bool ShardQueue::finish_shard(std::string& err) {
    const std::string path = dir_ + "/" + shard_name(shard_index_);
    shard_open_ = false;
    if (!writer_.close(err)) return false;
    if (!platform::rename_replace(path + kPart, path)) { err = "cannot rename " + path + kPart; return false; }
    ++shard_index_;
    shard_tokens_ = 0;
    return true;
}

bool ShardQueue::close(std::string& err) {
    if (!writer_thread_.joinable()) { err = "shard queue is not open"; return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        items_.push_back(Item{});
    }
    cv_items_.notify_one();
    writer_thread_.join();
    bool ok = !failed_;
    std::string why = error_;
    if (ok && in_sequence_) {
        ok = false;
        why = "the last sequence was not ended";
    }
    if (ok && shard_open_) ok = finish_shard(why);
    release();
    if (!ok) err = why;
    return ok;
}

bool manifest_check(const std::string& dir, const std::map<std::string, std::string>& settings, std::string& err) {
    if (dir.empty() || settings.empty()) { err = "the export's directory or its settings are missing"; return false; }
    const std::string path = dir + "/" + kManifest;
    uint64_t bytes = 0;
    if (!platform::file_size(path, bytes)) {
        // absent: this run's settings become the directory's
        if (!platform::make_dirs(dir)) { err = "cannot create " + dir; return false; }
        std::string text = "{\n";
        size_t written = 0;
        for (const auto& [key, value] : settings)
            text += "  " + program::json_quote(key) + ": " + program::json_quote(value) + (++written < settings.size() ? ",\n" : "\n");
        text += "}\n";
        std::FILE* f = platform::open_file_utf8(path + kPart, "wb");
        if (f == nullptr) { err = "cannot create " + path; return false; }
        const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
        if (std::fclose(f) != 0 || !ok || !platform::rename_replace(path + kPart, path)) { err = "cannot write " + path; return false; }
        return true;
    }
    if (bytes == 0 || bytes > kManifestMaxBytes) { err = path + " is not an export's settings file"; return false; }
    std::string text((size_t) bytes, '\0');
    std::FILE* f = platform::open_file_utf8(path, "rb");
    if (f == nullptr) { err = "cannot read " + path; return false; }
    const bool got = std::fread(text.data(), 1, text.size(), f) == text.size();
    std::fclose(f);
    core::Json stored;
    if (!got || !core::parse_json(text.data(), text.size(), stored, err) || !stored.is_object()) {
        err = path + " is not an export's settings file" + (err.empty() ? "" : ": " + err);
        return false;
    }
    for (const auto& [key, value] : settings) {
        const core::Json* was = stored.get(key);
        if (was == nullptr || !was->is_string()) { err = path + ": no \"" + key + "\" (this run's is \"" + value + "\")"; return false; }
        if (was->s != value) { err = path + ": \"" + key + "\" is \"" + was->s + "\", this run's is \"" + value + "\""; return false; }
    }
    for (const auto& member : stored.members)
        if (settings.find(member.first) == settings.end()) { err = path + ": \"" + member.first + "\" is not a setting of this export"; return false; }
    return true;
}

}  // namespace basalt::models::qwen4exp::generate
