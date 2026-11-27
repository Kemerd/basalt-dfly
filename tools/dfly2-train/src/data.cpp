// tools/dfly2-train/src/data.cpp - the shard set, the training order and the read-ahead thread (data.hpp).
#include "data.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace basalt::dfly2 {

namespace {

/// splitmix64's finaliser: spreads a counter over 64 bits.
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/// Whether `name` ends with `tail`.
bool ends_with(const std::string& name, const std::string& tail) {
    return name.size() > tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0;
}

/// The anchors of an ascending list whose block of `block` rows lies inside a sequence of `tokens` rows.
uint64_t count_inside(const int32_t* anchors, uint64_t n, uint64_t tokens, int64_t block) {
    uint64_t usable = 0;
    while (usable < n && (uint64_t) anchors[usable] + (uint64_t) block <= tokens) ++usable;
    return usable;
}

/// Checks a shard's header against the drafter and the pack; false with the reason in `err`.
bool check_header(const ShardHeader& h, const DrafterShape& s, const PackIdentity& pack, std::string& err) {
    if ((int64_t) h.hidden != s.hidden || (int64_t) h.vocab != s.vocab || (int64_t) h.n_head_kv != s.n_head_kv || (int64_t) h.head_dim != s.head_dim) {
        err = "its geometry (hidden " + std::to_string(h.hidden) + ", vocab " + std::to_string(h.vocab) + ", " + std::to_string(h.n_head_kv) + " key/value heads of " +
              std::to_string(h.head_dim) + ") is not the drafter's";
        return false;
    }
    if ((int64_t) h.n_rot != s.n_rot || std::abs(h.rope_base - s.rope_base) > 1e-9 * s.rope_base) { err = "its rotation (width " + std::to_string(h.n_rot) + ") is not the drafter's"; return false; }
    if ((int64_t) h.n_layers != s.n_layers) { err = "it carries " + std::to_string(h.n_layers) + " paired layers, the drafter has " + std::to_string(s.n_layers); return false; }
    for (uint32_t l = 0; l < h.n_layers; ++l)
        if ((int64_t) h.layers[l] != (int64_t) s.target_layers[l]) { err = "its paired layer " + std::to_string(l) + " is target layer " + std::to_string(h.layers[l]) + ", the drafter's is " + std::to_string(s.target_layers[l]); return false; }
    if ((int64_t) h.sel_cap > s.sel_cap) { err = "its selections are up to " + std::to_string(h.sel_cap) + " cells wide, the drafter reads " + std::to_string(s.sel_cap); return false; }
    if (h.pack_bytes != pack.bytes || std::memcmp(h.pack_meta_sha256, pack.meta_sha256.data(), 32) != 0) {
        err = "it was exported from another pack (" + std::string(h.pack_name, strnlen(h.pack_name, sizeof h.pack_name)) + ", " + std::to_string(h.pack_bytes) + " bytes)";
        return false;
    }
    return true;
}

}  // namespace

std::vector<uint64_t> shuffled_order(uint64_t n, uint64_t seed) {
    std::vector<uint64_t> order((size_t) n);
    for (uint64_t i = 0; i < n; ++i) order[(size_t) i] = i;
    // Fisher-Yates over a generator the standard fixes bit for bit; the remainder's bias is far below one draw
    std::mt19937_64 rng(seed);
    for (uint64_t i = n; i > 1; --i) std::swap(order[(size_t) (i - 1)], order[(size_t) (rng() % i)]);
    return order;
}

uint64_t epoch_seed(uint64_t seed, int64_t epoch) { return mix64(seed ^ mix64((uint64_t) epoch + 1)); }

uint64_t sequence_seed(uint64_t seed, int64_t epoch, uint64_t sequence) { return mix64(epoch_seed(seed, epoch) ^ mix64(sequence + 0x5851f42d4c957f2dull)); }

// ---------------------------------------------------------------- the shard set

bool ShardSet::open(const std::string& dir, const DrafterShape& shape, const PackIdentity& pack, std::string& err) {
    shape_ = shape;
    paths_.clear();
    refs_.clear();
    open_.clear();
    blocks_ = tokens_ = 0;
    std::vector<platform::DirFile> files;
    if (!platform::list_files(dir, files)) { err = "cannot read the shard directory " + dir; return false; }
    // by name, so the set's sequence numbering does not depend on the order the OS lists files in
    std::sort(files.begin(), files.end(), [](const platform::DirFile& a, const platform::DirFile& b) { return a.name < b.name; });
    std::vector<int32_t> anchors;
    for (const platform::DirFile& f : files) {
        if (!ends_with(f.name, kShardExtension)) continue;
        const std::string path = dir + "/" + f.name;
        ShardReader r;
        if (!r.open(path, err)) return false;
        if (!check_header(r.header(), shape, pack, err)) { err = path + ": " + err; return false; }
        const uint32_t shard = (uint32_t) paths_.size();
        paths_.push_back(path);
        for (uint64_t i = 0; i < r.sequences(); ++i) {
            const SequenceHeader* info = r.info(i);
            if (info == nullptr) { err = path + ": sequence " + std::to_string(i) + " is missing"; return false; }
            anchors.resize((size_t) info->n_anchors);
            if (!r.read_anchors(i, anchors.data(), err)) { err = path + ": " + err; return false; }
            const uint64_t usable = count_inside(anchors.data(), info->n_anchors, info->n_tokens, shape.block);
            if (usable == 0) continue;
            refs_.push_back({shard, i, info->n_tokens, usable});
            blocks_ += usable;
            tokens_ += info->n_tokens;
        }
    }
    if (paths_.empty()) { err = "no *" + std::string(kShardExtension) + " file in " + dir; return false; }
    if (refs_.empty()) { err = "the shards of " + dir + " hold no block of " + std::to_string(shape.block) + " rows that fits its sequence"; return false; }
    return true;
}

ShardReader* ShardSet::reader(uint32_t shard, std::string& err) {
    for (auto it = open_.begin(); it != open_.end(); ++it) {
        if (it->first != shard) continue;
        open_.splice(open_.begin(), open_, it);
        return open_.front().second.get();
    }
    if (shard >= paths_.size()) { err = "no such shard " + std::to_string(shard); return nullptr; }
    auto r = std::make_unique<ShardReader>();
    if (!r->open(paths_[shard], err)) return nullptr;
    open_.emplace_front(shard, std::move(r));
    if (open_.size() > kOpenShards) open_.pop_back();
    return open_.front().second.get();
}

bool ShardSet::load(size_t ref_index, uint64_t seed, uint64_t first, LoadedSequence& out, std::string& err) {
    if (ref_index >= refs_.size()) { err = "no such sequence " + std::to_string(ref_index); return false; }
    const SequenceRef& ref = refs_[ref_index];
    ShardReader* r = reader(ref.shard, err);
    if (r == nullptr) return false;
    const SequenceHeader* info = r->info(ref.index);
    if (info == nullptr || info->n_tokens != ref.tokens) { err = r->path() + ": sequence " + std::to_string(ref.index) + " changed since the shard was counted"; return false; }
    const ShardHeader& h = r->header();
    const int64_t L = (int64_t) info->n_tokens, A = (int64_t) info->n_anchors, layers = (int64_t) h.n_layers;

    // the arrays are read straight into the tensors' memory
    torch::Tensor tokens = torch::empty({L}, torch::kInt32), anchors = torch::empty({A}, torch::kInt32);
    torch::Tensor selection = torch::empty({A, layers, (int64_t) h.sel_cap}, torch::kInt32), counts = torch::empty({A, layers}, torch::kInt32);
    out = LoadedSequence{};
    out.ctx.hidden = torch::empty({L, (int64_t) h.hidden}, h.hidden_dtype == (uint32_t) ShardDtype::f16 ? torch::kFloat16 : torch::kFloat32);
    SequenceSpans spans;
    spans.tokens = tokens.data_ptr<int32_t>();
    spans.anchors = anchors.data_ptr<int32_t>();
    spans.selection = selection.data_ptr<int32_t>();
    spans.sel_count = counts.data_ptr<int32_t>();
    spans.hidden = out.ctx.hidden.data_ptr();
    for (int64_t l = 0; l < layers; ++l) {
        out.ctx.k.push_back(torch::empty({L, (int64_t) h.n_head_kv, (int64_t) h.head_dim}, torch::kFloat16));
        out.ctx.v.push_back(torch::empty({L, (int64_t) h.n_head_kv, (int64_t) h.head_dim}, torch::kFloat16));
        spans.k[l] = static_cast<uint16_t*>(out.ctx.k.back().data_ptr());
        spans.v[l] = static_cast<uint16_t*>(out.ctx.v.back().data_ptr());
    }
    if (!r->read(ref.index, spans, err)) { err = r->path() + ": sequence " + std::to_string(ref.index) + ": " + err; return false; }

    // the blocks that fit, in this epoch's order, from where the run stands
    const uint64_t usable = count_inside(anchors.data_ptr<int32_t>(), (uint64_t) A, info->n_tokens, shape_.block);
    if (usable != ref.blocks) { err = r->path() + ": sequence " + std::to_string(ref.index) + " changed since the shard was counted"; return false; }
    if (first >= usable) { err = "the run's place (block " + std::to_string(first) + ") is past sequence " + std::to_string(ref.index) + " of " + r->path(); return false; }
    const std::vector<uint64_t> order = shuffled_order(usable, seed);
    torch::Tensor take = torch::empty({(int64_t) (usable - first)}, torch::kLong);
    for (uint64_t i = first; i < usable; ++i) take.data_ptr<int64_t>()[i - first] = (int64_t) order[(size_t) i];
    out.anchors = anchors.to(torch::kLong).index_select(0, take);
    out.selection = selection.index_select(0, take);
    out.sel_count = counts.index_select(0, take);
    out.ctx.tokens = tokens.to(torch::kLong);
    out.ctx.pos0 = info->pos0;
    out.first = first;
    out.bytes = out.ctx.tokens.nbytes() + out.ctx.hidden.nbytes() + out.anchors.nbytes() + out.selection.nbytes() + out.sel_count.nbytes();
    for (int64_t l = 0; l < layers; ++l) out.bytes += out.ctx.k[(size_t) l].nbytes() + out.ctx.v[(size_t) l].nbytes();
    return true;
}

// ---------------------------------------------------------------- the read-ahead thread

SequenceLoader::SequenceLoader(ShardSet& set, uint64_t seed, int64_t epoch, uint64_t position, uint64_t first, int64_t end_epoch, uint64_t queue_bytes)
    : set_(set), seed_(seed), queue_bytes_(queue_bytes) {
    thread_ = std::thread([this, epoch, position, first, end_epoch] { run(epoch, position, first, end_epoch); });
}

SequenceLoader::~SequenceLoader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    can_push_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void SequenceLoader::run(int64_t epoch, uint64_t position, uint64_t first, int64_t end_epoch) {
    std::string err;
    try {
        const uint64_t count = set_.sequences().size();
        for (; err.empty() && (end_epoch < 0 || epoch < end_epoch); ++epoch, position = 0) {
            const std::vector<uint64_t> order = shuffled_order(count, epoch_seed(seed_, epoch));
            for (; position < count; ++position, first = 0) {
                LoadedSequence s;
                const uint64_t ref = order[(size_t) position];
                if (!set_.load((size_t) ref, sequence_seed(seed_, epoch, ref), first, s, err)) break;
                s.epoch = epoch;
                s.position = position;
                std::unique_lock<std::mutex> lock(mutex_);
                can_push_.wait(lock, [this] { return stop_ || queue_.empty() || queued_bytes_ < queue_bytes_; });
                if (stop_) return;
                queued_bytes_ += s.bytes;
                queue_.push_back(std::move(s));
                lock.unlock();
                can_pop_.notify_one();
            }
        }
    } catch (const std::exception& e) {
        err = std::string("reading a sequence failed: ") + e.what();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = err;
        done_ = true;
    }
    can_pop_.notify_all();
}

bool SequenceLoader::next(LoadedSequence& out, std::string& err) {
    std::unique_lock<std::mutex> lock(mutex_);
    can_pop_.wait(lock, [this] { return done_ || !queue_.empty(); });
    // what was read before a failure is still handed out, in order
    if (queue_.empty()) {
        err = error_;
        return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    queued_bytes_ -= out.bytes;
    lock.unlock();
    can_push_.notify_one();
    return true;
}

}  // namespace basalt::dfly2
