// tests/models/qwen4exp/load/dfly2_shard_queue_test.cpp - the teacher export's writer on synthetic pieces: what goes in
// page-major comes back cell-major through the shard reader, a piece may start and end inside a cache page, a
// selection belongs to the piece that holds the row before its anchor, the pieces are the only buffers, shards roll
// and a restart counts them, and the settings file refuses a changed setting. Host only, no model.
#include "models/qwen4exp/load/dfly2_shard_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace basalt::dfly2;
using basalt::models::qwen4exp::generate::ShardQueue;
using basalt::models::qwen4exp::generate::manifest_check;

namespace {
int g_fail = 0;
// `why` is read here, after the checked call has written it: the reason a check that should hold does not
void check(bool ok, const std::string& what, const std::string& why = std::string()) {
    std::printf("  %-86s %s\n", (ok || why.empty() ? what : what + ": " + why).c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
// a refusal: its text is shown whether or not the check holds
void refused(bool ok, const std::string& what, const std::string& why) {
    std::printf("  %-86s %s\n", (what + ": " + why).c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("basalt-dfly2-queue-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
    std::string sub(const char* name) const { return (path / name).string(); }
};

// ---- plain memory for the pieces, counted
std::atomic<int> g_taken{0}, g_given{0};
void* take_plain(size_t bytes) { ++g_taken; return std::malloc(bytes); }
void give_plain(void* p) { ++g_given; std::free(p); }
void* take_none(size_t) { return nullptr; }
const ShardQueue::Memory kPlain{&take_plain, &give_plain};

constexpr int64_t kPage = 4;            // cells of a cache page
constexpr uint16_t kNotMine = 0xFFFF;   // what a page holds where the piece has no cell
constexpr int32_t kPastCount = 77777;   // what a selection row holds past its ids

ShardHeader tiny_header() {
    ShardHeader h{};
    h.hidden = 6;
    h.vocab = 50;
    h.n_head_kv = 2;
    h.head_dim = 4;
    h.n_layers = 2;
    h.layers[0] = 3;
    h.layers[1] = 7;
    h.sel_cap = 5;
    h.n_rot = 2;
    h.rope_base = 1e7;
    h.hidden_dtype = (uint32_t) ShardDtype::f16;
    h.kv_dtype = (uint32_t) ShardDtype::f16;
    h.pack_bytes = 123456789;
    for (int i = 0; i < 32; ++i) h.pack_meta_sha256[i] = (uint8_t) (i * 3 + 1);
    std::memcpy(h.pack_name, "tiny.basalt", sizeof "tiny.basalt");
    return h;
}

ShardQueue::Limits tiny_limits() {
    ShardQueue::Limits l;
    l.rows = 7;
    l.anchors = 8;
    l.page = kPage;
    l.pieces = 2;
    l.shard_tokens = 20;
    return l;
}

// ---- what the synthetic target computed for sequence `seq`
int32_t token_of(uint64_t seq, int64_t row) { return (int32_t) ((seq * 7 + (uint64_t) row * 3) % 50); }
uint16_t hidden_of(uint64_t seq, int64_t row, uint32_t d) { return (uint16_t) (1 + (seq * 131 + (uint64_t) row * 17 + d) % 0xE000); }
uint16_t cell_of(uint64_t seq, uint32_t layer, bool value, int64_t cell, uint32_t head, uint32_t d) {
    return (uint16_t) (1 + (seq * 977 + layer * 4099 + (value ? 2053 : 0) + (uint64_t) cell * 37 + head * 11 + d) % 0xE000);
}
// row anchor - 1's cells in `layer`: every cell while they fit, else spread over the cells below the anchor
std::vector<int32_t> selection_of(const ShardHeader& h, uint32_t layer, int32_t anchor) {
    std::vector<int32_t> ids;
    const int32_t count = anchor < (int32_t) h.sel_cap ? anchor : (int32_t) h.sel_cap;
    for (int32_t j = 0; j < count; ++j)
        ids.push_back(anchor <= (int32_t) h.sel_cap ? j : (int32_t) ((int64_t) j * (anchor - 1 - (int32_t) layer % 2) / (count - 1)));
    return ids;
}

/// Fills `p` with rows [row0, row0 + rows) of sequence `seq`; `anchors` are the sequence's.
void fill_piece(ShardQueue::Piece& p, const ShardHeader& h, uint64_t seq, int64_t row0, int64_t rows, const std::vector<int32_t>& anchors) {
    p.row0 = row0;
    p.rows = rows;
    p.page0 = row0 / kPage;
    p.pages = (row0 + rows - 1) / kPage - p.page0 + 1;
    for (int64_t r = 0; r < rows; ++r)
        for (uint32_t d = 0; d < h.hidden; ++d) p.hidden[r * h.hidden + d] = hidden_of(seq, row0 + r, d);
    // the pages as a cache holds them: [page][kv_head][slot][head_dim], cells outside the piece marked
    for (uint32_t l = 0; l < h.n_layers; ++l)
        for (int64_t pg = 0; pg < p.pages; ++pg)
            for (uint32_t head = 0; head < h.n_head_kv; ++head)
                for (int64_t slot = 0; slot < kPage; ++slot) {
                    const int64_t cell = (p.page0 + pg) * kPage + slot;
                    const bool mine = cell >= row0 && cell < row0 + rows;
                    const size_t at = (size_t) (((pg * h.n_head_kv + head) * kPage + slot) * h.head_dim);
                    for (uint32_t d = 0; d < h.head_dim; ++d) {
                        p.k[l][at + d] = mine ? cell_of(seq, l, false, cell, head, d) : kNotMine;
                        p.v[l][at + d] = mine ? cell_of(seq, l, true, cell, head, d) : kNotMine;
                    }
                }
    p.n_sel = 0;
    for (size_t a = 0; a < anchors.size(); ++a) {
        if (anchors[a] - 1 < row0 || anchors[a] - 1 >= row0 + rows) continue;
        p.sel_anchor[p.n_sel] = (int32_t) a;
        for (uint32_t l = 0; l < h.n_layers; ++l) {
            int32_t* row = p.sel + ((size_t) p.n_sel * h.n_layers + l) * h.sel_cap;
            const std::vector<int32_t> ids = selection_of(h, l, anchors[a]);
            for (uint32_t j = 0; j < h.sel_cap; ++j) row[j] = j < ids.size() ? ids[j] : kPastCount;
        }
        ++p.n_sel;
    }
}

/// Hands sequence `seq` of `n` rows to the queue in pieces of the lengths `cuts` cycles through.
bool put_sequence(ShardQueue& q, const ShardHeader& h, uint64_t seq, int64_t n, const std::vector<int32_t>& anchors,
                  const std::vector<int64_t>& cuts, std::string& err) {
    std::vector<int32_t> tokens((size_t) n);
    for (int64_t r = 0; r < n; ++r) tokens[(size_t) r] = token_of(seq, r);
    if (!q.begin(tokens, anchors, err)) return false;
    size_t c = 0;
    for (int64_t row0 = 0; row0 < n;) {
        const int64_t rows = std::min(cuts[c++ % cuts.size()], n - row0);
        ShardQueue::Piece* p = q.take();
        if (p == nullptr) { err = "no piece"; return false; }
        fill_piece(*p, h, seq, row0, rows, anchors);
        if (!q.put(p, err)) return false;
        row0 += rows;
    }
    return q.end(err);
}

/// Sequence `i` of the shard at `path` is sequence `seq` as the synthetic target computed it.
bool same_sequence(const std::string& path, uint64_t i, const ShardHeader& h, uint64_t seq, int64_t n, const std::vector<int32_t>& anchors) {
    ShardReader r;
    std::string err;
    SequenceData d;
    if (!r.open(path, err) || !r.read(i, d, err)) { std::printf("    %s\n", err.c_str()); return false; }
    if ((int64_t) d.info.n_tokens != n || d.info.pos0 != 0 || d.anchors != anchors) return false;
    const uint16_t* hidden = (const uint16_t*) d.hidden.data();
    for (int64_t row = 0; row < n; ++row) {
        if (d.tokens[(size_t) row] != token_of(seq, row)) return false;
        for (uint32_t x = 0; x < h.hidden; ++x)
            if (hidden[row * h.hidden + x] != hidden_of(seq, row, x)) return false;
        // cell-major: [cell][kv_head][head_dim]
        for (uint32_t l = 0; l < h.n_layers; ++l)
            for (uint32_t head = 0; head < h.n_head_kv; ++head)
                for (uint32_t x = 0; x < h.head_dim; ++x) {
                    const size_t at = (size_t) ((row * h.n_head_kv + head) * h.head_dim + x);
                    if (d.k[l][at] != cell_of(seq, l, false, row, head, x) || d.v[l][at] != cell_of(seq, l, true, row, head, x)) return false;
                }
    }
    for (size_t a = 0; a < anchors.size(); ++a)
        for (uint32_t l = 0; l < h.n_layers; ++l) {
            const std::vector<int32_t> ids = selection_of(h, l, anchors[a]);
            const size_t slot = a * h.n_layers + l;
            if (d.sel_count[slot] != (int32_t) ids.size()) return false;
            for (uint32_t j = 0; j < h.sel_cap; ++j)
                if (d.selection[slot * h.sel_cap + j] != (j < ids.size() ? ids[j] : -1)) return false;
        }
    return true;
}

uint64_t sequences_in(const std::string& path) {
    ShardReader r;
    std::string err;
    return r.open(path, err) ? r.sequences() : 0;
}

// the five sequences every part of the test writes: rows, anchors
struct Seq { int64_t n; std::vector<int32_t> anchors; };
const std::vector<Seq> kSeqs = {{9, {1, 4, 8}}, {12, {2, 5, 6, 7, 11}}, {5, {}}, {30, {1, 3, 9, 10, 17, 25, 29}}, {3, {2}}};
}  // namespace

int main() {
    const ShardHeader h = tiny_header();
    std::string err;

    std::printf("1. pieces in, sequences out\n");
    TempDir d;
    const std::string dir = d.sub("export");
    {
        ShardQueue q;
        check(q.open(dir, h, tiny_limits(), kPlain, err), "open", err);
        check(q.sequences_done() == 0 && q.piece_pages() == 3, "a fresh directory holds no sequence; 7 rows span at most 3 pages");
        const int taken = g_taken.load();
        check(taken == 2, "the pieces are taken once, at open");
        // piece lengths that start and end inside a page, and whole pages
        bool ok = true;
        for (size_t s = 0; s < kSeqs.size(); ++s)
            ok = ok && put_sequence(q, h, s, kSeqs[s].n, kSeqs[s].anchors, s % 2 ? std::vector<int64_t>{3, 7, 1, 4} : std::vector<int64_t>{7, 2, 5}, err);
        check(ok, "five sequences are handed over", err);
        check(q.close(err), "close", err);
        check(g_taken.load() == taken && g_given.load() == taken, "no buffer is taken per piece, and close gives the pieces back");
    }
    // 20 tokens close a shard at a sequence's end: 9 + 12 | 5 + 30 | 3
    const std::string s0 = dir + "/" + ShardQueue::shard_name(0), s1 = dir + "/" + ShardQueue::shard_name(1),
                      s2 = dir + "/" + ShardQueue::shard_name(2);
    check(ShardQueue::shard_name(0) == "shard-00000.dfly2", "a shard's name");
    check(sequences_in(s0) == 2 && sequences_in(s1) == 2 && sequences_in(s2) == 1, "shards roll at a sequence boundary");
    check(same_sequence(s0, 0, h, 0, kSeqs[0].n, kSeqs[0].anchors) && same_sequence(s0, 1, h, 1, kSeqs[1].n, kSeqs[1].anchors) &&
          same_sequence(s1, 0, h, 2, kSeqs[2].n, kSeqs[2].anchors) && same_sequence(s1, 1, h, 3, kSeqs[3].n, kSeqs[3].anchors) &&
          same_sequence(s2, 0, h, 4, kSeqs[4].n, kSeqs[4].anchors),
          "every row, cell and selection reads back, a cell's heads together");
    {
        bool part = false;
        for (const auto& e : fs::directory_iterator(dir)) part = part || e.path().extension() == ".part";
        check(!part, "no unfinished file is left");
    }

    std::printf("2. a restart\n");
    {
        std::ofstream(dir + "/" + ShardQueue::shard_name(3) + ".part") << "half a shard";
        ShardQueue q;
        check(q.open(dir, h, tiny_limits(), kPlain, err), "open again", err);
        check(q.sequences_done() == 5, "the closed shards' sequences are counted");
        check(!fs::exists(dir + "/" + ShardQueue::shard_name(3) + ".part"), "an unfinished shard is deleted");
        check(put_sequence(q, h, 5, 6, {1, 5}, {4, 2}, err) && q.close(err), "one more sequence", err);
        check(sequences_in(dir + "/" + ShardQueue::shard_name(3)) == 1 &&
              same_sequence(dir + "/" + ShardQueue::shard_name(3), 0, h, 5, 6, {1, 5}), "it lands in the next shard");
    }
    {
        ShardQueue q;
        ShardHeader other = h;
        other.hidden = 8;
        refused(!q.open(dir, other, tiny_limits(), kPlain, err) && err.find("hidden width") != std::string::npos, "another geometry is refused", err);
        other = h;
        other.pack_bytes += 1;
        refused(!q.open(dir, other, tiny_limits(), kPlain, err) && err.find("pack") != std::string::npos, "another pack is refused", err);
        other = h;
        other.rope_base = 1e4;
        refused(!q.open(dir, other, tiny_limits(), kPlain, err) && err.find("rotation") != std::string::npos, "another rope base is refused", err);
        fs::copy_file(dir + "/" + ShardQueue::shard_name(3), dir + "/" + ShardQueue::shard_name(7));
        refused(!q.open(dir, h, tiny_limits(), kPlain, err) && err.find("without a gap") != std::string::npos, "a gap in the numbering is refused", err);
        fs::remove(dir + "/" + ShardQueue::shard_name(7));
    }

    std::printf("3. the bound\n");
    {
        const std::string dir3 = d.sub("bound");
        ShardQueue q;
        check(q.open(dir3, h, tiny_limits(), kPlain, err), "open", err);
        check(q.begin({1, 2, 3, 4, 5, 6, 7, 8}, {}, err), "begin", err);
        ShardQueue::Piece *a = q.take(), *b = q.take();
        check(a != nullptr && b != nullptr && a != b, "two pieces are free");
        std::atomic<bool> got{false};
        ShardQueue::Piece* c = nullptr;
        std::thread waiter([&] { c = q.take(); got = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        check(!got.load(), "a third take waits while both are in flight");
        fill_piece(*a, h, 9, 0, 4, {});
        check(q.put(a, err), "put", err);
        waiter.join();
        check(got.load() && c == a, "it returns the piece the writer gave back");
        // an over-long piece is refused and stays usable
        b->row0 = 4; b->rows = 8; b->page0 = 1; b->pages = 2; b->n_sel = 0;
        refused(!q.put(b, err) && err.find("limits") != std::string::npos, "a piece above the limits is refused", err);
        ShardQueue::Piece* again = q.take();
        check(again == b, "the refused piece is free again");
        again->row0 = 4; again->rows = 4; again->page0 = 2; again->pages = 1; again->n_sel = 0;
        check(!q.put(again, err), "a piece whose pages do not hold its rows is refused");
        again = q.take();
        q.give_back(again);
        check(q.take() == again, "a piece given back unwritten is free again");
        fill_piece(*again, h, 9, 4, 4, {});
        check(q.put(again, err) && q.end(err), "the sequence completes", err);
        ShardQueue::Piece other{};
        check(!q.put(&other, err) && !q.put(nullptr, err), "a piece that is not the queue's is refused");
        check(q.close(err), "close", err);
        // one piece in flight: every take but the first waits for the writer
        ShardQueue::Limits one = tiny_limits();
        one.pieces = 1;
        ShardQueue q1;
        check(q1.open(d.sub("one"), h, one, kPlain, err) && put_sequence(q1, h, 3, kSeqs[3].n, kSeqs[3].anchors, {2, 3}, err) && q1.close(err),
              "one piece carries a whole sequence", err);
        check(same_sequence(d.sub("one") + "/" + ShardQueue::shard_name(0), 0, h, 3, kSeqs[3].n, kSeqs[3].anchors), "and it reads back");
    }

    std::printf("4. refusals\n");
    {
        for (int which = 0; which < 5; ++which) {
            ShardQueue::Limits l = tiny_limits();
            if (which == 0) l.rows = 0;
            if (which == 1) l.anchors = 0;
            if (which == 2) l.page = 0;
            if (which == 3) l.pieces = 0;
            if (which == 4) l.shard_tokens = 0;
            ShardQueue q;
            refused(!q.open(d.sub("refused"), h, l, kPlain, err), "a zero limit is refused", err);
        }
        ShardQueue q;
        refused(!q.open(d.sub("refused"), h, tiny_limits(), ShardQueue::Memory{}, err), "no memory functions", err);
        refused(!q.open(d.sub("refused"), h, tiny_limits(), ShardQueue::Memory{&take_none, &give_plain}, err), "no memory", err);
        ShardHeader f32 = h;
        f32.hidden_dtype = (uint32_t) ShardDtype::f32;
        refused(!q.open(d.sub("refused"), f32, tiny_limits(), kPlain, err), "f32 hidden rows are refused", err);
        check(q.take() == nullptr && !q.begin({1}, {}, err) && !q.end(err) && !q.close(err), "a queue that is not open does nothing");
    }
    {
        // rows out of order: the writer fails, and every later call reports it
        ShardQueue q;
        check(q.open(d.sub("order"), h, tiny_limits(), kPlain, err) && q.begin({1, 2, 3, 4, 5, 6, 7, 8}, {}, err), "open and begin", err);
        ShardQueue::Piece* p = q.take();
        fill_piece(*p, h, 1, 4, 4, {});
        check(q.put(p, err), "the piece is queued");
        refused(!q.close(err) && err.find("out of order") != std::string::npos, "close reports the rows out of order", err);
    }
    {
        // a selection in a piece that does not hold the row before its anchor
        ShardQueue q;
        check(q.open(d.sub("anchor"), h, tiny_limits(), kPlain, err) && q.begin({1, 2, 3, 4, 5, 6, 7, 8}, {6}, err), "open and begin", err);
        ShardQueue::Piece* p = q.take();
        fill_piece(*p, h, 1, 0, 4, {});
        p->n_sel = 1;
        p->sel_anchor[0] = 0;
        check(q.put(p, err), "the piece is queued");
        refused(!q.close(err) && err.find("row before anchor 6") != std::string::npos, "close reports the misplaced selection", err);
    }
    {
        // a sequence that ends short of its rows
        ShardQueue q;
        check(q.open(d.sub("short"), h, tiny_limits(), kPlain, err) && q.begin({1, 2, 3, 4, 5, 6, 7, 8}, {}, err), "open and begin", err);
        ShardQueue::Piece* p = q.take();
        fill_piece(*p, h, 1, 0, 4, {});
        check(q.put(p, err) && q.end(err), "queued");
        refused(!q.close(err) && err.find("missing") != std::string::npos, "close reports the missing rows", err);
        check(q.take() == nullptr, "a closed queue hands out no piece");
    }

    std::printf("5. the settings file\n");
    {
        const std::string dir5 = d.sub("settings");
        const std::map<std::string, std::string> settings = {{"seed", "42"}, {"window_rows", "512"}, {"pack_name", "a \"quoted\" name.basalt"}};
        check(manifest_check(dir5, settings, err) && fs::exists(dir5 + "/export.json"), "written when absent", err);
        check(manifest_check(dir5, settings, err), "the same settings are accepted", err);
        for (const auto& [key, value] : settings) {
            std::map<std::string, std::string> changed = settings;
            changed[key] = value + "0";
            refused(!manifest_check(dir5, changed, err) && err.find("\"" + key + "\"") != std::string::npos &&
                  err.find(value) != std::string::npos, "a changed " + key + " is refused with the stored value", err);
        }
        std::map<std::string, std::string> more = settings, fewer = settings;
        more["prefill_chunk"] = "2048";
        fewer.erase("seed");
        refused(!manifest_check(dir5, more, err) && err.find("prefill_chunk") != std::string::npos, "a setting the file lacks is refused", err);
        refused(!manifest_check(dir5, fewer, err) && err.find("seed") != std::string::npos, "a setting this run lacks is refused", err);
        check(!manifest_check(dir5, {}, err) && !manifest_check("", settings, err), "no settings or no directory");
        std::ofstream(dir5 + "/export.json") << "not json";
        refused(!manifest_check(dir5, settings, err), "a file that is not the settings is refused", err);
    }

    if (g_fail) {
        std::printf("dfly2_shard_queue_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("dfly2_shard_queue_test: ok\n");
    return 0;
}
