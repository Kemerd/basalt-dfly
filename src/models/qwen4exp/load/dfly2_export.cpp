// src/models/qwen4exp/load/dfly2_export.cpp - the teacher export (--dfly2-export): what the target computes for each
// document of a token file, written as the shards a DFly2 drafter is trained on (basalt/artifact/dfly2_shard.hpp).
// A document's first rows are read by the batched prompt path and its last ones through verify windows, as a served
// request's are. The compute thread fills page-locked pieces; the shard queue's thread repacks and writes them.
#include "basalt/dfly2/anchors.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/models/qwen4exp/kernels/rope_scaling.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"
#include "dfly2_shard_queue.hpp"

#include <cstring>

namespace basalt::models::qwen4exp::generate {

namespace {

constexpr uint64_t kSeed = 42;                             // the anchors' seed
constexpr int kPrefixEvery = 64;                           // one anchor per this many prompt-path rows
constexpr int kWindowEvery = 8;                            // one anchor per this many window rows
constexpr int kBlock = basalt::kernels::kVerifyMaxT;       // rows of a drafted block, and of a verify window
constexpr uint64_t kShardTokens = 262144;                  // a shard is closed once it holds this many tokens
constexpr int kPieces = 2;                                 // piece buffers in flight
constexpr int kPaired[] = {31, 35, 39, 43, 47};            // the target layers a drafter layer is paired with
constexpr int kNPaired = (int) (sizeof kPaired / sizeof kPaired[0]);
constexpr size_t kReadBytes = 1u << 20;                    // the documents file is read this much at a time

/// The index of `layer` in the paired layers, or -1.
int paired_index(int64_t layer) {
    for (int k = 0; k < kNPaired; ++k)
        if (kPaired[k] == layer) return k;
    return -1;
}

/// The last component of a path.
std::string leaf_name(const std::string& path) {
    const size_t at = path.find_last_of("/\\");
    return at == std::string::npos ? path : path.substr(at + 1);
}

/// "what: the CUDA error", clearing it.
std::string cuda_text(const char* what) { return std::string(what) + ": " + cudaGetErrorString(cudaGetLastError()); }

void* take_pinned(size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return p;
}
void give_pinned(void* p) { cudaFreeHost(p); }

/// The documents file, a line at a time: a document may be longer than any fixed buffer.
class DocsFile {
public:
    DocsFile() = default;
    DocsFile(const DocsFile&) = delete;
    DocsFile& operator=(const DocsFile&) = delete;
    ~DocsFile() {
        if (f_ != nullptr) std::fclose(f_);
    }
    bool open(const std::string& path) {
        f_ = basalt::platform::open_file_utf8(path, "rb");
        buf_.resize(kReadBytes);
        return f_ != nullptr;
    }
    /// The next line without its end; false at the end of the file, or when a read failed (`failed`).
    bool next(std::string& line) {
        line.clear();
        for (;;) {
            if (at_ == got_) {
                if (f_ == nullptr || done_) break;
                got_ = std::fread(buf_.data(), 1, buf_.size(), f_);
                at_ = 0;
                if (got_ < buf_.size()) {
                    done_ = true;
                    failed_ = std::ferror(f_) != 0;
                }
                if (got_ == 0) break;
            }
            const char* from = buf_.data() + at_;
            const char* nl = (const char*) std::memchr(from, '\n', got_ - at_);
            if (nl == nullptr) {   // the line goes on in the next read
                line.append(from, got_ - at_);
                at_ = got_;
                continue;
            }
            line.append(from, (size_t) (nl - from));
            at_ += (size_t) (nl - from) + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        // the last line may end without a newline
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return !failed_ && !line.empty();
    }
    bool failed() const { return failed_; }

private:
    std::FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t at_ = 0, got_ = 0;
    bool done_ = false, failed_ = false;
};

/// One run of the export: the engine's parts it reads with, the writer, and the document being read.
struct Export {
    Context& context;
    Options& o;
    const basalt::core::ModelGeometry& g;
    basalt::core::SessionState& ss;
    cudaStream_t cs;                     ///< the run's main stream: the prompt path's, and the export's own copies

    basalt::prefill::Prefill prefill;
    basalt::core::Verifier ver;
    ShardQueue queue;
    ShardQueue::Limits limits;
    bool windows = false;                ///< a document's last rows go through verify windows
    int64_t N = 0, cap = 0, page = 0;    ///< a hidden row's width, a selection's, a cache page's cells
    size_t page_values = 0;              ///< fp16 values of one cache page of K (or V)
    uint16_t* d_rows16 = nullptr;        ///< device: one window's head rows as fp16

    // the prompt loan: the cache slots that hold the prompt path's buffers while a document's first rows are read
    LentSlots lent;
    int64_t chunk_cap = 0;               ///< the largest chunk, the one `prefill` was initialized with
    bool loaned = false, lent_out = false;
    int64_t refills = 0;
    double ms_refill = 0;

    // the document being read
    int64_t n = 0, m = 0;                ///< its rows, and the first one the windows read
    std::vector<int32_t> anchors;
    bool writing = false;                ///< its pieces go to the writer (else they are only digested)
    bool digesting = false;
    uint64_t digest = 0;
    ShardQueue::Piece* cur = nullptr;    ///< the piece being filled
    uint32_t cur_layers = 0;             ///< the paired layers whose selections `cur` has

    int64_t rows_prompt = 0, rows_window = 0;
    double ms_prompt = 0, ms_window = 0;

    ~Export() {
        if (d_rows16 != nullptr) cudaFree(d_rows16);
    }

    /// Takes a free piece for rows [row0, row0 + rows) and lists the anchors whose row before lies in them.
    bool piece_open(int64_t row0, int64_t rows, std::string& e) {
        if (row0 < 0 || rows < 1 || rows > limits.rows || row0 + rows > n) {
            e = "rows " + std::to_string(row0) + "+" + std::to_string(rows) + " are not a piece of the document";
            return false;
        }
        cur = queue.take();
        if (cur == nullptr) {   // the writer failed: its first failure is the reason
            if (queue.close(e)) e = "the export's writer stopped";
            return false;
        }
        cur->row0 = row0;
        cur->rows = rows;
        cur->page0 = row0 / page;
        cur->pages = (row0 + rows - 1) / page - cur->page0 + 1;
        cur->n_sel = 0;
        cur_layers = 0;
        size_t i = (size_t) (std::lower_bound(anchors.begin(), anchors.end(), (int32_t) (row0 + 1)) - anchors.begin());
        for (; i < anchors.size() && anchors[i] - 1 < row0 + rows; ++i) {
            if (cur->n_sel >= limits.anchors) {
                e = "a piece of " + std::to_string(rows) + " rows holds more anchors than its selection area";
                return false;
            }
            cur->sel_anchor[cur->n_sel++] = (int32_t) i;
        }
        return true;
    }

    /// Queues the copies of paired layer `k`'s selections for the piece's anchors whose row before is one of the
    /// `rows` rows from `row0` that `ids` holds (device, `cap` ids a row).
    bool copy_selections(int k, const int32_t* ids, int64_t row0, int64_t rows, std::string& e) {
        for (int i = 0; i < cur->n_sel; ++i) {
            const int64_t a = anchors[(size_t) cur->sel_anchor[i]];
            if (a - 1 < row0 || a - 1 >= row0 + rows) continue;
            // row a - 1 selects among a cells
            const size_t width = (size_t) std::min<int64_t>(a, cap);
            if (cudaMemcpyAsync(cur->sel + ((size_t) i * kNPaired + (size_t) k) * (size_t) cap, ids + (a - 1 - row0) * cap,
                                width * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                e = cuda_text("the export's copy of a selection");
                return false;
            }
        }
        return true;
    }

    /// Queues the copies of the cache pages the piece's rows lie in, as the cache holds them, for every paired layer.
    bool copy_cells(std::string& e) {
        const size_t at = (size_t) cur->page0 * page_values, bytes = (size_t) cur->pages * page_values * sizeof(uint16_t);
        for (int k = 0; k < kNPaired; ++k) {
            const basalt::core::QsaState& st = ss.qsa_states[kPaired[k] / g.qsa_interval];
            if (cur->page0 + cur->pages > st.n_pages) {
                e = "rows past the cache of layer " + std::to_string(kPaired[k]);
                return false;
            }
            if (cudaMemcpyAsync(cur->k[k], st.k_pool + at, bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                cudaMemcpyAsync(cur->v[k], st.v_pool + at, bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                e = cuda_text("the export's copy of a layer's cells");
                return false;
            }
        }
        return true;
    }

    /// Adds the piece's rows, its own cells and its selections to the document's digest.
    void digest_piece(const ShardQueue::Piece& p) {
        const auto add = [&](const void* at, size_t bytes) {
            digest = basalt::core::fnv1a64((const uint8_t*) at, (uint64_t) bytes, digest);
        };
        add(&p.row0, sizeof p.row0);
        add(&p.rows, sizeof p.rows);
        add(p.hidden, (size_t) p.rows * (size_t) N * sizeof(uint16_t));
        const size_t heads = (size_t) g.n_head_kv, dim = (size_t) g.head_dim;
        for (int k = 0; k < kNPaired; ++k)
            for (int64_t r = 0; r < p.rows; ++r) {
                const size_t cell = (size_t) (p.row0 + r), pg = cell / (size_t) page - (size_t) p.page0;
                for (size_t h = 0; h < heads; ++h) {
                    const size_t from = ((pg * heads + h) * (size_t) page + cell % (size_t) page) * dim;
                    add(p.k[k] + from, dim * sizeof(uint16_t));
                    add(p.v[k] + from, dim * sizeof(uint16_t));
                }
            }
        for (int i = 0; i < p.n_sel; ++i) {
            const int64_t a = anchors[(size_t) p.sel_anchor[i]];
            add(&a, sizeof a);
            for (int k = 0; k < kNPaired; ++k)
                add(p.sel + ((size_t) i * kNPaired + (size_t) k) * (size_t) cap,
                    (size_t) std::min<int64_t>(a, cap) * sizeof(int32_t));
        }
    }

    /// Hands the filled piece to the writer, or back to the queue when the document is not written.
    bool piece_done(std::string& e) {
        ShardQueue::Piece* p = cur;
        cur = nullptr;
        if (digesting) digest_piece(*p);
        if (!writing) {
            queue.give_back(p);
            return true;
        }
        return queue.put(p, e);
    }

    /// Prefill::on_selection: a paired layer's selections of the chunk's rows before an anchor, queued for the host.
    bool on_selection(int64_t layer, int64_t p0, int64_t T, const int32_t* ids, int64_t ids_cap, std::string& e) {
        const int k = paired_index(layer);
        if (k < 0) return true;
        if (ids == nullptr || ids_cap != cap) {
            e = "the prompt path selects " + std::to_string(ids_cap) + " cells a row, the export writes " + std::to_string(cap);
            return false;
        }
        if (cur == nullptr && !piece_open(p0, T, e)) return false;
        if (cur->row0 != p0 || cur->rows != T) {
            e = "the prompt path reported a layer of another chunk than the one being exported";
            return false;
        }
        cur_layers |= 1u << k;
        return copy_selections(k, ids, p0, T, e);
    }

    /// Prefill::on_chunk: the chunk's rows are final. Its cells and head rows complete the piece.
    bool on_chunk(const float* R_rows, int64_t T, int64_t p0, std::string& e) {
        if (cur == nullptr && !piece_open(p0, T, e)) return false;
        if (cur->row0 != p0 || cur->rows != T) {
            e = "the prompt path reported another chunk than the one being exported";
            return false;
        }
        if (cur->n_sel > 0 && cur_layers != (1u << kNPaired) - 1) {
            e = "the prompt path did not report every paired layer's selections";
            return false;
        }
        // the head's rows last: its wait covers the copies queued before it
        if (!copy_cells(e) || !prefill.head_input(R_rows, T, cur->hidden, e)) return false;
        rows_prompt += T;
        return piece_done(e);
    }

    /// Rows [m, n) of `doc` through verify windows, every row kept, a piece at a time.
    bool read_windows(const std::vector<int64_t>& doc, std::string& e) {
        auto& drive = *context.drive;
        int32_t win[kBlock] = {}, picks[kBlock] = {};
        const int64_t piece_rows = limits.rows / kBlock * kBlock;   // whole windows, so a piece never cuts one
        for (int64_t q = m; q < n;) {
            const int64_t rows = std::min(piece_rows, n - q);
            if (!piece_open(q, rows, e)) return false;
            for (int64_t w = q; w < q + rows;) {
                const int T = (int) std::min<int64_t>(kBlock, q + rows - w);
                for (int t = 0; t < T; ++t) win[t] = (int32_t) doc[(size_t) (w + t)];
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                if (!ver.run(T, win, w, &drive_pool_multi, &drive, picks, e) || drive.d.failed) {
                    if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                    return false;
                }
                if (!ver.commit(T, e)) return false;
                // the window's head rows, rounded on the device, and its rows' selections before an anchor
                basalt::kernels::f32_to_f16_bulk(ver.head_rows(), d_rows16, (int64_t) T * N, cs);
                if (cudaMemcpyAsync(cur->hidden + (size_t) (w - q) * (size_t) N, d_rows16,
                                    (size_t) T * (size_t) N * sizeof(uint16_t), cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                    e = cuda_text("the export's copy of a window's rows");
                    return false;
                }
                for (int k = 0; k < kNPaired; ++k)
                    if (!copy_selections(k, ver.selection(k), w, T, e)) return false;
                if (cudaStreamSynchronize(cs) != cudaSuccess) {
                    e = cuda_text("the export's copies of a window");
                    return false;
                }
                w += T;
            }
            // the windows wrote their cells into the cache: read once the last commit has landed
            if (!ver.wait_commit(e) || !copy_cells(e)) return false;
            if (cudaStreamSynchronize(cs) != cudaSuccess) {
                e = cuda_text("the export's copy of the windows' cells");
                return false;
            }
            rows_window += rows;
            if (!piece_done(e)) return false;
            q += rows;
        }
        return true;
    }

    /// The loan for a document whose first `rows` rows the prompt path reads: no more slots than its chunk needs.
    bool lend(int64_t rows, std::string& e) {
        if (!loaned) return true;
        const int64_t want = serve_loans::request_chunk(rows, chunk_cap);
        if (lent_out && want == prefill.chunk()) return true;   // the loan in place covers it
        if (lent_out && !refill(e)) return false;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        lent.clear();
        lend_cache_tail(context, serve_loans::lend_slots(*context.xcache, g, ss, want), lent, borrow, borrow_bytes);
        lent_out = true;
        return prefill.relayout(want, borrow, borrow_bytes, e);
    }

    /// Gives the lent slots back to the cache, their experts read again.
    bool refill(std::string& e) {
        double ms = 0;
        if (refill_lent(context, lent, ms) != 0) {
            e = "refilling the cache slots lent to the prompt path failed";
            return false;
        }
        lent_out = false;
        ++refills;
        ms_refill += ms;
        return true;
    }

    /// Reads `doc` (sequence `line` of the export) from an empty sequence; `write` hands its pieces to the writer.
    bool read_document(const std::vector<int64_t>& doc, uint64_t line, bool write, std::string& e) {
        n = (int64_t) doc.size();
        m = basalt::dfly2::window_from(n, windows ? o.dfly2_window_rows : 0);
        // prefix anchors follow a prompt-path row, window anchors a window row; every block is whole
        anchors = basalt::dfly2::sample_anchors(kSeed, line, 1, std::min(m, n - kBlock), kPrefixEvery);
        const std::vector<int32_t> tail = basalt::dfly2::sample_anchors(kSeed, line, m + 1, n - kBlock, kWindowEvery);
        anchors.insert(anchors.end(), tail.begin(), tail.end());
        writing = write;
        if (write) {
            std::vector<int32_t> ids(doc.size());
            for (size_t i = 0; i < doc.size(); ++i) ids[i] = (int32_t) doc[i];
            if (!queue.begin(std::move(ids), anchors, e)) return false;
        }

        // an empty sequence: the last document's commit has landed, then every layer's state is zero
        if (windows && !ver.wait_commit(e)) return false;
        basalt::core::session_zero(ss, g, nullptr, cs);
        if (cudaStreamSynchronize(cs) != cudaSuccess) {
            e = cuda_text("zeroing the session state");
            return false;
        }
        ss.ple_prev[0] = ss.ple_prev[1] = -1;

        if (!lend(m, e)) return false;
        const Clock::time_point t_prompt = Clock::now();
        if (!prefill.run(doc.data(), m, 0, e)) return false;
        ms_prompt += std::chrono::duration<double, std::milli>(Clock::now() - t_prompt).count();
        if (n > m) {
            // the windows read the cache's slots: the lent ones hold their experts again first
            if (lent_out && !refill(e)) return false;
            ss.ple_prev[0] = m >= 2 ? (int32_t) doc[(size_t) (m - 2)] : -1;
            ss.ple_prev[1] = (int32_t) doc[(size_t) (m - 1)];
            const Clock::time_point t_window = Clock::now();
            if (!read_windows(doc, e)) return false;
            ms_window += std::chrono::duration<double, std::milli>(Clock::now() - t_window).count();
        }
        return !write || queue.end(e);
    }
};

/// Rows a second, 0 when no time was measured.
double rate(int64_t rows, double ms) { return ms > 0.0 ? 1000.0 * (double) rows / ms : 0.0; }

}  // namespace

int dfly2_export(Context& context) {
    auto& o = *context.o;
    auto& g = *context.g;
    auto& ss = *context.ss;
    auto& err = *context.err;
    const int64_t n_vocab = *context.n_vocab;
    const auto refuse = [](const std::string& why) {
        basalt::log::error("basalt generate", "--dfly2-export %s", why.c_str());
        return 2;
    };

    // ---- what the export cannot read
    if (*context.multi_gpu || *context.split_same)
        return refuse("reads a document on one card's layers: not with gpu-mode split");
    const basalt::kernels::RopeScaling& rope = basalt::kernels::rope_scaling();
    if (rope.type != basalt::kernels::RopeScalingType::None || rope.freq_scale() != 1.0 || rope.mscale() != 1.0)
        return refuse("exports cells with the model's plain rotation: not with a rope scaling");
    if (o.prefill_chunk <= 0) return refuse("reads a document's first rows through the batched prompt path: not with --prefill 0");
    const bool windows = o.dfly2_window_rows > 0;
    if (windows && context.thits->d_res == nullptr)
        return refuse("--dfly2-window-rows needs the device residency table (an expert profile, --expert-cache and the "
                      "token graph); 0 reads every row through the prompt path");
    basalt::kernels::QsaShapes shapes = basalt::kernels::qsa_real_shapes();
    shapes.n_head_kv = g.n_head_kv;
    shapes.head_dim = g.head_dim;
    for (int k = 0; k < kNPaired; ++k) {
        const int64_t ord = kPaired[k] / g.qsa_interval;
        if (kPaired[k] >= g.n_layers || !basalt::core::is_qsa_layer(g, kPaired[k]) || ord < ss.qsa_ord0 ||
            ord >= ss.qsa_ord0 + ss.qsa_alloc)
            return refuse("pairs layer " + std::to_string(kPaired[k]) + ", which is not an attention layer of this model");
        const basalt::core::QsaState& st = ss.qsa_states[ord];
        if (st.k_pool == nullptr || st.v_pool == nullptr || st.kv_mode != 0 || st.kv_int8 || st.kv_q4 || st.kv_hybrid ||
            st.kv_rot || st.kv_arena != nullptr || st.max_cells < o.max_context)
            return refuse("reads whole fp16 cells from VRAM, which layer " + std::to_string(kPaired[k]) +
                          "'s cache does not hold (--kv fp16, no --kv-resident, no elastic KV)");
    }
    const basalt::core::ModelFile::Section* meta = context.model_file->find("meta");
    uint64_t docs_bytes = 0;
    DocsFile docs;
    if (meta == nullptr) return refuse("needs the model file's meta section");
    if (!basalt::platform::file_size(o.dfly2_docs, docs_bytes) || !docs.open(o.dfly2_docs))
        return refuse("cannot read --dfly2-docs " + o.dfly2_docs);

    Export x{context, o, g, ss, (cudaStream_t) *context.main_cs};
    x.windows = windows;
    x.N = g.n_embd;
    x.page = shapes.page_size;
    x.cap = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, shapes);
    x.page_values = (size_t) (g.n_head_kv * shapes.page_size * g.head_dim);

    // ---- the windows, as the one-shot loop sets them up, with the paired layers' selections kept
    if (windows) {
        x.ver.keep_selections(std::vector<int>(kPaired, kPaired + kNPaired));
        if (const int rc = window_init(context, x.ver, kBlock)) return rc;
        x.ver.set_head_sampling(false);   // every row is kept and no pick is read
        if (x.ver.selection_cap() != x.cap || x.ver.selection(kNPaired - 1) == nullptr)
            return refuse("and the verify window disagree on a selection's width");
        if (cudaMalloc((void**) &x.d_rows16, (size_t) kBlock * (size_t) x.N * sizeof(uint16_t)) != cudaSuccess)
            return fatal_start_error("--dfly2-export: no room for a window's rows" + vram_free_note());
    }

    // ---- the prompt path, with the loan of the longest document
    int64_t loan_slots = 0;
    if (!prompt_path_init(context, x.prefill, o.max_context, x.lent, err, &loan_slots)) return fatal_start_error(err);
    x.chunk_cap = x.prefill.chunk();
    x.loaned = x.lent_out = loan_slots > 0;
    if (x.chunk_cap < kBlock) return refuse("needs prompt chunks of " + std::to_string(kBlock) + " tokens or more");
    x.prefill.on_selection = [&x](int64_t layer, int64_t p0, int64_t T, const int32_t* ids, int64_t ids_cap, std::string& e) {
        return x.on_selection(layer, p0, T, ids, ids_cap, e);
    };
    x.prefill.on_chunk = [&x](const float* R_rows, int64_t T, int64_t p0, std::string& e) {
        return x.on_chunk(R_rows, T, p0, e);
    };

    // ---- the shards' header, and the settings a restart must repeat
    const std::string pack_name = leaf_name(o.model);
    basalt::dfly2::ShardHeader header{};
    header.hidden = (uint32_t) g.n_embd;
    header.vocab = (uint32_t) n_vocab;
    header.n_head_kv = (uint32_t) g.n_head_kv;
    header.head_dim = (uint32_t) g.head_dim;
    header.n_layers = (uint32_t) kNPaired;
    header.sel_cap = (uint32_t) x.cap;
    header.n_rot = (uint32_t) shapes.n_rot;
    header.hidden_dtype = header.kv_dtype = (uint32_t) basalt::dfly2::ShardDtype::f16;
    header.rope_base = rope.freq_base;
    for (int k = 0; k < kNPaired; ++k) header.layers[k] = (uint32_t) kPaired[k];
    header.pack_bytes = context.model_file->file_bytes();
    std::memcpy(header.pack_meta_sha256, meta->sha256, sizeof header.pack_meta_sha256);
    std::memcpy(header.pack_name, pack_name.data(), std::min(pack_name.size(), sizeof header.pack_name - 1));
    char text[80];
    std::string meta_hex;
    for (const uint8_t b : meta->sha256) {
        std::snprintf(text, sizeof text, "%02x", (unsigned) b);
        meta_hex += text;
    }
    std::map<std::string, std::string> settings;
    settings["pack_name"] = pack_name;
    settings["pack_bytes"] = std::to_string(header.pack_bytes);
    settings["pack_meta_sha256"] = meta_hex;
    settings["docs_name"] = leaf_name(o.dfly2_docs);
    settings["docs_bytes"] = std::to_string(docs_bytes);
    settings["seed"] = std::to_string(kSeed);
    settings["prefix_every"] = std::to_string(kPrefixEvery);
    settings["window_every"] = std::to_string(kWindowEvery);
    settings["block"] = std::to_string(kBlock);
    settings["window_rows"] = std::to_string(o.dfly2_window_rows);
    settings["max_context"] = std::to_string(o.max_context);
    settings["prefill_chunk"] = std::to_string(x.chunk_cap);
    std::snprintf(text, sizeof text, "%g", o.batch <= 0 && !o.no_pool ? o.prefill_cpu_share : 0.0);
    settings["prefill_cpu_share"] = text;
    settings["expert_cache_slots"] = std::to_string(context.xcache->slots());
    settings["kv"] = "fp16";
    std::snprintf(text, sizeof text, "%.17g", rope.freq_base);
    settings["rope_base"] = text;
    const auto failed = [](const std::string& why) {
        basalt::log::error("basalt generate", "--dfly2-export: %s", why.c_str());
        return 1;
    };
    if (!manifest_check(o.dfly2_export, settings, err)) return failed(err);

    // ---- the writer: a piece holds a chunk's rows, or as many window rows
    x.limits.rows = x.chunk_cap;
    x.limits.anchors = std::max(x.chunk_cap / kPrefixEvery, std::min(o.dfly2_window_rows, x.chunk_cap) / kWindowEvery) + 2;
    x.limits.page = x.page;
    x.limits.pieces = kPieces;
    x.limits.shard_tokens = kShardTokens;
    if (!x.queue.open(o.dfly2_export, header, x.limits, ShardQueue::Memory{&take_pinned, &give_pinned}, err)) return failed(err);
    const uint64_t skip = x.queue.sequences_done();
    basalt::log::info("basalt generate", "--dfly2-export: %s, prompt chunks of up to %lld tokens, the last %lld rows of a "
                         "document through verify windows%s",
                 o.dfly2_export.c_str(), (long long) x.chunk_cap, (long long) (windows ? o.dfly2_window_rows : 0),
                 skip > 0 ? (", continuing after " + std::to_string(skip) + " documents").c_str() : "");

    // ---- the documents, a line each; sequence k of the export is line k
    std::string line_text, parse_err;
    std::vector<int64_t> doc, first_doc;
    uint64_t line = 0, first_line = 0, first_digest = 0;
    int64_t documents = 0, tokens = 0, truncated = 0;
    while (docs.next(line_text)) {
        const uint64_t at = line++;
        if (at < skip) continue;   // written by an earlier run
        if (line_text.find('\0') != std::string::npos || !cli::parse_i64_list(line_text.c_str(), doc, parse_err))
            return failed(o.dfly2_docs + " line " + std::to_string(at + 1) + ": " +
                          (parse_err.empty() ? "not a list of token ids" : parse_err));
        for (const int64_t id : doc)
            if (id >= n_vocab)
                return failed(o.dfly2_docs + " line " + std::to_string(at + 1) + ": token " + std::to_string(id) +
                              " is not in the vocabulary of " + std::to_string(n_vocab));
        if ((int64_t) doc.size() > o.max_context) {
            doc.resize((size_t) o.max_context);
            ++truncated;
        }
        // the run's first document is digested: it is read once more at the end
        x.digesting = documents == 0;
        x.digest = 0;
        if (!x.read_document(doc, at, true, err)) return failed("line " + std::to_string(at + 1) + ": " + err);
        if (documents == 0) {
            first_doc = doc;
            first_line = at;
            first_digest = x.digest;
            basalt::log::info("basalt generate", "--dfly2-export: the first document's %lld prompt-path rows at %.0f rows/s, "
                                 "its %lld window rows at %.1f rows/s",
                         (long long) x.rows_prompt, rate(x.rows_prompt, x.ms_prompt), (long long) x.rows_window,
                         rate(x.rows_window, x.ms_window));
        }
        ++documents;
        tokens += (int64_t) doc.size();
    }
    if (docs.failed()) return failed("reading " + o.dfly2_docs + " failed");
    if (line < skip)
        return failed(o.dfly2_docs + " has " + std::to_string(line) + " documents, the directory's shards hold " +
                      std::to_string(skip));
    const int64_t rows_prompt = x.rows_prompt, rows_window = x.rows_window;
    const double ms_prompt = x.ms_prompt, ms_window = x.ms_window;

    // ---- the first document again, from the state the last one left: when both readings give the same rows, cells
    // and selections, the reset between documents leaves nothing behind
    if (documents > 0) {
        x.digesting = true;
        x.digest = 0;
        if (!x.read_document(first_doc, first_line, false, err)) return failed("reading the first document again: " + err);
        if (x.digest == first_digest)
            basalt::log::info("basalt generate", "--dfly2-export: the first document read again after the last gave the "
                                 "same rows, cells and selections");
        else
            basalt::log::warn("basalt generate", "--dfly2-export: the first document (line %llu) read again after the last "
                                 "gave other rows, cells or selections: a document's result depends on the ones read "
                                 "before it (the session's reset, or the expert tiers)",
                         (unsigned long long) (first_line + 1));
    }
    if (!x.queue.close(err)) return failed(err);
    basalt::log::info("basalt generate", "--dfly2-export: %lld documents, %lld tokens (%lld documents cut to %lld tokens); "
                         "%lld prompt-path rows at %.0f rows/s, %lld window rows at %.1f rows/s; the lent cache slots "
                         "refilled %lld times in %.0f ms",
                 (long long) documents, (long long) tokens, (long long) truncated, (long long) o.max_context,
                 (long long) rows_prompt, rate(rows_prompt, ms_prompt), (long long) rows_window, rate(rows_window, ms_window),
                 (long long) x.refills, x.ms_refill);
    return 0;
}

}  // namespace basalt::models::qwen4exp::generate
