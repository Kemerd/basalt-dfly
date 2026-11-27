// src/models/qwen4exp/load/dfly2_export.cpp - the teacher export (--dfly2-export): what the target computes for each
// document of a token file, written as the shards a DFly2 drafter is trained on (basalt/artifact/dfly2_shard.hpp).
// A document's first rows are read by the batched prompt path and its last ones through verify windows, as a served
// request's are. The compute thread fills page-locked pieces; the shard queue's thread repacks and writes them.
#include "basalt/core/diag.hpp"
#include "basalt/dfly2/anchors.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/kernels/rope_scaling.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"
#include "dfly2_docs.hpp"
#include "dfly2_shard_queue.hpp"

#include <cmath>
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
// --exp dfly2_export_check=N: every Nth document's window rows are also read by the prompt path, and the two compared
constexpr int64_t kCheckContext[] = {2051, 8192, 32768};   // rows are summed by cells of context: up to each, and above
constexpr int kNCheckContext = (int) (sizeof kCheckContext / sizeof kCheckContext[0]) + 1;
constexpr int64_t kCheckGather = 2048;                     // cells the check gathers from the cache at a time
// prefix anchors stand for window anchors past 2,051 cells when no layer differs in more of its selected cells than
// kGateDiffer and the head rows' mean cosine against the windows' is at least kGateCosine
constexpr double kGateCosine = 0.999;
constexpr double kGateDiffer = 0.01;
// the prompt path's head rows against the decode's mix of the same residual rows differ by rounding only: a row whose
// cosine is under this is a wrong mix
constexpr double kMixCosine = 0.9999;
constexpr int64_t kMixRun = 256;                           // rows the check mixes the decode's way per copy

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

/// What one path computed for a document's last rows, on the host.
struct TailRows {
    int64_t row0 = 0, rows = 0;
    std::vector<uint16_t> hidden;                    ///< [rows][hidden]
    std::vector<uint16_t> k[kNPaired], v[kNPaired];  ///< per paired layer [rows][n_head_kv][head_dim]
    std::vector<int32_t> sel[kNPaired];              ///< per paired layer [rows][cap]: every row's selection
};

/// The check's sums over the rows of one range of context lengths: the prompt path's rows against the windows'.
struct CheckSums {
    int64_t rows = 0;
    double cos_sum = 0, cos_min = 1, row_err_max = 0;           ///< the head's rows
    int64_t off_rows = 0;
    double off_cos_sum = 0;                                     ///< a prompt-path head row against the window row before
    int64_t selected[kNPaired] = {}, differ[kNPaired] = {};     ///< selected cells, and those only one path selected
    double k_err_sum[kNPaired] = {}, k_err_max[kNPaired] = {};  ///< a cell's relative error
    double v_err_sum[kNPaired] = {}, v_err_max[kNPaired] = {};
};

/// |a - b| / |b| over `n` fp16 values, and their cosine when `cosine` is given.
double f16_error(const uint16_t* a, const uint16_t* b, size_t n, double* cosine = nullptr) {
    double dd = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = basalt::kernels::f32_from_f16(a[i]), y = basalt::kernels::f32_from_f16(b[i]);
        dd += (x - y) * (x - y);
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if (cosine != nullptr) *cosine = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : dd > 0 ? 0.0 : 1.0;
    return bb > 0 ? std::sqrt(dd / bb) : dd > 0 ? 1.0 : 0.0;
}

/// The anchors of document `line` of `n` rows whose windows start at row `m`: a prefix anchor follows a prompt-path
/// row, a window anchor a window row, and every anchor has a whole block after it.
std::vector<int32_t> document_anchors(uint64_t line, int64_t n, int64_t m) {
    std::vector<int32_t> anchors = basalt::dfly2::sample_anchors(kSeed, line, 1, std::min(m, n - kBlock), kPrefixEvery);
    const std::vector<int32_t> tail = basalt::dfly2::sample_anchors(kSeed, line, m + 1, n - kBlock, kWindowEvery);
    anchors.insert(anchors.end(), tail.begin(), tail.end());
    return anchors;
}

/// A document's rows, cells and selections as one number: each array is summed in the order a shard holds it, so
/// the pieces a document arrives in do not change it and a sequence read back gives the same one.
struct DocDigest {
    uint64_t hidden, sel, k[kNPaired], v[kNPaired];
    DocDigest() { reset(); }
    void reset() {
        hidden = sel = basalt::core::fnv1a64(nullptr, 0);
        for (int l = 0; l < kNPaired; ++l) k[l] = v[l] = hidden;
    }
    static void add(uint64_t& sum, const void* at, size_t bytes) {
        sum = basalt::core::fnv1a64((const uint8_t*) at, (uint64_t) bytes, sum);
    }
    /// One anchor's selection in one paired layer: its first `count` ids.
    void selection(int64_t anchor, const int32_t* ids, size_t count) {
        add(sel, &anchor, sizeof anchor);
        add(sel, ids, count * sizeof(int32_t));
    }
    uint64_t value() const {
        uint64_t sum = hidden;
        add(sum, &sel, sizeof sel);
        add(sum, k, sizeof k);
        add(sum, v, sizeof v);
        return sum;
    }
    /// The digest of a sequence as a shard holds it.
    static uint64_t of(const basalt::dfly2::ShardHeader& h, const basalt::dfly2::SequenceData& d) {
        DocDigest sum;
        add(sum.hidden, d.hidden.data(), d.hidden.size());
        for (uint32_t l = 0; l < h.n_layers && l < (uint32_t) kNPaired; ++l) {
            add(sum.k[l], d.k[l].data(), d.k[l].size() * sizeof(uint16_t));
            add(sum.v[l], d.v[l].data(), d.v[l].size() * sizeof(uint16_t));
        }
        for (size_t a = 0; a < d.anchors.size(); ++a)
            for (uint32_t l = 0; l < h.n_layers; ++l) {
                const size_t slot = a * h.n_layers + l;
                sum.selection(d.anchors[a], d.selection.data() + slot * h.sel_cap, (size_t) d.sel_count[slot]);
            }
        return sum.value();
    }
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
    DocDigest digest;
    ShardQueue::Piece* cur = nullptr;    ///< the piece being filled
    uint32_t cur_layers = 0;             ///< the paired layers whose selections `cur` has

    int64_t rows_prompt = 0, rows_window = 0;
    double ms_prompt = 0, ms_window = 0;

    // the check: a document's window rows by both paths, and every piece's cells against the cache's own gather
    basalt::kernels::QsaShapes shapes;
    bool checking = false;               ///< the document being read is a checked one
    TailRows* keep = nullptr;            ///< where the rows being read are kept, when they are
    TailRows by_prompt, by_window;
    CheckSums sums[kNCheckContext];
    int64_t check_documents = 0, check_pieces = 0, check_runs_differ = 0;
    std::vector<uint16_t> cells_k, cells_v;      ///< a piece's cells of one layer, as the writer repacks them
    std::vector<uint16_t> gather_k, gather_v;    ///< one gathered run
    std::vector<int32_t> gather_ids;
    int32_t* d_gather_ids = nullptr;
    uint16_t *d_gather_k = nullptr, *d_gather_v = nullptr;
    // the prompt path's head rows against the decode's mix (lm_head_mix) of the same residual rows
    basalt::core::BlockBuffers mix_bb;           ///< only R, mixed, inject and gr are set: what lm_head_mix reads
    void* d_mix_ws = nullptr;                    ///< device: the mix's workspace and injection
    float* d_mix = nullptr;                      ///< device: [kMixRun][N]
    uint16_t* d_mix16 = nullptr;
    std::vector<uint16_t> mix16;
    int64_t mix_rows = 0;
    double mix_cos_sum = 0, mix_cos_min = 1, mix_err_max = 0;

    ~Export() {
        if (d_rows16 != nullptr) cudaFree(d_rows16);
        if (d_mix_ws != nullptr) cudaFree(d_mix_ws);
        if (d_mix != nullptr) cudaFree(d_mix);
        if (d_mix16 != nullptr) cudaFree(d_mix16);
        if (d_gather_ids != nullptr) cudaFree(d_gather_ids);
        if (d_gather_k != nullptr) cudaFree(d_gather_k);
        if (d_gather_v != nullptr) cudaFree(d_gather_v);
    }

    /// The check's buffers: a piece's cells on the host, and one gathered run on the device.
    bool check_init(std::string& e) {
        const size_t cell = (size_t) (g.n_head_kv * g.head_dim), run = (size_t) kCheckGather * cell;
        cells_k.resize((size_t) limits.rows * cell);
        cells_v.resize((size_t) limits.rows * cell);
        gather_k.resize(run);
        gather_v.resize(run);
        gather_ids.resize((size_t) kCheckGather);
        if (cudaMalloc((void**) &d_gather_ids, (size_t) kCheckGather * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_gather_k, run * sizeof(uint16_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_gather_v, run * sizeof(uint16_t)) != cudaSuccess) {
            e = cuda_text("the check's gather buffers");
            return false;
        }
        // the decode's mix: its workspace, then the injection it may write, 16-byte aligned
        const basalt::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
        const size_t ws_bytes = (basalt::kernels::gr_workspace_bytes(gs) + 15) / 16 * 16;
        if (cudaMalloc(&d_mix_ws, ws_bytes + (size_t) g.hc * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_mix, (size_t) kMixRun * (size_t) N * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_mix16, (size_t) kMixRun * (size_t) N * sizeof(uint16_t)) != cudaSuccess) {
            e = cuda_text("the check's mix buffers");
            return false;
        }
        basalt::kernels::gr_workspace_init(gs, d_mix_ws, mix_bb.gr);
        mix_bb.inject = (float*) ((char*) d_mix_ws + ws_bytes);
        mix16.resize((size_t) kMixRun * (size_t) N);
        return true;
    }

    /// The `T` head rows `hidden` the prompt path made from residual rows `R_rows` (device, [T][hc][n_embd]) against
    /// the decode's mix of the same rows, rounded to fp16 as a window's are.
    bool check_mix(const float* R_rows, int64_t T, const uint16_t* hidden, std::string& e) {
        if (context.wt == nullptr || d_mix == nullptr) {
            e = "the check's mix has no weights or buffers";
            return false;
        }
        const size_t D = (size_t) (g.hc * g.n_embd);
        for (int64_t r0 = 0; r0 < T; r0 += kMixRun) {
            const int64_t run = std::min(kMixRun, T - r0);
            for (int64_t i = 0; i < run; ++i) {
                mix_bb.R = const_cast<float*>(R_rows + (size_t) (r0 + i) * D);
                mix_bb.mixed = d_mix + (size_t) i * (size_t) N;
                if (!basalt::core::lm_head_mix(*context.wt, g, mix_bb, cs, e)) return false;
            }
            basalt::kernels::f32_to_f16_bulk(d_mix, d_mix16, run * N, cs);
            if (cudaMemcpyAsync(mix16.data(), d_mix16, (size_t) run * (size_t) N * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                cs) != cudaSuccess ||
                cudaStreamSynchronize(cs) != cudaSuccess) {
                e = cuda_text("the check's mix of a chunk's rows");
                return false;
            }
            for (int64_t i = 0; i < run; ++i) {
                double cosine = 1;
                const double err = f16_error(hidden + (size_t) (r0 + i) * (size_t) N, mix16.data() + (size_t) i * (size_t) N,
                                             (size_t) N, &cosine);
                ++mix_rows;
                mix_cos_sum += cosine;
                mix_cos_min = std::min(mix_cos_min, cosine);
                mix_err_max = std::max(mix_err_max, err);
            }
        }
        return true;
    }

    /// Keeps rows [row0, row0 + rows) of the document read next in `to`.
    void keep_rows(TailRows& to, int64_t row0, int64_t rows) {
        const size_t cell = (size_t) (g.n_head_kv * g.head_dim);
        to.row0 = row0;
        to.rows = rows;
        to.hidden.resize((size_t) rows * (size_t) N);
        for (int k = 0; k < kNPaired; ++k) {
            to.k[k].resize((size_t) rows * cell);
            to.v[k].resize((size_t) rows * cell);
            to.sel[k].resize((size_t) rows * (size_t) cap);
        }
        keep = &to;
    }

    /// Copies the kept rows' whole selections of paired layer `k` among the `rows` rows from `row0` that `ids` holds.
    bool keep_selections(int k, const int32_t* ids, int64_t row0, int64_t rows, std::string& e) {
        const int64_t lo = std::max(row0, keep->row0), hi = std::min(row0 + rows, keep->row0 + keep->rows);
        if (lo >= hi) return true;
        if (cudaMemcpyAsync(keep->sel[k].data() + (size_t) (lo - keep->row0) * (size_t) cap, ids + (lo - row0) * cap,
                            (size_t) (hi - lo) * (size_t) cap * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            e = cuda_text("the check's copy of a layer's selections");
            return false;
        }
        return true;
    }

    /// A filled piece of a checked document: its cells as the writer repacks them against the cache's own gather of
    /// the same cells, and its kept rows.
    bool check_piece(const ShardQueue::Piece& p, std::string& e) {
        const size_t cell = (size_t) (g.n_head_kv * g.head_dim);
        const int64_t lo = keep != nullptr ? std::max(p.row0, keep->row0) : 0;
        const int64_t hi = keep != nullptr ? std::min(p.row0 + p.rows, keep->row0 + keep->rows) : 0;
        for (int k = 0; k < kNPaired; ++k) {
            const basalt::core::QsaState& st = ss.qsa_states[kPaired[k] / g.qsa_interval];
            ShardQueue::piece_cells(p, (uint32_t) k, (uint64_t) page, (uint64_t) g.n_head_kv, (uint64_t) g.head_dim,
                                    cells_k.data(), cells_v.data());
            for (int64_t r = 0; r < p.rows; r += kCheckGather) {
                const int64_t run = std::min(kCheckGather, p.rows - r);
                const size_t bytes = (size_t) run * cell * sizeof(uint16_t);
                for (int64_t i = 0; i < run; ++i) gather_ids[(size_t) i] = (int32_t) (p.row0 + r + i);
                if (cudaMemcpyAsync(d_gather_ids, gather_ids.data(), (size_t) run * sizeof(int32_t), cudaMemcpyHostToDevice,
                                    cs) != cudaSuccess) {
                    e = cuda_text("the check's cell ids");
                    return false;
                }
                basalt::kernels::kv_gather(st.k_pool, st.v_pool, st.page_table, d_gather_ids, run, shapes, d_gather_k,
                                           d_gather_v, cs);
                if (cudaMemcpyAsync(gather_k.data(), d_gather_k, bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                    cudaMemcpyAsync(gather_v.data(), d_gather_v, bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                    cudaStreamSynchronize(cs) != cudaSuccess) {
                    e = cuda_text("the check's gather of a layer's cells");
                    return false;
                }
                if (std::memcmp(gather_k.data(), cells_k.data() + (size_t) r * cell, bytes) != 0 ||
                    std::memcmp(gather_v.data(), cells_v.data() + (size_t) r * cell, bytes) != 0)
                    ++check_runs_differ;
            }
            if (lo < hi) {
                const size_t at = (size_t) (lo - keep->row0) * cell, from = (size_t) (lo - p.row0) * cell;
                std::memcpy(keep->k[k].data() + at, cells_k.data() + from, (size_t) (hi - lo) * cell * sizeof(uint16_t));
                std::memcpy(keep->v[k].data() + at, cells_v.data() + from, (size_t) (hi - lo) * cell * sizeof(uint16_t));
            }
        }
        if (lo < hi)
            std::memcpy(keep->hidden.data() + (size_t) (lo - keep->row0) * (size_t) N,
                        p.hidden + (size_t) (lo - p.row0) * (size_t) N, (size_t) (hi - lo) * (size_t) N * sizeof(uint16_t));
        ++check_pieces;
        return true;
    }

    /// Adds the prompt path's kept rows against the windows' to the sums of each row's context length.
    void compare() {
        const size_t cell = (size_t) (g.n_head_kv * g.head_dim);
        for (int64_t r = 0; r < by_window.rows; ++r) {
            const int64_t n_kv = by_window.row0 + r + 1;   // the cells row r reads among
            int b = 0;
            while (b < kNCheckContext - 1 && n_kv > kCheckContext[b]) ++b;
            CheckSums& c = sums[b];
            ++c.rows;
            double cosine = 1;
            const double row_err = f16_error(by_prompt.hidden.data() + (size_t) r * (size_t) N,
                                             by_window.hidden.data() + (size_t) r * (size_t) N, (size_t) N, &cosine);
            c.cos_sum += cosine;
            c.cos_min = std::min(c.cos_min, cosine);
            c.row_err_max = std::max(c.row_err_max, row_err);
            // the same against the window's row before: what a head row one off would give
            if (r > 0) {
                double off = 1;
                f16_error(by_prompt.hidden.data() + (size_t) r * (size_t) N, by_window.hidden.data() + (size_t) (r - 1) * (size_t) N,
                          (size_t) N, &off);
                ++c.off_rows;
                c.off_cos_sum += off;
            }
            const int64_t width = std::min(n_kv, cap);
            for (int k = 0; k < kNPaired; ++k) {
                // both lists ascend: a cell of the prompt path's that the window's lacks
                const int32_t *a = by_prompt.sel[k].data() + (size_t) r * (size_t) cap,
                              *w = by_window.sel[k].data() + (size_t) r * (size_t) cap;
                int64_t same = 0;
                for (int64_t i = 0, j = 0; i < width && j < width;) {
                    if (a[i] == w[j]) ++same, ++i, ++j;
                    else if (a[i] < w[j]) ++i;
                    else ++j;
                }
                c.selected[k] += width;
                c.differ[k] += width - same;
                const size_t at = (size_t) r * cell;
                const double k_err = f16_error(by_prompt.k[k].data() + at, by_window.k[k].data() + at, cell);
                const double v_err = f16_error(by_prompt.v[k].data() + at, by_window.v[k].data() + at, cell);
                c.k_err_sum[k] += k_err;
                c.k_err_max[k] = std::max(c.k_err_max[k], k_err);
                c.v_err_sum[k] += v_err;
                c.v_err_max[k] = std::max(c.v_err_max[k], v_err);
            }
        }
        ++check_documents;
    }

    /// Logs the check's sums and whether prefix anchors pass the gate past 2,051 cells. False when a prompt-path
    /// head row is not the decode's mix of its residual row (a cosine under kMixCosine), when, where every cell is
    /// selected, the two paths selected other cells or the head rows match the window's row before as well as their
    /// own, or when a piece's cells were not the cache's.
    bool check_report() const {
        bool ok = check_runs_differ == 0, gate = true;
        int64_t gate_rows = 0;
        const double mix_mean = mix_rows > 0 ? mix_cos_sum / (double) mix_rows : 0.0;
        basalt::log::info("basalt generate", "--dfly2-export check: %lld prompt-path head rows against the decode's mix of the "
                             "same residual rows: cosine mean %.8f, least %.8f, largest relative error %.3g",
                     (long long) mix_rows, mix_mean, mix_cos_min, mix_err_max);
        if (mix_rows == 0 || !(mix_cos_min >= kMixCosine)) {
            ok = false;
            basalt::log::error("basalt generate", "--dfly2-export check: a prompt-path head row's cosine against the decode's "
                                  "mix of its residual row is under %.4f: the head input is not the mix",
                               kMixCosine);
        }
        for (int b = 0; b < kNCheckContext; ++b) {
            const CheckSums& c = sums[b];
            if (c.rows == 0) continue;
            const double cos_mean = c.cos_sum / (double) c.rows;
            const double off_mean = c.off_rows > 0 ? c.off_cos_sum / (double) c.off_rows : 0.0;
            if (b == 0 && c.off_rows > 0 && !(cos_mean > off_mean)) {
                ok = false;
                basalt::log::error("basalt generate", "--dfly2-export check: where both paths read every cell, the head "
                                      "rows' mean cosine %.6f is not above %.6f, theirs against the window row before: the "
                                      "rows are not aligned", cos_mean, off_mean);
            }
            if (b > 0) {
                gate_rows += c.rows;
                gate = gate && cos_mean >= kGateCosine;
                for (int k = 0; k < kNPaired; ++k)
                    gate = gate && (double) c.differ[k] <= kGateDiffer * (double) c.selected[k];
            }
            char range[48];
            if (b < kNCheckContext - 1) std::snprintf(range, sizeof range, "up to %lld", (long long) kCheckContext[b]);
            else std::snprintf(range, sizeof range, "above %lld", (long long) kCheckContext[b - 1]);
            basalt::log::info("basalt generate", "--dfly2-export check: %lld rows with %s cells of context, the prompt path "
                                 "against verify windows: head rows' cosine mean %.6f (%.6f against the row before), least "
                                 "%.6f, largest relative error %.3g",
                         (long long) c.rows, range, cos_mean, off_mean, c.cos_min, c.row_err_max);
            for (int k = 0; k < kNPaired; ++k) {
                basalt::log::info("basalt generate", "--dfly2-export check:   layer %d: %.4f%% of the selected cells differ "
                                     "(%lld of %lld); K relative error mean %.3g, largest %.3g; V mean %.3g, largest %.3g",
                             kPaired[k], c.selected[k] > 0 ? 100.0 * (double) c.differ[k] / (double) c.selected[k] : 0.0,
                             (long long) c.differ[k], (long long) c.selected[k], c.k_err_sum[k] / (double) c.rows,
                             c.k_err_max[k], c.v_err_sum[k] / (double) c.rows, c.v_err_max[k]);
                if (b == 0 && c.differ[k] != 0) ok = false;
            }
        }
        if (check_runs_differ == 0)
            basalt::log::info("basalt generate", "--dfly2-export check: %lld documents; the cells of %lld pieces as written "
                                 "equal the cache's gather of them",
                         (long long) check_documents, (long long) check_pieces);
        else
            basalt::log::error("basalt generate", "--dfly2-export check: %lld runs of cells as written differ from the "
                                  "cache's gather of them (%lld pieces)",
                          (long long) check_runs_differ, (long long) check_pieces);
        for (int k = 0; k < kNPaired; ++k)
            if (sums[0].differ[k] != 0) {
                basalt::log::error("basalt generate", "--dfly2-export check: the two paths selected other cells where a "
                                      "row selects every cell");
                break;
            }
        if (gate_rows > 0)
            basalt::log::info("basalt generate", "--dfly2-export check: past %lld cells the prompt path's rows %s for "
                                 "window rows (at most %.0f%% of a layer's selected cells differ, head rows' mean cosine at "
                                 "least %.3f): prefix anchors %s",
                              (long long) kCheckContext[0], gate ? "stand" : "do not stand", 100.0 * kGateDiffer,
                              kGateCosine, gate ? "may be trained on" : "are best left out of training");
        return ok;
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

    /// Adds the piece's rows, its own cells and its selections to the document's digest, each read from where the
    /// target left it.
    void digest_piece(const ShardQueue::Piece& p) {
        DocDigest::add(digest.hidden, p.hidden, (size_t) p.rows * (size_t) N * sizeof(uint16_t));
        const size_t heads = (size_t) g.n_head_kv, dim = (size_t) g.head_dim;
        for (int k = 0; k < kNPaired; ++k)
            for (int64_t r = 0; r < p.rows; ++r) {
                const size_t cell = (size_t) (p.row0 + r), pg = cell / (size_t) page - (size_t) p.page0;
                for (size_t h = 0; h < heads; ++h) {
                    const size_t from = ((pg * heads + h) * (size_t) page + cell % (size_t) page) * dim;
                    DocDigest::add(digest.k[k], p.k[k] + from, dim * sizeof(uint16_t));
                    DocDigest::add(digest.v[k], p.v[k] + from, dim * sizeof(uint16_t));
                }
            }
        for (int i = 0; i < p.n_sel; ++i) {
            const int64_t a = anchors[(size_t) p.sel_anchor[i]];
            for (int k = 0; k < kNPaired; ++k)
                digest.selection(a, p.sel + ((size_t) i * kNPaired + (size_t) k) * (size_t) cap,
                                 (size_t) std::min<int64_t>(a, cap));
        }
    }

    /// Hands the filled piece to the writer, or back to the queue when the document is not written.
    bool piece_done(std::string& e) {
        ShardQueue::Piece* p = cur;
        cur = nullptr;
        if (digesting) digest_piece(*p);
        if (checking && !check_piece(*p, e)) {
            queue.give_back(p);
            return false;
        }
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
        if (keep != nullptr && !keep_selections(k, ids, p0, T, e)) return false;
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
        if (checking && !check_mix(R_rows, T, cur->hidden, e)) return false;
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
                    if (!copy_selections(k, ver.selection(k), w, T, e) ||
                        (keep != nullptr && !keep_selections(k, ver.selection(k), w, T, e)))
                        return false;
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

    /// Reads `doc` (sequence `line` of the export) from an empty sequence, its last `window_rows` rows through verify
    /// windows; `write` hands its pieces to the writer.
    bool read_document(const std::vector<int64_t>& doc, uint64_t line, int64_t window_rows, bool write, std::string& e) {
        n = (int64_t) doc.size();
        m = basalt::dfly2::window_from(n, window_rows);
        anchors = document_anchors(line, n, m);
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
    uint64_t docs_bytes = 0, doc_lines = 0;
    DocsFile docs;
    if (meta == nullptr) return refuse("needs the model file's meta section");
    if (!basalt::platform::file_size(o.dfly2_docs, docs_bytes) || !docs.open(o.dfly2_docs))
        return refuse("cannot read --dfly2-docs " + o.dfly2_docs);
    // every line is a document before the first one is computed: a bad line costs no shard
    if (!documents_check(o.dfly2_docs, n_vocab, doc_lines, err)) {
        basalt::log::error("basalt generate", "--dfly2-export: %s", err.c_str());
        return 2;
    }

    Export x{context, o, g, ss, (cudaStream_t) *context.main_cs};
    x.windows = windows;
    x.shapes = shapes;
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
    // what else decides which experts compute a row, and so its rounding
    settings["fixed_experts"] = o.diag_fixed_experts ? "1" : "0";
    std::snprintf(text, sizeof text, "%g", o.pcie_frac);
    settings["pcie_frac"] = text;
    std::snprintf(text, sizeof text, "%g", (double) o.route.prior);
    settings["route_prior"] = text;
    settings["route_tail_skip"] = std::to_string(o.route.tail_skip);
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
    // a failure between two documents closes the shard first, so the documents before it are kept
    const auto failed_between = [&](const std::string& why) {
        std::string ignored;
        x.queue.close(ignored);
        return failed(why);
    };
    if (doc_lines < skip)
        return failed_between(o.dfly2_docs + " has " + std::to_string(doc_lines) + " documents, the directory's shards hold " +
                              std::to_string(skip));
    // every Nth document's window rows are read by both paths and compared
    const int64_t check_every = windows ? (int64_t) basalt::core::exp_flag("dfly2_export_check") : 0;
    if (check_every > 0 && !x.check_init(err)) return failed(err);
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
        // the file was checked whole at the start: a line that fails now was changed since
        if (!document_parse(line_text, n_vocab, doc, parse_err))
            return failed_between(o.dfly2_docs + " line " + std::to_string(at + 1) + " changed during the run: " + parse_err);
        if ((int64_t) doc.size() > o.max_context) {
            doc.resize((size_t) o.max_context);
            ++truncated;
        }
        const int64_t window_rows = windows ? o.dfly2_window_rows : 0;
        const int64_t from = basalt::dfly2::window_from((int64_t) doc.size(), window_rows);
        x.checking = check_every > 0 && documents % check_every == 0 && from < (int64_t) doc.size();
        if (x.checking) {
            // first the whole document by the prompt path, its window rows kept and nothing written or counted
            const int64_t rows_before = x.rows_prompt;
            const double ms_before = x.ms_prompt;
            x.digesting = false;
            x.keep_rows(x.by_prompt, from, (int64_t) doc.size() - from);
            if (!x.read_document(doc, at, 0, false, err))
                return failed("line " + std::to_string(at + 1) + ", read by the prompt path alone: " + err);
            x.rows_prompt = rows_before;
            x.ms_prompt = ms_before;
            x.keep_rows(x.by_window, from, (int64_t) doc.size() - from);
        }
        // the run's first document is digested: it is read once more at the end
        x.digesting = documents == 0;
        x.digest.reset();
        if (!x.read_document(doc, at, window_rows, true, err)) return failed("line " + std::to_string(at + 1) + ": " + err);
        if (x.checking) x.compare();
        x.checking = false;
        x.keep = nullptr;
        if (documents == 0) {
            first_doc = doc;
            first_line = at;
            first_digest = x.digest.value();
            basalt::log::info("basalt generate", "--dfly2-export: the first document's %lld prompt-path rows at %.0f rows/s, "
                                 "its %lld window rows at %.1f rows/s",
                         (long long) x.rows_prompt, rate(x.rows_prompt, x.ms_prompt), (long long) x.rows_window,
                         rate(x.rows_window, x.ms_window));
        }
        ++documents;
        tokens += (int64_t) doc.size();
    }
    if (docs.failed()) return failed_between("reading " + o.dfly2_docs + " failed");
    if (line != doc_lines)
        return failed_between(o.dfly2_docs + " changed during the run: " + std::to_string(line) + " documents, " +
                              std::to_string(doc_lines) + " at the start");
    const int64_t rows_prompt = x.rows_prompt, rows_window = x.rows_window;
    const double ms_prompt = x.ms_prompt, ms_window = x.ms_window;

    // ---- the first document again, from the state the last one left: when both readings give the same rows, cells
    // and selections, the reset between documents leaves nothing behind
    bool first_again = true;
    if (documents > 0) {
        x.digesting = true;
        x.digest.reset();
        if (!x.read_document(first_doc, first_line, windows ? o.dfly2_window_rows : 0, false, err))
            return failed("reading the first document again: " + err);
        if (x.digest.value() == first_digest) {
            basalt::log::info("basalt generate", "--dfly2-export: the first document read again after the last gave the "
                                 "same rows, cells and selections");
        } else if (o.diag_fixed_experts) {
            // every expert row runs the same way: only state the reset left behind can change a row
            first_again = false;
            basalt::log::error("basalt generate", "--dfly2-export: the first document (line %llu) read again after the "
                                  "last gave other rows, cells or selections with the experts fixed: the reset between "
                                  "documents leaves state behind",
                               (unsigned long long) (first_line + 1));
        } else {
            basalt::log::warn("basalt generate", "--dfly2-export: the first document (line %llu) read again after the last "
                                 "gave other rows, cells or selections: a document's result depends on the ones read "
                                 "before it (the session's reset, or where the experts ran; --diag-fixed-experts tells "
                                 "them apart)",
                         (unsigned long long) (first_line + 1));
        }
    }
    if (!x.queue.close(err)) return failed(err);

    // ---- every closed shard back through the format's reader, against the documents file: each sequence's tokens,
    // the anchors drawn for it and the form of its selections, and this run's first document as it was computed
    DocsFile docs_again;
    if (!docs_again.open(o.dfly2_docs)) return failed("cannot read " + o.dfly2_docs + " again");
    uint64_t line_again = 0, digest_back = 0;
    const SequenceSource expected = [&](uint64_t seq, std::vector<int32_t>& ids, std::vector<int32_t>& anchors,
                                        std::string& e) {
        // sequence k is line k, cut as it was when it was read
        if (seq != line_again || !docs_again.next(line_text)) {
            e = "the documents file has no line for it";
            return false;
        }
        ++line_again;
        if (!document_parse(line_text, n_vocab, doc, e)) return false;
        if ((int64_t) doc.size() > o.max_context) doc.resize((size_t) o.max_context);
        ids.resize(doc.size());
        for (size_t i = 0; i < doc.size(); ++i) ids[i] = (int32_t) doc[i];
        anchors = document_anchors(seq, (int64_t) doc.size(),
                                   basalt::dfly2::window_from((int64_t) doc.size(), windows ? o.dfly2_window_rows : 0));
        return true;
    };
    const SequenceSink seen = [&](uint64_t seq, const basalt::dfly2::SequenceData& data) {
        if (documents > 0 && seq == first_line) digest_back = DocDigest::of(header, data);
    };
    ReadBack back;
    const Clock::time_point t_back = Clock::now();
    bool whole = shards_read_back(o.dfly2_export, header, shapes.idx_block, line, expected, seen, back, err);
    if (!whole) {
        basalt::log::error("basalt generate", "--dfly2-export: reading the shards back: %s", err.c_str());
    } else {
        const bool first_same = documents == 0 || digest_back == first_digest;
        if (back.off_form != 0)
            basalt::log::error("basalt generate", "--dfly2-export: %llu of %llu selections read back do not have the "
                                  "form of the index's selection; the first: %s",
                          (unsigned long long) back.off_form, (unsigned long long) back.selections,
                          back.first_off_form.c_str());
        if (!first_same)
            basalt::log::error("basalt generate", "--dfly2-export: the first document (line %llu) read back from its "
                                  "shard is not the rows, cells and selections that were computed",
                          (unsigned long long) (first_line + 1));
        basalt::log::info("basalt generate", "--dfly2-export: %llu shards read back in %.1f s: %llu sequences, %llu tokens "
                             "and %llu anchors are the documents'%s",
                     (unsigned long long) back.shards, std::chrono::duration<double>(Clock::now() - t_back).count(),
                     (unsigned long long) back.sequences, (unsigned long long) back.tokens,
                     (unsigned long long) back.anchors,
                     documents > 0 && first_same ? "; this run's first document is as computed" : "");
        whole = back.off_form == 0 && first_same;
    }
    const bool checked = (check_every <= 0 || x.check_report()) && whole && first_again;
    basalt::log::info("basalt generate", "--dfly2-export: %lld documents, %lld tokens (%lld documents cut to %lld tokens); "
                         "%lld prompt-path rows at %.0f rows/s, %lld window rows at %.1f rows/s; the lent cache slots "
                         "refilled %lld times in %.0f ms",
                 (long long) documents, (long long) tokens, (long long) truncated, (long long) o.max_context,
                 (long long) rows_prompt, rate(rows_prompt, ms_prompt), (long long) rows_window, rate(rows_window, ms_window),
                 (long long) x.refills, x.ms_refill);
    return checked ? 0 : 1;
}

}  // namespace basalt::models::qwen4exp::generate
