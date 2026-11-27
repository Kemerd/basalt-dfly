// tests/models/qwen4exp/decode/dfly_real.cpp - the DFly2 block pass of a pack's drafter against the trainer's forward over
// the same inputs.
//
// A fixture of `basalt-dfly2-train fixture` holds blocks of teacher data (the source row's head input, each paired layer's
// selected cells with their cached keys and values) and what the trainer's drafter computed from them. Each block runs
// through the engine's pass bound to the pack's head and embedding table, with the cells placed at their ids in fp16 pools
// as a session holds them. Reported per block and in all: the error of the rows after the final norm, of the corrected rows
// and of the confidence logits (rows whose previous token is the same in both), the share of drafts that agree per draft
// position, how many of the engine's picks are the argmax of its own row over the whole head, and the difference of the
// two embedding tables. A misread equation moves the rows by their own size; the Q8_0 projections, the q8_1 inputs and
// the two-stage pick move them by their rounding.
//
// usage: qwen4exp_dfly_real PACK.basalt FIXTURE.safetensors   (GPU; the pack carries the drafter the fixture was written for)
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/device_copy.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/models/qwen4exp/decode/dfly.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/rope_scaling.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/mapped_file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace basalt::core {

/// A weight table row by name: the head's shape the bind checks, without loading the target's weights
struct WeightTableFixture {
    static void put(WeightTable& t, const std::string& name, int64_t ne0, int64_t ne1) {
        WeightRef r;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.elements = ne0 * ne1;
        t.table_[name] = r;
    }
};

}  // namespace basalt::core

using namespace basalt;
namespace bk = basalt::kernels;

namespace {

constexpr const char* kTag = "dfly_real";
constexpr int kWinRows = 8;          ///< rows of the window the pass is bound to; a block's source row is one of them
/// Mean relative error of z over the blocks past which an equation is read differently: the Q8_0 projections and q8_1
/// inputs move it by 3 to 6% (a block's largest by up to 12%) for a 40-step and a step-0 drafter on teacher data.
constexpr double kMisread = 0.1;

/// False, with the call named, for a failed CUDA call.
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s: %s\n", kTag, what, cudaGetErrorString(e));
    return false;
}

/// The fixture: a safetensors file mapped read-only.
struct Fixture {
    /// A tensor's type, shape and byte range from the start of the data.
    struct Entry {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t begin = 0, end = 0;
    };
    platform::MappedFile map;
    uint64_t data = 0;   ///< where the tensors' bytes start
    std::map<std::string, Entry> t;
    std::map<std::string, std::string> meta;

    /// Maps `path` and reads its header; false with `err` for a file that is not safetensors.
    bool open(const std::string& path, std::string& err) {
        auto m = platform::MappedFile::open_ro(path, &err);
        if (!m) return false;
        map = std::move(*m);
        uint64_t n = 0;
        if (map.size() < 8) { err = path + ": not a safetensors file"; return false; }
        std::memcpy(&n, map.data(), 8);
        if (n > map.size() - 8) { err = path + ": its header runs past the file"; return false; }
        core::Json j;
        if (!core::parse_json((const char*) map.data() + 8, (size_t) n, j, err) || !j.is_object()) {
            err = path + ": the header is not a JSON object" + (err.empty() ? "" : ": " + err);
            return false;
        }
        data = 8 + n;
        for (const auto& [name, v] : j.members) {
            if (name == "__metadata__") {
                for (const auto& [k, s] : v.members)
                    if (s.is_string()) meta[k] = s.s;
                continue;
            }
            const core::Json *dt = v.get("dtype"), *sh = v.get("shape"), *off = v.get("data_offsets");
            if (!dt || !dt->is_string() || !sh || !sh->is_array() || !off || !off->is_array() || off->items.size() != 2) {
                err = path + ": tensor " + name + " is not described in full";
                return false;
            }
            Entry e;
            e.dtype = dt->s;
            for (const core::Json& d : sh->items) e.shape.push_back(d.i64());
            e.begin = off->items[0].u64();
            e.end = off->items[1].u64();
            if (e.begin > e.end || e.end > map.size() - data) {
                err = path + ": tensor " + name + " lies outside the file";
                return false;
            }
            t[name] = std::move(e);
        }
        return true;
    }
    /// The tensor `name` of type `dtype` copied out, `count` elements of T (0: any); false with `err` otherwise.
    template <class T> bool get(const std::string& name, const char* dtype, size_t count, std::vector<T>& out, std::string& err) const {
        const auto it = t.find(name);
        if (it == t.end() || it->second.dtype != dtype) {
            err = "the fixture has no " + std::string(dtype) + " tensor " + name;
            return false;
        }
        const uint64_t bytes = it->second.end - it->second.begin;
        if (count != 0 && bytes != count * sizeof(T)) {
            err = "the fixture's " + name + " holds " + std::to_string(bytes) + " bytes, not " + std::to_string(count * sizeof(T));
            return false;
        }
        out.resize((size_t) (bytes / sizeof(T)));
        std::memcpy(out.data(), map.data() + data + it->second.begin, (size_t) bytes);
        return true;
    }
    /// A metadata value as a number; -1 when it is missing.
    int64_t meta_int(const std::string& key) const {
        const auto it = meta.find(key);
        return it == meta.end() ? -1 : std::atoll(it->second.c_str());
    }
};

/// The paired layers' caches: fp16 pools in the cache's page-major layout with the identity table.
struct Pools {
    std::vector<std::vector<uint16_t>> k, v;   ///< per layer, host mirrors
    std::vector<uint16_t*> dk, dv;
    int32_t* table = nullptr;
    int64_t cells = 0;
};

/// The row of KV head h of cell c under the identity table.
size_t prow(const bk::QsaShapes& s, int64_t c, int64_t h) {
    return (size_t) (((c / s.page_size) * s.n_head_kv + h) * s.page_size + c % s.page_size);
}

/// The largest difference over the reference's largest magnitude, over n values.
double rel_err(const float* got, const float* want, size_t n) {
    double scale = 0, e = 0;
    for (size_t i = 0; i < n; ++i) {
        scale = std::max(scale, std::fabs((double) want[i]));
        e = std::max(e, std::fabs((double) got[i] - (double) want[i]));
    }
    return e / (scale + 1e-12);
}

/// The largest difference of two rows of n values.
double max_abs_diff(const float* a, const float* b, size_t n) {
    double e = 0;
    for (size_t i = 0; i < n; ++i) e = std::max(e, std::fabs((double) a[i] - (double) b[i]));
    return e;
}

/// Reports a failure; the program's exit code.
int fail(const std::string& what) {
    std::fprintf(stderr, "%s: %s\n", kTag, what.c_str());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    basalt::models::register_builtin_models();
    if (argc != 3) {
        std::fprintf(stderr, "usage: qwen4exp_dfly_real PACK.basalt FIXTURE.safetensors\n");
        return 2;
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("%s: no CUDA device, skipped\n", kTag);
        return 77;
    }
    const std::string pack_path = argv[1], fix_path = argv[2];
    const std::vector<std::string> shards = {pack_path};
    const bk::QsaShapes s = bk::qsa_real_shapes();
    std::string err;

    // ---- the fixture's shape
    Fixture fx;
    if (!fx.open(fix_path, err)) return fail(err);
    const int64_t n_blocks = fx.meta_int("blocks"), R = fx.meta_int("block"), layers = fx.meta_int("layers"), hidden = fx.meta_int("hidden");
    if (fx.meta["format"] != "dfly2-fixture" || n_blocks < 1 || R < 2 || layers < 1 || hidden < 1)
        return fail(fix_path + " is not a fixture of basalt-dfly2-train (format, blocks, block, layers, hidden)");

    // ---- the pack: its head's type and vocabulary, and its drafter's record
    core::ModelFile mf;
    if (!mf.open(pack_path, err) || !mf.load_meta(err) || !mf.register_gguf_view(err)) return fail(pack_path + ": " + err);
    uint64_t head_bytes = 0;
    int head_type = -1;
    if (!core::NativeHead::plan_bytes(shards, head_bytes, err, &head_type)) return fail(err);
    int64_t n_vocab = 0;
    try {
        const basalt::GgufModel model(shards);
        const basalt::TensorInfo* wo = model.find("output.weight");
        if (wo == nullptr || wo->shape.size() != 2) return fail("the pack has no output.weight matrix");
        n_vocab = (int64_t) wo->shape[1];
    } catch (const std::exception& e) {
        return fail(std::string("the pack's tensors: ") + e.what());
    }
    const core::DflyTarget target = {n_vocab, (int) bk::qsa_selection_width(bk::kTopkMaxCells, s), (int) s.n_rot};
    core::DflyPack pack;
    if (!core::dfly_read_pack(mf, core::ModelGeometry{}, target, pack, err)) return fail(pack_path + ": " + err);
    if (!pack.present) return fail(pack_path + " carries no DFly2 drafter (attach one with tools/convert/convert.py --dfly)");
    const core::DflyDims& d = pack.dims;
    const int N = d.n_embd, KV = d.n_head_kv, HD = d.head_dim;
    if (layers != d.layers || hidden != N || R > d.block || R > core::kDflyBlockMax)
        return fail("the fixture's blocks are not this drafter's (layers, width or block rows)");
    std::printf("%s: drafter of %d layers at step %lld, head type %d over %lld rows, %lld blocks of %lld rows (fixture of step %s, %s)\n", kTag,
                d.layers, (long long) d.step, head_type, (long long) n_vocab, (long long) n_blocks, (long long) R, fx.meta["step"].c_str(),
                fx.meta["weight_dtype"].c_str());
    if (fx.meta_int("step") != d.step) std::printf("%s: the fixture's checkpoint is of another step than the pack's drafter\n", kTag);

    // the rotation the drafter was trained with: the process's, set before the graph is captured
    bk::RopeScaling rs = bk::rope_scaling();
    rs.freq_base = d.rope_base;
    bk::rope_scaling_set(rs);
    // the confidence head's bias, as the record stores it (BF16)
    float bias = 0;
    {
        const auto it = pack.rec.tensors.find("confidence_head.proj.bias");
        uint16_t bits = 0;
        const int fd = platform::open_fd_utf8(pack_path);
        const bool read = fd >= 0 && it != pack.rec.tensors.end() && platform::pread_all(fd, &bits, 2, it->second.offset);
        if (fd >= 0) platform::close_fd(fd);
        if (!read) return fail("the confidence head's bias cannot be read from the pack");
        const uint32_t u = (uint32_t) bits << 16;
        std::memcpy(&bias, &u, 4);
    }

    // ---- the head, the embedding table and the drafter, read in one pass
    core::NativeHead head;
    core::NativeEmbed embed;
    core::DflyDrafter drafter;
    {
        core::LoadPlan plan;
        core::Loader loader;
        if (!head.load(shards, N, n_vocab, plan, err) || !embed.load(shards, N, n_vocab, plan, err) ||
            !drafter.load(pack, pack_path, (int) R, false, plan, err) || !loader.run(plan, err) || !head.finish(err))
            return fail("load: " + err);
    }
    core::set_native_embed(&embed);

    // ---- the window's buffers the pass is bound to: head-input rows and each layer's selections
    const int64_t cap = target.sel_width;
    float* d_rows = nullptr;
    if (!ck(cudaMalloc((void**) &d_rows, (size_t) kWinRows * N * 4), "the head rows") ||
        !ck(cudaMemset(d_rows, 0, (size_t) kWinRows * N * 4), "the head rows"))
        return 1;
    std::vector<int32_t*> d_sel((size_t) d.layers, nullptr);
    for (int32_t*& p : d_sel)
        if (!ck(cudaMalloc((void**) &p, (size_t) kWinRows * cap * 4), "the selections") ||
            !ck(cudaMemset(p, 0xff, (size_t) kWinRows * cap * 4), "the selections"))
            return 1;
    {
        core::WeightTable wt;
        core::WeightTableFixture::put(wt, "output.weight", N, n_vocab);
        std::vector<const int32_t*> sel(d_sel.begin(), d_sel.end());
        if (!drafter.bind(wt, &head, d_rows, sel.data(), cap, kWinRows, err)) return fail("bind: " + err);
    }

    // ---- the paired layers' pools, wide enough for every block's positions
    Pools pools;
    {
        int64_t max_pos = 0;
        for (int64_t b = 0; b < n_blocks; ++b) {
            std::vector<int64_t> pos;
            if (!fx.get("block." + std::to_string(b) + ".pos", "I64", 1, pos, err)) return fail(err);
            max_pos = std::max(max_pos, pos[0]);
        }
        pools.cells = (max_pos + s.page_size - 1) / s.page_size * s.page_size;
        const size_t rows = (size_t) pools.cells * (size_t) KV;
        std::vector<int32_t> table((size_t) (pools.cells / s.page_size));
        std::iota(table.begin(), table.end(), 0);
        if (!ck(cudaMalloc((void**) &pools.table, std::max<size_t>(table.size() * 4, 4)), "the page table") ||
            !ck(core::device_copy(pools.table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "the page table"))
            return 1;
        for (int l = 0; l < d.layers; ++l) {
            pools.k.emplace_back(rows * HD, 0);
            pools.v.emplace_back(rows * HD, 0);
            pools.dk.push_back(nullptr);
            pools.dv.push_back(nullptr);
            if (!ck(cudaMalloc((void**) &pools.dk.back(), rows * HD * 2), "a K pool") ||
                !ck(cudaMalloc((void**) &pools.dv.back(), rows * HD * 2), "a V pool"))
                return 1;
        }
    }
    // the pools as a session of the target holds them
    const core::ModelGeometry g;
    std::vector<core::QsaState> states((size_t) g.n_qsa_layers());
    for (int l = 0; l < d.layers; ++l) {
        core::QsaState& q = states[(size_t) (d.target_layers[(size_t) l] / g.qsa_interval)];
        q.page_table = pools.table;
        q.k_pool = pools.dk[(size_t) l];
        q.v_pool = pools.dv[(size_t) l];
    }
    core::SessionState ss;
    ss.qsa_states = states.data();
    ss.layer_lo = 0;
    ss.layer_hi = g.n_layers;
    ss.qsa_ord0 = 0;
    ss.qsa_alloc = g.n_qsa_layers();
    std::vector<bk::QsaAttnPools> attn((size_t) d.layers);
    if (!core::dfly_session_pools(ss, g, d, false, attn.data(), err)) return fail("the session's pools: " + err);
    const int seq = drafter.add_sequence(attn.data(), err);
    if (seq < 0) return fail(err);

    // scratch for the engine's embedding rows and for the whole head's exact logits of a corrected row, on a stream of
    // their own (the head's product takes no null stream)
    float *d_emb = nullptr, *d_logits = nullptr;
    void* d_xq = nullptr;
    cudaStream_t score_stream = nullptr;
    if (!ck(cudaMalloc((void**) &d_emb, (size_t) N * 4), "an embedding row") ||
        !ck(cudaMalloc((void**) &d_logits, (size_t) n_vocab * 4), "the logits") ||
        !ck(cudaMalloc(&d_xq, bk::native_q8_1_bytes(N, 1)), "a q8_1 row") ||
        !ck(cudaStreamCreateWithFlags(&score_stream, cudaStreamNonBlocking), "the scoring stream"))
        return 1;

    // ---- every block
    const int drafts = (int) R - 1;
    std::vector<int64_t> agree((size_t) drafts, 0), chain((size_t) drafts, 0);
    double z_max = 0, z_sum = 0, h_max = 0, c_max = 0, emb_max = 0;
    int64_t h_rows = 0, exact = 0, picks = 0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const std::string p = "block." + std::to_string(b) + ".";
        std::vector<int32_t> anchor, ids_t;
        std::vector<int64_t> posv;
        std::vector<float> hs, e_anchor, e_mask, e_prev, z_t, h_t, c_t;
        if (!fx.get(p + "anchor", "I32", 1, anchor, err) || !fx.get(p + "pos", "I64", 1, posv, err) ||
            !fx.get(p + "h_s", "F32", (size_t) N, hs, err) || !fx.get(p + "emb_anchor", "F32", (size_t) N, e_anchor, err) ||
            !fx.get(p + "emb_mask", "F32", (size_t) N, e_mask, err) || !fx.get(p + "emb_prev", "F32", (size_t) (drafts * N), e_prev, err) ||
            !fx.get(p + "z", "F32", (size_t) (R * N), z_t, err) || !fx.get(p + "h_corrected", "F32", (size_t) (drafts * N), h_t, err) ||
            !fx.get(p + "conf_logit", "F32", (size_t) drafts, c_t, err) || !fx.get(p + "draft_ids", "I32", (size_t) drafts, ids_t, err))
            return fail(err);
        const int64_t pos = posv[0];
        if (pos < 1 || pos + R > INT32_MAX || anchor[0] < 0 || anchor[0] >= n_vocab) return fail(p + "anchor or pos is out of range");
        const int64_t width = bk::qsa_selection_width(pos, s);
        // ---- the source row's head input at the window's row `row`
        const int row = (int) std::min<int64_t>(kWinRows - 1, pos - 1);
        if (!ck(core::device_copy(d_rows + (size_t) row * N, hs.data(), (size_t) N * 4, cudaMemcpyHostToDevice), "a head row")) return 1;
        // ---- each layer's selection at that row, and its cells at their ids
        for (int l = 0; l < d.layers; ++l) {
            const std::string q = p + "layer." + std::to_string(l) + ".";
            std::vector<int32_t> sel;
            std::vector<uint16_t> kc, vc;
            if (!fx.get(q + "sel", "I32", 0, sel, err)) return fail(err);
            const size_t n = sel.size();
            if ((int64_t) n != width)
                return fail(q + "sel holds " + std::to_string(n) + " ids where the pass reads " + std::to_string(width));
            if (!fx.get(q + "k_ctx", "F16", n * (size_t) (KV * HD), kc, err) || !fx.get(q + "v_ctx", "F16", n * (size_t) (KV * HD), vc, err))
                return fail(err);
            std::vector<int32_t> rowsel((size_t) cap, -1);
            for (size_t j = 0; j < n; ++j) {
                if (sel[j] < 0 || sel[j] >= pos) return fail(q + "sel names a cell at or after the anchor");
                rowsel[j] = sel[j];
                for (int h = 0; h < KV; ++h) {
                    const size_t at = prow(s, sel[j], h) * (size_t) HD, from = (j * (size_t) KV + (size_t) h) * (size_t) HD;
                    std::memcpy(pools.k[(size_t) l].data() + at, kc.data() + from, (size_t) HD * 2);
                    std::memcpy(pools.v[(size_t) l].data() + at, vc.data() + from, (size_t) HD * 2);
                }
            }
            const size_t pool_bytes = pools.k[(size_t) l].size() * 2;
            if (!ck(core::device_copy(d_sel[(size_t) l] + (size_t) row * cap, rowsel.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "a selection") ||
                !ck(core::device_copy(pools.dk[(size_t) l], pools.k[(size_t) l].data(), pool_bytes, cudaMemcpyHostToDevice), "a K pool") ||
                !ck(core::device_copy(pools.dv[(size_t) l], pools.v[(size_t) l].data(), pool_bytes, cudaMemcpyHostToDevice), "a V pool"))
                return 1;
        }
        // ---- the block
        bool queued = false;
        if (!drafter.draft(seq, anchor[0], pos, pos - 1 - row, 0, kWinRows, queued, err) || !queued || !drafter.wait(err))
            return fail(p + "the block: " + (err.empty() ? "the window does not hold its source row" : err));
        std::vector<int32_t> ids_e((size_t) drafts);
        std::vector<float> conf((size_t) drafts), z_e((size_t) (R * N)), h_e((size_t) (drafts * N));
        drafter.read(seq, ids_e.data(), conf.data());
        const float* raw = drafter.raw_conf(seq);
        if (!ck(core::device_copy(z_e.data(), drafter.block_rows(), z_e.size() * 4, cudaMemcpyDeviceToHost), "the block's rows") ||
            !ck(core::device_copy(h_e.data(), drafter.corrected_rows(), h_e.size() * 4, cudaMemcpyDeviceToHost), "the corrected rows"))
            return 1;
        // ---- the engine's embedding rows against the trainer's: the anchor, the mask and each drafted row's previous token
        std::vector<std::pair<int64_t, const float*>> rows_e = {{anchor[0], e_anchor.data()}, {d.mask_id, e_mask.data()}};
        for (int j = 0; j < drafts; ++j) rows_e.push_back({j == 0 ? anchor[0] : ids_t[(size_t) j - 1], e_prev.data() + (size_t) j * N});
        std::vector<float> e((size_t) N);
        for (const auto& [tok, want] : rows_e) {
            if (tok < 0 || tok >= n_vocab) return fail(p + "a draft id is outside the vocabulary");
            embed.gather_one(tok, d_emb, score_stream);
            if (!ck(cudaStreamSynchronize(score_stream), "an embedding row") ||
                !ck(core::device_copy(e.data(), d_emb, (size_t) N * 4, cudaMemcpyDeviceToHost), "an embedding row"))
                return 1;
            emb_max = std::max(emb_max, max_abs_diff(e.data(), want, (size_t) N));
        }
        // ---- the rows, and the drafted rows whose previous token is the same in both
        const double ez = rel_err(z_e.data(), z_t.data(), z_e.size());
        z_max = std::max(z_max, ez);
        z_sum += ez;
        double eh = 0, ec = 0;
        int same_prev = 0, first_miss = -1, agreed = 0;
        for (int j = 0; j < drafts; ++j) {
            // row j reads the draft before it, so it is compared while every earlier draft agrees
            if (first_miss < 0) {
                eh = std::max(eh, rel_err(h_e.data() + (size_t) j * N, h_t.data() + (size_t) j * N, (size_t) N));
                ec = std::max(ec, std::fabs((double) raw[j] + (double) bias - (double) c_t[(size_t) j]));
                ++same_prev;
            }
            agree[(size_t) j] += ids_e[(size_t) j] == ids_t[(size_t) j];
            agreed += ids_e[(size_t) j] == ids_t[(size_t) j];
            if (ids_e[(size_t) j] != ids_t[(size_t) j] && first_miss < 0) first_miss = j;
            if (first_miss < 0) ++chain[(size_t) j];
        }
        h_max = std::max(h_max, eh);
        c_max = std::max(c_max, ec);
        h_rows += same_prev;
        // ---- each pick against the whole head's exact logits of the engine's own corrected row
        try {
            for (int j = 0; j < drafts; ++j) {
                bk::native_quantize_q8_1(drafter.corrected_rows() + (size_t) j * N, d_xq, N, 1, score_stream);
                head.score_rows(head.weights(), d_xq, d_logits, (int) n_vocab, 1, score_stream);
                std::vector<float> logits((size_t) n_vocab);
                if (!ck(cudaStreamSynchronize(score_stream), "the logits") ||
                    !ck(core::device_copy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost), "the logits"))
                    return 1;
                exact += std::max_element(logits.begin(), logits.end()) - logits.begin() == ids_e[(size_t) j];
                ++picks;
            }
        } catch (const std::exception& e) {
            return fail(std::string("scoring the head: ") + e.what());
        }
        std::printf("%s: block %lld at %lld, %lld cells: z rel err %.3g, corrected %.3g and confidence logit abs err %.3g over %d rows, "
                    "drafts agree %d of %d, first miss %d\n", kTag, (long long) b, (long long) pos, (long long) width, ez, eh, ec, same_prev,
                    agreed, drafts, first_miss);
    }

    // ---- in all
    std::printf("%s: z rel err max %.3g, mean %.3g over %lld blocks\n", kTag, z_max, z_sum / (double) n_blocks, (long long) n_blocks);
    std::printf("%s: corrected rows rel err max %.3g, confidence logit abs err max %.3g, over %lld rows with the same previous token\n", kTag,
                h_max, c_max, (long long) h_rows);
    std::printf("%s: drafts that agree, by position:", kTag);
    for (int j = 0; j < drafts; ++j) std::printf(" %d:%.3f", j + 1, (double) agree[(size_t) j] / (double) n_blocks);
    std::printf("\n%s: drafts that agree with every draft before them, by position:", kTag);
    for (int j = 0; j < drafts; ++j) std::printf(" %d:%.3f", j + 1, (double) chain[(size_t) j] / (double) n_blocks);
    std::printf("\n%s: %lld of %lld engine picks are the whole head's argmax of the engine's own row\n", kTag, (long long) exact, (long long) picks);
    std::printf("%s: embedding rows, max |engine - trainer| %.3g\n", kTag, emb_max);

    core::set_native_embed(nullptr);
    cudaFree(d_emb);
    cudaStreamDestroy(score_stream);
    cudaFree(d_logits);
    cudaFree(d_xq);
    cudaFree(d_rows);
    for (int32_t* q : d_sel) cudaFree(q);
    for (uint16_t* q : pools.dk) cudaFree(q);
    for (uint16_t* q : pools.dv) cudaFree(q);
    cudaFree(pools.table);
    if (z_sum / (double) n_blocks > kMisread) return fail("z is further from the trainer's than rounding moves it");
    std::printf("%s: ok\n", kTag);
    return 0;
}
