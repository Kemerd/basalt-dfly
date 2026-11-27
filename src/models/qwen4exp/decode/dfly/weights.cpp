// src/models/qwen4exp/decode/dfly/weights.cpp - the DFly2 drafter's memory on the card: its weights, rounded as they load,
// and the working memory of a block.
#include "basalt/models/qwen4exp/decode/dfly.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/q5k_head.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <map>

namespace basalt::core {

namespace bk = basalt::kernels;

namespace {

constexpr uint64_t kAlign = 256;

// a bump allocator over one arena; with no base it only counts
struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + kAlign - 1) & ~(kAlign - 1);
        return p;
    }
};

// the bytes every weight takes on the card, each tensor on its own boundary
uint64_t weight_bytes(const DflyDims& d) {
    uint64_t n = 0;
    for (const DflySpec& s : dfly_specs(d)) n += (s.bytes() + kAlign - 1) & ~(kAlign - 1);
    return n;
}

}  // namespace

uint64_t DflyDrafter::carve(const DflyDims& d, int rows, bool kv_int8, uint8_t* base, Buffers& b) {
    const uint64_t R = (uint64_t) rows, N = (uint64_t) d.n_embd, I = (uint64_t) d.ffn, C = (uint64_t) d.corr_ffn;
    const uint64_t NH = (uint64_t) d.n_head, NKV = (uint64_t) d.n_head_kv, HD = (uint64_t) d.head_dim;
    // the private pool holds the widest selection and a full block, in whole pages
    const bk::QsaShapes s = shapes_of(d);
    const uint64_t cap_p = (uint64_t) d.sel_width + kDflyBlockMax, ps = (uint64_t) s.page_size;
    const uint64_t pages = (cap_p + ps - 1) / ps, cells = pages * ps;
    const int widest = (int) std::max(std::max(N, I), NH * HD);
    Bump m;
    m.base = base;
    b.rec = m.take<uint8_t>(record_bytes(d, rows));
    b.out = m.take<int32_t>(3 * kDflyBlockMax);
    b.iota = m.take<int32_t>(R * cap_p);
    b.ident = m.take<int32_t>(pages);
    b.mask_ids = m.take<int32_t>(kDflyBlockMax);
    b.mask_rows = m.take<float>(R * N);
    b.hs = m.take<float>(N); b.emb0 = m.take<float>(N); b.cat = m.take<float>(2 * N); b.fused = m.take<float>(N);
    b.x = m.take<float>(R * N); b.y = m.take<float>(R * N); b.o = m.take<float>(R * N); b.z = m.take<float>(R * N);
    b.hp = m.take<float>(R * N);
    b.qfull = m.take<float>(R * NH * 2 * HD); b.qcur = m.take<float>(R * NH * HD);
    b.kcur = m.take<float>(R * NKV * HD); b.vcur = m.take<float>(R * NKV * HD);
    b.attn = m.take<float>(R * NH * HD); b.attn_gated = m.take<float>(R * NH * HD);
    b.attn_scratch = m.take<float>(R * bk::qsa_decode_attn_scratch_floats((int64_t) cap_p, s));
    b.g = m.take<float>(R * I); b.u = m.take<float>(R * I); b.h = m.take<float>(R * I);
    b.pe = m.take<float>(N); b.cg = m.take<float>(C); b.cu = m.take<float>(C); b.ch = m.take<float>(C);
    b.delta = m.take<float>(N);
    b.xq = m.take<uint8_t>(bk::native_q8_1_bytes(widest, rows));
    b.cat_q8 = m.take<uint8_t>(bk::native_q8_1_bytes((int) (2 * N), 1));
    b.ch_q8 = m.take<uint8_t>(bk::native_q8_1_bytes((int) C, 1));
    b.hq = m.take<uint8_t>(bk::native_q8_1_bytes((int) N, 1));
    // the pool in the cache's form: fp16 rows, or int8 codes with a scale per group
    const uint64_t row = HD * (kv_int8 ? 1 : 2);
    b.pk = m.take<uint8_t>(cells * NKV * row);
    b.pv = m.take<uint8_t>(cells * NKV * row);
    if (kv_int8) {
        const uint64_t nsc = HD / bk::KV_Q8_GROUP;
        b.pks = m.take<uint8_t>(cells * NKV * nsc * 2);
        b.pvs = m.take<uint8_t>(cells * NKV * nsc * 2);
    }
    return m.used;
}

uint64_t DflyDrafter::plan_bytes(const DflyDims& d, int rows, bool kv_int8) {
    Buffers none;
    return weight_bytes(d) + carve(d, rows, kv_int8, nullptr, none);
}

uint64_t DflyDrafter::bind_bytes(int head_type, int64_t n_vocab) {
    if (!NativeHead::has_draft_top1(head_type) || n_vocab <= 0) return 0;
    return bk::q5k_proxy_bytes(n_vocab) + bk::q5k_proxy_scratch_bytes(n_vocab) + (uint64_t) n_vocab * 4;
}

DflyDrafter::~DflyDrafter() {
    for (Sequence& q : seqs_) {
        if (q.rec) cudaFreeHost(q.rec);
        if (q.out) cudaFreeHost(q.out);
    }
    if (exec_) cudaGraphExecDestroy((cudaGraphExec_t) exec_);
    if (done_) cudaEventDestroy((cudaEvent_t) done_);
    if (ds_) cudaStreamDestroy((cudaStream_t) ds_);
    if (proxy_) cudaFree(proxy_);
    if (proxy_scratch_) cudaFree(proxy_scratch_);
    if (map_) cudaFree(map_);
    if (arena_) cudaFree(arena_);
    if (weights_) cudaFree(weights_);
}

bool DflyDrafter::load(const DflyPack& pack, const std::string& path, int rows, bool kv_int8, LoadPlan& reads,
                       std::string& err) {
    if (loaded()) { err = "dfly: the drafter is already loaded"; return false; }
    if (!pack.present || path.empty()) { err = "dfly: the pack carries no drafter"; return false; }
    if (rows < 2 || rows > kDflyBlockMax || rows > pack.dims.block) {
        err = "dfly: a block of " + std::to_string(rows) + " rows (the drafter's blocks hold 2.." + std::to_string(pack.dims.block) + ")";
        return false;
    }
    d_ = pack.dims;
    rows_ = rows;
    kv_int8_ = kv_int8;
    cudaGetDevice(&device_);
    // ---- the weights: one allocation, each tensor at its own boundary, rounded by the loader on the way in
    const std::vector<DflySpec> specs = dfly_specs(d_);
    const uint64_t wb = weight_bytes(d_);
    if (cudaMalloc(&weights_, wb) != cudaSuccess) {
        cudaGetLastError();
        weights_ = nullptr;
        err = "dfly: the drafter's weights (" + std::to_string(wb >> 20) + " MiB) do not fit";
        return false;
    }
    std::map<std::string, void*> slot;
    const int file = reads.file(path);
    uint64_t at = 0;
    for (const DflySpec& s : specs) {
        const auto it = pack.rec.tensors.find(s.name);
        if (it == pack.rec.tensors.end()) { err = "dfly: no tensor " + s.name; return false; }
        LoadItem item;
        item.file = file;
        item.off = it->second.offset;
        item.bytes = it->second.bytes;
        item.dev = (uint8_t*) weights_ + at;
        item.device = device_;
        item.conv = s.f32 ? Conv::Bf16ToF32 : Conv::Bf16ToQ8_0;
        reads.add(item);
        slot[s.name] = item.dev;
        at += (s.bytes() + kAlign - 1) & ~(kAlign - 1);
    }
    auto w = [&](const std::string& n) { return (const void*) slot.at(n); };
    auto f = [&](const std::string& n) { return (const float*) slot.at(n); };
    layer_.assign((size_t) d_.layers, Layer{});
    for (int i = 0; i < d_.layers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        Layer& l = layer_[(size_t) i];
        l.in_ln = f(p + "input_layernorm.weight");
        l.q = w(p + "self_attn.q_proj.weight"); l.k = w(p + "self_attn.k_proj.weight");
        l.v = w(p + "self_attn.v_proj.weight"); l.o = w(p + "self_attn.o_proj.weight");
        l.q_norm = f(p + "self_attn.q_norm.weight"); l.k_norm = f(p + "self_attn.k_norm.weight");
        l.post_ln = f(p + "post_attention_layernorm.weight");
        l.gate = w(p + "mlp.gate_proj.weight"); l.up = w(p + "mlp.up_proj.weight"); l.down = w(p + "mlp.down_proj.weight");
    }
    final_norm_ = f("final_norm.weight");
    fusion_.eh_proj = w("mtp_fusion.eh_proj.weight");
    fusion_.enorm = f("mtp_fusion.enorm.weight");
    fusion_.hnorm = f("mtp_fusion.hnorm.weight");
    fusion_.gate = f("mtp_fusion.gate");
    corr_.gate = w("hidden_correction.gate_proj.weight");
    corr_.up = w("hidden_correction.up_proj.weight");
    corr_.down = w("hidden_correction.down_proj.weight");
    corr_.hidden_norm = f("hidden_correction.hidden_norm.weight");
    corr_.embed_norm = f("hidden_correction.embed_norm.weight");
    corr_.ffn = d_.corr_ffn;
    conf_w_ = f("confidence_head.proj.weight");
    conf_b_ = f("confidence_head.proj.bias");
    // ---- the working memory of a block
    Buffers count;
    const uint64_t ab = carve(d_, rows_, kv_int8_, nullptr, count);
    if (cudaMalloc(&arena_, ab) != cudaSuccess) {
        cudaGetLastError();
        arena_ = nullptr;
        err = "dfly: the drafter's working memory (" + std::to_string(ab >> 20) + " MiB) does not fit";
        return false;
    }
    if (cudaMemset(arena_, 0, ab) != cudaSuccess) { err = "dfly: clearing the working memory failed"; return false; }
    carve(d_, rows_, kv_int8_, (uint8_t*) arena_, b_);
    vram_ = wb + ab;
    basalt::log::info("basalt dfly", "drafter: %d layers paired with target layers %d..%d, blocks of %d rows, %.0f MiB on "
                      "the GPU (Q8_0 projections), step %lld", d_.layers, d_.target_layers.front(), d_.target_layers.back(),
                      rows_, (double) vram_ / 1048576.0, (long long) d_.step);
    return true;
}

}  // namespace basalt::core
