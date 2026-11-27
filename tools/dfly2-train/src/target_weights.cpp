// tools/dfly2-train/src/target_weights.cpp - the target's weights from a .basalt pack (target_weights.hpp).
#include "target_weights.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/model_format.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace basalt::dfly2 {

namespace {

/// One row of `cols` values of GGUF type `type` at `src`, as FP32. The caller has checked the type with row_bytes.
void dequantize_row(uint32_t type, const uint8_t* src, float* dst, int64_t cols) {
    switch (type) {
    case 0: dequantize_f32(src, dst, (int) cols); break;
    case 1: dequantize_f16(src, dst, (int) cols); break;
    case 30: dequantize_bf16(src, dst, (int) cols); break;
    case 8: for (int64_t b = 0; b < cols / 32; ++b) dequantize_q8_0(src + b * 34, dst + b * 32); break;
    case 12: for (int64_t b = 0; b < cols / 256; ++b) dequantize_q4_K(src + b * 144, dst + b * 256); break;
    case 13: for (int64_t b = 0; b < cols / 256; ++b) dequantize_q5_K(src + b * 176, dst + b * 256); break;
    case 14: for (int64_t b = 0; b < cols / 256; ++b) dequantize_q6_K(src + b * 210, dst + b * 256); break;
    default: break;
    }
}

/// Bytes of one row of `cols` values of `type`; 0 for a type this reader does not dequantise or a row that is not
/// whole blocks.
uint64_t row_bytes(uint32_t type, int64_t cols) {
    if (type != 0 && type != 1 && type != 30 && type != 8 && type != 12 && type != 13 && type != 14) return 0;
    int elems = 0, bytes = 0;
    if (!block_geometry(type, elems, bytes) || cols <= 0 || cols % elems != 0) return 0;
    return (uint64_t) (cols / elems) * (uint64_t) bytes;
}

/// A metadata number, or false when the key is missing or not a number.
bool meta_num(const GgufFile& g, const std::string& key, double& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr || !v->is_num()) { err = "the pack's metadata has no number " + key; return false; }
    out = v->num();
    return true;
}

}  // namespace

std::string PackIdentity::meta_sha256_hex() const {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (const uint8_t b : meta_sha256) {
        out += digits[b >> 4];
        out += digits[b & 15];
    }
    return out;
}

int64_t TargetGeometry::selection_width(int64_t layer) const {
    if (layer < 0 || layer >= (int64_t) compress.size() || compress[(size_t) layer] <= 0) return 0;
    return top_k + compress[(size_t) layer] - 1;
}

std::vector<int> TargetGeometry::last_sparse_layers(int64_t count) const {
    std::vector<int> out;
    for (int64_t l = (int64_t) compress.size() - 1; l >= 0 && (int64_t) out.size() < count; --l)
        if (compress[(size_t) l] > 0) out.insert(out.begin(), (int) l);
    if ((int64_t) out.size() != count) out.clear();
    return out;
}

bool TargetPack::open(const std::string& path, std::string& err) {
    gguf_.reset();
    identity_ = PackIdentity{};
    geometry_ = TargetGeometry{};
    if (!file_.open(path, err) || !file_.load_meta(err) || !file_.register_gguf_view(err)) return false;
    try {
        gguf_ = std::make_unique<GgufFile>(path);
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }

    // identity: the name, the size, and the digest the section table holds for `meta`
    const size_t slash = path.find_last_of("/\\");
    identity_.name = slash == std::string::npos ? path : path.substr(slash + 1);
    identity_.bytes = file_.file_bytes();
    std::vector<uint8_t> head((size_t) core::kHeaderBytes);
    if (!core::read_range_at(core::FileRange(path), 0, head.data(), core::kHeaderBytes, err)) return false;
    core::FileHeader h;
    std::memcpy(&h, head.data(), sizeof h);
    for (uint32_t i = 0; i < h.n_sections && i < core::kMaxSections; ++i) {
        core::SectionEntry e;
        std::memcpy(&e, head.data() + sizeof(core::FileHeader) + (size_t) i * sizeof(core::SectionEntry), sizeof e);
        if (std::strncmp(e.name, "meta", sizeof e.name) == 0) std::memcpy(identity_.meta_sha256.data(), e.sha256, 32);
    }

    // geometry: the source model's metadata keys under its architecture name
    const MetaValue* arch = gguf_->get("general.architecture");
    if (arch == nullptr || arch->type != MetaType::STRING || arch->s.empty()) { err = path + ": the pack's metadata has no general.architecture"; return false; }
    TargetGeometry& g = geometry_;
    g.arch = arch->s;
    const std::string a = g.arch + ".";
    double layers = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, n_rot = 0, top_k = 0;
    if (!meta_num(*gguf_, a + "block_count", layers, err) || !meta_num(*gguf_, a + "embedding_length", hidden, err) ||
        !meta_num(*gguf_, a + "attention.head_count", heads, err) || !meta_num(*gguf_, a + "attention.head_count_kv", kv_heads, err) ||
        !meta_num(*gguf_, a + "attention.key_length", head_dim, err) || !meta_num(*gguf_, a + "rope.dimension_count", n_rot, err) ||
        !meta_num(*gguf_, a + "rope.freq_base", g.rope_base, err) || !meta_num(*gguf_, a + "attention.layer_norm_rms_epsilon", g.attn_norm_eps, err) ||
        !meta_num(*gguf_, a + "attention.indexer.top_k", top_k, err)) {
        err = path + ": " + err;
        return false;
    }
    g.n_layers = (int64_t) layers;
    g.hidden = (int64_t) hidden;
    g.n_head = (int64_t) heads;
    g.n_head_kv = (int64_t) kv_heads;
    g.head_dim = (int64_t) head_dim;
    g.n_rot = (int64_t) n_rot;
    g.top_k = (int64_t) top_k;
    const MetaValue* ratios = gguf_->get(a + "attention.compress_ratios");
    if (ratios == nullptr || ratios->type != MetaType::ARRAY || (int64_t) ratios->count != g.n_layers || ratios->items.size() != ratios->count) {
        err = path + ": the pack's " + a + "attention.compress_ratios does not list its " + std::to_string(g.n_layers) + " layers";
        return false;
    }
    for (const MetaValue& r : ratios->items) g.compress.push_back(r.is_num() ? (int64_t) r.num() : 0);
    if (g.n_layers <= 0 || g.hidden <= 0 || g.n_head <= 0 || g.n_head_kv <= 0 || g.head_dim <= 0 || g.n_rot < 0 || g.top_k <= 0) {
        err = path + ": the pack's geometry has a dimension that is not positive";
        return false;
    }
    // the vocabulary is the embedding's row count
    const TensorInfo* embd = gguf_->find("token_embd.weight");
    if (embd == nullptr || embd->shape.size() != 2 || (int64_t) embd->shape[0] != g.hidden || embd->shape[1] == 0) {
        err = path + ": token_embd.weight is missing or is not [vocab][" + std::to_string(g.hidden) + "]";
        return false;
    }
    g.vocab = (int64_t) embd->shape[1];
    return true;
}

bool TargetPack::tensor(const std::string& name, torch::Tensor& out, std::string& err) const {
    if (gguf_ == nullptr) { err = "no pack is open"; return false; }
    const TensorInfo* t = gguf_->find(name);
    if (t == nullptr) { err = file_.path() + " has no tensor " + name; return false; }
    if (t->shape.empty() || t->shape.size() > 2 || t->shape[0] == 0 || t->shape[0] > (1ull << 31) || (t->shape.size() == 2 && (t->shape[1] == 0 || t->shape[1] > (1ull << 31)))) {
        err = name + " is not a vector or a matrix";
        return false;
    }
    // GGUF lists the fastest dimension first: a matrix is [in][out] there and [out][in] here
    const int64_t cols = (int64_t) t->shape[0], rows = t->shape.size() == 2 ? (int64_t) t->shape[1] : 1;
    const uint64_t per_row = row_bytes(t->type, cols);
    if (per_row == 0) { err = name + " is stored as " + t->type_name() + ", which the trainer does not dequantise"; return false; }
    const uint64_t bytes = per_row * (uint64_t) rows, size = gguf_->file_size();
    if (t->offset > size || bytes > size - t->offset) { err = name + " runs past the end of " + file_.path(); return false; }
    const uint8_t* src = gguf_->tensor_data(*t);
    out = t->shape.size() == 2 ? torch::empty({rows, cols}, torch::kFloat32) : torch::empty({cols}, torch::kFloat32);
    float* dst = out.data_ptr<float>();
    const uint32_t type = t->type;
    // rows are whole blocks, so they dequantise independently
    const int64_t workers = std::max<int64_t>(1, std::min<int64_t>({(int64_t) std::thread::hardware_concurrency(), 16, rows / 64}));
    std::vector<std::thread> pool;
    for (int64_t w = 0; w < workers; ++w)
        pool.emplace_back([=] {
            for (int64_t r = rows * w / workers; r < rows * (w + 1) / workers; ++r) dequantize_row(type, src + (uint64_t) r * per_row, dst + r * cols, cols);
        });
    for (std::thread& th : pool) th.join();
    return true;
}

bool TargetPack::attention(int64_t layer, TargetAttention& out, std::string& err) const {
    const TargetGeometry& g = geometry_;
    if (g.selection_width(layer) == 0) { err = "target layer " + std::to_string(layer) + " has no sparse attention to pair with"; return false; }
    const std::string p = "blk." + std::to_string(layer) + ".";
    if (!tensor(p + "attn_q.weight", out.q, err) || !tensor(p + "attn_k.weight", out.k, err) || !tensor(p + "attn_v.weight", out.v, err) ||
        !tensor(p + "attn_output.weight", out.o, err) || !tensor(p + "attn_q_norm.weight", out.q_norm, err) || !tensor(p + "attn_k_norm.weight", out.k_norm, err))
        return false;
    const auto is = [](const torch::Tensor& t, std::initializer_list<int64_t> shape) { return t.sizes() == torch::IntArrayRef(shape); };
    if (!is(out.q, {2 * g.n_head * g.head_dim, g.hidden}) || !is(out.k, {g.n_head_kv * g.head_dim, g.hidden}) || !is(out.v, {g.n_head_kv * g.head_dim, g.hidden}) ||
        !is(out.o, {g.hidden, g.n_head * g.head_dim}) || !is(out.q_norm, {g.head_dim}) || !is(out.k_norm, {g.head_dim})) {
        err = "target layer " + std::to_string(layer) + "'s attention tensors do not have the shapes its metadata gives";
        return false;
    }
    return true;
}

bool TargetPack::embedding(torch::Tensor& out, std::string& err) const {
    if (!tensor("token_embd.weight", out, err)) return false;
    if (out.dim() != 2 || out.size(0) != geometry_.vocab || out.size(1) != geometry_.hidden) { err = "token_embd.weight is not [vocab][hidden]"; return false; }
    return true;
}

bool TargetPack::head(torch::Tensor& out, std::string& err) const {
    if (!tensor("output.weight", out, err)) return false;
    if (out.dim() != 2 || out.size(0) != geometry_.vocab || out.size(1) != geometry_.hidden) { err = "output.weight is not [vocab][hidden]"; return false; }
    return true;
}

bool TargetPack::drafter_shape(const std::vector<int>& layers, int64_t n_layers, int64_t block, int64_t mask_token_id,
                               double rms_norm_eps, DrafterShape& out, std::string& err) const {
    const TargetGeometry& g = geometry_;
    out = DrafterShape{};
    out.target_layers = layers.empty() ? g.last_sparse_layers(n_layers) : layers;
    if (out.target_layers.empty()) { err = "the target has fewer than " + std::to_string(n_layers) + " layers with the sparse attention"; return false; }
    out.n_layers = (int64_t) out.target_layers.size();
    // every paired layer must cache cells and select them; the drafter reads the widest selection
    for (const int l : out.target_layers) {
        const int64_t width = g.selection_width(l);
        if (width == 0) { err = "target layer " + std::to_string(l) + " has no sparse attention to pair with"; return false; }
        out.sel_cap = std::max(out.sel_cap, width);
    }
    out.hidden = g.hidden;
    out.intermediate = 3 * g.hidden;
    out.correction = g.hidden;
    out.n_head = g.n_head;
    out.n_head_kv = g.n_head_kv;
    out.head_dim = g.head_dim;
    out.n_rot = g.n_rot;
    out.rope_base = g.rope_base;
    out.attn_norm_eps = g.attn_norm_eps;
    out.rms_norm_eps = rms_norm_eps;
    out.vocab = g.vocab;
    out.block = block;
    out.mask_token_id = mask_token_id;
    return drafter_check_shape(out, err);
}

}  // namespace basalt::dfly2
