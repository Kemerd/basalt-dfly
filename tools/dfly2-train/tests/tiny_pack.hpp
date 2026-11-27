// tools/dfly2-train/tests/tiny_pack.hpp - writes a small .basalt pack for the tests: a target of eight layers, two of
// them with the sparse attention, whose tensors are stored in the encodings a real pack uses (Q6_K and Q8_0
// projections, Q8_0 embedding, Q6_K head, F32 norms). The values each tensor must dequantise to are kept beside it.
#pragma once

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/model_format.hpp"

#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace dfly2_test {

/// The tiny target's geometry.
struct TinyTarget {
    int64_t hidden = 256, vocab = 48, n_head = 4, n_head_kv = 2, head_dim = 8, n_rot = 4, n_layers = 8, top_k = 8, compress = 4;
    std::vector<int> sparse = {3, 7};   ///< the layers with the sparse attention
};

/// A written pack: its path and what every tensor dequantises to ([out][in] for a matrix).
struct TinyPack {
    std::string path;
    std::map<std::string, torch::Tensor> expect;
    uint8_t meta_sha256[32] = {};
};

namespace pack_detail {

/// The FP16 bit pattern of `v` and the value it holds.
inline uint16_t half_bits(float v, float& held) {
    const torch::Tensor h = torch::tensor(v).to(torch::kFloat16);
    uint16_t bits = 0;
    std::memcpy(&bits, h.data_ptr(), 2);
    held = h.to(torch::kFloat32).item<float>();
    return bits;
}

/// rows x cols values as Q8_0: per 32 values an FP16 scale and 32 int8 codes. The expectation is computed here.
inline std::string encode_q8_0(int64_t rows, int64_t cols, std::mt19937& rng, torch::Tensor& expect) {
    std::string out;
    expect = torch::empty({rows, cols}, torch::kFloat32);
    float* e = expect.data_ptr<float>();
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.001f, 0.01f);
    for (int64_t b = 0; b < rows * cols / 32; ++b) {
        float d = 0;
        const uint16_t bits = half_bits(scale(rng), d);
        out.append((const char*) &bits, 2);
        for (int j = 0; j < 32; ++j) {
            const int8_t q = (int8_t) code(rng);
            out.push_back((char) q);
            e[b * 32 + j] = d * (float) q;
        }
    }
    return out;
}

/// rows x cols values as Q6_K blocks of random codes and scales; the expectation is the engine's dequantiser's.
inline std::string encode_q6_k(int64_t rows, int64_t cols, std::mt19937& rng, torch::Tensor& expect) {
    std::string out;
    expect = torch::empty({rows, cols}, torch::kFloat32);
    std::uniform_int_distribution<int> byte(0, 255), scales(-60, 60);
    std::uniform_real_distribution<float> scale(0.00002f, 0.0002f);
    for (int64_t b = 0; b < rows * cols / 256; ++b) {
        std::string block;
        for (int j = 0; j < 128 + 64; ++j) block.push_back((char) byte(rng));   // low nibbles, then high bit pairs
        for (int j = 0; j < 16; ++j) block.push_back((char) (int8_t) scales(rng));
        float d = 0;
        const uint16_t bits = half_bits(scale(rng), d);
        block.append((const char*) &bits, 2);
        basalt::dequantize_q6_K((const uint8_t*) block.data(), expect.data_ptr<float>() + b * 256);
        out += block;
    }
    return out;
}

/// `values` as raw F32.
inline std::string encode_f32(const torch::Tensor& values) {
    const torch::Tensor c = values.contiguous();
    return std::string((const char*) c.data_ptr<float>(), (size_t) c.numel() * 4);
}

}  // namespace pack_detail

/// Writes the pack to `path` (UTF-8). The section table's digest for `meta` is a fixed pattern.
inline TinyPack write_tiny_pack(const std::string& path, const TinyTarget& t, unsigned seed) {
    using namespace basalt::core;
    TinyPack pack;
    pack.path = path;
    std::mt19937 rng(seed);
    struct Entry { std::string name, type; int type_id; std::vector<int64_t> ne; std::string bytes; };
    std::vector<Entry> entries;
    const auto matrix = [&](const std::string& name, int64_t out, int64_t in) {
        // the converter's rule: Q6_K when the input width is whole 256-value blocks, else Q8_0
        torch::Tensor expect;
        if (in % 256 == 0) entries.push_back({name, "Q6_K", 14, {in, out}, pack_detail::encode_q6_k(out, in, rng, expect)});
        else entries.push_back({name, "Q8_0", 8, {in, out}, pack_detail::encode_q8_0(out, in, rng, expect)});
        pack.expect[name] = expect;
    };
    const auto norm = [&](const std::string& name, int64_t n) {
        const torch::Tensor w = torch::rand({n}) + 0.5;
        entries.push_back({name, "F32", 0, {n}, pack_detail::encode_f32(w)});
        pack.expect[name] = w;
    };
    {
        torch::Tensor expect;
        entries.push_back({"token_embd.weight", "Q8_0", 8, {t.hidden, t.vocab}, pack_detail::encode_q8_0(t.vocab, t.hidden, rng, expect)});
        pack.expect["token_embd.weight"] = expect;
    }
    matrix("output.weight", t.vocab, t.hidden);
    for (const int l : t.sparse) {
        const std::string p = "blk." + std::to_string(l) + ".";
        matrix(p + "attn_q.weight", 2 * t.n_head * t.head_dim, t.hidden);
        matrix(p + "attn_k.weight", t.n_head_kv * t.head_dim, t.hidden);
        matrix(p + "attn_v.weight", t.n_head_kv * t.head_dim, t.hidden);
        matrix(p + "attn_output.weight", t.hidden, t.n_head * t.head_dim);
        norm(p + "attn_q_norm.weight", t.head_dim);
        norm(p + "attn_k_norm.weight", t.head_dim);
    }

    // the gguf section: each tensor at a multiple of kTensorAlign
    std::string gguf, tensors_json;
    for (const Entry& e : entries) {
        gguf.resize((gguf.size() + kTensorAlign - 1) / kTensorAlign * kTensorAlign, '\0');
        std::string shape;
        for (size_t i = 0; i < e.ne.size(); ++i) shape += (i != 0 ? "," : "") + std::to_string(e.ne[i]);
        tensors_json += std::string(tensors_json.empty() ? "" : ",") + "\"" + e.name + "\":{\"type\":\"" + e.type + "\",\"type_id\":" + std::to_string(e.type_id) +
                        ",\"shape\":[" + shape + "],\"bytes\":" + std::to_string(e.bytes.size()) + ",\"offset\":" + std::to_string(gguf.size()) + "}";
        gguf += e.bytes;
    }
    std::string ratios;
    for (int64_t l = 0; l < t.n_layers; ++l) {
        bool sparse = false;
        for (const int s : t.sparse) sparse = sparse || s == l;
        ratios += (l != 0 ? "," : "") + std::to_string(sparse ? t.compress : 0);
    }
    const std::string meta =
        "{\"format\":\"basalt\",\"version\":2,\"name\":\"tiny\",\"gguf_kv\":{\"general.architecture\":\"qwen4exp\","
        "\"qwen4exp.block_count\":" + std::to_string(t.n_layers) + ",\"qwen4exp.embedding_length\":" + std::to_string(t.hidden) +
        ",\"qwen4exp.attention.head_count\":" + std::to_string(t.n_head) + ",\"qwen4exp.attention.head_count_kv\":" + std::to_string(t.n_head_kv) +
        ",\"qwen4exp.attention.key_length\":" + std::to_string(t.head_dim) + ",\"qwen4exp.rope.dimension_count\":" + std::to_string(t.n_rot) +
        ",\"qwen4exp.rope.freq_base\":10000000.0,\"qwen4exp.attention.layer_norm_rms_epsilon\":1e-06,\"qwen4exp.attention.indexer.top_k\":" +
        std::to_string(t.top_k) + ",\"qwen4exp.attention.compress_ratios\":[" + ratios + "]},\"tensors\":{" + tensors_json + "}}";

    // the container: header, section table, sections at kSectionAlign
    std::vector<char> file((size_t) kHeaderBytes, 0);
    FileHeader h{};
    std::memcpy(h.magic, kModelMagic, 8);
    h.version = kModelVersion;
    h.n_sections = 2;
    const std::pair<const char*, const std::string*> sections[2] = {{"meta", &meta}, {"gguf", &gguf}};
    for (size_t i = 0; i < 2; ++i) {
        const uint64_t off = (file.size() + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
        file.resize((size_t) off, 0);
        file.insert(file.end(), sections[i].second->begin(), sections[i].second->end());
        SectionEntry e{};
        std::memcpy(e.name, sections[i].first, std::strlen(sections[i].first));
        e.offset = off;
        e.bytes = sections[i].second->size();
        if (i == 0) {
            for (int j = 0; j < 32; ++j) e.sha256[j] = pack.meta_sha256[j] = (uint8_t) (0xa0 + j);
        }
        std::memcpy(file.data() + sizeof h + i * sizeof e, &e, sizeof e);
    }
    h.file_bytes = file.size();
    std::memcpy(file.data(), &h, sizeof h);
    std::ofstream(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary).write(file.data(), (std::streamsize) file.size());
    return pack;
}

}  // namespace dfly2_test
