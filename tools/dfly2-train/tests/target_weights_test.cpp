// tools/dfly2-train/tests/target_weights_test.cpp - reading the target from a pack: a synthetic .basalt whose tensors
// are stored as a real pack stores them reads back with the geometry, the identity and the dequantised values, and a
// drafter built from it starts with the paired layers' attention. No model, no GPU.
#include "target_weights.hpp"

#include "test_util.hpp"
#include "tiny.hpp"
#include "tiny_pack.hpp"

using namespace basalt::dfly2;
using dfly2_test::check;
using dfly2_test::refused;
using dfly2_test::max_abs_diff;

int main() {
    return dfly2_test::run("target_weights_test", []() -> int {
    torch::manual_seed(3);
    dfly2_test::TempDir dir("basalt-dfly2-target-test");
    const dfly2_test::TinyTarget tt;
    const dfly2_test::TinyPack pack = dfly2_test::write_tiny_pack(dir.file("tiny.basalt"), tt, 17);
    std::string err;

    std::printf("1. geometry and identity\n");
    TargetPack target;
    const bool opened = target.open(pack.path, err);
    check(opened, "the pack opens", err);
    if (!opened) return dfly2_test::finish("target_weights_test");
    const TargetGeometry& g = target.geometry();
    check(g.arch == "qwen4exp" && g.n_layers == 8 && g.hidden == 256 && g.vocab == 48 && g.n_head == 4 && g.n_head_kv == 2 && g.head_dim == 8 &&
              g.n_rot == 4 && g.rope_base == 1e7 && g.top_k == 8, "the geometry is the metadata's and the vocabulary the embedding's rows");
    check(std::abs(g.attn_norm_eps - 1e-6) < 1e-12, "the attention norm's eps is the metadata's");
    check(g.selection_width(3) == 11 && g.selection_width(7) == 11 && g.selection_width(2) == 0, "a sparse layer selects top_k + compress - 1 cells, another layer none");
    check(g.last_sparse_layers(2) == std::vector<int>({3, 7}) && g.last_sparse_layers(1) == std::vector<int>({7}) && g.last_sparse_layers(3).empty(), "the last sparse layers");
    const PackIdentity& id = target.identity();
    check(id.name == "tiny.basalt" && id.bytes == std::filesystem::file_size(dir.path / "tiny.basalt") && std::memcmp(id.meta_sha256.data(), pack.meta_sha256, 32) == 0,
          "the identity is the file's name, its size and the section table's digest of meta");
    check(id.meta_sha256_hex().substr(0, 8) == "a0a1a2a3" && id.meta_sha256_hex().size() == 64, "the digest in hex");

    std::printf("2. dequantised tensors\n");
    {
        bool same = true;
        for (const auto& [name, expect] : pack.expect) {
            torch::Tensor got;
            const bool ok = target.tensor(name, got, err) && max_abs_diff(got, expect) == 0.0;
            if (!ok) std::printf("    differs: %s %s\n", name.c_str(), err.c_str());
            same = same && ok;
        }
        check(same, "every tensor (Q6_K, Q8_0, F32) has the engine dequantiser's values, a matrix as [out][in]");
        torch::Tensor e, h, missing;
        check(target.embedding(e, err) && e.size(0) == 48 && e.size(1) == 256 && target.head(h, err) && h.size(0) == 48 && h.size(1) == 256, "the embedding and the head are [vocab][hidden]");
        check(e.abs().max().item<float>() > 0.01f && h.abs().max().item<float>() > 0.001f, "they hold values");
        refused(!target.tensor("blk.0.attn_q.weight", missing, err), "a tensor the pack does not have", err);
    }

    std::printf("3. a drafter from the pack\n");
    {
        DrafterShape shape;
        check(target.drafter_shape({}, 2, 8, 47, 1e-5, shape, err), "the default pairing", err);
        check(shape.target_layers == std::vector<int>({3, 7}) && shape.hidden == 256 && shape.intermediate == 768 && shape.correction == 256 && shape.n_head == 4 &&
                  shape.n_head_kv == 2 && shape.head_dim == 8 && shape.n_rot == 4 && shape.vocab == 48 && shape.sel_cap == 11 && shape.block == 8 && shape.mask_token_id == 47,
              "the shape pairs the last sparse layers and takes the target's attention geometry");
        Drafter drafter(shape);
        bool copied = true;
        for (int64_t i = 0; i < shape.n_layers; ++i) {
            TargetAttention t;
            copied = copied && target.attention(shape.target_layers[(size_t) i], t, err) && drafter->init_attention(i, t, err);
        }
        check(copied, "each paired layer's attention copies in", err);
        const DraftAttentionImpl& a = *drafter->layers->ptr<DraftLayerImpl>(1)->self_attn;
        check(max_abs_diff(a.q_proj->weight, pack.expect.at("blk.7.attn_q.weight")) == 0.0 && max_abs_diff(a.k_proj->weight, pack.expect.at("blk.7.attn_k.weight")) == 0.0 &&
                  max_abs_diff(a.v_proj->weight, pack.expect.at("blk.7.attn_v.weight")) == 0.0 && max_abs_diff(a.o_proj->weight, pack.expect.at("blk.7.attn_output.weight")) == 0.0 &&
                  max_abs_diff(a.q_norm->weight, pack.expect.at("blk.7.attn_q_norm.weight")) == 0.0 && max_abs_diff(a.k_norm->weight, pack.expect.at("blk.7.attn_k_norm.weight")) == 0.0,
              "draft layer 1 holds target layer 7's projections and norms");
        TargetAttention t;
        refused(!target.attention(2, t, err), "a layer without the sparse attention is refused", err);
        refused(!target.drafter_shape({2, 7}, 2, 8, 47, 1e-5, shape, err), "pairing with such a layer is refused", err);
        refused(!target.drafter_shape({}, 3, 8, 47, 1e-5, shape, err), "more pairs than the target has sparse layers", err);
        refused(!target.drafter_shape({}, 2, 8, 48, 1e-5, shape, err), "a mask token outside the vocabulary", err);
    }

    std::printf("4. refusals\n");
    {
        TargetPack bad;
        refused(!bad.open(dir.file("missing.basalt"), err), "a missing file", err);
        const std::string junk = dir.file("junk.basalt");
        std::ofstream(std::filesystem::path(std::u8string(junk.begin(), junk.end())), std::ios::binary) << std::string(100000, 'x');
        refused(!bad.open(junk, err), "a file that is not a pack", err);
        torch::Tensor none;
        refused(!bad.tensor("output.weight", none, err), "a tensor of a pack that did not open", err);
    }
    return dfly2_test::finish("target_weights_test");
    });
}
