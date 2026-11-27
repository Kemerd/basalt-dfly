// tools/dfly2-train/src/target_weights.hpp - what the trainer reads of the target from a .basalt pack: its geometry,
// the token embedding, the output head, and the attention projections and norms of the layers a drafter pairs with.
// The pack is read through the engine's own reader and dequantisers, so a weight here has the value the engine
// serves.
#pragma once

#include "drafter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace basalt::dfly2 {

/// What names a pack in a shard's header and a checkpoint's metadata.
struct PackIdentity {
    std::string name;                         ///< file name without its directory, UTF-8
    uint64_t bytes = 0;                       ///< the pack's size
    std::array<uint8_t, 32> meta_sha256{};    ///< the digest its section table records for `meta`
    /// The digest as 64 lower-case hex digits.
    std::string meta_sha256_hex() const;
};

/// The target's shape, from the pack's metadata and its tensors' shapes.
struct TargetGeometry {
    std::string arch;                 ///< general.architecture
    int64_t n_layers = 0;             ///< target layers
    int64_t hidden = 0, vocab = 0;
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    double rope_base = 0, attn_norm_eps = 0;
    int64_t top_k = 0;                ///< the indexer's selection budget
    std::vector<int64_t> compress;    ///< per layer: cells one indexer key pools, 0 for a layer without the sparse attention
    /// The widest selection of layer `layer`: top_k + compress - 1 (the budget in whole pools plus an unfilled tail).
    int64_t selection_width(int64_t layer) const;
    /// The last `count` layers that have the sparse attention, ascending; empty when there are fewer.
    std::vector<int> last_sparse_layers(int64_t count) const;
};

class TargetPack {
public:
    /// Opens `path`, parses its metadata and reads the geometry. False with `err` when the file is not a pack the
    /// trainer can use.
    bool open(const std::string& path, std::string& err);
    const PackIdentity& identity() const { return identity_; }
    const TargetGeometry& geometry() const { return geometry_; }

    /// The tensor `name`, dequantised to FP32 on the CPU: a matrix as [out][in], a vector as [n].
    bool tensor(const std::string& name, torch::Tensor& out, std::string& err) const;
    /// Target layer `layer`'s attention projections and norms, checked against the geometry.
    bool attention(int64_t layer, TargetAttention& out, std::string& err) const;
    /// token_embd.weight as [vocab][hidden].
    bool embedding(torch::Tensor& out, std::string& err) const;
    /// output.weight as [vocab][hidden].
    bool head(torch::Tensor& out, std::string& err) const;

    /// A drafter shape for this target: `layers` paired layers (the last sparse ones when empty), the contract's
    /// dense widths (an MLP of 3 x hidden, a correction of hidden) and the given block and mask token.
    bool drafter_shape(const std::vector<int>& layers, int64_t n_layers, int64_t block, int64_t mask_token_id,
                       double rms_norm_eps, DrafterShape& out, std::string& err) const;

private:
    core::ModelFile file_;
    std::unique_ptr<GgufFile> gguf_;
    PackIdentity identity_;
    TargetGeometry geometry_;
};

}  // namespace basalt::dfly2
