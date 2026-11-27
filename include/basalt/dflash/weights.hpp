// include/basalt/dflash/weights.hpp - a DFlash 2 block drafter as a pack holds it: its shape (the `<arch>.dflash.*`
// metadata) and its weights (the `dflash.*` tensors), for every model whose pack carries one.
//
// The target's rows at a few layers, projected and normed, are the context (K/V in a sliding window) of a few non-causal
// attention layers that read a block [anchor, mask, ...] and give every slot's candidates at once; a selector walks one
// path through them.  The embedding and the head are the target's.
//
// Nothing here needs a device: a model reads the shape, then names its own loaders for the tensors.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/linear.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace basalt::dflash {

constexpr int kDraftQ = 20;   ///< candidates a position keeps at most (the selector's top_k)

/// The drafter's shape, as the pack's metadata gives it.
struct Meta {
    int n_embd = 0;                          ///< the target's width, which the drafter shares
    int block = 0, mask_id = 0, conv_group = 0, rank = 0, top_k = 0, window = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    float theta = 0.0f, eps = 1e-6f;
    std::vector<int> target_layers;          ///< the target layers whose outputs it reads, in order
    int n_layers = 0, n_ff = 0;              ///< its own layers and their FFN width
};

/// The drafter with its weights on the device.
struct BlockDrafter : Meta {
    struct Layer {
        const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        const float *base_a = nullptr, *base_f = nullptr;   // the two convolutions' base kernels [2][2][n_embd]
        model::Linear q, k, v, o, gate, up, down, conv_a, conv_f;
    };
    bool present = false;
    std::vector<model::Linear> fc;           ///< per target layer, its block of the feature projection
    const float *hidden_norm = nullptr, *norm = nullptr;
    std::vector<Layer> layers;
    const uint16_t *sel_h = nullptr, *pred = nullptr, *succ = nullptr;   // bf16 [rank][n_embd], [vocab][rank] x 2
};

/// Reads the `<arch>.dflash.*` keys of a pack's metadata `kv` into `out`, for a target `n_embd` wide.  False with `err`
/// (led by `arch`) when the shape is one the engine does not run: a block outside 2..8 rows, no target layer, a
/// convolution of other than two taps or whose groups do not divide the width, heads other than 128 or 256 wide or not
/// grouped evenly, more than kDraftQ candidates, or a count that is not positive.
bool read_meta(const std::map<std::string, MetaValue>& kv, const std::string& arch, int n_embd, Meta& out, std::string& err);

/// What a model gives read_weights to place the pack's tensors: each returns the tensor on the device, null (false for a
/// matrix) when the pack lacks it or holds another type or size.
struct Loaders {
    std::function<const float*(const std::string& name, int64_t elems)> floats;      ///< an F32 tensor of `elems` values
    std::function<const uint16_t*(const std::string& name, int64_t elems)> bf16;     ///< a BF16 tensor of `elems` values
    /// The matrix `<base>.weight` of rows x cols (and its scale, where its type has one).
    std::function<bool(const std::string& base, int64_t rows, int64_t cols, model::Linear& out)> linear;
};

/// Places the pack's `dflash.*` tensors in `d` (its shape already read) through `load`, in this order: the feature
/// projection `dflash.fc.N` per target layer, the two norms, each layer `dflash.blk.L.*`, then the selector over a
/// vocabulary of `n_vocab`.  `d.present` is set once every tensor is in; false when one is not.
bool read_weights(BlockDrafter& d, int64_t n_vocab, const Loaders& load);

}  // namespace basalt::dflash
