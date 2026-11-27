#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {
class LoadPlan;

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (basalt::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    /// Allocates the head and queues its rows' read in `reads` (the caller's pass places them).
    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, LoadPlan& reads, std::string& err);
    /// Once the rows have landed: the forms some kernels read are made from them (a Q8_0 head's bit planes, a Q6_K
    /// head's 2-bit copy).
    bool finish(std::string& err);
    /// The bytes `load` uploads for `output.weight`, from the GGUF headers only; `type`, when given, gets its ggml type,
    /// which decides the head's kernels before the head is loaded.
    static bool plan_bytes(const std::vector<std::string>& shards, uint64_t& out, std::string& err, int* type = nullptr);
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    /// Rows [0, n) of the head rows at `w` (weights(), or a copy of a run of its rows on the stream's device) against
    /// T q8_1 columns of the head's input width: T rows of n logits in `y`, each the bits of the whole head's.  The
    /// kernel is the one the head's type and T select; launches only.  Throws std::invalid_argument on an unloaded
    /// head, a null buffer or a count out of range, and whatever the kernel throws.
    void score_rows(const void* w, const void* x_q8_1, float* y, int n, int T, void* stream) const;
    /// Whether a head of this ggml type has the two-stage argmax of draft_top1.
    static bool has_draft_top1(int type) { return type == 13 || type == 14 || type == 8; }
    /// One q8_1 column's argmax over the head rows `map` names and its probability, by the 2-bit copy `proxy` of those
    /// rows (rows near the best are scored exactly from `rows`), with the kernel of the head's type. `rows_by_map`:
    /// `rows` is the whole head and the copy's row r is its row map[r]; a Q5_K head has no such form, its `rows` are
    /// the copy's rows in order. False: this type has none (nothing launched).
    bool draft_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap, const int32_t* map,
                    int* out_id, float* out_prob, void* scratch, void* stream, bool rows_by_map) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }
    /// A Q6_K head's 2-bit copy for the verify window's two-stage head (q6k_vhead); null for other types or when it
    /// did not fit.
    const void* vproxy() const { return vproxy_; }
    /// A Q8_0 head held as bit planes (q80_vhead.hpp): weights() is in that layout for every reader.
    bool planes() const { return planes_; }

private:
    void* weights_ = nullptr;
    void* vproxy_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
    bool planes_ = false;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token.  The rows of the lowest ids (the byte-level BPE gives the most frequent tokens the
/// lowest ids) also have a VRAM copy, `kHotRows` of them, which the gathers read instead: it is allocated before the
/// expert cache is sized, so it comes out of that cache.  `embed_row`, the verify window and the MTP drafter use it
/// instead of the canonical table when it is set (`set_native_embed`).
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    /// The table in pinned host memory (or VRAM) and its hot rows' copy, read in the caller's pass (`reads`).
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, LoadPlan& reads, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    /// Rows with a VRAM copy (0: none), and its bytes.
    int64_t hot_rows() const { return n_hot_; }
    uint64_t hot_bytes() const { return (uint64_t) n_hot_ * row_; }
    static constexpr int64_t kHotRows = 8192;

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    void* hot_ = nullptr;
    int hot_device_ = -1;   ///< the device that holds hot_; the other devices read every row from the mapped table
    int64_t n_hot_ = 0;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace basalt::core
