// include/basalt/dfly2/record.hpp - what the DFly2 drafters share that needs no device: the record a pack keeps of its `dfly`
// section (meta.dfly2), the row of a decode window a block starts from, and the order a block's drafts are made in.
//
// The record holds the settings every drafter has and the section's tensors. A model reads its own keys from meta.dfly2
// itself and decides which of the shared values it runs: nothing here refuses a setting a drafter could hold.
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace basalt::dfly2 {

constexpr int kTapRows = 16;    ///< rows of a window the tap keeps (the draft policy's widest window)
constexpr int kMaxLayers = 8;   ///< the most draft layers a record may name

/// One tensor of the section, as the record lists it.
struct Tensor {
    uint64_t offset = 0, bytes = 0;   ///< absolute in the file
    std::vector<uint64_t> shape;      ///< safetensors order, outermost first
    std::string dtype;                ///< the stored type as the record names it; empty when it names none
};

/// The pack's record of a drafter: the settings shared by every drafter, and its tensors.
struct Record {
    bool present = false;
    int layers = 0;                   ///< draft layers
    int mask_id = 0;                  ///< the mask token (mask_token_id)
    int topk = 0;                     ///< the target's selection width (target_index_topk); 0 when the record has none
    int local_window = 0;             ///< rows of the context's near window (context_local_window); 0: none
    int block = 0;                    ///< rows of a block (block_size); 0 when the record has none
    std::vector<int> kv_layers;       ///< the target layer whose cache each draft layer reads
    std::vector<int> idx_layers;      ///< the target layer whose selection each draft layer reads
    std::map<std::string, Tensor> tensors;   ///< by the name the record gives each
};

/// Reads meta.dfly2 (null or absent: present stays false) of a pack whose `dfly` section starts at `section_offset`;
/// offsets in the record are relative to it. False with `err` when the record is malformed: the layer count, the two
/// layer lists or the mask token are missing or disagree, or a tensor lacks its offset, bytes or shape.
bool parse_record(const core::Json& dfly2, uint64_t section_offset, Record& out, std::string& err);

/// The window row whose output chose the token at `pos`: row pos - 1 - pos0 of a window that began at `pos0`, when its hidden
/// state (bit r of `hid_ok`) and every one of the `layers` selections (bits of sel_ok[r], kTapRows words) were kept; else -1.
inline int source_row(int64_t pos0, unsigned hid_ok, const unsigned* sel_ok, int layers, int64_t pos) {
    const int64_t r = pos - 1 - pos0;
    if (!sel_ok || pos0 < 0 || r < 0 || r >= kTapRows || !(hid_ok & (1u << r)) || sel_ok[r] != (1u << layers) - 1u) return -1;
    return (int) r;
}

/// The drafts of a block in order, `step(j, previous)` giving draft j after the token before it (the anchor first), or
/// ending the block before it (`go` false): `made` drafts in `ids` and `conf` (room for `n` each). The loop a drafter's
/// heads run, apart from the device, so a test can run it over fake logits. False when a step fails.
bool draft_loop(int n, int32_t anchor,
                const std::function<bool(int j, int32_t previous, int32_t& token, float& conf, bool& go)>& step,
                int32_t* ids, float* conf, int& made);

}  // namespace basalt::dfly2
