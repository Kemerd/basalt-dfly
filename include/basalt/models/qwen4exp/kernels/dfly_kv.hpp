// include/basalt/models/qwen4exp/kernels/dfly_kv.hpp - a DFly2 draft layer's context: the cells its paired target layer
// selected, copied out of that layer's K/V cache into the drafter's own pool, where the block's own rows follow them.
#pragma once

#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace basalt::kernels {

/// A block's values every kernel of its pass reads from device memory (int32 each).
enum DflyBlk : int {
    kDflyAnchor = 0,   ///< the anchor token
    kDflyRow = 1,      ///< the source row: the row of the last window whose output chose the anchor
    kDflyNctx = 2,     ///< the context cells of the block: the selection's width at the anchor's position
    kDflyBlkCount = 3,
};

/// Where a block's context cells go: fp16 pools, or int8 codes with their scales, in the cache's page-major layout
/// with the identity table.
struct DflyKvDst {
    uint16_t *k_pool = nullptr, *v_pool = nullptr;
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_scale = nullptr, *v_scale = nullptr;
};

/// Cells ids[row * stride + j], j < n, of the pools `*src` (device) names, copied as stored (fp16 rows, or int8
/// codes and their scales) through its page table into cells [0, n) of `dst`; row = blk[kDflyRow] and
/// n = blk[kDflyNctx] (device), n <= cap. A cell whose page is not backed reads as zeros. The pools of `dst` say which
/// form is copied; all bases are 16-byte aligned. Capturable: no per-block value is an argument.
void dfly_kv_gather(const QsaAttnPools* src, const int32_t* ids, int64_t stride, const int32_t* blk, int64_t cap,
                    const QsaShapes& s, const DflyKvDst& dst, void* stream);

}  // namespace basalt::kernels
