// include/basalt/dfly2/anchors.hpp - where a teacher export places a sequence's anchors: one in each stride of
// positions, drawn from a seed with integer arithmetic alone, so every build and every restart draws the same ones.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace basalt::dfly2 {

/// One splitmix64 step of `x`: the draw of a stride.
inline uint64_t anchor_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/// One anchor in each stride of `every` positions of [lo, hi] (the last stride may be shorter), ascending; empty
/// when hi < lo, when lo is negative, when hi is past an int32 or when `every` is below 1. Stride k's anchor is its
/// first position plus anchor_mix(seed, seq, that position) % its length: integer arithmetic only, each stride on
/// its own.
inline std::vector<int32_t> sample_anchors(uint64_t seed, uint64_t seq, int64_t lo, int64_t hi, int every) {
    std::vector<int32_t> out;
    if (every < 1 || lo < 0 || hi < lo || hi > (std::numeric_limits<int32_t>::max)()) return out;
    out.reserve((size_t) ((hi - lo) / every + 1));
    for (int64_t first = lo; first <= hi; first += every) {
        const uint64_t len = (uint64_t) std::min<int64_t>(every, hi - first + 1);
        const uint64_t draw = anchor_mix(seed ^ ((seq + 1) * 0x9E3779B97F4A7C15ull) ^ ((uint64_t) first * 0xD1B54A32D192ED03ull));
        out.push_back((int32_t) (first + (int64_t) (draw % len)));
    }
    return out;
}

/// The first row of a sequence of `n_tokens` rows that an export reads through verify windows: its last
/// `window_rows` rows, never row 0. `n_tokens` when no row is.
inline int64_t window_from(int64_t n_tokens, int64_t window_rows) {
    if (n_tokens <= 0) return 0;
    return n_tokens - std::min(std::max<int64_t>(window_rows, 0), n_tokens - 1);
}

}  // namespace basalt::dfly2
