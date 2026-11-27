// src/dflash/seq.cpp - a sequence's context range and its buffers: what of the block drafter's core needs no device.
#include "basalt/dflash/core.hpp"

#include <algorithm>

namespace basalt::dflash {

void Seq::reset(int64_t q) {
    from = upto = q;
    hi = q - 1;   // what the cells held before is no position of this context: nothing written counts against it
}

void Seq::truncate(int64_t q) {
    // a cell keeps the last position written to it: positions at or below hi - cap have lost theirs
    const int64_t kept = std::max(from, hi + 1 - cap);
    if (q > upto || q <= kept) {
        reset(q);
        return;
    }
    from = kept;
    upto = q;
}

void Seq::extend(int64_t pos, int64_t n) {
    if (n <= 0) return;
    upto = pos + n;
    mark(pos, n);
}

void Seq::mark(int64_t pos, int64_t n) {
    if (n > 0) hi = std::max(hi, pos + n - 1);
}

int64_t Seq::window(int64_t limit, int64_t pos) const { return std::max<int64_t>(1, std::min<int64_t>(limit, pos - from + 1)); }

void Seq::take_range(const Seq& other) {
    from = other.from;
    upto = other.upto;
    hi = other.hi;
}

bool take_seq(const BlockDrafter& bd, int rows, const std::function<void*(size_t)>& take, Seq& seq) {
    seq = Seq{};
    if (!take || !bd.present || rows < 1 || bd.layers.empty() || bd.n_embd <= 0) return false;
    seq.cap = ring_cells(bd);
    const size_t ring = ring_bytes(bd), R = (size_t) rows;
    // each layer's K, then its V
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        seq.k.push_back(static_cast<uint16_t*>(take(ring)));
        seq.v.push_back(static_cast<uint16_t*>(take(ring)));
        if (!seq.k.back() || !seq.v.back()) return false;
    }
    seq.acc = static_cast<float*>(take(R * (size_t) bd.n_embd * sizeof(float)));
    seq.ids = static_cast<int32_t*>(take(R * sizeof(int32_t)));
    seq.conf = static_cast<float*>(take(R * sizeof(float)));
    seq.q_ids = static_cast<int32_t*>(take(R * kDraftQ * sizeof(int32_t)));
    seq.q_p = static_cast<float*>(take(R * kDraftQ * sizeof(float)));
    seq.anchor = static_cast<int32_t*>(take(sizeof(int32_t)));
    return seq.acc && seq.ids && seq.conf && seq.q_ids && seq.q_p && seq.anchor;
}

}  // namespace basalt::dflash
