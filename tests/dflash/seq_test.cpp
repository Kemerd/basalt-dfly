// tests/dflash/seq_test.cpp - a block drafter sequence's context range: what a reset, new context rows, a block's own
// rows and a rewind leave of it once the ring has wrapped, the window a pass reads, and the order and sizes of the
// sequence's buffers. Host only.
#include "basalt/dflash/core.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace basalt;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// the drafter released for Qwen3.8-Flash-Next: a window of 2048 positions over rings of 2304 cells
constexpr int kWindow = 2048;
constexpr int64_t kCap = kWindow + 256;

dflash::BlockDrafter drafter() {
    dflash::BlockDrafter bd;
    bd.n_embd = 2560;
    bd.block = 8;
    bd.window = kWindow;
    bd.n_head = 32;
    bd.n_kv = 8;
    bd.head_dim = 128;
    bd.n_layers = 5;
    bd.layers.assign(5, {});
    bd.present = true;
    return bd;
}

dflash::Seq ring() {
    dflash::Seq q;
    q.cap = kCap;
    return q;
}

bool is(const dflash::Seq& q, int64_t from, int64_t upto, int64_t hi) { return q.from == from && q.upto == upto && q.hi == hi; }

void test_range() {
    dflash::Seq q = ring();
    check(is(q, 0, 0, -1), "a new sequence holds nothing");
    check(q.window(kWindow, 0) == 1, "its first pass reads its own position");
    q.extend(0, 16);
    q.extend(16, 5);
    check(is(q, 0, 21, 20), "context rows end the range after them");
    check(q.window(kWindow, 21) == 22, "a pass reads every position there is, itself included");
    q.extend(21, 0);
    check(is(q, 0, 21, 20), "no rows change nothing");
    // a block at 21 writes its own eight rows' cells: no context, but the cells are taken
    q.mark(21, 8);
    check(is(q, 0, 21, 28), "a block's rows are marked and are no context");
    q.mark(21, 4);
    check(is(q, 0, 21, 28), "a narrower block leaves the mark");
    q.extend(21, 3);
    check(is(q, 0, 24, 28), "context goes on under a block's cells");
    // far along, the window is the drafter's
    q.extend(24, 5000 - 24);
    check(is(q, 0, 5000, 4999), "a long context");
    check(q.window(kWindow, 5000) == kWindow, "a pass reads its window at most");
    check(q.window(16, 5000) == 16, "or the limit given");
}

void test_reset() {
    dflash::Seq q = ring();
    q.extend(0, 3000);
    q.reset(700);
    check(is(q, 700, 700, 699), "a reset leaves nothing, from its position");
    check(q.window(kWindow, 700) == 1, "a pass after it reads itself alone");
    q.extend(700, 10);
    check(q.window(kWindow, 710) == 11, "and then what was read since");
    check(q.window(kWindow, 690) == 1, "a pass below the context still reads one position");
}

void test_truncate() {
    {   // inside a ring that has not wrapped: the positions below stay
        dflash::Seq q = ring();
        q.extend(0, 1000);
        q.mark(1000, 8);
        q.truncate(400);
        check(is(q, 0, 400, 1007), "a rewind keeps the positions below it");
        check(q.window(kWindow, 400) == 401, "and a pass reads them");
        q.truncate(400);
        check(is(q, 0, 400, 1007), "a rewind to where the context ends changes nothing");
    }
    {   // the ring wrapped: positions whose cells later ones took are gone
        dflash::Seq q = ring();
        q.extend(0, 5000);
        // cells hold 5000 - 2304 = 2696 .. 4999
        q.truncate(4000);
        check(is(q, 2696, 4000, 4999), "a rewind in a wrapped ring keeps what the cells still hold");
        check(q.window(kWindow, 4000) == 4000 - 2696 + 1, "and a pass reads no further back");
        // new rows from 4000 take no cell of a kept position
        q.extend(4000, 16);
        check(is(q, 2696, 4016, 4999), "the context goes on from the rewind");
        q.extend(4016, 2000);
        check(is(q, 2696, 6016, 6015), "past the old mark");
        q.truncate(6000);
        check(is(q, 3712, 6000, 6015), "the kept range follows the mark");
    }
    {   // a block's own rows count: their cells are taken too
        dflash::Seq q = ring();
        q.extend(0, 2300);
        q.mark(2300, 8);   // cells of positions 0 .. 3 are the block's now
        q.truncate(2000);
        check(is(q, 4, 2000, 2307), "a block's rows take cells from the oldest positions");
    }
    {   // a rewind at or below everything the cells hold: nothing is left
        dflash::Seq q = ring();
        q.extend(0, 5000);
        q.truncate(2696);
        check(is(q, 2696, 2696, 2695), "a rewind to the oldest kept position leaves nothing");
        q = ring();
        q.extend(0, 5000);
        q.truncate(100);
        check(is(q, 100, 100, 99), "a rewind below the ring leaves nothing");
        check(q.window(kWindow, 100) == 1, "and a pass reads itself alone");
    }
    {   // a position past the context: rows are missing between, so nothing is this context's
        dflash::Seq q = ring();
        q.extend(0, 100);
        q.truncate(101);
        check(is(q, 101, 101, 100), "a position past the context leaves nothing");
    }
    {   // after a reset, earlier cells do not count against the new context
        dflash::Seq q = ring();
        q.extend(0, 5000);
        q.reset(10);
        q.extend(10, 30);
        q.truncate(20);
        check(is(q, 10, 20, 39), "a reset forgets the old mark");
    }
    {   // another sequence's range, for rings copied from it
        dflash::Seq a = ring(), b = ring();
        a.extend(0, 5000);
        a.truncate(4000);
        b.take_range(a);
        check(is(b, a.from, a.upto, a.hi) && b.cap == kCap, "a copied sequence takes the range");
    }
}

void test_buffers() {
    const dflash::BlockDrafter bd = drafter();
    check(dflash::ring_cells(bd) == kCap, "a ring is the window and 256 cells");
    check(dflash::ring_bytes(bd) == (size_t) kCap * 8 * 128 * 2, "of fp16 K or V rows");
    // each take is recorded; the buffers are told apart by the addresses handed out
    std::vector<size_t> sizes;
    std::vector<char> pool(1 << 16);
    const auto take = [&](size_t n) -> void* {
        sizes.push_back(n);
        return &pool[sizes.size()];
    };
    dflash::Seq q;
    check(dflash::take_seq(bd, 16, take, q), "a sequence's buffers are carved");
    const size_t ring_b = dflash::ring_bytes(bd);
    bool ok = sizes.size() == 16;
    for (size_t i = 0; ok && i < 10; ++i) ok = sizes[i] == ring_b;
    ok = ok && sizes[10] == (size_t) 16 * 2560 * 4 && sizes[11] == 16 * 4 && sizes[12] == 16 * 4 &&
         sizes[13] == (size_t) 16 * dflash::kDraftQ * 4 && sizes[14] == (size_t) 16 * dflash::kDraftQ * 4 && sizes[15] == 4;
    check(ok, "each layer's K then V, the features, the drafts, their candidates, the anchor");
    ok = q.k.size() == 5 && q.v.size() == 5;
    for (size_t l = 0; ok && l < 5; ++l)
        ok = (void*) q.k[l] == (void*) &pool[2 * l + 1] && (void*) q.v[l] == (void*) &pool[2 * l + 2];
    ok = ok && (void*) q.acc == (void*) &pool[11] && (void*) q.ids == (void*) &pool[12] && (void*) q.conf == (void*) &pool[13] &&
         (void*) q.q_ids == (void*) &pool[14] && (void*) q.q_p == (void*) &pool[15] && (void*) q.anchor == (void*) &pool[16];
    check(ok, "in that order");
    check(q.cap == kCap && is(q, 0, 0, -1), "the new sequence is empty over its ring");

    // memory that runs out, and what cannot be carved
    int left = 7;
    const auto short_take = [&](size_t) -> void* { return left-- > 0 ? pool.data() : nullptr; };
    check(!dflash::take_seq(bd, 16, short_take, q), "memory that runs out fails");
    check(!dflash::take_seq(bd, 0, take, q), "no rows: nothing is carved");
    check(!dflash::take_seq(bd, 16, nullptr, q), "no memory: nothing is carved");
    dflash::BlockDrafter none;
    check(!dflash::take_seq(none, 16, take, q), "no drafter: nothing is carved");
}

}  // namespace

int main() {
    test_range();
    test_reset();
    test_truncate();
    test_buffers();
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
