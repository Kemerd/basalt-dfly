// tests/dfly2/anchors_test.cpp - CPU tests of a teacher export's anchors (basalt/dfly2/anchors.hpp): one per stride,
// inside the range, ascending, the same for the same seed, and the rows an export reads through verify windows.
#include "basalt/dfly2/anchors.hpp"

#include <cstdio>
#include <vector>

using namespace basalt;

static int g_fail = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "anchors_test: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                                \
        }                                                                            \
    } while (0)

namespace {

// Every stride of `every` positions of [lo, hi] holds exactly one anchor, and that anchor lies in it.
bool one_per_stride(const std::vector<int32_t>& a, int64_t lo, int64_t hi, int every) {
    const size_t strides = hi < lo ? 0 : (size_t) ((hi - lo) / every + 1);
    if (a.size() != strides) return false;
    for (size_t k = 0; k < a.size(); ++k) {
        const int64_t first = lo + (int64_t) k * every, last = first + every - 1 < hi ? first + every - 1 : hi;
        if (a[k] < first || a[k] > last) return false;
        if (k > 0 && a[k] <= a[k - 1]) return false;
    }
    return true;
}

void test_mix() {
    // splitmix64's first outputs for the states 0 and 42
    CHECK(dfly2::anchor_mix(0) == 0xe220a8397b1dcdafull);
    CHECK(dfly2::anchor_mix(42) == 0xbdd732262feb6e95ull);
}

void test_fixed() {
    // the draws are part of the export's contract: a restart must place the same anchors
    CHECK(dfly2::sample_anchors(42, 0, 1, 300, 64) == (std::vector<int32_t>{60, 119, 166, 232, 299}));
    CHECK(dfly2::sample_anchors(42, 7, 1, 300, 64) == (std::vector<int32_t>{4, 70, 174, 212, 271}));
    CHECK(dfly2::sample_anchors(42, 0, 1001, 1040, 8) == (std::vector<int32_t>{1003, 1015, 1021, 1026, 1035}));
    CHECK(dfly2::sample_anchors(1, 3, 5, 5, 8) == (std::vector<int32_t>{5}));
}

void test_ranges() {
    for (int every : {1, 8, 64}) {
        for (int64_t lo : {0, 1, 63, 64, 1000}) {
            for (int64_t span : {0, 1, 7, 8, 9, 63, 64, 65, 129, 5000}) {
                const int64_t hi = lo + span;
                for (uint64_t seq : {0ull, 1ull, 999ull}) {
                    const std::vector<int32_t> a = dfly2::sample_anchors(42, seq, lo, hi, every);
                    CHECK(one_per_stride(a, lo, hi, every));
                    CHECK(a == dfly2::sample_anchors(42, seq, lo, hi, every));
                }
            }
        }
    }
    // a stride's draw depends on its first position alone: a longer range keeps the anchors of the whole strides
    const std::vector<int32_t> a = dfly2::sample_anchors(42, 5, 1, 640, 64), b = dfly2::sample_anchors(42, 5, 1, 6400, 64);
    CHECK(a.size() == 10 && b.size() == 100);
    for (size_t k = 0; k < a.size(); ++k) CHECK(a[k] == b[k]);
    // another sequence or another seed draws other anchors
    CHECK(dfly2::sample_anchors(42, 5, 1, 6400, 64) != dfly2::sample_anchors(42, 6, 1, 6400, 64));
    CHECK(dfly2::sample_anchors(42, 5, 1, 6400, 64) != dfly2::sample_anchors(43, 5, 1, 6400, 64));
}

void test_empty() {
    CHECK(dfly2::sample_anchors(42, 0, 10, 9, 64).empty());            // hi below lo
    CHECK(dfly2::sample_anchors(42, 0, 1, -3, 64).empty());
    CHECK(dfly2::sample_anchors(42, 0, -1, 100, 64).empty());          // a negative position
    CHECK(dfly2::sample_anchors(42, 0, 1, 100, 0).empty());            // no stride
    CHECK(dfly2::sample_anchors(42, 0, 1, 100, -8).empty());
    CHECK(dfly2::sample_anchors(42, 0, 1, (int64_t) 1 << 31, 64).empty());   // past an int32
    CHECK(dfly2::sample_anchors(42, 0, 0, 0, 8) == (std::vector<int32_t>{0}));
}

void test_window_from() {
    CHECK(dfly2::window_from(0, 512) == 0);
    CHECK(dfly2::window_from(-4, 512) == 0);
    CHECK(dfly2::window_from(1, 512) == 1);        // one row: the prompt path's
    CHECK(dfly2::window_from(2, 512) == 1);        // row 0 is never a window row
    CHECK(dfly2::window_from(512, 512) == 1);
    CHECK(dfly2::window_from(513, 512) == 1);
    CHECK(dfly2::window_from(514, 512) == 2);
    CHECK(dfly2::window_from(4096, 512) == 3584);
    CHECK(dfly2::window_from(4096, 0) == 4096);    // no window rows
    CHECK(dfly2::window_from(4096, -5) == 4096);
}

}  // namespace

int main() {
    test_mix();
    test_fixed();
    test_ranges();
    test_empty();
    test_window_from();
    if (g_fail) return 1;
    std::printf("anchors_test: ok\n");
    return 0;
}
