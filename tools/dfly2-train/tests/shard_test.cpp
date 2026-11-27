// tools/dfly2-train/tests/shard_test.cpp - the teacher shard (basalt/artifact/dfly2_shard.hpp): what the writer puts
// down in pieces is what the reader hands back, and the reader and writer refuse what they must. No model, no GPU.
#include "basalt/artifact/dfly2_shard.hpp"

#include "test_util.hpp"

#include <cstring>
#include <fstream>
#include <random>

using namespace basalt::dfly2;
using dfly2_test::check;

namespace {

/// One sequence's arrays as the test made them.
struct Fixture {
    std::vector<int32_t> tokens, anchors;
    std::vector<float> hidden;
    std::vector<std::vector<uint16_t>> k, v;
    std::vector<std::vector<int32_t>> selection;   ///< [anchor * n_layers + layer]
    int64_t pos0 = 0;
};

ShardHeader tiny_header() {
    ShardHeader h{};
    h.hidden = 6;
    h.vocab = 50;
    h.n_head_kv = 2;
    h.head_dim = 4;
    h.n_layers = 2;
    h.layers[0] = 3;
    h.layers[1] = 7;
    h.sel_cap = 5;
    h.n_rot = 2;
    h.rope_base = 1e7;
    h.hidden_dtype = (uint32_t) ShardDtype::f32;
    h.kv_dtype = (uint32_t) ShardDtype::f16;
    h.pack_bytes = 123456789;
    for (int i = 0; i < 32; ++i) h.pack_meta_sha256[i] = (uint8_t) (i * 3 + 1);
    std::memcpy(h.pack_name, "tiny.basalt", sizeof "tiny.basalt");
    return h;
}

Fixture make_sequence(const ShardHeader& h, uint64_t n_tokens, const std::vector<int32_t>& anchors, int64_t pos0, std::mt19937& rng) {
    Fixture f;
    f.pos0 = pos0;
    f.anchors = anchors;
    std::uniform_int_distribution<int32_t> tok(0, (int32_t) h.vocab - 1);
    std::uniform_real_distribution<float> val(-2.0f, 2.0f);
    std::uniform_int_distribution<uint32_t> half(0, 0x7bffu);
    f.tokens.resize(n_tokens);
    for (auto& t : f.tokens) t = tok(rng);
    f.hidden.resize(n_tokens * h.hidden);
    for (auto& x : f.hidden) x = val(rng);
    f.k.resize(h.n_layers);
    f.v.resize(h.n_layers);
    for (uint32_t l = 0; l < h.n_layers; ++l) {
        f.k[l].resize(n_tokens * h.n_head_kv * h.head_dim);
        f.v[l].resize(f.k[l].size());
        for (auto& x : f.k[l]) x = (uint16_t) half(rng);
        for (auto& x : f.v[l]) x = (uint16_t) half(rng);
    }
    // each row's selection: the last min(anchor, sel_cap) cells, or every other cell when that still fits
    for (int32_t a : anchors)
        for (uint32_t l = 0; l < h.n_layers; ++l) {
            std::vector<int32_t> ids;
            const int32_t step = l == 0 ? 1 : 2;
            for (int32_t c = a - 1; c >= 0 && ids.size() < h.sel_cap; c -= step) ids.insert(ids.begin(), c);
            f.selection.push_back(std::move(ids));
        }
    return f;
}

/// Writes `f` in pieces of `piece` rows, the way an exporter hands over prompt chunks.
bool write_sequence(ShardWriter& w, const ShardHeader& h, const Fixture& f, uint64_t piece, std::string& err) {
    const uint64_t n = f.tokens.size();
    if (!w.begin_sequence(f.tokens.data(), n, f.anchors.data(), f.anchors.size(), f.pos0, err)) return false;
    const uint64_t cell = (uint64_t) h.n_head_kv * h.head_dim;
    for (uint64_t r = 0; r < n; r += piece) {
        const uint64_t rows = std::min(piece, n - r);
        if (!w.put_hidden(r, rows, f.hidden.data() + r * h.hidden, err)) return false;
        for (uint32_t l = 0; l < h.n_layers; ++l)
            if (!w.put_kv(l, r, rows, f.k[l].data() + r * cell, f.v[l].data() + r * cell, err)) return false;
    }
    for (uint64_t a = 0; a < f.anchors.size(); ++a)
        for (uint32_t l = 0; l < h.n_layers; ++l) {
            const auto& ids = f.selection[a * h.n_layers + l];
            if (!w.put_selection(a, l, ids.data(), (uint32_t) ids.size(), err)) return false;
        }
    return w.end_sequence(err);
}

bool same(const ShardHeader& h, const Fixture& f, const SequenceData& d) {
    if (d.info.n_tokens != f.tokens.size() || d.info.n_anchors != f.anchors.size() || d.info.pos0 != f.pos0) return false;
    if (d.tokens != f.tokens || d.anchors != f.anchors) return false;
    if (d.hidden.size() != f.hidden.size() * sizeof(float) || std::memcmp(d.hidden.data(), f.hidden.data(), d.hidden.size()) != 0) return false;
    for (uint32_t l = 0; l < h.n_layers; ++l)
        if (d.k[l] != f.k[l] || d.v[l] != f.v[l]) return false;
    for (size_t s = 0; s < f.selection.size(); ++s) {
        const auto& ids = f.selection[s];
        if (d.sel_count[s] != (int32_t) ids.size()) return false;
        for (uint32_t j = 0; j < h.sel_cap; ++j)
            if (d.selection[s * h.sel_cap + j] != (j < ids.size() ? ids[j] : -1)) return false;
    }
    return true;
}

}  // namespace

int main() {
    dfly2_test::TempDir dir("basalt-dfly2-shard-test");
    std::mt19937 rng(7);
    const ShardHeader h = tiny_header();
    std::string err;

    std::printf("1. round trip\n");
    const Fixture a = make_sequence(h, 23, {0, 1, 4, 9, 22}, 0, rng);
    const Fixture b = make_sequence(h, 5, {}, 100, rng);
    const Fixture c = make_sequence(h, 70, {3, 64, 69}, 7, rng);
    const std::string path = dir.file("teacher.dfly2");
    {
        ShardWriter w;
        check(w.open(path, h, err), "writer opens: " + err);
        check(write_sequence(w, h, a, 4, err), "a sequence in pieces of 4 rows: " + err);
        check(write_sequence(w, h, b, 64, err), "a sequence with no anchors: " + err);
        check(write_sequence(w, h, c, 1, err), "a sequence a row at a time: " + err);
        check(w.close(err) && w.sequences() == 3, "writer closes with three sequences: " + err);
    }
    {
        ShardReader r;
        check(r.open(path, err), "reader opens: " + err);
        const ShardHeader& g = r.header();
        check(r.sequences() == 3 && g.hidden == h.hidden && g.vocab == h.vocab && g.n_layers == 2 && g.layers[1] == 7 &&
                  g.sel_cap == 5 && g.rope_base == 1e7 && g.pack_bytes == h.pack_bytes &&
                  std::memcmp(g.pack_meta_sha256, h.pack_meta_sha256, 32) == 0 && std::string(g.pack_name) == "tiny.basalt",
              "the header carries the geometry and the pack's identity");
        check(r.info(1) != nullptr && r.info(1)->n_tokens == 5 && r.info(3) == nullptr, "sequence headers by index");
        SequenceData d;
        check(r.read(0, d, err) && same(h, a, d), "sequence 0 reads back: " + err);
        check(r.read(1, d, err) && same(h, b, d), "sequence 1 reads back: " + err);
        check(r.read(2, d, err) && same(h, c, d), "sequence 2 reads back: " + err);
        check(!r.read(3, d, err), "a sequence past the end is refused");
    }

    std::printf("2. the writer's refusals\n");
    {
        ShardWriter w;
        ShardHeader bad = h;
        bad.n_layers = 0;
        check(!w.open(dir.file("bad.dfly2"), bad, err), "a header with no paired layer: " + err);
        bad = h;
        bad.layers[1] = 3;
        check(!w.open(dir.file("bad.dfly2"), bad, err), "paired layers out of order: " + err);
        check(w.open(dir.file("w.dfly2"), h, err), "writer opens");
        const std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6}, anchors = {2, 5}, wide = {1, 60};
        const std::vector<int32_t> unordered = {4, 2};
        check(!w.begin_sequence(wide.data(), 2, nullptr, 0, 0, err), "a token outside the vocabulary: " + err);
        check(!w.begin_sequence(tokens.data(), 6, unordered.data(), 2, 0, err), "anchors out of order: " + err);
        check(w.begin_sequence(tokens.data(), 6, anchors.data(), 2, 0, err), "a sequence begins");
        const std::vector<float> hid(6 * h.hidden, 0.5f);
        const std::vector<uint16_t> cells(6 * h.n_head_kv * h.head_dim, 0x3c00);
        check(!w.put_hidden(1, 2, hid.data(), err), "hidden rows that skip a row: " + err);
        check(w.put_hidden(0, 6, hid.data(), err), "hidden rows");
        check(!w.put_kv(2, 0, 6, cells.data(), cells.data(), err), "cells of a layer the shard does not pair: " + err);
        check(w.put_kv(0, 0, 6, cells.data(), cells.data(), err) && w.put_kv(1, 0, 6, cells.data(), cells.data(), err), "cells");
        const std::vector<int32_t> past = {0, 2}, down = {1, 0}, fine = {0, 1};
        check(!w.put_selection(0, 0, past.data(), 2, err), "a selected cell at or after the anchor: " + err);
        check(!w.put_selection(0, 0, down.data(), 2, err), "selected cells out of order: " + err);
        check(w.put_selection(0, 0, fine.data(), 2, err), "a selection");
        check(!w.put_selection(0, 0, fine.data(), 2, err), "a selection written twice: " + err);
        check(!w.end_sequence(err), "a sequence with selections missing does not end: " + err);
        check(!w.close(err), "a writer with an open sequence does not close: " + err);
    }

    std::printf("3. the reader's refusals\n");
    {
        ShardReader r;
        // the writer above never closed: no index
        check(!r.open(dir.file("w.dfly2"), err), "a shard its writer did not close: " + err);
        check(!r.open(dir.file("missing.dfly2"), err), "a missing file: " + err);
        std::ifstream in(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto save = [&](const std::string& name, const std::vector<char>& data) {
            const std::string p = dir.file(name);
            std::ofstream(std::filesystem::path(std::u8string(p.begin(), p.end())), std::ios::binary).write(data.data(), (std::streamsize) data.size());
            return p;
        };
        std::vector<char> cut(bytes.begin(), bytes.end() - 9);
        check(!r.open(save("cut.dfly2", cut), err), "a truncated shard: " + err);
        std::vector<char> magic = bytes;
        magic[0] = 'X';
        check(!r.open(save("magic.dfly2", magic), err), "a bad magic: " + err);
        // a selected id moved past its anchor, and an id written after the padding
        SequenceLayout lay;
        check(shard_sequence_layout(h, 23, 5, lay, err), "layout of sequence 0");
        const size_t sel = sizeof(ShardHeader) + (size_t) lay.selection;
        std::vector<char> past = bytes;
        const int32_t big = 22;
        std::memcpy(past.data() + sel + (size_t) (1 * h.n_layers + 0) * h.sel_cap * 4, &big, 4);   // anchor 1's only id
        check(r.open(save("past.dfly2", past), err), "a shard with one bad selection still opens");
        SequenceData d;
        check(!r.read(0, d, err), "a selected cell at or after its anchor: " + err);
        std::vector<char> pad = bytes;
        const int32_t zero = 0;
        std::memcpy(pad.data() + sel + (size_t) (1 * h.n_layers + 0) * h.sel_cap * 4 + 3 * 4, &zero, 4);
        check(r.open(save("pad.dfly2", pad), err) && !r.read(0, d, err), "an id after the padding: " + err);
        std::vector<char> tok = bytes;
        const int32_t out_of_vocab = 50;
        std::memcpy(tok.data() + sizeof(ShardHeader) + (size_t) lay.tokens, &out_of_vocab, 4);
        check(r.open(save("tok.dfly2", tok), err) && !r.read(0, d, err), "a token outside the vocabulary: " + err);
    }
    return dfly2_test::finish("shard_test");
}
