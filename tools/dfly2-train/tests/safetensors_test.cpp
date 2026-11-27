// tools/dfly2-train/tests/safetensors_test.cpp - the safetensors reader and writer: a file written here reads back
// bit for bit, and malformed files are refused. No model, no GPU.
#include "safetensors.hpp"

#include "test_util.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

using namespace basalt::dfly2;
using dfly2_test::check;

namespace {

/// A hand-built file: `json` as the header and `data` after it.
std::string raw_file(const dfly2_test::TempDir& dir, const std::string& name, const std::string& json, const std::string& data) {
    const std::string p = dir.file(name);
    std::ofstream out(std::filesystem::path(std::u8string(p.begin(), p.end())), std::ios::binary);
    const uint64_t n = json.size();
    out.write((const char*) &n, 8);
    out.write(json.data(), (std::streamsize) json.size());
    out.write(data.data(), (std::streamsize) data.size());
    return p;
}

}  // namespace

int main() {
    dfly2_test::TempDir dir("basalt-dfly2-safetensors-test");
    std::string err;

    std::printf("1. round trip\n");
    std::vector<float> a(2 * 3 * 4);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.25f * (float) i - 1.5f;
    const std::vector<uint16_t> b = {0x3f80, 0xbf80, 0x0000, 0x7f80, 0x4000};
    const std::vector<int64_t> c = {-5, 1ll << 40};
    const float scalar = 3.5f;
    const std::map<std::string, std::string> meta = {{"format", "test"}, {"quoted \"key\"", "back\\slash\nnewline"}, {"name", "h\xc3\xa9"}};
    const std::vector<StWrite> tensors = {
        {"layers.0.weight", "F32", {2, 3, 4}, a.data(), a.size() * 4},
        {"b", "BF16", {5}, b.data(), b.size() * 2},
        {"empty", "F32", {0, 7}, nullptr, 0},
        {"c", "I64", {2}, c.data(), c.size() * 8},
        {"scalar", "F32", {}, &scalar, 4},
    };
    const std::string path = dir.file("m.safetensors");
    check(safetensors_write(path, meta, tensors, err), "write: " + err);
    {
        SafetensorsFile f;
        check(f.open(path, err), "open: " + err);
        check(f.metadata() == meta, "the metadata reads back, escapes and UTF-8 included");
        check(f.tensors().size() == 5, "five tensors");
        bool same = true;
        for (const StWrite& w : tensors) {
            const StTensor* t = f.find(w.name);
            same = same && t != nullptr && t->dtype == w.dtype && t->shape == w.shape && t->end - t->begin == w.bytes &&
                   (w.bytes == 0 || std::memcmp(f.data(*t), w.data, (size_t) w.bytes) == 0);
        }
        check(same, "every tensor's dtype, shape and bytes read back");
        check(f.find("missing") == nullptr, "an absent name");
        check(f.tensors()[0].name == "layers.0.weight" && f.tensors()[0].begin == 0, "tensors come in data order");
    }
    // a second write over the same path replaces it whole
    check(safetensors_write(path, {}, {{"only", "U8", {3}, "abc", 3}}, err), "overwrite: " + err);
    {
        SafetensorsFile f;
        check(f.open(path, err) && f.tensors().size() == 1 && f.metadata().empty() && std::memcmp(f.data(f.tensors()[0]), "abc", 3) == 0,
              "the replaced file reads back: " + err);
    }
    check(!std::filesystem::exists(std::filesystem::path(std::u8string(path.begin(), path.end())).concat(".tmp")), "no temporary file is left");

    std::printf("2. the writer's refusals\n");
    check(!safetensors_write(dir.file("x.safetensors"), {}, {{"a", "F32", {2}, a.data(), 4}}, err), "bytes that do not match the shape: " + err);
    check(!safetensors_write(dir.file("x.safetensors"), {}, {{"a", "F99", {1}, a.data(), 4}}, err), "an unknown dtype: " + err);
    check(!safetensors_write(dir.file("x.safetensors"), {}, {{"a", "F32", {1}, a.data(), 4}, {"a", "F32", {1}, a.data(), 4}}, err), "a repeated name: " + err);
    check(!safetensors_write(dir.file("x.safetensors"), {}, {{"a", "F32", {1}, nullptr, 4}}, err), "missing data: " + err);

    std::printf("3. the reader's refusals\n");
    {
        SafetensorsFile f;
        const std::string eight(8, 'x');
        check(!f.open(dir.file("missing.safetensors"), err), "a missing file: " + err);
        check(!f.open(raw_file(dir, "short.safetensors", "", ""), err) || f.tensors().empty(), "an empty header is refused or empty");
        check(!f.open(raw_file(dir, "json.safetensors", "{\"a\":", ""), err), "broken JSON: " + err);
        check(!f.open(raw_file(dir, "array.safetensors", "[1,2]", ""), err), "a header that is not an object: " + err);
        check(!f.open(raw_file(dir, "range.safetensors", R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", "1234"), err), "a tensor past the end of the data: " + err);
        check(!f.open(raw_file(dir, "size.safetensors", R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", eight), err), "a shape that does not match its bytes: " + err);
        check(!f.open(raw_file(dir, "gap.safetensors", R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}})", eight), err), "a gap before the first tensor: " + err);
        check(!f.open(raw_file(dir, "overlap.safetensors", R"({"a":{"dtype":"U8","shape":[6],"data_offsets":[0,6]},"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}})", eight), err), "two tensors that overlap: " + err);
        check(!f.open(raw_file(dir, "tail.safetensors", R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", eight), err), "bytes after the last tensor: " + err);
        check(!f.open(raw_file(dir, "dtype.safetensors", R"({"a":{"dtype":"Q7","shape":[8],"data_offsets":[0,8]}})", eight), err), "an unknown dtype: " + err);
        check(!f.open(raw_file(dir, "neg.safetensors", R"({"a":{"dtype":"U8","shape":[-8],"data_offsets":[0,8]}})", eight), err), "a negative dimension: " + err);
        check(!f.open(raw_file(dir, "meta.safetensors", R"({"__metadata__":{"k":1}})", ""), err), "metadata that is not a string: " + err);
        // a length that claims more header than the file holds
        const std::string p = dir.file("len.safetensors");
        {
            std::ofstream out(std::filesystem::path(std::u8string(p.begin(), p.end())), std::ios::binary);
            const uint64_t n = 1000;
            out.write((const char*) &n, 8);
            out.write("{}", 2);
        }
        check(!f.open(p, err), "a header length past the end of the file: " + err);
        check(f.open(raw_file(dir, "ok.safetensors", R"({"a":{"dtype":"U8","shape":[2,4],"data_offsets":[0,8]}})", eight), err) && f.find("a") != nullptr, "a well-formed hand-built file opens: " + err);
    }
    return dfly2_test::finish("safetensors_test");
}
