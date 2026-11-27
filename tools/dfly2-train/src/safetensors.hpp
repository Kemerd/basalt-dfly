// tools/dfly2-train/src/safetensors.hpp - reads and writes safetensors files, the form of the trainer's checkpoints.
//
// The layout: an 8-byte little-endian length N, N bytes of JSON (an object; this writer pads it with spaces to a
// multiple of 8), then the tensors' bytes. The JSON maps each tensor's name to {"dtype", "shape", "data_offsets":
// [begin, end]} with offsets from the end of the JSON, and may hold "__metadata__", an object of strings. Tensors are
// little-endian and row-major, and together cover the data without a gap or an overlap.
#pragma once

#include "basalt/platform/mapped_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::dfly2 {

/// A tensor of an open file.
struct StTensor {
    std::string name;
    std::string dtype;             ///< the file's spelling: F32, BF16, I64, ...
    std::vector<int64_t> shape;    ///< outermost first; empty for a scalar
    uint64_t begin = 0, end = 0;   ///< byte range inside the data
    uint64_t elements() const;
};

/// Bytes of one element of a safetensors dtype; 0 for a spelling this reader does not know.
uint64_t st_dtype_bytes(const std::string& dtype);

/// A safetensors file mapped read-only. open() checks the length, the JSON and every tensor's type, shape and range.
class SafetensorsFile {
public:
    bool open(const std::string& path, std::string& err);
    /// Unmaps the file; a mapped file cannot be replaced on every platform.
    void close();
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    /// The tensors in data order.
    const std::vector<StTensor>& tensors() const { return tensors_; }
    /// The tensor named `name`, or nullptr.
    const StTensor* find(const std::string& name) const;
    /// A tensor's bytes, valid while the file is open.
    const uint8_t* data(const StTensor& t) const { return map_.data() + data_start_ + t.begin; }

private:
    platform::MappedFile map_;
    uint64_t data_start_ = 0;
    std::vector<StTensor> tensors_;
    std::map<std::string, size_t> by_name_;
    std::map<std::string, std::string> metadata_;
};

/// A tensor to write: `bytes` at `data` must be the shape's element count times the dtype's size.
struct StWrite {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    uint64_t bytes = 0;
};

/// Writes `tensors` in the order given with `metadata`, to a temporary file beside `path` that replaces `path` once
/// it is whole and on the device, so a reader never sees half a file.
bool safetensors_write(const std::string& path, const std::map<std::string, std::string>& metadata,
                       const std::vector<StWrite>& tensors, std::string& err);

}  // namespace basalt::dfly2
