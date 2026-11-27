// tools/dfly2-train/src/safetensors.cpp - the safetensors reader and writer (safetensors.hpp).
#include "safetensors.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/platform/file.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

namespace basalt::dfly2 {

namespace {

constexpr uint64_t kMaxHeaderBytes = 100ull << 20;   ///< the format's bound on the JSON

/// a * b, false on overflow.
bool mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (b != 0 && a > (std::numeric_limits<uint64_t>::max)() / b) return false;
    out = a * b;
    return true;
}

/// `s` as a JSON string, quotes included.
std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += (char) c;
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        } else {
            out += (char) c;   // UTF-8 bytes pass through
        }
    }
    return out + "\"";
}

/// The element count of `shape`; false for a negative dimension or an overflow.
bool shape_elements(const std::vector<int64_t>& shape, uint64_t& out) {
    out = 1;
    for (const int64_t d : shape)
        if (d < 0 || !mul_u64(out, (uint64_t) d, out)) return false;
    return true;
}

}  // namespace

uint64_t st_dtype_bytes(const std::string& dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8") return 1;
    if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "I32" || dtype == "U32" || dtype == "F32") return 4;
    if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
    return 0;
}

uint64_t StTensor::elements() const {
    uint64_t n = 1;
    return shape_elements(shape, n) ? n : 0;
}

void SafetensorsFile::close() {
    map_.reset();
    data_start_ = 0;
    tensors_.clear();
    by_name_.clear();
    metadata_.clear();
}

bool SafetensorsFile::open(const std::string& path, std::string& err) {
    close();
    std::string why;
    auto m = platform::MappedFile::open_ro(path, &why);
    if (!m) { err = "cannot open " + path + (why.empty() ? "" : ": " + why); return false; }
    map_ = std::move(*m);
    const uint64_t size = map_.size();
    if (size < 8) { err = path + ": too small for a safetensors file"; return false; }
    uint64_t n = 0;
    std::memcpy(&n, map_.data(), 8);
    if (n > kMaxHeaderBytes || n > size - 8) { err = path + ": safetensors header length " + std::to_string(n) + " does not fit the file"; return false; }
    data_start_ = 8 + n;
    const uint64_t data_bytes = size - data_start_;

    core::Json root;
    if (!core::parse_json((const char*) map_.data() + 8, (size_t) n, root, err)) { err = path + ": safetensors header: " + err; return false; }
    if (!root.is_object()) { err = path + ": safetensors header is not a JSON object"; return false; }
    for (const auto& [name, v] : root.members) {
        if (name == "__metadata__") {
            if (!v.is_object()) { err = path + ": __metadata__ is not an object"; return false; }
            for (const auto& [key, text] : v.members) {
                if (!text.is_string()) { err = path + ": metadata '" + key + "' is not a string"; return false; }
                metadata_[key] = text.s;
            }
            continue;
        }
        const core::Json* dtype = v.get("dtype");
        const core::Json* shape = v.get("shape");
        const core::Json* offs = v.get("data_offsets");
        if (!v.is_object() || dtype == nullptr || !dtype->is_string() || shape == nullptr || !shape->is_array() ||
            shape->count != shape->items.size() || offs == nullptr || !offs->is_array() || offs->count != 2 || offs->items.size() != 2) {
            err = path + ": tensor '" + name + "' is malformed";
            return false;
        }
        StTensor t;
        t.name = name;
        t.dtype = dtype->s;
        const uint64_t elem = st_dtype_bytes(t.dtype);
        if (elem == 0) { err = path + ": tensor '" + name + "' has an unknown dtype " + t.dtype; return false; }
        for (const core::Json& d : shape->items) {
            if (d.type != core::Json::Type::Int || d.neg || d.u > (uint64_t) (std::numeric_limits<int64_t>::max)()) { err = path + ": tensor '" + name + "' has a bad shape"; return false; }
            t.shape.push_back((int64_t) d.u);
        }
        const core::Json &b = offs->items[0], &e = offs->items[1];
        if (b.type != core::Json::Type::Int || e.type != core::Json::Type::Int || b.neg || e.neg) { err = path + ": tensor '" + name + "' has bad offsets"; return false; }
        t.begin = b.u;
        t.end = e.u;
        uint64_t count = 0, bytes = 0;
        if (t.begin > t.end || t.end > data_bytes || !shape_elements(t.shape, count) || !mul_u64(count, elem, bytes) || bytes != t.end - t.begin) {
            err = path + ": tensor '" + name + "' does not match its byte range";
            return false;
        }
        tensors_.push_back(std::move(t));
    }
    // the tensors tile the data: sorted by offset, each starts where the last ended
    std::sort(tensors_.begin(), tensors_.end(), [](const StTensor& a, const StTensor& b) { return a.begin != b.begin ? a.begin < b.begin : a.end < b.end; });
    uint64_t at = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].begin != at) { err = path + ": tensor '" + tensors_[i].name + "' leaves a gap or overlaps another"; return false; }
        at = tensors_[i].end;
        by_name_[tensors_[i].name] = i;
    }
    if (at != data_bytes) { err = path + ": " + std::to_string(data_bytes - at) + " bytes after the last tensor"; return false; }
    return true;
}

const StTensor* SafetensorsFile::find(const std::string& name) const {
    const auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

bool safetensors_write(const std::string& path, const std::map<std::string, std::string>& metadata,
                       const std::vector<StWrite>& tensors, std::string& err) {
    // the JSON: the metadata, then each tensor at the running offset
    std::string json = "{\"__metadata__\":{";
    bool first = true;
    for (const auto& [key, text] : metadata) {
        json += (first ? "" : ",") + json_string(key) + ":" + json_string(text);
        first = false;
    }
    json += "}";
    std::set<std::string> names;
    uint64_t at = 0;
    for (const StWrite& t : tensors) {
        const uint64_t elem = st_dtype_bytes(t.dtype);
        uint64_t count = 0, bytes = 0;
        if (elem == 0 || !shape_elements(t.shape, count) || !mul_u64(count, elem, bytes) || bytes != t.bytes || (bytes != 0 && t.data == nullptr)) {
            err = "tensor '" + t.name + "' does not match its dtype and shape";
            return false;
        }
        if (t.name.empty() || t.name == "__metadata__" || !names.insert(t.name).second) { err = "tensor name '" + t.name + "' is empty, reserved or repeated"; return false; }
        json += "," + json_string(t.name) + ":{\"dtype\":" + json_string(t.dtype) + ",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) json += (i != 0 ? "," : "") + std::to_string(t.shape[i]);
        json += "],\"data_offsets\":[" + std::to_string(at) + "," + std::to_string(at + bytes) + "]}";
        at += bytes;
    }
    json += "}";
    json.resize((json.size() + 7) / 8 * 8, ' ');
    if (json.size() > kMaxHeaderBytes) { err = "safetensors header is too large"; return false; }

    const std::string tmp = path + ".tmp";
    const int fd = platform::open_write_utf8(tmp);
    if (fd < 0) { err = "cannot create " + tmp; return false; }
    const uint64_t n = json.size();
    bool ok = platform::pwrite_all(fd, &n, 8, 0) && platform::pwrite_all(fd, json.data(), n, 8);
    uint64_t off = 8 + n;
    for (size_t i = 0; ok && i < tensors.size(); ++i) {
        ok = tensors[i].bytes == 0 || platform::pwrite_all(fd, tensors[i].data, tensors[i].bytes, off);
        off += tensors[i].bytes;
    }
    ok = ok && platform::sync_fd(fd);
    platform::close_fd(fd);
    if (!ok) {
        platform::remove_file(tmp);
        err = "cannot write " + tmp;
        return false;
    }
    if (!platform::rename_replace(tmp, path)) { err = "cannot move " + tmp + " to " + path; return false; }
    return true;
}

}  // namespace basalt::dfly2
