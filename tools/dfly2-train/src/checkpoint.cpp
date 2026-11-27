// tools/dfly2-train/src/checkpoint.cpp - a run's drafter and state files (checkpoint.hpp).
#include "checkpoint.hpp"

#include "safetensors.hpp"

#include "basalt/platform/file.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

namespace basalt::dfly2 {

namespace {

constexpr const char* kStateFormat = "basalt-dfly2-train-state";
constexpr const char* kStateVersion = "1";
constexpr const char* kDrafterTail = ".safetensors";
constexpr const char* kStateTail = ".state.safetensors";

/// The tensors of `named` as safetensors entries, FP32 and contiguous; `keep` holds what the entries point at.
bool to_entries(const std::vector<std::pair<std::string, torch::Tensor>>& named, std::vector<torch::Tensor>& keep, std::vector<StWrite>& out, std::string& err) {
    keep.reserve(named.size());
    for (const auto& [name, t] : named) {
        if (!t.defined() || t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) { err = "tensor " + name + " is not FP32 on the CPU"; return false; }
        keep.push_back(t.contiguous());
        const torch::Tensor& c = keep.back();
        out.push_back({name, "F32", std::vector<int64_t>(c.sizes().begin(), c.sizes().end()), c.data_ptr<float>(), (uint64_t) c.nbytes()});
    }
    return true;
}

/// Copies an open file's tensor out of the mapping.
bool to_tensor(const SafetensorsFile& f, const StTensor& t, torch::Tensor& out, std::string& err) {
    if (t.dtype != "F32") { err = "tensor " + t.name + " is " + t.dtype + ", not F32"; return false; }
    out = torch::empty(t.shape, torch::kFloat32);
    if ((uint64_t) out.nbytes() != t.end - t.begin) { err = "tensor " + t.name + " does not have its shape's size"; return false; }
    if (out.nbytes() != 0) std::memcpy(out.data_ptr<float>(), f.data(t), out.nbytes());
    return true;
}

/// A metadata value; false with `err` when the key is missing.
bool meta_text(const std::map<std::string, std::string>& m, const char* key, std::string& out, std::string& err) {
    const auto it = m.find(key);
    if (it == m.end()) { err = std::string("the file's metadata has no ") + key; return false; }
    out = it->second;
    return true;
}

/// A metadata value that is a whole non-negative integer.
bool meta_u64(const std::map<std::string, std::string>& m, const char* key, uint64_t& out, std::string& err) {
    std::string text;
    if (!meta_text(m, key, text, err)) return false;
    char* end = nullptr;
    const unsigned long long v = text.empty() || text[0] == '-' ? 0 : std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') { err = std::string("the file's metadata ") + key + " is not a count: " + text; return false; }
    out = (uint64_t) v;
    return true;
}

/// "step-" and nine digits.
std::string step_stem(int64_t step) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "step-%09lld", (long long) step);
    return buf;
}

/// The step a file name of `tail` names, or -1.
int64_t step_of(const std::string& name, const std::string& tail) {
    if (name.size() != 5 + 9 + tail.size() || name.compare(0, 5, "step-") != 0 || name.compare(14, tail.size(), tail) != 0) return -1;
    int64_t step = 0;
    for (size_t i = 5; i < 14; ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
        step = step * 10 + (name[i] - '0');
    }
    return step;
}

}  // namespace

// ---------------------------------------------------------------- the drafter file

bool write_drafter_file(const std::string& path, const DrafterFile& file, std::string& err) {
    if (!drafter_check_shape(file.shape, err)) return false;
    if (file.weight_dtype != "BF16" && file.weight_dtype != "F32") { err = "weight dtype " + file.weight_dtype + " is not BF16 or F32"; return false; }
    // the contract's tensors in its order, each with the contract's shape
    std::vector<std::pair<std::string, torch::Tensor>> named;
    for (const DrafterTensor& t : drafter_tensors(file.shape)) {
        const auto it = file.tensors.find(t.name);
        if (it == file.tensors.end() || !it->second.defined() || !(it->second.sizes() == torch::IntArrayRef(t.shape))) { err = "the drafter has no tensor " + t.name + " of the contract's shape"; return false; }
        named.emplace_back(t.name, it->second);
    }
    std::vector<torch::Tensor> keep;
    std::vector<StWrite> entries;
    if (!to_entries(named, keep, entries, err)) return false;
    std::map<std::string, std::string> meta = drafter_metadata(file.shape);
    meta[kKeyWeightDtype] = file.weight_dtype;
    meta[kKeyStep] = std::to_string(file.step);
    meta[kKeyPackName] = file.pack_name;
    meta[kKeyPackBytes] = std::to_string(file.pack_bytes);
    meta[kKeyPackMetaSha256] = file.pack_meta_sha256;
    return safetensors_write(path, meta, entries, err);
}

bool read_drafter_file(const std::string& path, DrafterFile& out, std::string& err) {
    SafetensorsFile f;
    if (!f.open(path, err)) return false;
    out = DrafterFile{};
    uint64_t step = 0;
    if (!drafter_shape_from_metadata(f.metadata(), out.shape, err) || !meta_text(f.metadata(), kKeyWeightDtype, out.weight_dtype, err) ||
        !meta_u64(f.metadata(), kKeyStep, step, err) || !meta_text(f.metadata(), kKeyPackName, out.pack_name, err) ||
        !meta_u64(f.metadata(), kKeyPackBytes, out.pack_bytes, err) || !meta_text(f.metadata(), kKeyPackMetaSha256, out.pack_meta_sha256, err)) {
        err = path + ": " + err;
        return false;
    }
    out.step = (int64_t) step;
    if (out.weight_dtype != "BF16" && out.weight_dtype != "F32") { err = path + ": weight dtype " + out.weight_dtype + " is not BF16 or F32"; return false; }
    const std::vector<DrafterTensor> expect = drafter_tensors(out.shape);
    if (f.tensors().size() != expect.size()) { err = path + ": " + std::to_string(f.tensors().size()) + " tensors, the drafter's shape has " + std::to_string(expect.size()); return false; }
    for (const DrafterTensor& t : expect) {
        const StTensor* st = f.find(t.name);
        if (st == nullptr || st->shape != t.shape) { err = path + ": no tensor " + t.name + " of the contract's shape"; return false; }
        if (!to_tensor(f, *st, out.tensors[t.name], err)) { err = path + ": " + err; return false; }
    }
    return true;
}

bool drafter_file_matches(const DrafterFile& file, const PackIdentity& pack, std::string& err) {
    if (file.pack_bytes == pack.bytes && file.pack_meta_sha256 == pack.meta_sha256_hex()) return true;
    err = "the drafter was built on another pack (" + file.pack_name + ", " + std::to_string(file.pack_bytes) + " bytes), not " + pack.name;
    return false;
}

// ---------------------------------------------------------------- the state file

bool write_train_state(const std::string& path, const TrainState& state, std::string& err) {
    std::vector<std::pair<std::string, torch::Tensor>> named(state.buffers.begin(), state.buffers.end());
    std::vector<torch::Tensor> keep;
    std::vector<StWrite> entries;
    if (!to_entries(named, keep, entries, err)) return false;
    char elapsed[64];
    std::snprintf(elapsed, sizeof elapsed, "%.3f", state.elapsed);
    const std::map<std::string, std::string> meta = {
        {"format", kStateFormat},
        {"version", kStateVersion},
        {"step", std::to_string(state.step)},
        {"adam_steps", std::to_string(state.adam_steps)},
        {"epoch", std::to_string(state.epoch)},
        {"position", std::to_string(state.position)},
        {"first", std::to_string(state.first)},
        {"elapsed", elapsed},
        {"seed", std::to_string(state.seed)},
        {"blocks_per_step", std::to_string(state.blocks_per_step)},
        {"total_steps", std::to_string(state.total_steps)},
        {"dataset_blocks", std::to_string(state.dataset_blocks)},
        {"dataset_sequences", std::to_string(state.dataset_sequences)},
    };
    return safetensors_write(path, meta, entries, err);
}

bool read_train_state(const std::string& path, TrainState& out, std::string& err) {
    SafetensorsFile f;
    if (!f.open(path, err)) return false;
    const auto& m = f.metadata();
    std::string format, version, elapsed;
    out = TrainState{};
    uint64_t step = 0, adam = 0, epoch = 0, bps = 0, total = 0;
    if (!meta_text(m, "format", format, err) || !meta_text(m, "version", version, err)) { err = path + ": " + err; return false; }
    if (format != kStateFormat || version != kStateVersion) { err = path + ": not a training state this build reads (" + format + " " + version + ")"; return false; }
    if (!meta_u64(m, "step", step, err) || !meta_u64(m, "adam_steps", adam, err) || !meta_u64(m, "epoch", epoch, err) || !meta_u64(m, "position", out.position, err) ||
        !meta_u64(m, "first", out.first, err) || !meta_text(m, "elapsed", elapsed, err) || !meta_u64(m, "seed", out.seed, err) || !meta_u64(m, "blocks_per_step", bps, err) ||
        !meta_u64(m, "total_steps", total, err) || !meta_u64(m, "dataset_blocks", out.dataset_blocks, err) || !meta_u64(m, "dataset_sequences", out.dataset_sequences, err)) {
        err = path + ": " + err;
        return false;
    }
    out.step = (int64_t) step;
    out.adam_steps = (int64_t) adam;
    out.epoch = (int64_t) epoch;
    out.blocks_per_step = (int64_t) bps;
    out.total_steps = (int64_t) total;
    if (!detail::parse_real(elapsed, out.elapsed) || !(out.elapsed >= 0.0)) { err = path + ": elapsed is not a time: " + elapsed; return false; }
    for (const StTensor& t : f.tensors())
        if (!to_tensor(f, t, out.buffers[t.name], err)) { err = path + ": " + err; return false; }
    return true;
}

// ---------------------------------------------------------------- the run directory

std::string drafter_path(const std::string& run, int64_t step) { return run + "/" + step_stem(step) + kDrafterTail; }

std::string state_path(const std::string& run, int64_t step) { return run + "/" + step_stem(step) + kStateTail; }

bool list_checkpoints(const std::string& run, std::vector<int64_t>& steps, std::string& err) {
    steps.clear();
    std::vector<platform::DirFile> files;
    if (!platform::list_files(run, files)) { err = "cannot read the run directory " + run; return false; }
    std::set<int64_t> drafters, states;
    for (const platform::DirFile& f : files) {
        if (const int64_t state = step_of(f.name, kStateTail); state >= 0) states.insert(state);
        if (const int64_t drafter = step_of(f.name, kDrafterTail); drafter >= 0) drafters.insert(drafter);
    }
    for (const int64_t s : drafters)
        if (states.count(s) != 0) steps.push_back(s);
    return true;
}

void prune_checkpoints(const std::string& run, int64_t keep) {
    std::vector<int64_t> steps;
    std::string ignored;
    if (keep <= 0 || !list_checkpoints(run, steps, ignored)) return;
    // the drafter file first: a step without it is no checkpoint, so nothing half-removed is ever resumed from
    for (size_t i = 0; i + (size_t) keep < steps.size(); ++i) {
        platform::remove_file(drafter_path(run, steps[i]));
        platform::remove_file(state_path(run, steps[i]));
    }
}

}  // namespace basalt::dfly2
