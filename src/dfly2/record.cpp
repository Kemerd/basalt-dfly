// src/dfly2/record.cpp - the pack's record of a DFly2 drafter and the draft loop (record.hpp). No device is needed.
#include "basalt/dfly2/record.hpp"

#include <utility>

namespace basalt::dfly2 {

namespace {

using core::Json;

// a list of layer numbers; anything else reads as an empty list, which the layer count then refuses
std::vector<int> ints(const Json* j) {
    std::vector<int> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back((int) x.num());
    return v;
}

}  // namespace

bool parse_record(const Json& d, uint64_t section_offset, Record& out, std::string& err) {
    out = Record{};
    // no record: the pack has no drafter, which is not an error
    if (!d.is_object()) return true;
    auto num = [&](const char* k, double dflt) {
        const Json* j = d.get(k);
        return j && j->is_number() ? j->num() : dflt;
    };
    // the settings every drafter has; the optional ones stay 0 when the record leaves them out
    out.layers = (int) num("layers", 0);
    out.mask_id = (int) num("mask_token_id", -1);
    out.topk = (int) num("target_index_topk", 0);
    out.local_window = (int) num("context_local_window", 0);
    out.block = (int) num("block_size", 0);
    out.kv_layers = ints(d.get("target_kv_layers"));
    out.idx_layers = ints(d.get("target_index_layers"));
    // each draft layer names one target layer for its cache and one for its selection
    if (out.layers < 1 || out.layers > kMaxLayers || (int) out.kv_layers.size() != out.layers ||
        (int) out.idx_layers.size() != out.layers || out.mask_id < 0) {
        err = "meta.dfly2: layers, target_kv_layers, target_index_layers or mask_token_id are missing or disagree";
        return false;
    }
    const Json* tj = d.get("tensors");
    if (!tj || !tj->is_object()) {
        err = "meta.dfly2 has no tensors";
        return false;
    }
    for (const auto& [name, t] : tj->members) {
        Tensor r;
        const Json *off = t.get("offset"), *by = t.get("bytes"), *sh = t.get("shape"), *dt = t.get("dtype");
        if (!off || !by || !sh || !sh->is_array()) {
            err = "meta.dfly2: " + name + " has no offset, bytes or shape";
            return false;
        }
        // the stored type is kept as named: which types a drafter reads is the model's to check
        if (dt && dt->is_string()) r.dtype = dt->s;
        r.offset = section_offset + off->u64();
        r.bytes = by->u64();
        for (const Json& x : sh->items) r.shape.push_back(x.u64());
        out.tensors.emplace(name, std::move(r));
    }
    out.present = true;
    return true;
}

bool draft_loop(int n, int32_t anchor, const std::function<bool(int, int32_t, int32_t&, float&, bool&)>& step, int32_t* ids,
                float* conf, int& made) {
    made = 0;
    if (!step || !ids || !conf) return false;
    // each draft follows the token before it: the anchor, then the draft just made
    int32_t prev = anchor;
    for (; made < n; ++made) {
        bool go = true;
        if (!step(made, prev, ids[made], conf[made], go)) return false;
        if (!go) break;
        prev = ids[made];
    }
    return true;
}

}  // namespace basalt::dfly2
