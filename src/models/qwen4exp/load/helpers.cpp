#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

// A byte count in GiB with one decimal, rounded down (a safe limit must not round up past itself).
std::string gib_text(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", std::floor((double) bytes / 1073741824.0 * 10.0) / 10.0);
    return buf;
}

// A start-ending failure: one `basalt error:` line, which the server shows as the reason.  Returns the exit code.
int fatal_start_error(const std::string& text) {
    basalt::log::error("basalt", "%s", text.c_str());
    return 1;
}
// " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() {
    (void) cudaGetLastError();
    size_t free_b = 0, total_b = 0;
    if (basalt::core::gpu_mem_info(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}

/// Calls `f` with the tensor name of every row of a pack index (the first field of each non-comment line).
bool for_each_index_name(const basalt::core::FileRange& index, const std::function<void(const std::string&)>& f) {
    std::vector<uint8_t> raw;
    std::string err;
    if (!basalt::core::read_range(index, raw, err)) return false;
    const std::string text(raw.begin(), raw.end());
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        char name[256];
        if (line.empty() || line[0] == '#' || std::sscanf(line.c_str(), "%255s", name) != 1) continue;
        f(name);
    }
    return true;
}


int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share for a x16 link) from 20 GB/s
// up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about what the
// x16 share costs.  Continuous, so a reading near either edge does not move the share by a large part of its range.
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

void lend_cache_tail(Context& context, int64_t k, LentSlots& lent, void*& borrow, uint64_t& borrow_bytes) {
    auto& xcache = *context.xcache;
    auto& host_res = *context.host_res;
    const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
    const int32_t first = (int32_t) (xcache.slots() - k);
    for (size_t i = 0; i < host_res.size(); ++i)
        if (host_res[i] >= first) {
            lent.emplace_back((int32_t) i, host_res[i]);
            host_res[i] = basalt::core::kNotResident;
        }
    context.res_put(*context.d_res);
    borrow = xcache.device_slot(first);
    borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                         : (uint64_t) k * (uint64_t) blob;
}

bool prompt_path_init(Context& context, basalt::prefill::Prefill& prefill, int64_t rows, LentSlots& lent, std::string& err) {
    auto& o = *context.o;
    auto& g = *context.g;
    auto& ss = *context.ss;
    auto& xcache = *context.xcache;
    auto& host_res = *context.host_res;
    void* borrow = nullptr;
    uint64_t borrow_bytes = 0;
    if (!host_res.empty() && *context.d_res != nullptr) {
        int64_t chunk = o.prefill_chunk;
        int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, *context.auto_ceiling,
                                           (int) *context.auto_lend_pct);   // auto: the largest that fits; fixed: halved
        const int64_t request_sized = serve_loans::request_chunk(rows, chunk);
        if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
            chunk = request_sized;
            k = serve_loans::lend_slots(xcache, g, ss, chunk);
            if (k + 128 > xcache.slots()) k = 0;
            if (!o.prefill_auto) o.prefill_chunk = chunk;
        }
        if (o.prefill_auto) {
            o.prefill_chunk = k > 0 ? chunk : serve_loans::request_chunk(rows, 1024);
            basalt::log::info("basalt generate", "prompt chunk auto: %lld tokens", (long long) o.prefill_chunk);
        } else if (chunk != o.prefill_chunk) {
            k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
        }
        if (k > 0) {   // the lent slots are refilled after the prompt
            lend_cache_tail(context, k, lent, borrow, borrow_bytes);
            basalt::log::info("basalt generate", "prompt path borrows %lld cache slots (%.2f GiB)", (long long) k,
                         (double) borrow_bytes / 1073741824.0);
        }
    }
    if (borrow == nullptr)
        basalt::log::info("basalt generate", "prompt path allocates its own buffers (no cache slots to borrow)");
    if (!*context.multi_gpu && o.batch <= 0 && !o.no_pool) {
        prefill.set_cpu_pool(context.pool);
        prefill.set_cpu_share(o.prefill_cpu_share);
    }
    return prefill.init(*context.wt, g, ss, *context.srcp, o.expert_cache > 0 ? &xcache : nullptr,
                        host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, *context.main_cs, err, borrow,
                        borrow_bytes);
}

int refill_lent(Context& context, const LentSlots& lent, double& ms) {
    auto& g = *context.g;
    auto& xcache = *context.xcache;
    auto& host_res = *context.host_res;
    auto& srcp = *context.srcp;
    std::string& err = *context.err;
    const Clock::time_point tr = Clock::now();
    for (const auto& [i, slot] : lent) {   // queued, one wait
        const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
        const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
        if (b == nullptr || !(xcache.fill_slot_queued(slot, b, err, nb))) {
            basalt::log::error("basalt generate", "refilling a lent slot failed: %s", err.c_str());
            return 1;
        }
        host_res[(size_t) i] = slot;
    }
    if (!xcache.sync_queued(err)) {
        basalt::log::error("basalt generate", "refilling the lent slots failed: %s", err.c_str());
        return 1;
    }
    context.res_put(*context.d_res);
    ms = std::chrono::duration<double, std::milli>(Clock::now() - tr).count();
    return 0;
}

int window_init(Context& context, basalt::core::Verifier& ver, int max_t) {
    auto& o = *context.o;
    auto& thits = *context.thits;
    auto& host_res = *context.host_res;
    auto& xcache = *context.xcache;
    auto& native_head = *context.native_head;
    auto& drive = *context.drive;
    if (thits.d_res == nullptr) {
        basalt::log::error("basalt generate", "--spec needs the device residency table (an expert profile, "
                             "--expert-cache and the token graph)");
        return 2;
    }
    basalt::core::VerifyHits vh;
    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
    vh.n_slots = xcache.slots();
    ver.set_cpu_warm(drive.d.pool, drive.d.src);
    ver.set_route_tail_skip(o.route.tail_skip);
    ver.set_route_prior(o.route.prior);
    if (!ver.init(*context.wt, *context.g, *context.ss, vh, native_head.loaded() ? &native_head : nullptr, max_t, *context.err)) {
        return fatal_start_error(*context.err);
    }
    drive.d.plan = ver.plan_sink();
    drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
    if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
    if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
    return 0;
}


}  // namespace
