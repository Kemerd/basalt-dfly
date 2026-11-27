#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

int prefill(Context& context) {
    auto& o = *context.o;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& mtp = *context.mtp;
    auto& native_head = *context.native_head;
    auto& prefill_ms = *context.prefill_ms;
    auto& n_prompt = *context.n_prompt;
    auto& tok = *context.tok;

    // ---- the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop then starts at the
    // last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    basalt::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    LentSlots lent;                                    // the cache slots lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        if (!prompt_path_init(context, prefill, n_batched, lent, err)) {
            return fatal_start_error(err);
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                return fatal_start_error(err);
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // the draft layer's K/V from the batched prompt path
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            return fatal_start_error(err);
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            double refill_ms = 0;
            if (const int rc = refill_lent(context, lent, refill_ms)) return rc;
            basalt::log::info("basalt generate", "%zu lent slots refilled in %.1f ms", lent.size(), refill_ms);
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const basalt::prefill::PrefillStats& ps = prefill.stats();
        basalt::log::info("basalt generate", "prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld, on the CPU %lld (share %.2f); PLE %.1f ms",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident,
                     (long long) ps.experts_cpu, ps.cpu_share, ps.ms_ple);
    }

    // The last prompt token is the first verify window (T = 1).
    context.spec_pos = &pos_start;
    return decode(context);
}

}
