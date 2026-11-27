// include/basalt/models/qwen4exp/draft_source.hpp - which of a pack's drafters fills a run's verify windows.
//
// A pack may carry more than one drafter.  A run drafts with one, chosen once when the pack is opened; every place that
// asks "is a drafter on, and which" reads that choice.  Another drafter is one more DraftSource, one more field of
// DraftPack and one more line of choose_draft_source.
#pragma once

namespace basalt::models::qwen4exp {

/// What drafts a window's later rows.
enum class DraftSource {
    None,    ///< no drafter: every window is one token, or a prompt lookup's
    Mtp,     ///< the model's draft layer (the pack's mtp sections)
    Block,   ///< the pack's block drafter (its dflash section)
};

/// The section a pack holds its block drafter in.
inline constexpr const char* kBlockDrafterSection = "dflash";

/// The drafters a pack carries.
struct DraftPack {
    bool mtp = false;     ///< the draft layer's sections are all there
    bool block = false;   ///< the block drafter's section is there
};

/// What of a run decides the drafter.
struct DraftRun {
    bool serve = false;            ///< the request loop (--serve)
    int batch = 0;                 ///< batch slots asked for
    bool layer_split = false;      ///< the layers run across cards (gpu-mode split)
    bool teacher_export = false;   ///< the run writes training shards and generates nothing
};

/// The drafter a run uses.  The block drafter where it runs: one conversation at a time on the request loop, every
/// layer on one card (its context is each window's rows after some layers, read on the card that ran them).  Elsewhere
/// the draft layer when the pack has it.
inline DraftSource choose_draft_source(const DraftPack& pack, const DraftRun& run) {
    if (run.teacher_export) return DraftSource::None;
    if (pack.block && run.serve && run.batch <= 0 && !run.layer_split) return DraftSource::Block;
    return pack.mtp ? DraftSource::Mtp : DraftSource::None;
}

/// The name a run reports its drafter by.
inline const char* draft_source_name(DraftSource s) {
    switch (s) {
        case DraftSource::Mtp: return "mtp";
        case DraftSource::Block: return "block";
        case DraftSource::None: break;
    }
    return "none";
}

}  // namespace basalt::models::qwen4exp
