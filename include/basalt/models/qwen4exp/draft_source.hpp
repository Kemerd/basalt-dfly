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
    Dfly,    ///< the pack's DFly2 drafter (its dfly section)
};

/// The section a pack holds its block drafter in.
inline constexpr const char* kBlockDrafterSection = "dflash";
/// The section a pack holds its DFly2 drafter in.
inline constexpr const char* kDflyDrafterSection = "dfly";

/// The drafters a pack carries.
struct DraftPack {
    bool mtp = false;     ///< the draft layer's sections are all there
    bool block = false;   ///< the block drafter's section is there
    bool dfly = false;    ///< the DFly2 drafter's section is there
};

/// What of a run decides the drafter.
struct DraftRun {
    bool serve = false;            ///< the request loop (--serve)
    int batch = 0;                 ///< batch slots asked for
    bool layer_split = false;      ///< the layers run across cards (gpu-mode split)
    bool teacher_export = false;   ///< the run writes training shards and generates nothing
    bool dfly = false;             ///< the DFly2 drafter is asked for (--exp dfly2=1) and nothing of the start rules it out
};

/// The drafter a run uses.  A block drafter where it runs: one conversation at a time on the request loop, every
/// layer on one card (its context is each window's rows after some layers, read on the card that ran them); the DFly2
/// drafter first when the run asks for it, then the DFlash 2 one.  Elsewhere the draft layer when the pack has it.
inline DraftSource choose_draft_source(const DraftPack& pack, const DraftRun& run) {
    if (run.teacher_export) return DraftSource::None;
    const bool solo = run.serve && run.batch <= 0 && !run.layer_split;
    if (pack.dfly && run.dfly && solo) return DraftSource::Dfly;
    if (pack.block && solo) return DraftSource::Block;
    return pack.mtp ? DraftSource::Mtp : DraftSource::None;
}

/// Whether the drafter drafts a block at a time, each draft with the chance it states (the window is priced from them).
inline bool drafts_blocks(DraftSource s) { return s == DraftSource::Block || s == DraftSource::Dfly; }

/// The name a run reports its drafter by.
inline const char* draft_source_name(DraftSource s) {
    switch (s) {
        case DraftSource::Mtp: return "mtp";
        case DraftSource::Block: return "block";
        case DraftSource::Dfly: return "dfly2";
        case DraftSource::None: break;
    }
    return "none";
}

}  // namespace basalt::models::qwen4exp
