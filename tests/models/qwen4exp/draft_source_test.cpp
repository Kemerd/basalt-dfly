// tests/models/qwen4exp/draft_source_test.cpp - which of a pack's drafters a run drafts with: the block drafter where it
// runs, the draft layer elsewhere, none without one or in a teacher export. Host only.
#include "basalt/models/qwen4exp/draft_source.hpp"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt::models::qwen4exp;
    const DraftPack neither{}, mtp_only{true, false}, block_only{false, true}, both{true, true};
    const DraftRun serve{true, 0, false, false};
    DraftRun batch = serve, split = serve, one_shot = serve, teacher = serve;
    batch.batch = 2;
    split.layer_split = true;
    one_shot.serve = false;
    teacher.teacher_export = true;

    check(choose_draft_source(neither, serve) == DraftSource::None, "a pack without a drafter drafts with none");
    check(choose_draft_source(mtp_only, serve) == DraftSource::Mtp, "a pack without the block section runs the draft layer");
    check(choose_draft_source(both, serve) == DraftSource::Block, "a served solo run takes the block drafter when the pack has it");
    check(choose_draft_source(block_only, serve) == DraftSource::Block, "also when the pack has no draft layer");
    // where the block drafter does not run, the draft layer does, as before
    check(choose_draft_source(both, batch) == DraftSource::Mtp, "batch slots run the draft layer");
    check(choose_draft_source(both, split) == DraftSource::Mtp, "a layer split runs the draft layer");
    check(choose_draft_source(both, one_shot) == DraftSource::Mtp, "a one-shot run runs the draft layer");
    check(choose_draft_source(block_only, batch) == DraftSource::None && choose_draft_source(block_only, split) == DraftSource::None &&
              choose_draft_source(block_only, one_shot) == DraftSource::None,
          "and with no draft layer in the pack, none");
    DraftRun one_slot = serve;
    one_slot.batch = 1;
    check(choose_draft_source(both, one_slot) == DraftSource::Mtp, "one batch slot is batch serving");
    for (const DraftPack& p : {neither, mtp_only, block_only, both})
        check(choose_draft_source(p, teacher) == DraftSource::None, "a teacher export runs no drafter");

    check(std::strcmp(draft_source_name(DraftSource::None), "none") == 0 && std::strcmp(draft_source_name(DraftSource::Mtp), "mtp") == 0 &&
              std::strcmp(draft_source_name(DraftSource::Block), "block") == 0,
          "the names a run reports");
    check(std::strcmp(kBlockDrafterSection, "dflash") == 0, "the block drafter's section, as the converter names it");
    return failures == 0 ? 0 : 1;
}
