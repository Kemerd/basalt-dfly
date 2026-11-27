#pragma once

#include <cstdint>
#include <string>

namespace basalt::models::qwen4exp::cli {

/// The flags a teacher export (--dfly2-export) is checked against before anything loads.
struct ExportFlags {
    bool docs = false;             ///< --dfly2-docs names a file
    int64_t window_rows = 0;       ///< --dfly2-window-rows
    bool serve = false;            ///< --serve
    std::string kv;                ///< --kv; empty when it was not given
    int64_t kv_resident = 0;       ///< --kv-resident
    bool no_prompt_path = false;   ///< --prefill 0
    bool control_vector = false;   ///< a --control-vector or --control-vector-scaled file
};

// A teacher export writes whole fp16 cells and the model's own rows, from one process that serves nothing: every
// other cache format holds no fp16 cell, streamed cells are not whole in VRAM, a control vector edits the rows, and
// a document's first rows are read by the batched prompt path. Returns why the export does not run with `f` (what
// follows "--dfly2-export " in the error), empty when it does.
inline std::string export_refusal(const ExportFlags& f) {
    if (!f.docs) return "needs --dfly2-docs FILE (one document a line, as token ids)";
    if (f.window_rows < 0) return "takes --dfly2-window-rows of 0 or more";
    if (f.serve) return "is a one-shot run: not with --serve";
    if (!f.kv.empty() && f.kv != "fp16") return "exports fp16 cells: not with --kv " + f.kv + " (--kv fp16)";
    if (f.kv_resident > 0) return "reads whole cells from VRAM: not with --kv-resident";
    if (f.no_prompt_path) return "reads a document's first rows through the batched prompt path: not with --prefill 0";
    if (f.control_vector) return "exports the model's own rows: not with a control vector";
    return "";
}

}  // namespace basalt::models::qwen4exp::cli
