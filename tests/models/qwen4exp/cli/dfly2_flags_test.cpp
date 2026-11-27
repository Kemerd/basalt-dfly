// tests/models/qwen4exp/cli/dfly2_flags_test.cpp - what a teacher export (--dfly2-export) runs with, and the refusal
// for the rest: it names the flag, and for a cache format the one that works. Host only.
#include "models/qwen4exp/cli/dfly2_flags.hpp"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    using basalt::models::qwen4exp::cli::ExportFlags;
    using basalt::models::qwen4exp::cli::export_refusal;
    ExportFlags ok;
    ok.docs = true;
    ok.window_rows = 512;
    check(export_refusal(ok).empty(), "documents and window rows run");
    ExportFlags f = ok;
    f.kv = "fp16";
    f.window_rows = 0;
    check(export_refusal(f).empty(), "--kv fp16 and no window rows run");

    check(has(export_refusal(ExportFlags{}), "--dfly2-docs FILE"), "no documents file is refused with the flag that names one");
    f = ok;
    f.window_rows = -1;
    check(has(export_refusal(f), "--dfly2-window-rows"), "negative window rows are refused");
    f = ok;
    f.serve = true;
    check(has(export_refusal(f), "--serve"), "--serve is refused");
    // every cache format but fp16, by name and with the one that works
    for (const char* kv : {"int8", "q4_0", "q4", "k8v4", "fp8"}) {
        f = ok;
        f.kv = kv;
        const std::string r = export_refusal(f);
        check(has(r, ("--kv " + std::string(kv)).c_str()) && has(r, "(--kv fp16)") && has(r, "fp16 cells"),
              "another cache format is refused by name, with the one that works");
    }
    f = ok;
    f.kv_resident = 20480;
    check(has(export_refusal(f), "--kv-resident"), "streamed cells are refused");
    f = ok;
    f.no_prompt_path = true;
    check(has(export_refusal(f), "--prefill 0"), "--prefill 0 is refused");
    f = ok;
    f.control_vector = true;
    check(has(export_refusal(f), "control vector"), "a control vector is refused");
    // the first reason wins: the documents file before anything else, the cache format before the streaming
    f = ExportFlags{};
    f.serve = true;
    f.kv = "int8";
    check(has(export_refusal(f), "--dfly2-docs"), "a missing documents file is named first");
    f = ok;
    f.kv = "int8";
    f.kv_resident = 4096;
    check(has(export_refusal(f), "--kv int8"), "the cache format is named before the streaming");
    return failures == 0 ? 0 : 1;
}
