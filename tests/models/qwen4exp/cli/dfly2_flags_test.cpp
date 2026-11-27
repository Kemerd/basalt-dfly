// tests/models/qwen4exp/cli/dfly2_flags_test.cpp - what a teacher export (--dfly2-export) runs with, and the refusal
// for the rest: it names the flag, and for a cache format the one that works. Then the same through the model's own
// flag parser: the three flags, the exit code and the window the run is left with. Host only.
#include "models/qwen4exp/adapter.hpp"
#include "models/qwen4exp/cli/dfly2_flags.hpp"

#include "basalt/model/flags.hpp"
#include "basalt/models/qwen4exp/options.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
using basalt::models::qwen4exp::Options;

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// The exit code the command line `words` ends with (0: accepted, -1: a flag the model does not take); `out` gets the
// options it leaves.
int finish(std::vector<std::string> words, Options* out = nullptr, bool serve = false) {
    Options o;
    o.serve = serve;   // a flag of every model, read before the model's own
    words.insert(words.begin(), "basalt");
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    for (int i = 1; i < (int) argv.size(); ++i) {
        int code = 0;
        basalt::model::FlagArgs args((int) argv.size(), argv.data(), i);
        if (!basalt::models::qwen4exp::parse_flag(argv[i], args, o, code) || code != 0) return -1;
    }
    const int code = basalt::models::qwen4exp::finish_flags(o).value_or(0);
    if (out != nullptr) *out = o;
    return code;
}
}  // namespace

int main() {
    using basalt::models::qwen4exp::cli::ExportFlags;
    using basalt::models::qwen4exp::cli::export_refusal;
    std::printf("1. the refusals\n");
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
              "--kv " + std::string(kv) + " is refused by name, with the format that works");
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

    std::printf("2. the command line\n");
    Options o;
    check(finish({"--dfly2-export", "out", "--dfly2-docs", "docs.txt", "--kv", "fp16", "--prefill", "2048"}, &o) == 0 &&
              o.dfly2_export == "out" && o.dfly2_docs == "docs.txt" && o.dfly2_window_rows == 512 && o.spec == 8 && o.spec_given,
          "--dfly2-export with its documents is taken: 512 window rows, windows of 8");
    check(finish({"--dfly2-export", "out", "--dfly2-docs", "docs.txt", "--dfly2-window-rows", "0"}, &o) == 0 &&
              o.dfly2_window_rows == 0,
          "--dfly2-window-rows 0 is taken");
    check(finish({"--dfly2-export", "out", "--dfly2-docs", "docs.txt", "--dfly2-window-rows", "4096", "--spec", "3"}, &o) == 0 &&
              o.dfly2_window_rows == 4096 && o.spec == 8,
          "the export runs its own windows whatever --spec says");
    check(finish({}, &o) == 0 && o.spec == 0 && !o.spec_given && o.dfly2_export.empty(), "without --dfly2-export the window is left alone");
    check(finish({"--spec", "5"}, &o) == 0 && o.spec == 5, "and --spec keeps its value");
    check(finish({"--dfly2-export", "out"}) == 2, "--dfly2-export without --dfly2-docs is refused");
    check(finish({"--dfly2-docs", "docs.txt"}) == 2, "--dfly2-docs without --dfly2-export is refused");
    check(finish({"--dfly2-window-rows", "64"}, &o) == 0 && o.dfly2_export.empty(), "window rows alone start no export");
    check(finish({"--dfly2-export", "out", "--dfly2-docs", "docs.txt"}, nullptr, true) == 2, "--dfly2-export is refused with --serve");
    for (const std::vector<std::string>& more : {std::vector<std::string>{"--dfly2-window-rows", "-1"}, {"--kv", "int8"},
                                                 {"--kv", "q4_0"}, {"--kv", "k8v4"}, {"--kv-resident", "20480"},
                                                 {"--prefill", "0"}, {"--control-vector", "v.gguf"},
                                                 {"--control-vector-scaled", "v.gguf:0.5"}}) {
        std::vector<std::string> words = {"--dfly2-export", "out", "--dfly2-docs", "docs.txt"};
        words.insert(words.end(), more.begin(), more.end());
        check(finish(words) == 2, "--dfly2-export is refused with " + more[0] + " " + more[1]);
    }
    return failures == 0 ? 0 : 1;
}
