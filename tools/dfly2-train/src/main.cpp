// tools/dfly2-train/src/main.cpp - basalt-dfly2-train: the command line of the DFly2 drafter trainer.
//
//   basalt-dfly2-train init  --model PACK --out FILE
//   basalt-dfly2-train train --model PACK --shards DIR --run DIR [--init FILE]
//   basalt-dfly2-train eval  --model PACK --shards DIR --checkpoint FILE
//   basalt-dfly2-train fixture --model PACK --shards DIR --checkpoint FILE --out FILE
#include "trainer.hpp"

#include "basalt/log.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace basalt::dfly2;

constexpr const char* kTag = "basalt dfly2-train";

/// Set by the first interrupt; the training loop reads it between forwards.
volatile std::sig_atomic_t g_stop = 0;

/// The first interrupt asks the run to stop at its last whole step; a second one ends the process as usual.
extern "C" void on_interrupt(int sig) {
    g_stop = 1;
    std::signal(sig, SIG_DFL);
}

void usage() {
    std::fputs(
        "basalt-dfly2-train: trains a DFly2 drafter for a target in a .basalt pack\n"
        "\n"
        "  basalt-dfly2-train init  --model PACK --out FILE [options]\n"
        "  basalt-dfly2-train train --model PACK --shards DIR --run DIR [--init FILE] [options]\n"
        "  basalt-dfly2-train eval  --model PACK --shards DIR --checkpoint FILE [options]\n"
        "  basalt-dfly2-train fixture --model PACK --shards DIR --checkpoint FILE --out FILE [options]\n"
        "\n"
        "init: a new drafter, its attention copied from the target layers it pairs with\n"
        "  --model PACK           the target's pack\n"
        "  --out FILE             the drafter file to write (safetensors)\n"
        "  --layers A,B,...       target layers to pair with (default: the last --draft-layers sparse ones)\n"
        "  --draft-layers N       drafter layers (5)\n"
        "  --block N              rows of a block: the anchor and N - 1 drafts (8)\n"
        "  --mask-id N            the mask token, an id the tokenizer never produces (248077)\n"
        "  --rms-eps X            eps of the drafter's own norms (1e-5)\n"
        "  --seed N               seed of the random initialisation (42)\n"
        "\n"
        "train: resumes the newest checkpoint of --run, or starts a run from --init\n"
        "  --model PACK           the target's pack\n"
        "  --shards DIR           directory of teacher shards (*.dfly2)\n"
        "  --run DIR              the run directory: step-N.safetensors, step-N.state.safetensors, train.log\n"
        "  --init FILE            the drafter a new run starts from\n"
        "  --steps N              optimiser steps of the run (default: from --epochs)\n"
        "  --epochs X             passes over the shards' blocks (1)\n"
        "  --blocks-per-step N    blocks behind one optimiser step (8192)\n"
        "  --blocks N             blocks of one forward and backward (256)\n"
        "  --loss-blocks N        blocks whose logits are alive at once (32)\n"
        "  --attention-blocks N   blocks whose attention scores are alive at once (64)\n"
        "  --device D             cpu, cuda or cuda:N (cuda)\n"
        "  --dtype T              the drafter's weights in the forward, bf16 or f32; the masters are FP32 (bf16)\n"
        "  --head-dtype T         the frozen output head, f32 or bf16 (f32)\n"
        "  --seed N               seed of the data order (42)\n"
        "  --save-every N         steps between checkpoints (250)\n"
        "  --keep N               checkpoints kept (3)\n"
        "  --log-every N          steps between progress lines (10)\n"
        "  --queue-gb X           host memory of the sequences read ahead (4)\n"
        "  --lr X  --min-lr X  --warmup-ratio X            the schedule (5e-4, 5e-5, 0.04)\n"
        "  --weight-decay X  --beta1 X  --beta2 X  --adam-eps X      AdamW (0.01, 0.9, 0.95, 1e-8)\n"
        "  --muon-weight-decay X  --muon-momentum X  --muon-ns-steps N  --muon-rms X      Muon (0.1, 0.95, 5, 0.2)\n"
        "  --muon-plain-momentum  Muon without Nesterov's look-ahead\n"
        "  --muon-head-rows N     rows of a query projection orthogonalised together; 0: the whole matrix (head_dim)\n"
        "  --grad-clip X          joint gradient norm clip (1)\n"
        "  --kl-topk N  --dpace-alpha X  --e2e-weight X  --confidence-weight X      the loss (10, 0.5, 1, 1)\n"
        "  --prefix-weight X      loss weight, 0 to 1, of a block whose anchor follows a row of the target's prompt path\n"
        "                         rather than a verify window; 0 leaves them out (1)\n"
        "  --kv-int8-share X      share, 0 to 1, of the blocks that read the target's cache and store their own keys and\n"
        "                         values as a target serving with --kv int8 does (0)\n"
        "\n"
        "eval: the loss and the accepted-length metrics of a drafter over shards\n"
        "  --model, --shards, --device, --head-dtype, --blocks, --loss-blocks, --attention-blocks, --seed, --queue-gb,\n"
        "  --kl-topk, --dpace-alpha, --e2e-weight, --confidence-weight, --prefix-weight, --kv-int8-share as for train\n"
        "  --checkpoint FILE      the drafter file to measure\n"
        "  --max-blocks N         measure this many blocks (default: all)\n"
        "\n"
        "fixture: blocks of the shards and the drafter's forward over them, for a runtime's block pass to be checked against\n"
        "  --model, --shards, --device, --head-dtype, --attention-blocks, --seed, --queue-gb as for eval\n"
        "  --checkpoint FILE      the drafter file whose forward is written\n"
        "  --out FILE             the safetensors file to write\n"
        "  --count N              blocks to write, at most four from a sequence (1)\n",
        stdout);
}

/// The arguments of one command, read left to right.
struct Args {
    int argc;
    char** argv;
    int i = 2;
    bool bad = false;

    /// The value after flag `flag`; marks the command line bad when there is none.
    const char* value(const char* flag) {
        if (i + 1 >= argc) {
            basalt::log::error(kTag, "%s needs a value", flag);
            bad = true;
            return "";
        }
        return argv[++i];
    }
    /// The value as a whole number of at least `least`.
    int64_t integer(const char* flag, int64_t least) {
        const char* text = value(flag);
        char* end = nullptr;
        const long long v = std::strtoll(text, &end, 10);
        if (bad) return least;
        if (*text == '\0' || *end != '\0' || v < least) {
            basalt::log::error(kTag, "%s takes a whole number of at least %lld, not '%s'", flag, (long long) least, text);
            bad = true;
            return least;
        }
        return (int64_t) v;
    }
    /// The value as a finite number of at least zero.
    double real(const char* flag) {
        const char* text = value(flag);
        char* end = nullptr;
        const double v = std::strtod(text, &end);
        if (bad) return 0.0;
        if (*text == '\0' || *end != '\0' || !(v >= 0.0) || !(v < 1e300)) {
            basalt::log::error(kTag, "%s takes a number of at least zero, not '%s'", flag, text);
            bad = true;
            return 0.0;
        }
        return v;
    }
};

/// "31,35,39" as layer numbers; false for anything else.
bool parse_layers(const std::string& text, std::vector<int>& out) {
    out.clear();
    size_t at = 0;
    while (at <= text.size()) {
        const size_t comma = text.find(',', at);
        const std::string item = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        char* end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || v < 0 || v > 1 << 20) return false;
        out.push_back((int) v);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return !out.empty();
}

/// Reads a flag `train` and `eval` share; false when `a` is not one of them.
bool run_flag(const std::string& a, Args& args, RunOptions& o) {
    if (a == "--model") o.model = args.value("--model");
    else if (a == "--shards") o.shards = args.value("--shards");
    else if (a == "--device") o.device = args.value("--device");
    else if (a == "--head-dtype") o.head_dtype = args.value("--head-dtype");
    else if (a == "--blocks") o.blocks = args.integer("--blocks", 1);
    else if (a == "--loss-blocks") o.loss.chunk_blocks = args.integer("--loss-blocks", 1);
    else if (a == "--attention-blocks") o.attention_blocks = args.integer("--attention-blocks", 1);
    else if (a == "--seed") o.seed = (uint64_t) args.integer("--seed", 0);
    else if (a == "--queue-gb") o.queue_bytes = (uint64_t) (args.real("--queue-gb") * 1073741824.0);
    else if (a == "--kl-topk") o.loss.kl_topk = args.integer("--kl-topk", 1);
    else if (a == "--dpace-alpha") o.loss.dpace_alpha = args.real("--dpace-alpha");
    else if (a == "--e2e-weight") o.loss.e2e_weight = args.real("--e2e-weight");
    else if (a == "--confidence-weight") o.loss.confidence_weight = args.real("--confidence-weight");
    else if (a == "--prefix-weight") o.prefix_weight = args.real("--prefix-weight");
    else if (a == "--kv-int8-share") o.kv_int8_share = args.real("--kv-int8-share");
    else return false;
    return true;
}

/// The exit code of a command line that named a flag the command does not have.
int unknown(const char* command, const std::string& flag) {
    basalt::log::error(kTag, "%s has no option %s (--help lists them)", command, flag.c_str());
    return 2;
}

int run_init(Args& args) {
    InitOptions o;
    for (; args.i < args.argc && !args.bad; ++args.i) {
        const std::string a = args.argv[args.i];
        if (a == "--model") o.model = args.value("--model");
        else if (a == "--out") o.out = args.value("--out");
        else if (a == "--layers") {
            const char* text = args.value("--layers");
            if (!args.bad && !parse_layers(text, o.layers)) { basalt::log::error(kTag, "--layers takes layer numbers separated by commas, not '%s'", text); return 2; }
        }
        else if (a == "--draft-layers") o.draft_layers = args.integer("--draft-layers", 1);
        else if (a == "--block") o.block = args.integer("--block", 2);
        else if (a == "--mask-id") o.mask_token_id = args.integer("--mask-id", 0);
        else if (a == "--rms-eps") o.rms_norm_eps = args.real("--rms-eps");
        else if (a == "--seed") o.seed = (uint64_t) args.integer("--seed", 0);
        else return unknown("init", a);
    }
    if (args.bad) return 2;
    std::string err;
    if (!init_drafter(o, err)) { basalt::log::error(kTag, "%s", err.c_str()); return 1; }
    return 0;
}

int run_train(Args& args) {
    TrainOptions o;
    for (; args.i < args.argc && !args.bad; ++args.i) {
        const std::string a = args.argv[args.i];
        if (run_flag(a, args, o)) continue;
        if (a == "--run") o.run = args.value("--run");
        else if (a == "--init") o.init = args.value("--init");
        else if (a == "--dtype") o.dtype = args.value("--dtype");
        else if (a == "--steps") o.steps = args.integer("--steps", 1);
        else if (a == "--epochs") o.epochs = args.real("--epochs");
        else if (a == "--blocks-per-step") o.blocks_per_step = args.integer("--blocks-per-step", 1);
        else if (a == "--save-every") o.save_every = args.integer("--save-every", 1);
        else if (a == "--keep") o.keep = args.integer("--keep", 1);
        else if (a == "--log-every") o.log_every = args.integer("--log-every", 1);
        else if (a == "--lr") o.optim.lr = args.real("--lr");
        else if (a == "--min-lr") o.optim.min_lr = args.real("--min-lr");
        else if (a == "--warmup-ratio") o.optim.warmup_ratio = args.real("--warmup-ratio");
        else if (a == "--weight-decay") o.optim.weight_decay = args.real("--weight-decay");
        else if (a == "--beta1") o.optim.beta1 = args.real("--beta1");
        else if (a == "--beta2") o.optim.beta2 = args.real("--beta2");
        else if (a == "--adam-eps") o.optim.adam_eps = args.real("--adam-eps");
        else if (a == "--muon-weight-decay") o.optim.muon_weight_decay = args.real("--muon-weight-decay");
        else if (a == "--muon-momentum") o.optim.muon_momentum = args.real("--muon-momentum");
        else if (a == "--muon-ns-steps") o.optim.muon_ns_steps = args.integer("--muon-ns-steps", 0);
        else if (a == "--muon-rms") o.optim.muon_rms = args.real("--muon-rms");
        else if (a == "--muon-plain-momentum") o.optim.muon_nesterov = false;
        else if (a == "--muon-head-rows") o.muon_head_rows = args.integer("--muon-head-rows", 0);
        else if (a == "--grad-clip") o.optim.max_grad_norm = args.real("--grad-clip");
        else return unknown("train", a);
    }
    if (args.bad) return 2;
    std::signal(SIGINT, on_interrupt);
    std::signal(SIGTERM, on_interrupt);
#ifdef SIGBREAK
    std::signal(SIGBREAK, on_interrupt);
#endif
    TrainResult result;
    std::string err;
    if (!train(o, [] { return g_stop != 0; }, result, err)) { basalt::log::error(kTag, "%s", err.c_str()); return 1; }
    return 0;
}

int run_fixture(Args& args) {
    FixtureOptions o;
    for (; args.i < args.argc && !args.bad; ++args.i) {
        const std::string a = args.argv[args.i];
        if (run_flag(a, args, o)) continue;
        if (a == "--checkpoint") o.checkpoint = args.value("--checkpoint");
        else if (a == "--out") o.out = args.value("--out");
        else if (a == "--count") o.count = args.integer("--count", 1);
        else return unknown("fixture", a);
    }
    if (args.bad) return 2;
    std::string err;
    if (!write_fixture(o, err)) { basalt::log::error(kTag, "%s", err.c_str()); return 1; }
    return 0;
}

int run_eval(Args& args) {
    EvalOptions o;
    for (; args.i < args.argc && !args.bad; ++args.i) {
        const std::string a = args.argv[args.i];
        if (run_flag(a, args, o)) continue;
        if (a == "--checkpoint") o.checkpoint = args.value("--checkpoint");
        else if (a == "--max-blocks") o.max_blocks = args.integer("--max-blocks", 1);
        else return unknown("eval", a);
    }
    if (args.bad) return 2;
    EvalResult r;
    std::string err;
    if (!evaluate(o, r, err)) { basalt::log::error(kTag, "%s", err.c_str()); return 1; }

    // the result is the program's output: one name and its numbers per line
    std::printf("drafter               %s (step %lld)\n", o.checkpoint.c_str(), (long long) r.step);
    std::printf("blocks                %lld, each drafting %lld tokens\n", (long long) r.blocks, (long long) r.drafts);
    std::printf("loss                  %.6f\n", r.loss);
    std::printf("  kl                  %.6f\n", r.kl);
    std::printf("  e2e                 %.6f\n", r.e2e);
    std::printf("  confidence          %.6f\n", r.confidence);
    std::printf("confidence error      %.6f\n", r.confidence_mae);
    std::printf("row accuracy, corpus ");
    for (const double a : r.accuracy_corpus) std::printf(" %.4f", a);
    std::printf("\nrow accuracy, teacher");
    for (const double a : r.accuracy_teacher) std::printf(" %.4f", a);
    std::printf("\naccepted drafts per block\n");
    std::printf("  teacher-forced, corpus token     %.4f\n", r.accept_corpus);
    std::printf("  teacher-forced, greedy           %.4f\n", r.accept_teacher);
    std::printf("  teacher-forced, sampling         %.4f\n", r.accept_sampling);
    std::printf("  serving chain, greedy            %.4f\n", r.accept_served);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) { usage(); return 0; }
    Args args{argc, argv};
    const std::string command = argv[1];
    if (command == "init") return run_init(args);
    if (command == "train") return run_train(args);
    if (command == "eval") return run_eval(args);
    if (command == "fixture") return run_fixture(args);
    basalt::log::error(kTag, "'%s' is not init, train, eval or fixture (--help lists the options)", command.c_str());
    return 2;
}
