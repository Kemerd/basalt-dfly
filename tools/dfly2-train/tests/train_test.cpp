// tools/dfly2-train/tests/train_test.cpp - the trainer end to end on the CPU: a tiny target pack, teacher shards written
// by the real shard writer, then init, eval, train, an interrupted and resumed run against an uninterrupted one, BF16
// weights, the refusals, a fixture, and the program itself when its path is given as the first argument.
//
// The fixture's corpus is a Markov chain the drafter can learn: a token's teacher row is a fixed vector of the token,
// and the next token is the teacher's likeliest most of the time.
#include "checkpoint.hpp"
#include "data.hpp"
#include "safetensors.hpp"
#include "trainer.hpp"

#include "test_util.hpp"
#include "tiny.hpp"
#include "tiny_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace basalt::dfly2;
using dfly2_test::check;
using dfly2_test::refused;

namespace {

constexpr int64_t kBlock = 4;

/// `t` as FP16 bit patterns, row-major.
std::vector<uint16_t> half_bits(const torch::Tensor& t) {
    const torch::Tensor h = t.to(torch::kFloat16).contiguous();
    std::vector<uint16_t> out((size_t) h.numel());
    std::memcpy(out.data(), h.data_ptr(), out.size() * 2);
    return out;
}

/// The shard header of the tiny target's sparse layers and `pack`.
ShardHeader shard_header(const dfly2_test::TinyTarget& t, const PackIdentity& pack) {
    ShardHeader h{};
    h.hidden = (uint32_t) t.hidden;
    h.vocab = (uint32_t) t.vocab;
    h.n_head_kv = (uint32_t) t.n_head_kv;
    h.head_dim = (uint32_t) t.head_dim;
    h.n_layers = (uint32_t) t.sparse.size();
    h.sel_cap = (uint32_t) (t.top_k + t.compress - 1);
    h.n_rot = (uint32_t) t.n_rot;
    h.hidden_dtype = (uint32_t) ShardDtype::f16;
    h.kv_dtype = (uint32_t) ShardDtype::f16;
    h.rope_base = 1e7;
    for (size_t l = 0; l < t.sparse.size(); ++l) h.layers[l] = (uint32_t) t.sparse[l];
    h.pack_bytes = pack.bytes;
    std::memcpy(h.pack_meta_sha256, pack.meta_sha256.data(), 32);
    std::snprintf(h.pack_name, sizeof h.pack_name, "%s", pack.name.c_str());
    return h;
}

/// Writes one sequence of `rows` rows of the chain into `w`: an anchor every second row from `first_anchor`, each
/// layer selecting the cells just before the anchor (every cell in the first layer, every other one in the rest).
bool write_sequence(ShardWriter& w, const ShardHeader& h, const torch::Tensor& state_of_token, const torch::Tensor& next_of_token, int64_t rows, int64_t first_anchor,
                    int64_t pos0, std::mt19937& rng, std::string& err) {
    std::uniform_int_distribution<int> any_token(0, (int) h.vocab - 2);   // the last id is the mask token
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::vector<int32_t> tokens((size_t) rows), anchors;
    tokens[0] = any_token(rng);
    for (int64_t t = 1; t < rows; ++t)
        tokens[(size_t) t] = coin(rng) < 0.85f ? (int32_t) next_of_token[tokens[(size_t) t - 1]].item<int64_t>() : any_token(rng);
    for (int64_t a = first_anchor; a < rows; a += 2) anchors.push_back((int32_t) a);
    if (!w.begin_sequence(tokens.data(), (uint64_t) rows, anchors.data(), anchors.size(), pos0, err)) return false;

    // the teacher's head input: the token's vector, and cached cells that are noise the drafter may ignore
    const torch::Tensor ids = torch::tensor(std::vector<int64_t>(tokens.begin(), tokens.end()), torch::kLong);
    const std::vector<uint16_t> hidden = half_bits(state_of_token.index_select(0, ids));
    if (!w.put_hidden(0, (uint64_t) rows, hidden.data(), err)) return false;
    for (uint32_t l = 0; l < h.n_layers; ++l) {
        const std::vector<uint16_t> k = half_bits(torch::randn({rows, (int64_t) h.n_head_kv, (int64_t) h.head_dim})), v = half_bits(torch::randn({rows, (int64_t) h.n_head_kv, (int64_t) h.head_dim}));
        if (!w.put_kv(l, 0, (uint64_t) rows, k.data(), v.data(), err)) return false;
    }
    for (size_t a = 0; a < anchors.size(); ++a)
        for (uint32_t l = 0; l < h.n_layers; ++l) {
            std::vector<int32_t> sel;
            for (int32_t c = anchors[a] - 1; c >= 0 && sel.size() < h.sel_cap; c -= (l == 0 ? 1 : 2)) sel.insert(sel.begin(), c);
            if (!w.put_selection(a, l, sel.data(), (uint32_t) sel.size(), err)) return false;
        }
    return w.end_sequence(err);
}

/// A drafter file's tensors against another's: the largest difference, 1e300 when they do not line up.
double file_diff(const std::string& a, const std::string& b) {
    DrafterFile fa, fb;
    std::string err;
    if (!read_drafter_file(a, fa, err) || !read_drafter_file(b, fb, err) || fa.tensors.size() != fb.tensors.size()) return 1e300;
    double worst = 0;
    for (const auto& [name, t] : fa.tensors) {
        const auto it = fb.tensors.find(name);
        if (it == fb.tensors.end() || !(it->second.sizes() == t.sizes())) return 1e300;
        worst = std::max(worst, (t - it->second).abs().max().item<double>());
    }
    return worst;
}

/// A file's text.
std::string slurp(const std::string& path) {
    std::ifstream f(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

/// Runs the trainer program with `args`, its output into `log`; the exit code.
int run_program(const std::string& exe, const std::string& args, const std::string& log) {
    std::string command = "\"" + exe + "\" " + args + " > \"" + log + "\" 2>&1";
#ifdef _WIN32
    command = "\"" + command + "\"";   // cmd strips one pair of quotes around the whole line
#endif
    const int code = std::system(command.c_str());
#ifdef _WIN32
    return code;
#else
    return code == -1 ? -1 : WEXITSTATUS(code);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    // a CPU test: no GPU is visible to it or to the program it starts, whatever the machine has
#ifdef _WIN32
    _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    return dfly2_test::run("train_test", [&]() -> int {
    torch::manual_seed(5);
    const dfly2_test::TempDir dir("basalt-dfly2-train");
    const dfly2_test::TinyTarget target;
    const std::string pack_path = dir.file("tiny.basalt");
    const dfly2_test::TinyPack tiny = dfly2_test::write_tiny_pack(pack_path, target, 11);
    std::string err;

    std::printf("1. the fixture: a pack and two shards through the shard writer\n");
    TargetPack pack;
    check(pack.open(pack_path, err), "the tiny pack opens", err);
    const std::string shards = dir.file("shards"), run_a = dir.file("run-a"), run_b = dir.file("run-b"), run_c = dir.file("run-c");
    std::filesystem::create_directories(dir.path / "shards");
    int64_t fixture_blocks = 0, fixture_sequences = 0;
    {
        // a token's teacher row scores one token far above the rest, and the chain follows it
        const torch::Tensor head = tiny.expect.at("output.weight");                           // [vocab][hidden]
        const torch::Tensor state = torch::randn({target.vocab, target.hidden}) * 3.0;
        const torch::Tensor next = torch::matmul(state.to(torch::kFloat16).to(torch::kFloat32), head.t()).slice(1, 0, target.vocab - 1).argmax(-1);
        const ShardHeader h = shard_header(target, pack.identity());
        std::mt19937 rng(3);
        // rows, first anchor, pos0: one sequence anchored at row 0, one too short for a block, one off position 0
        const std::vector<std::vector<int64_t>> plan[2] = {{{70, 1, 0}, {64, 0, 0}, {3, 1, 0}, {81, 1, 0}}, {{90, 1, 0}, {57, 2, 1000}, {76, 1, 0}}};
        bool ok = true;
        for (int f = 0; f < 2 && ok; ++f) {
            ShardWriter w;
            ok = w.open(shards + "/part-" + std::to_string(f) + kShardExtension, h, err);
            for (const auto& s : plan[f]) {
                ok = ok && write_sequence(w, h, state, next, s[0], s[1], s[2], rng, err);
                // an anchor counts when its block of four rows ends inside the sequence
                int64_t blocks = 0;
                for (int64_t a = s[1]; a + kBlock <= s[0]; a += 2) ++blocks;
                fixture_blocks += blocks;
                fixture_sequences += blocks > 0 ? 1 : 0;
            }
            ok = ok && w.close(err);
        }
        check(ok, "two shards of seven sequences are written", err);
    }

    std::printf("2. init\n");
    const std::string init_file = dir.file("init.safetensors");
    InitOptions io;
    io.model = pack_path;
    io.out = init_file;
    io.draft_layers = 2;
    io.block = kBlock;
    io.mask_token_id = target.vocab - 1;
    check(init_drafter(io, err), "a drafter is initialised from the pack", err);
    DrafterFile init;
    check(read_drafter_file(init_file, init, err), "its file reads back", err);
    check(init.step == 0 && init.weight_dtype == "F32" && init.shape.target_layers == std::vector<int>({3, 7}) && init.shape.sel_cap == 11 && init.shape.block == kBlock &&
              init.shape.intermediate == 3 * target.hidden && drafter_file_matches(init, pack.identity(), err),
          "it pairs with the two sparse layers, at step 0, and names the pack");
    check(dfly2_test::max_abs_diff(init.tensors.at("layers.1.self_attn.q_proj.weight"), tiny.expect.at("blk.7.attn_q.weight")) == 0.0 &&
              init.tensors.at("mtp_fusion.gate").abs().max().item<double>() == 0.0 && init.tensors.at("hidden_correction.down_proj.weight").abs().max().item<double>() == 0.0,
          "its attention is the target's, its fusion gate and correction output zero");
    {
        ShardSet set;
        check(set.open(shards, init.shape, pack.identity(), err), "the shards open for it", err);
        check((int64_t) set.blocks() == fixture_blocks && (int64_t) set.sequences().size() == fixture_sequences && set.shards() == 2,
              "the set counts the fixture's " + std::to_string(fixture_blocks) + " blocks in " + std::to_string(fixture_sequences) + " sequences");
        // one pass hands out every block once
        SequenceLoader loader(set, 9, 0, 0, 0, 1, 1);
        LoadedSequence s;
        int64_t seen = 0, sequences = 0;
        bool inside = true;
        while (loader.next(s, err)) {
            seen += s.anchors.size(0);
            ++sequences;
            inside = inside && s.anchors.max().item<int64_t>() + kBlock <= s.ctx.tokens.size(0) && s.selection.size(0) == s.anchors.size(0) && s.ctx.hidden.scalar_type() == torch::kFloat16;
        }
        check(err.empty() && seen == fixture_blocks && sequences == fixture_sequences && inside, "a pass of the loader hands out every block once, each inside its sequence", err);
    }

    std::printf("3. eval before training\n");
    EvalOptions eo;
    eo.model = pack_path;
    eo.shards = shards;
    eo.device = "cpu";
    eo.blocks = 24;
    eo.loss.chunk_blocks = 8;
    eo.checkpoint = init_file;
    EvalResult before;
    check(evaluate(eo, before, err), "the untrained drafter is measured", err);
    std::printf("  loss %.4f (kl %.4f, e2e %.4f, confidence %.4f); accepted of %lld: %.3f sampling, %.3f greedy, %.3f served\n", before.loss, before.kl, before.e2e,
                before.confidence, (long long) before.drafts, before.accept_sampling, before.accept_teacher, before.accept_served);
    check(before.blocks == fixture_blocks && before.drafts == kBlock - 1 && std::isfinite(before.loss) && before.accuracy_teacher.size() == 3, "over every block of the fixture");
    eo.max_blocks = 50;
    EvalResult part;
    check(evaluate(eo, part, err) && part.blocks == 50, "--max-blocks measures that many", err);
    eo.max_blocks = 0;

    std::printf("4. train\n");
    TrainOptions to;
    to.model = pack_path;
    to.shards = shards;
    to.device = "cpu";
    to.dtype = "f32";
    to.run = run_a;
    to.init = init_file;
    to.steps = 120;
    to.blocks_per_step = 64;
    to.blocks = 24;
    to.loss.chunk_blocks = 8;
    to.attention_blocks = 16;
    to.save_every = 25;
    to.keep = 2;
    to.log_every = 20;
    to.queue_bytes = 1 << 20;
    TrainResult ra;
    check(train(to, nullptr, ra, err), "120 steps of 64 blocks", err);
    std::printf("  step loss %.4f -> %.4f\n", ra.first_loss, ra.last_loss);
    std::vector<int64_t> kept;
    check(ra.steps == 120 && !ra.stopped && ra.checkpoint == drafter_path(run_a, 120) && list_checkpoints(run_a, kept, err) && kept == std::vector<int64_t>({100, 120}),
          "the run ends at its last step and keeps its two newest checkpoints");
    const std::string log_text = slurp(run_a + "/train.log");
    check(std::count(log_text.begin(), log_text.end(), '\n') == 6 && log_text.find("step 120/120") != std::string::npos, "train.log has a line per 20 steps");
    eo.checkpoint = ra.checkpoint;
    EvalResult after;
    check(evaluate(eo, after, err), "the trained drafter is measured", err);
    std::printf("  loss %.4f (kl %.4f, e2e %.4f, confidence %.4f); accepted of %lld: %.3f sampling, %.3f greedy, %.3f served\n", after.loss, after.kl, after.e2e,
                after.confidence, (long long) after.drafts, after.accept_sampling, after.accept_teacher, after.accept_served);
    check(after.step == 120 && after.loss < 0.5 * before.loss && after.kl < 0.5 * before.kl, "the loss over the fixture fell to under half");
    check(after.accept_served > before.accept_served + 0.5 && after.accept_sampling > before.accept_sampling + 0.5, "and the accepted lengths rose by more than half a draft");

    std::printf("5. an interrupted run, resumed\n");
    to.run = run_b;
    TrainResult rb;
    int asked = 0;
    check(train(to, [&] { return ++asked > 150; }, rb, err), "a run stopped after 150 forwards", err);
    std::printf("  stopped at step %lld\n", (long long) rb.steps);
    check(rb.stopped && rb.steps > 0 && rb.steps < 120 && rb.checkpoint == drafter_path(run_b, rb.steps), "it stops short of the run's end and keeps its last whole step");
    to.init.clear();
    check(train(to, nullptr, rb, err), "the same command resumes it", err);
    const double diff = file_diff(drafter_path(run_a, 120), drafter_path(run_b, 120));
    std::printf("  max |resumed - uninterrupted| over every weight at step 120: %.3e\n", diff);
    check(rb.steps == 120 && !rb.stopped && diff == 0.0, "and ends bit for bit where the uninterrupted run did");
    check(train(to, nullptr, rb, err) && rb.steps == 120, "a finished run resumes to nothing", err);
    TrainState state;
    check(read_train_state(state_path(run_b, 120), state, err) && state.step == 120 && state.total_steps == 120 && (int64_t) state.dataset_blocks == fixture_blocks,
          "its state names the step, the schedule and the data", err);

    std::printf("6. refusals\n");
    {
        TrainOptions bad = to;
        bad.seed = 7;
        refused(!train(bad, nullptr, rb, err), "a run resumed with another seed", err);
        bad = to;
        bad.dtype = "bf16";
        refused(!train(bad, nullptr, rb, err), "a run resumed with another weight type", err);
        bad = to;
        bad.run = run_c;
        refused(!train(bad, nullptr, rb, err), "a new run without --init", err);
        bad.init = init_file;
        bad.device = "cuda";
        refused(!train(bad, nullptr, rb, err), "a CUDA device where none is visible", err);
        // shards exported from another pack
        PackIdentity other = pack.identity();
        other.bytes += 1;
        const std::string elsewhere = dir.file("other-shards");
        std::filesystem::create_directories(dir.path / "other-shards");
        ShardWriter w;
        std::mt19937 rng(4);
        const ShardHeader h = shard_header(target, other);
        const bool wrote = w.open(elsewhere + "/part-0" + kShardExtension, h, err) &&
                           write_sequence(w, h, torch::randn({target.vocab, target.hidden}), torch::zeros({target.vocab}, torch::kLong), 20, 1, 0, rng, err) && w.close(err);
        check(wrote, "a shard naming another pack is written", err);
        bad.device = "cpu";
        bad.shards = elsewhere;
        refused(!train(bad, nullptr, rb, err), "shards exported from another pack", err);
        bad.shards = dir.file("no-such-directory");
        refused(!train(bad, nullptr, rb, err), "a shard directory that does not exist", err);
        EvalOptions bad_eval = eo;
        bad_eval.checkpoint = state_path(run_b, 120);
        refused(!evaluate(bad_eval, part, err), "a state file given as a drafter", err);
    }

    std::printf("7. BF16 weights behind FP32 masters\n");
    {
        TrainOptions bf = to;
        bf.run = run_c;
        bf.init = init_file;
        bf.dtype = "bf16";
        bf.steps = 6;
        bf.save_every = 3;
        TrainResult rc;
        check(train(bf, nullptr, rc, err), "six steps with BF16 weights", err);
        DrafterFile file;
        check(read_drafter_file(rc.checkpoint, file, err) && file.weight_dtype == "BF16" && file.step == 6 && std::isfinite(rc.last_loss), "the checkpoint says BF16 and the loss is finite", err);
        // the file holds the masters: values a BF16 weight cannot hold
        const torch::Tensor w = file.tensors.at("layers.0.mlp.up_proj.weight");
        check((w - w.to(torch::kBFloat16).to(torch::kFloat32)).abs().max().item<double>() > 0.0, "its weights are the FP32 masters, not their BF16 rounding");
        eo.checkpoint = rc.checkpoint;
        EvalResult bf_eval;
        check(evaluate(eo, bf_eval, err) && std::isfinite(bf_eval.loss), "eval runs it in BF16", err);
        std::printf("  step loss %.4f -> %.4f, eval loss %.4f\n", rc.first_loss, rc.last_loss, bf_eval.loss);
    }

    std::printf("8. a fixture of the untrained drafter\n");
    {
        FixtureOptions fo;
        fo.model = pack_path;
        fo.shards = shards;
        fo.device = "cpu";
        fo.checkpoint = init_file;
        fo.out = dir.file("fixture.safetensors");
        fo.count = 6;
        check(write_fixture(fo, err), "six blocks are written", err);
        SafetensorsFile f;
        check(f.open(fo.out, err), "the file reads back", err);
        // a tensor of the file as `type`, undefined when it is missing or of another size
        auto tensor = [&](const std::string& name, torch::Dtype type) -> torch::Tensor {
            const StTensor* t = f.find(name);
            if (t == nullptr || t->end - t->begin != t->elements() * (uint64_t) torch::elementSize(type)) return {};
            return torch::from_blob((void*) f.data(*t), t->shape, type).clone();
        };
        const torch::Tensor head = tiny.expect.at("output.weight"), embd = tiny.expect.at("token_embd.weight");
        const torch::Tensor cw = init.tensors.at("confidence_head.proj.weight"), cb = init.tensors.at("confidence_head.proj.bias");
        const int64_t H = target.hidden, rows = kBlock - 1;
        bool shapes = f.metadata().count("blocks") == 1 && f.metadata().at("blocks") == "6", identity = true, chain = true;
        for (int b = 0; b < 6 && shapes; ++b) {
            const std::string p = "block." + std::to_string(b) + ".";
            const torch::Tensor z = tensor(p + "z", torch::kFloat32), hc = tensor(p + "h_corrected", torch::kFloat32),
                                conf = tensor(p + "conf_logit", torch::kFloat32), ids = tensor(p + "draft_ids", torch::kInt32),
                                prev = tensor(p + "emb_prev", torch::kFloat32), anchor = tensor(p + "anchor", torch::kInt32),
                                pos = tensor(p + "pos", torch::kInt64), hs = tensor(p + "h_s", torch::kFloat32);
            shapes = z.defined() && hc.defined() && conf.defined() && ids.defined() && prev.defined() && anchor.defined() && pos.defined() &&
                     hs.defined() && z.sizes() == torch::IntArrayRef({kBlock, H}) && hc.sizes() == torch::IntArrayRef({rows, H}) &&
                     prev.sizes() == torch::IntArrayRef({rows, H}) && ids.size(0) == rows && pos.item<int64_t>() >= 1;
            for (int64_t l = 0; shapes && l < 2; ++l) {
                const std::string q = p + "layer." + std::to_string(l) + ".";
                const torch::Tensor sel = tensor(q + "sel", torch::kInt32), k = tensor(q + "k_ctx", torch::kFloat16), v = tensor(q + "v_ctx", torch::kFloat16);
                // the cells selected for the row before the anchor, each with its cached key and value
                shapes = sel.defined() && k.defined() && v.defined() && sel.dim() == 1 && sel.size(0) >= 1 && sel.min().item<int32_t>() >= 0 &&
                         sel.max().item<int64_t>() < pos.item<int64_t>() && k.sizes() == torch::IntArrayRef({sel.size(0), target.n_head_kv, target.head_dim}) &&
                         v.sizes() == k.sizes();
            }
            if (!shapes) break;
            // the untrained correction is the identity, so each drafted row is z's next row
            identity = identity && hc.equal(z.slice(0, 1, kBlock));
            // each draft is the head's pick over its corrected row, its confidence logit the head's with the bias, and
            // the row after it reads its embedding
            const torch::Tensor want_conf = torch::matmul(hc, cw.t()).squeeze(-1) + cb;
            chain = chain && torch::matmul(hc, head.t()).argmax(-1).to(torch::kInt32).equal(ids) && (want_conf - conf).abs().max().item<double>() < 1e-5 &&
                    prev[0].equal(embd[anchor[0].item<int64_t>()]) && prev.slice(0, 1, rows).equal(embd.index_select(0, ids.slice(0, 0, rows - 1).to(torch::kLong)));
        }
        check(shapes, "every block holds its inputs and outputs in their shapes, after the sequence's first row");
        check(identity, "the untrained correction leaves z's drafted rows as they are");
        check(chain, "each draft is the head's argmax, its confidence the head's logit, and the next row reads it");
        FixtureOptions bad = fo;
        bad.count = 0;
        refused(!write_fixture(bad, err), "a fixture of no blocks", err);
    }

    if (argc > 1) {
        std::printf("9. the program\n");
        const std::string exe = argv[1], out = dir.file("program.txt"), cli_init = dir.file("cli-init.safetensors"), run_d = dir.file("run-d");
        check(run_program(exe, "--help", out) == 0 && slurp(out).find("basalt-dfly2-train init") != std::string::npos, "--help prints the commands");
        check(run_program(exe, "init --model \"" + pack_path + "\" --out \"" + cli_init + "\" --draft-layers 2 --block 4 --mask-id 47", out) == 0, "init", slurp(out));
        check(file_diff(cli_init, init_file) == 0.0, "writes the drafter the library call wrote");
        const std::string common = " --model \"" + pack_path + "\" --shards \"" + shards + "\" --device cpu --blocks 24 --loss-blocks 8 --attention-blocks 16";
        check(run_program(exe, "train" + common + " --run \"" + run_d + "\" --init \"" + cli_init + "\" --dtype f32 --steps 120 --blocks-per-step 64 --save-every 25 --keep 2 --log-every 20 --queue-gb 0.001", out) == 0,
              "train", slurp(out));
        const double cli_diff = file_diff(drafter_path(run_d, 120), drafter_path(run_a, 120));
        std::printf("  max |program - library| over every weight at step 120: %.3e\n", cli_diff);
        check(cli_diff == 0.0 && slurp(out).find("step 120/120") != std::string::npos, "trains to the weights the library call reached and prints its progress");
        check(run_program(exe, "eval" + common + " --checkpoint \"" + drafter_path(run_d, 120) + "\"", out) == 0 && slurp(out).find("serving chain, greedy") != std::string::npos, "eval prints the accepted lengths", slurp(out));
        check(run_program(exe, "train --no-such-flag", out) == 2 && run_program(exe, "train --steps many", out) == 2 && run_program(exe, "bake", out) == 2, "an unknown option, a bad number and an unknown command exit with 2");
        check(run_program(exe, "eval" + common + " --checkpoint \"" + dir.file("missing.safetensors") + "\"", out) == 1, "a missing drafter file exits with 1");
        const std::string cli_fixture = dir.file("cli-fixture.safetensors");
        check(run_program(exe, "fixture --model \"" + pack_path + "\" --shards \"" + shards + "\" --device cpu --checkpoint \"" + cli_init + "\" --out \"" + cli_fixture +
                                   "\" --count 3", out) == 0 && std::filesystem::exists(std::filesystem::path(std::u8string(cli_fixture.begin(), cli_fixture.end()))),
              "fixture writes its file", slurp(out));
    } else {
        std::printf("9. the program: skipped, no path to it was given\n");
    }
    return dfly2_test::finish("train_test");
    });
}
