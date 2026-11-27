// tools/dfly2-train/src/trainer.cpp - init, train, eval and fixture (trainer.hpp).
#include "trainer.hpp"

#include "checkpoint.hpp"
#include "data.hpp"
#include "safetensors.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>

namespace basalt::dfly2 {

namespace {

constexpr const char* kTag = "basalt dfly2-train";

/// "bf16" or "f32" as a type and as the checkpoint's spelling.
bool parse_dtype(const std::string& text, torch::Dtype& out, std::string& name) {
    if (text == "bf16") { out = torch::kBFloat16; name = "BF16"; return true; }
    if (text == "f32") { out = torch::kFloat32; name = "F32"; return true; }
    return false;
}

/// "cpu", "cuda" or "cuda:N" as a device that exists.
bool parse_device(const std::string& text, torch::Device& out, std::string& err) {
    if (text != "cpu" && text != "cuda" && text.rfind("cuda:", 0) != 0) { err = "device '" + text + "' is not cpu, cuda or cuda:N"; return false; }
    try {
        out = torch::Device(text);
    } catch (const std::exception&) {
        err = "device '" + text + "' is not cpu, cuda or cuda:N";
        return false;
    }
    if (!out.is_cuda()) return true;
    if (!torch::cuda::is_available()) { err = "no CUDA device is available; --device cpu runs on the processor"; return false; }
    if (out.has_index() && (size_t) out.index() >= torch::cuda::device_count()) { err = "there is no " + text + " (" + std::to_string(torch::cuda::device_count()) + " CUDA devices)"; return false; }
    return true;
}

/// What train and eval both hold: the target's frozen tables, the drafter and the data.
struct Session {
    TargetPack pack;
    torch::Device device{torch::kCPU};
    Drafter drafter{nullptr};
    torch::Tensor embedding, head;   ///< [vocab][hidden] on the device
    ShardSet shards;
};

/// Opens the pack and the shards for the drafter in `file`, builds the drafter with weights of `dtype` on the
/// device and loads the target's tables beside it.
bool open_session(const RunOptions& o, const DrafterFile& file, torch::Dtype dtype, Session& s, std::string& err) {
    torch::Dtype head_dtype = torch::kFloat32;
    std::string ignored;
    if (o.model.empty() || o.shards.empty()) { err = "--model and --shards are needed"; return false; }
    if (!parse_dtype(o.head_dtype, head_dtype, ignored)) { err = "head dtype '" + o.head_dtype + "' is not f32 or bf16"; return false; }
    if (o.blocks <= 0 || o.attention_blocks <= 0 || o.loss.chunk_blocks <= 0 || o.loss.kl_topk <= 0) { err = "a block count or the KL's token count is not positive"; return false; }
    if (!(o.loss.dpace_alpha >= 0.0 && o.loss.dpace_alpha <= 1.0) || !(o.loss.e2e_weight >= 0.0) || !(o.loss.confidence_weight >= 0.0)) { err = "a loss weight is out of range"; return false; }
    if (!parse_device(o.device, s.device, err)) return false;
    if (!s.pack.open(o.model, err) || !drafter_file_matches(file, s.pack.identity(), err)) return false;
    if (!s.shards.open(o.shards, file.shape, s.pack.identity(), err)) return false;

    // the drafter: every parameter from the file, then moved and rounded to the forward's type
    s.drafter = Drafter(file.shape);
    {
        torch::NoGradGuard no_grad;
        const auto params = s.drafter->named_parameters();
        if (params.size() != file.tensors.size()) { err = "the drafter file does not hold the drafter's " + std::to_string(params.size()) + " tensors"; return false; }
        for (const auto& p : params) {
            const auto it = file.tensors.find(p.key());
            if (it == file.tensors.end() || !(it->second.sizes() == p.value().sizes())) { err = "the drafter file has no tensor " + p.key() + " of the drafter's shape"; return false; }
            p.value().copy_(it->second);
        }
    }
    s.drafter->to(s.device, dtype);

    // the target's embedding in the drafter's type (it only feeds the drafter), its head in the head's own
    torch::Tensor table;
    if (!s.pack.embedding(table, err)) return false;
    s.embedding = table.to(s.device, dtype);
    if (!s.pack.head(table, err)) return false;
    s.head = table.to(s.device, head_dtype);
    return true;
}

/// A sequence's teacher data on the device, in the types the shard stores.
SequenceContext to_device(const SequenceContext& c, const torch::Device& device) {
    SequenceContext out;
    out.pos0 = c.pos0;
    out.tokens = c.tokens.to(device);
    out.hidden = c.hidden.to(device);
    for (const torch::Tensor& k : c.k) out.k.push_back(k.to(device));
    for (const torch::Tensor& v : c.v) out.v.push_back(v.to(device));
    return out;
}

/// Blocks [i0, i1) of a loaded sequence, their selections cut to the widest row among them.
BlockBatch make_batch(const LoadedSequence& s, int64_t i0, int64_t i1, const torch::Device& device) {
    BlockBatch b;
    b.anchors = s.anchors.slice(0, i0, i1).to(device);
    for (int64_t l = 0; l < s.selection.size(1); ++l) {
        const int64_t width = std::max<int64_t>(1, s.sel_count.slice(0, i0, i1).select(1, l).max().item<int32_t>());
        b.sel.push_back(s.selection.slice(0, i0, i1).select(1, l).slice(1, 0, width).to(device, torch::kLong));
    }
    return b;
}

/// Loss terms of several forwards, weighted by their blocks.
struct LossSum {
    double blocks = 0, loss = 0, kl = 0, e2e = 0, confidence = 0, mae = 0, tv_accept = 0;
    std::vector<double> hits_corpus, hits_teacher;

    void add(const LossTerms& t) {
        blocks += t.blocks;
        loss += t.loss * t.blocks;
        kl += t.kl * t.blocks;
        e2e += t.e2e * t.blocks;
        confidence += t.confidence * t.blocks;
        mae += t.confidence_mae * t.blocks;
        tv_accept += t.tv_accept;
        hits_corpus.resize(t.hits_corpus.size(), 0.0);
        hits_teacher.resize(t.hits_teacher.size(), 0.0);
        for (size_t i = 0; i < t.hits_corpus.size(); ++i) hits_corpus[i] += t.hits_corpus[i];
        for (size_t i = 0; i < t.hits_teacher.size(); ++i) hits_teacher[i] += t.hits_teacher[i];
    }
    /// A summed term per block.
    double per_block(double v) const { return blocks > 0 ? v / blocks : 0.0; }
};

/// 1234567 as "1.23M".
std::string short_count(double v) {
    char buf[32];
    if (v >= 1e9) std::snprintf(buf, sizeof buf, "%.2fG", v / 1e9);
    else if (v >= 1e6) std::snprintf(buf, sizeof buf, "%.2fM", v / 1e6);
    else if (v >= 1e3) std::snprintf(buf, sizeof buf, "%.1fk", v / 1e3);
    else std::snprintf(buf, sizeof buf, "%.0f", v);
    return buf;
}

/// Seconds as "1h02m" or "3m05s".
std::string short_time(double seconds) {
    const long long s = (long long) std::max(0.0, seconds);
    char buf[32];
    if (s >= 3600) std::snprintf(buf, sizeof buf, "%lldh%02lldm", s / 3600, s % 3600 / 60);
    else std::snprintf(buf, sizeof buf, "%lldm%02llds", s / 60, s % 60);
    return buf;
}

/// A copy of `t` on the CPU that later updates of `t` do not reach.
torch::Tensor snapshot(const torch::Tensor& t) {
    return t.device().is_cpu() ? t.detach().clone() : t.detach().cpu();
}

/// Writes a run's checkpoints on a thread of its own, one at a time, so a step waits for the copy of the weights
/// and not for the disk.
class CheckpointWriter {
public:
    CheckpointWriter(std::string run, int64_t keep) : run_(std::move(run)), keep_(keep) {}
    ~CheckpointWriter() {
        if (pending_.valid()) pending_.wait();
    }

    /// Waits for the write before it, then starts this one. False with `err` when the earlier write failed.
    bool submit(DrafterFile&& drafter, TrainState&& state, std::string& err) {
        if (!finish(err)) return false;
        pending_ = std::async(std::launch::async, [run = run_, keep = keep_, d = std::move(drafter), s = std::move(state)]() -> std::string {
            std::string e;
            try {
                if (!write_train_state(state_path(run, s.step), s, e) || !write_drafter_file(drafter_path(run, d.step), d, e)) return e;
                prune_checkpoints(run, keep);
            } catch (const std::exception& ex) {
                e = std::string("writing a checkpoint failed: ") + ex.what();
            }
            return e;
        });
        return true;
    }

    /// Waits for the write in flight. False with `err` when it failed.
    bool finish(std::string& err) {
        if (!pending_.valid()) return true;
        const std::string e = pending_.get();
        if (e.empty()) return true;
        err = e;
        return false;
    }

private:
    std::string run_;
    int64_t keep_;
    std::future<std::string> pending_;
};

/// Writes a progress line to the log and appends it to the run's train.log.
void progress(const std::string& run, const std::string& text) {
    const std::string line = basalt::log::render(basalt::log::Level::info, kTag, "%s", text.c_str());
    basalt::log::raw("%s", line.c_str());
    if (std::FILE* f = platform::open_file_utf8(run + "/train.log", "ab")) {
        std::fwrite(line.data(), 1, line.size(), f);
        std::fclose(f);
    }
}

bool init_impl(const InitOptions& o, std::string& err) {
    if (o.model.empty() || o.out.empty()) { err = "--model and --out are needed"; return false; }
    TargetPack pack;
    DrafterShape shape;
    if (!pack.open(o.model, err) || !pack.drafter_shape(o.layers, o.draft_layers, o.block, o.mask_token_id, o.rms_norm_eps, shape, err)) return false;
    torch::manual_seed(o.seed);
    Drafter drafter(shape);
    // each draft layer's attention starts as its paired target layer's, so its own cells match the cached ones
    for (int64_t i = 0; i < shape.n_layers; ++i) {
        TargetAttention attention;
        if (!pack.attention(shape.target_layers[(size_t) i], attention, err) || !drafter->init_attention(i, attention, err)) return false;
    }
    DrafterFile file;
    file.shape = shape;
    file.pack_name = pack.identity().name;
    file.pack_bytes = pack.identity().bytes;
    file.pack_meta_sha256 = pack.identity().meta_sha256_hex();
    for (const auto& p : drafter->named_parameters()) file.tensors[p.key()] = p.value().detach();
    if (!write_drafter_file(o.out, file, err)) return false;

    std::string layers;
    int64_t count = 0;
    for (const int l : shape.target_layers) layers += (layers.empty() ? "" : ",") + std::to_string(l);
    for (const auto& p : drafter->named_parameters()) count += p.value().numel();
    basalt::log::info(kTag, "%s: a drafter of %lld layers paired with target layers %s of %s, %s parameters, blocks of %lld rows", o.out.c_str(),
                      (long long) shape.n_layers, layers.c_str(), pack.identity().name.c_str(), short_count((double) count).c_str(), (long long) shape.block);
    basalt::log::info(kTag, "mask token %lld: it must be an id the tokenizer never produces", (long long) shape.mask_token_id);
    return true;
}

bool train_impl(const TrainOptions& o, const std::function<bool()>& stop, TrainResult& result, std::string& err) {
    result = TrainResult{};
    torch::Dtype dtype = torch::kBFloat16;
    std::string dtype_name;
    if (o.run.empty()) { err = "--run is needed"; return false; }
    if (!parse_dtype(o.dtype, dtype, dtype_name)) { err = "dtype '" + o.dtype + "' is not bf16 or f32"; return false; }
    if (o.blocks_per_step <= 0 || o.save_every <= 0 || o.keep <= 0 || o.log_every <= 0 || o.steps < 0 || !(o.epochs > 0.0)) { err = "a step, epoch or checkpoint count is out of range"; return false; }
    if (!platform::make_dirs(o.run)) { err = "cannot create the run directory " + o.run; return false; }

    // the run's newest checkpoint, or the file a new run starts from
    std::vector<int64_t> saved;
    if (!list_checkpoints(o.run, saved, err)) return false;
    const bool resumed = !saved.empty();
    DrafterFile file;
    TrainState state;
    if (resumed) {
        if (!read_drafter_file(drafter_path(o.run, saved.back()), file, err) || !read_train_state(state_path(o.run, saved.back()), state, err)) return false;
        if (state.step != saved.back() || file.step != saved.back()) { err = "the checkpoint of step " + std::to_string(saved.back()) + " in " + o.run + " names another step"; return false; }
        if (file.weight_dtype != dtype_name) { err = "the run was trained with " + file.weight_dtype + " weights; pass the same --dtype to resume it"; return false; }
    } else {
        if (o.init.empty()) { err = o.run + " holds no checkpoint to resume; --init names the drafter a new run starts from"; return false; }
        if (!read_drafter_file(o.init, file, err)) return false;
    }
    Session s;
    if (!open_session(o, file, dtype, s, err)) return false;
    const DrafterShape& shape = file.shape;

    // the schedule's length, and a resumed run's settings against the ones it was started with
    const int64_t total = o.steps > 0 ? o.steps : std::max<int64_t>(1, (int64_t) std::ceil(o.epochs * (double) s.shards.blocks() / (double) o.blocks_per_step));
    if (resumed && (state.seed != o.seed || state.blocks_per_step != o.blocks_per_step || state.total_steps != total ||
                    state.dataset_blocks != s.shards.blocks() || state.dataset_sequences != s.shards.sequences().size())) {
        err = "the run in " + o.run + " was started with seed " + std::to_string(state.seed) + ", " + std::to_string(state.blocks_per_step) + " blocks per step, " +
              std::to_string(state.total_steps) + " steps and shards of " + std::to_string(state.dataset_blocks) + " blocks in " + std::to_string(state.dataset_sequences) +
              " sequences; it resumes only with the same";
        return false;
    }
    OptimConfig cfg = o.optim;
    cfg.total_steps = total;
    cfg.query_head_rows = o.muon_head_rows < 0 ? shape.head_dim : o.muon_head_rows;
    std::vector<std::pair<std::string, torch::Tensor>> params;
    for (const auto& p : s.drafter->named_parameters()) params.emplace_back(p.key(), p.value());
    DrafterOptimizer opt(std::move(params), cfg);
    // the masters are the file's FP32 weights, not the forward's rounded copies of them
    std::map<std::string, torch::Tensor> buffers = state.buffers;
    if (!resumed)
        for (const auto& [name, t] : opt.buffers()) buffers[name] = t;
    if (!opt.restore(file.tensors, buffers, resumed ? state.step : 0, resumed ? state.adam_steps : 0, err)) return false;

    const int64_t drafts = shape.block - 1;
    const double dataset_blocks = (double) s.shards.blocks();
    basalt::log::info(kTag, "%zu shards: %zu sequences, %s tokens, %s blocks of %lld rows", s.shards.shards(), s.shards.sequences().size(),
                      short_count((double) s.shards.tokens()).c_str(), short_count(dataset_blocks).c_str(), (long long) shape.block);
    basalt::log::info(kTag, "%s at step %lld of %lld: %lld blocks per step in forwards of up to %lld, %s weights on %s", resumed ? "resuming" : "starting",
                      (long long) opt.steps(), (long long) total, (long long) o.blocks_per_step, (long long) o.blocks, dtype_name.c_str(), o.device.c_str());

    // the place the next step starts from; only a finished step moves it
    int64_t at_epoch = resumed ? state.epoch : 0;
    uint64_t at_position = resumed ? state.position : 0, at_first = resumed ? state.first : 0;
    SequenceLoader loader(s.shards, o.seed, at_epoch, at_position, at_first, -1, o.queue_bytes);
    CheckpointWriter writer(o.run, o.keep);
    int64_t last_saved = resumed ? state.step : 0;
    const double elapsed_before = resumed ? state.elapsed : 0.0;
    const auto started = std::chrono::steady_clock::now();
    const auto elapsed = [&] { return elapsed_before + std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };

    const auto save = [&](std::string& e) {
        DrafterFile snap;
        snap.shape = shape;
        snap.weight_dtype = dtype_name;
        snap.step = opt.steps();
        snap.pack_name = file.pack_name;
        snap.pack_bytes = file.pack_bytes;
        snap.pack_meta_sha256 = file.pack_meta_sha256;
        for (const auto& [name, t] : opt.masters()) snap.tensors[name] = snapshot(t);
        TrainState st;
        st.step = opt.steps();
        st.adam_steps = opt.adam_steps();
        st.epoch = at_epoch;
        st.position = at_position;
        st.first = at_first;
        st.elapsed = elapsed();
        st.seed = o.seed;
        st.blocks_per_step = o.blocks_per_step;
        st.total_steps = total;
        st.dataset_blocks = s.shards.blocks();
        st.dataset_sequences = s.shards.sequences().size();
        for (const auto& [name, t] : opt.buffers()) st.buffers[name] = snapshot(t);
        last_saved = opt.steps();
        return writer.submit(std::move(snap), std::move(st), e);
    };

    LoadedSequence seq;
    SequenceContext ctx;
    int64_t have = 0, used = 0, in_step = 0;   // blocks of the sequence in hand, taken from it, and in the open step
    LossSum window, step_sum;
    auto window_started = std::chrono::steady_clock::now();
    bool took_step = false;
    while (opt.steps() < total) {
        if (stop && stop()) { result.stopped = true; break; }
        if (used == have) {
            if (!loader.next(seq, err)) {
                if (err.empty()) err = "the shards ended before the run did";
                return false;
            }
            ctx = to_device(seq.ctx, s.device);
            have = seq.anchors.size(0);
            used = 0;
        }
        // one forward and backward; its gradient stands for `take` blocks of the step
        const int64_t take = std::min({o.blocks, have - used, o.blocks_per_step - in_step});
        const BlockBatch batch = make_batch(seq, used, used + take, s.device);
        const torch::Tensor z = s.drafter->backbone(s.embedding, ctx, batch, o.attention_blocks);
        torch::Tensor grad_z;
        const LossTerms terms = chunked_loss(s.drafter, z, s.embedding, s.head, ctx, batch.anchors, o.loss, (double) take, &grad_z);
        z.backward(grad_z);
        opt.accumulate((double) take);
        used += take;
        in_step += take;
        window.add(terms);
        step_sum.add(terms);
        if (in_step < o.blocks_per_step) continue;

        const double lr = opt.next_lr();
        const double grad_norm = opt.step();
        const int64_t step = opt.steps();
        in_step = 0;
        if (used == have) {
            at_epoch = seq.epoch;
            at_position = seq.position + 1;
            at_first = 0;
        } else {
            at_epoch = seq.epoch;
            at_position = seq.position;
            at_first = seq.first + (uint64_t) used;
        }
        const double step_loss = step_sum.per_block(step_sum.loss);
        if (!took_step) result.first_loss = step_loss;
        result.last_loss = step_loss;
        took_step = true;
        step_sum = LossSum{};
        if (!std::isfinite(grad_norm) || !std::isfinite(step_loss)) basalt::log::warn(kTag, "step %lld: the loss or its gradient is not finite; the step changed nothing", (long long) step);

        if (step % o.log_every == 0 || step == total) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - window_started).count();
            const double rate = seconds > 0 ? window.blocks / seconds : 0.0, spent = elapsed();
            const double left = rate > 0 ? (double) (total - step) * (double) o.blocks_per_step / rate : 0.0;
            char text[512];
            std::snprintf(text, sizeof text,
                          "step %lld/%lld  epoch %.2f  tokens %s  loss %.4f (kl %.4f  e2e %.4f  conf %.4f)  accepted %.2f sampled, %.2f greedy of %lld  lr %.2e  grad %.3f  %.0f blocks/s  %s elapsed, %s left",
                          (long long) step, (long long) total, (double) step * (double) o.blocks_per_step / dataset_blocks,
                          short_count((double) step * (double) o.blocks_per_step * (double) drafts).c_str(), window.per_block(window.loss), window.per_block(window.kl),
                          window.per_block(window.e2e), window.per_block(window.confidence), window.per_block(window.tv_accept),
                          simulated_accept_length(window.hits_teacher, window.blocks), (long long) drafts, lr, grad_norm, rate, short_time(spent).c_str(), short_time(left).c_str());
            progress(o.run, text);
            window = LossSum{};
            window_started = std::chrono::steady_clock::now();
        }
        if ((step % o.save_every == 0 || step == total) && !save(err)) return false;
    }

    // a stop inside a step drops the step's part; the last whole step is what is kept
    opt.discard();
    if (opt.steps() > last_saved && !save(err)) return false;
    if (!writer.finish(err)) return false;
    result.steps = opt.steps();
    result.total_steps = total;
    if (last_saved > 0) result.checkpoint = drafter_path(o.run, last_saved);
    if (result.stopped) basalt::log::info(kTag, "stopped at step %lld of %lld; the same command resumes the run", (long long) result.steps, (long long) total);
    else basalt::log::info(kTag, "finished %lld steps: %s", (long long) result.steps, result.checkpoint.c_str());
    return true;
}

bool evaluate_impl(const EvalOptions& o, EvalResult& result, std::string& err) {
    result = EvalResult{};
    if (o.checkpoint.empty()) { err = "--checkpoint is needed"; return false; }
    if (o.max_blocks < 0) { err = "--max-blocks is negative"; return false; }
    DrafterFile file;
    if (!read_drafter_file(o.checkpoint, file, err)) return false;
    Session s;
    if (!open_session(o, file, file.weight_dtype == "BF16" ? torch::kBFloat16 : torch::kFloat32, s, err)) return false;

    torch::NoGradGuard no_grad;
    const double limit = o.max_blocks > 0 ? (double) o.max_blocks : std::numeric_limits<double>::infinity();
    SequenceLoader loader(s.shards, o.seed, 0, 0, 0, 1, o.queue_bytes);
    LossSum sum;
    double served = 0;
    LoadedSequence seq;
    while (sum.blocks < limit && loader.next(seq, err)) {
        const SequenceContext ctx = to_device(seq.ctx, s.device);
        const int64_t have = seq.anchors.size(0);
        for (int64_t used = 0; used < have && sum.blocks < limit;) {
            const int64_t take = (int64_t) std::min({(double) o.blocks, (double) (have - used), limit - sum.blocks});
            const BlockBatch batch = make_batch(seq, used, used + take, s.device);
            const torch::Tensor z = s.drafter->backbone(s.embedding, ctx, batch, o.attention_blocks);
            sum.add(chunked_loss(s.drafter, z, s.embedding, s.head, ctx, batch.anchors, o.loss, 1.0, nullptr));
            served += served_chain_accept(s.drafter, z, s.embedding, s.head, ctx, batch.anchors, o.loss.chunk_blocks);
            used += take;
        }
    }
    if (!err.empty()) return false;

    result.blocks = (int64_t) sum.blocks;
    result.drafts = file.shape.block - 1;
    result.step = file.step;
    result.loss = sum.per_block(sum.loss);
    result.kl = sum.per_block(sum.kl);
    result.e2e = sum.per_block(sum.e2e);
    result.confidence = sum.per_block(sum.confidence);
    result.confidence_mae = sum.per_block(sum.mae);
    for (const double h : sum.hits_corpus) result.accuracy_corpus.push_back(sum.per_block(h));
    for (const double h : sum.hits_teacher) result.accuracy_teacher.push_back(sum.per_block(h));
    result.accept_corpus = simulated_accept_length(sum.hits_corpus, sum.blocks);
    result.accept_teacher = simulated_accept_length(sum.hits_teacher, sum.blocks);
    result.accept_sampling = sum.per_block(sum.tv_accept);
    result.accept_served = sum.per_block(served);
    return true;
}

/// Blocks a fixture takes from one sequence at most, so its blocks come from several documents.
constexpr int64_t kFixturePerSequence = 4;

/// The safetensors spelling of a tensor's type; empty for a type a fixture does not write.
std::string st_dtype_of(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kFloat32: return "F32";
        case torch::kFloat16: return "F16";
        case torch::kInt32: return "I32";
        case torch::kInt64: return "I64";
        default: return {};
    }
}

bool fixture_impl(const FixtureOptions& o, std::string& err) {
    if (o.checkpoint.empty() || o.out.empty()) { err = "--checkpoint and --out are needed"; return false; }
    if (o.count < 1) { err = "--count is not positive"; return false; }
    DrafterFile file;
    if (!read_drafter_file(o.checkpoint, file, err)) return false;
    Session s;
    if (!open_session(o, file, file.weight_dtype == "BF16" ? torch::kBFloat16 : torch::kFloat32, s, err)) return false;
    const DrafterShape& sh = file.shape;
    const int64_t rows = sh.block - 1;
    torch::NoGradGuard no_grad;

    // every tensor of the file on the CPU and contiguous, held until it is written
    std::vector<std::pair<std::string, torch::Tensor>> out;
    auto put = [&](const std::string& name, const torch::Tensor& t) { out.emplace_back(name, t.detach().to(torch::kCPU).contiguous()); };
    const torch::Tensor mask = s.embedding.select(0, sh.mask_token_id).to(torch::kFloat32);
    SequenceLoader loader(s.shards, o.seed, 0, 0, 0, 1, o.queue_bytes);
    LoadedSequence seq;
    int64_t n = 0;
    while (n < o.count && loader.next(seq, err)) {
        // a runtime drafts only after a row: an anchor at row 0 has no source row
        const torch::Tensor pick = torch::nonzero(seq.anchors.gt(0)).flatten().slice(0, 0, std::min(o.count - n, kFixturePerSequence));
        const int64_t take = pick.size(0);
        if (take == 0) continue;
        LoadedSequence part = seq;
        part.anchors = seq.anchors.index_select(0, pick);
        part.selection = seq.selection.index_select(0, pick);
        part.sel_count = seq.sel_count.index_select(0, pick);
        const SequenceContext ctx = to_device(seq.ctx, s.device);
        const BlockBatch batch = make_batch(part, 0, take, s.device);
        const torch::Tensor z = s.drafter->backbone(s.embedding, ctx, batch, o.attention_blocks);
        const torch::Tensor tokens = ctx.tokens.index_select(0, batch.anchors);
        const ServedChain chain = served_chain(s.drafter, z, s.embedding, s.head, tokens);
        for (int64_t b = 0; b < take; ++b, ++n) {
            const std::string p = "block." + std::to_string(n) + ".";
            const int64_t a = part.anchors[b].item<int64_t>();
            // the token each drafted row reads: the anchor, then the drafts before it
            const torch::Tensor prev = torch::cat({tokens.slice(0, b, b + 1), chain.drafts[b].slice(0, 0, rows - 1)});
            put(p + "anchor", tokens.slice(0, b, b + 1).to(torch::kInt32));
            put(p + "pos", torch::full({1}, a + ctx.pos0, torch::kLong));
            put(p + "h_s", ctx.hidden[a - 1].to(torch::kFloat32));
            put(p + "emb_anchor", s.embedding[tokens[b]].to(torch::kFloat32));
            put(p + "emb_mask", mask);
            put(p + "emb_prev", s.embedding.index_select(0, prev).to(torch::kFloat32));
            for (int64_t l = 0; l < sh.n_layers; ++l) {
                // the cells the layer selected, in the order the shard lists them, and their cached values
                const torch::Tensor ids = part.selection[b][l].slice(0, 0, part.sel_count[b][l].item<int32_t>());
                const std::string q = p + "layer." + std::to_string(l) + ".";
                put(q + "sel", ids);
                put(q + "k_ctx", seq.ctx.k[(size_t) l].index_select(0, ids.to(torch::kLong)));
                put(q + "v_ctx", seq.ctx.v[(size_t) l].index_select(0, ids.to(torch::kLong)));
            }
            put(p + "z", z[b].to(torch::kFloat32));
            put(p + "h_corrected", chain.corrected[b].to(torch::kFloat32));
            put(p + "conf_logit", chain.confidence[b].to(torch::kFloat32));
            put(p + "draft_ids", chain.drafts[b].to(torch::kInt32));
        }
    }
    if (!err.empty()) return false;
    if (n == 0) { err = "the shards hold no block after a sequence's first row"; return false; }

    std::vector<StWrite> tensors;
    for (const auto& [name, t] : out) {
        StWrite w;
        w.name = name;
        w.dtype = st_dtype_of(t);
        if (w.dtype.empty()) { err = "fixture: " + name + " has a type the file does not take"; return false; }
        w.shape.assign(t.sizes().begin(), t.sizes().end());
        w.data = t.data_ptr();
        w.bytes = (uint64_t) t.nbytes();
        tensors.push_back(std::move(w));
    }
    const std::map<std::string, std::string> meta = {
        {"format", "dfly2-fixture"}, {"blocks", std::to_string(n)}, {"block", std::to_string(sh.block)},
        {"layers", std::to_string(sh.n_layers)}, {"hidden", std::to_string(sh.hidden)}, {"step", std::to_string(file.step)},
        {"weight_dtype", file.weight_dtype}, {"pack_name", file.pack_name}};
    if (!safetensors_write(o.out, meta, tensors, err)) return false;
    basalt::log::info(kTag, "%s: %lld blocks of %lld rows, the forward of step %lld in %s", o.out.c_str(), (long long) n, (long long) sh.block,
                      (long long) file.step, file.weight_dtype.c_str());
    return true;
}

/// Runs a command's body; a LibTorch or allocation failure becomes the command's error.
template <class Body> bool guarded(std::string& err, Body&& body) {
    try {
        return body();
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace

bool init_drafter(const InitOptions& o, std::string& err) {
    return guarded(err, [&] { return init_impl(o, err); });
}

bool train(const TrainOptions& o, const std::function<bool()>& stop, TrainResult& result, std::string& err) {
    return guarded(err, [&] { return train_impl(o, stop, result, err); });
}

bool evaluate(const EvalOptions& o, EvalResult& result, std::string& err) {
    return guarded(err, [&] { return evaluate_impl(o, result, err); });
}

bool write_fixture(const FixtureOptions& o, std::string& err) {
    return guarded(err, [&] { return fixture_impl(o, err); });
}

}  // namespace basalt::dfly2
