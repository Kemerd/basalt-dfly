// tools/dfly2-train/src/optimizer.hpp - the drafter's optimiser: Muon on the weight matrices, AdamW on everything
// else, both on FP32 master weights behind the model's (BF16) weights, with one warmup-and-cosine learning rate.
//
// A step is taken over the gradients of many micro-batches. Each micro-batch's gradient is moved out of the model
// into an FP32 sum as soon as its backward ends, so the sum does not round in the model's type; step() divides it by
// the summed weight, clips its norm over every parameter jointly, updates the masters and writes them back to the
// model in its own type.
//
//   Muon   buf = momentum * buf + g;  u = newton_schulz(g + momentum * buf);
//          W *= 1 - lr * muon_weight_decay;  W -= lr * sqrt(max(rows, cols)) * muon_rms * u
//          where rows x cols is the matrix the iteration saw: one head's rows of a projection that is orthogonalised
//          head by head, the whole matrix otherwise.
//   AdamW  the decoupled-decay Adam of torch.optim.AdamW.
#pragma once

#include <torch/torch.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace basalt::dfly2 {

struct OptimConfig {
    double lr = 5e-4;                 ///< peak learning rate
    double min_lr = 5e-5;             ///< learning rate at the end of the cosine
    double warmup_ratio = 0.04;       ///< share of the steps spent warming up from zero
    int64_t total_steps = 1;          ///< steps of the whole schedule
    double weight_decay = 0.01;       ///< AdamW's
    double muon_weight_decay = 0.1;
    double beta1 = 0.9, beta2 = 0.95, adam_eps = 1e-8;
    double max_grad_norm = 1.0;       ///< joint clip over every parameter
    double muon_momentum = 0.95;
    bool muon_nesterov = true;
    int64_t muon_ns_steps = 5;
    double muon_rms = 0.2;            ///< scales Muon's step to the size of an AdamW step
    int64_t query_head_rows = 0;      ///< rows of one head of a query projection, orthogonalised on their own; 0: whole
};

/// The learning rate of the step taken after `step` earlier ones: linear from zero over int(warmup_ratio *
/// total_steps) steps, then a cosine from lr to min_lr at total_steps, min_lr after it.
double scheduled_lr(const OptimConfig& cfg, int64_t step);

/// The Newton-Schulz iteration Muon orthogonalises with, on `g` [..][rows][cols]: `steps` rounds of
/// X = a X + (b A + c A A) X, A = X X^T, from X = g / (|g| + 1e-7), on the transpose when rows > cols.
torch::Tensor newton_schulz(const torch::Tensor& g, int64_t steps);

class DrafterOptimizer {
public:
    /// Takes the named parameters to train. A parameter of two or more dimensions is Muon's unless its name holds
    /// "confidence_head"; the rest are AdamW's.
    DrafterOptimizer(std::vector<std::pair<std::string, torch::Tensor>> params, const OptimConfig& cfg);

    /// Moves the parameters' gradients, which stand for `weight` units of data, into the step's sum and clears them.
    void accumulate(double weight);
    /// Takes the step over what was accumulated and empties the sum. Returns the gradient's norm before clipping;
    /// a zero or non-finite gradient updates nothing and still advances the schedule.
    double step();
    /// Drops what was accumulated without taking a step.
    void discard();

    int64_t steps() const { return step_; }
    double accumulated_weight() const { return weight_; }
    /// The learning rate the next step will use.
    double next_lr() const { return scheduled_lr(cfg_, step_); }
    const OptimConfig& config() const { return cfg_; }

    /// The master weights by parameter name (the model's own tensors when it trains in the masters' type).
    std::vector<std::pair<std::string, torch::Tensor>> masters() const;
    /// Everything but the masters that a resumed run needs: the momentum and moment buffers by name
    /// ("muon.<param>", "adam_m.<param>", "adam_v.<param>").
    std::vector<std::pair<std::string, torch::Tensor>> buffers() const;
    /// The step counters a resumed run needs.
    int64_t adam_steps() const { return adam_t_; }

    /// Restores a run: master weights and buffers by the names masters() and buffers() give, and the counters.
    /// False with `err` when a tensor is missing or misshapen; nothing is changed then.
    bool restore(const std::map<std::string, torch::Tensor>& masters, const std::map<std::string, torch::Tensor>& buffers,
                 int64_t steps, int64_t adam_steps, std::string& err);

private:
    struct Slot {
        std::string name;
        torch::Tensor param;      ///< the model's tensor
        torch::Tensor master;     ///< FP32 (FP64 for an FP64 model); the model's own tensor when the types agree
        torch::Tensor sum;        ///< gradients accumulated since the last step, in the master's type
        torch::Tensor buf;        ///< Muon: momentum
        torch::Tensor m, v;       ///< AdamW: first and second moments
        bool muon = false, shared = false;
        int64_t head_rows = 0;    ///< Muon: rows orthogonalised together, 0 for the whole matrix
    };
    void muon_update(Slot& s, const torch::Tensor& g, double lr);
    void adam_update(Slot& s, const torch::Tensor& g, double lr);

    OptimConfig cfg_;
    std::vector<Slot> slots_;
    int64_t step_ = 0, adam_t_ = 0;
    double weight_ = 0;
};

}  // namespace basalt::dfly2
