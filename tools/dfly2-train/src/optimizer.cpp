// tools/dfly2-train/src/optimizer.cpp - Muon and AdamW over master weights (optimizer.hpp).
#include "optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace basalt::dfly2 {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Whether `name` ends with `tail`.
bool ends_with(const std::string& name, const std::string& tail) {
    return name.size() >= tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

double scheduled_lr(const OptimConfig& cfg, int64_t step) {
    const int64_t warmup = (int64_t) (cfg.warmup_ratio * (double) cfg.total_steps);
    if (warmup > 0 && step <= warmup) return cfg.lr * (double) step / (double) warmup;
    if (step > cfg.total_steps) return cfg.min_lr;
    const double span = (double) std::max<int64_t>(cfg.total_steps - warmup, 1);
    const double ratio = std::min(1.0, std::max(0.0, (double) (step - warmup) / span));
    return cfg.min_lr + 0.5 * (1.0 + std::cos(kPi * ratio)) * (cfg.lr - cfg.min_lr);
}

torch::Tensor newton_schulz(const torch::Tensor& g, int64_t steps) {
    if (!g.defined() || g.dim() < 2) throw std::invalid_argument("newton_schulz: not a matrix");
    const double a = 3.4445, b = -4.7750, c = 2.0315;
    // on the smaller side: X X^T is then the smaller square
    const bool tall = g.size(-2) > g.size(-1);
    torch::Tensor x = tall ? g.transpose(-2, -1) : g;
    // the spectral norm is at most the Frobenius norm, so every singular value starts at or below 1
    x = x / (x.pow(2).sum({-2, -1}, true).sqrt() + 1e-7);
    for (int64_t i = 0; i < steps; ++i) {
        const torch::Tensor A = torch::matmul(x, x.transpose(-2, -1));
        const torch::Tensor B = b * A + c * torch::matmul(A, A);
        x = a * x + torch::matmul(B, x);
    }
    return tall ? x.transpose(-2, -1) : x;
}

DrafterOptimizer::DrafterOptimizer(std::vector<std::pair<std::string, torch::Tensor>> params, const OptimConfig& cfg) : cfg_(cfg) {
    if (params.empty()) throw std::invalid_argument("optimizer: no parameters");
    if (cfg.total_steps <= 0 || !(cfg.lr >= cfg.min_lr) || !(cfg.min_lr >= 0.0) || !(cfg.warmup_ratio >= 0.0 && cfg.warmup_ratio < 1.0) ||
        !(cfg.max_grad_norm > 0.0) || cfg.muon_ns_steps < 0 || !(cfg.beta1 >= 0.0 && cfg.beta1 < 1.0) || !(cfg.beta2 >= 0.0 && cfg.beta2 < 1.0))
        throw std::invalid_argument("optimizer: a setting is out of range");
    torch::NoGradGuard no_grad;
    for (auto& [name, p] : params) {
        if (!p.defined() || !p.is_floating_point()) throw std::invalid_argument("optimizer: parameter " + name + " is not a float tensor");
        Slot s;
        s.name = name;
        s.param = p;
        const torch::Dtype master_type = p.scalar_type() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
        s.shared = p.scalar_type() == master_type;
        // a model already in the masters' type is its own master; otherwise the master is a wider copy
        s.master = s.shared ? p.detach() : p.detach().to(master_type);
        s.sum = torch::zeros_like(s.master);
        s.muon = p.dim() >= 2 && name.find("confidence_head") == std::string::npos;
        if (s.muon) {
            s.buf = torch::zeros_like(s.master);
            if (cfg.query_head_rows > 0 && p.dim() == 2 && ends_with(name, "self_attn.q_proj.weight")) {
                if (p.size(0) % cfg.query_head_rows != 0) throw std::invalid_argument("optimizer: " + name + " does not split into heads of " + std::to_string(cfg.query_head_rows) + " rows");
                s.head_rows = cfg.query_head_rows;
            }
        } else {
            s.m = torch::zeros_like(s.master);
            s.v = torch::zeros_like(s.master);
        }
        slots_.push_back(std::move(s));
    }
}

void DrafterOptimizer::accumulate(double weight) {
    torch::NoGradGuard no_grad;
    for (Slot& s : slots_) {
        torch::Tensor g = s.param.grad();
        if (!g.defined()) continue;
        s.sum += g.to(s.sum.scalar_type());
        g.zero_();
    }
    weight_ += weight;
}

void DrafterOptimizer::discard() {
    torch::NoGradGuard no_grad;
    for (Slot& s : slots_) s.sum.zero_();
    weight_ = 0;
}

void DrafterOptimizer::muon_update(Slot& s, const torch::Tensor& g, double lr) {
    s.buf.mul_(cfg_.muon_momentum).add_(g);
    const torch::Tensor update = cfg_.muon_nesterov ? g + cfg_.muon_momentum * s.buf : s.buf;
    // a matrix of more than two dimensions is one matrix per leading index, flattened behind it
    const torch::Tensor flat = update.reshape({update.size(0), -1});
    torch::Tensor u;
    int64_t rows = flat.size(0), cols = flat.size(1);
    if (s.head_rows > 0) {
        rows = s.head_rows;
        u = newton_schulz(flat.reshape({-1, rows, cols}), cfg_.muon_ns_steps).reshape(flat.sizes());
    } else {
        u = newton_schulz(flat, cfg_.muon_ns_steps);
    }
    const double adjusted = lr * std::sqrt((double) std::max(rows, cols)) * cfg_.muon_rms;
    s.master.mul_(1.0 - lr * cfg_.muon_weight_decay);
    s.master.add_(u.reshape(s.master.sizes()), -adjusted);
}

void DrafterOptimizer::adam_update(Slot& s, const torch::Tensor& g, double lr) {
    const double t = (double) (adam_t_ + 1);
    s.master.mul_(1.0 - lr * cfg_.weight_decay);
    s.m.mul_(cfg_.beta1).add_(g, 1.0 - cfg_.beta1);
    s.v.mul_(cfg_.beta2).addcmul_(g, g, 1.0 - cfg_.beta2);
    const double correction1 = 1.0 - std::pow(cfg_.beta1, t), correction2 = 1.0 - std::pow(cfg_.beta2, t);
    const torch::Tensor denom = s.v.sqrt() / std::sqrt(correction2) + cfg_.adam_eps;
    s.master.addcdiv_(s.m, denom, -lr / correction1);
}

double DrafterOptimizer::step() {
    torch::NoGradGuard no_grad;
    const double inv = weight_ > 0 ? 1.0 / weight_ : 1.0;
    // the norm of the averaged gradient over every parameter
    double sq = 0;
    for (const Slot& s : slots_) {
        const double n = s.sum.norm().item<double>();
        sq += n * n;
    }
    const double norm = std::sqrt(sq) * inv;
    if (norm > 0.0 && std::isfinite(norm)) {
        const double lr = scheduled_lr(cfg_, step_);
        const double clip = std::min(1.0, cfg_.max_grad_norm / (norm + 1e-6));
        for (Slot& s : slots_) {
            const torch::Tensor g = s.sum * (inv * clip);
            if (s.muon) muon_update(s, g, lr);
            else adam_update(s, g, lr);
            // the model sees the master rounded to its own type
            if (!s.shared) s.param.copy_(s.master);
        }
        ++adam_t_;
    }
    for (Slot& s : slots_) s.sum.zero_();
    weight_ = 0;
    ++step_;
    return norm;
}

std::vector<std::pair<std::string, torch::Tensor>> DrafterOptimizer::masters() const {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    for (const Slot& s : slots_) out.emplace_back(s.name, s.master);
    return out;
}

std::vector<std::pair<std::string, torch::Tensor>> DrafterOptimizer::buffers() const {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    for (const Slot& s : slots_) {
        if (s.muon) {
            out.emplace_back("muon." + s.name, s.buf);
        } else {
            out.emplace_back("adam_m." + s.name, s.m);
            out.emplace_back("adam_v." + s.name, s.v);
        }
    }
    return out;
}

bool DrafterOptimizer::restore(const std::map<std::string, torch::Tensor>& masters, const std::map<std::string, torch::Tensor>& buffers,
                               int64_t steps, int64_t adam_steps, std::string& err) {
    if (steps < 0 || adam_steps < 0 || adam_steps > steps) { err = "optimizer state has step counters out of range"; return false; }
    // every tensor first, so a bad file changes nothing
    const auto find = [&](const std::map<std::string, torch::Tensor>& from, const std::string& name, const torch::Tensor& like, torch::Tensor& out) {
        const auto it = from.find(name);
        if (it == from.end() || !it->second.defined() || !(it->second.sizes() == like.sizes()) || !it->second.is_floating_point()) {
            err = "optimizer state has no tensor " + name + " of the right shape";
            return false;
        }
        out = it->second;
        return true;
    };
    struct Loaded { torch::Tensor master, a, b; };
    std::vector<Loaded> loaded(slots_.size());
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot& s = slots_[i];
        if (!find(masters, s.name, s.master, loaded[i].master)) return false;
        if (s.muon ? !find(buffers, "muon." + s.name, s.buf, loaded[i].a)
                   : !(find(buffers, "adam_m." + s.name, s.m, loaded[i].a) && find(buffers, "adam_v." + s.name, s.v, loaded[i].b)))
            return false;
    }
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot& s = slots_[i];
        s.master.copy_(loaded[i].master);
        if (!s.shared) s.param.copy_(s.master);
        if (s.muon) {
            s.buf.copy_(loaded[i].a);
        } else {
            s.m.copy_(loaded[i].a);
            s.v.copy_(loaded[i].b);
        }
        s.sum.zero_();
    }
    step_ = steps;
    adam_t_ = adam_steps;
    weight_ = 0;
    return true;
}

}  // namespace basalt::dfly2
