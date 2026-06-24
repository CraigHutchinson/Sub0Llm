#include "sub0diff/train/checkpointer.hpp"

#include "sub0llm/nn/checkpoint.hpp"

#include <simdjson.h>   // train_state.json read (forward on-demand) — kept in this TU only

#include <cstdint>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>

namespace sub0diff::train {

namespace fs = std::filesystem;
using sub0llm::autograd::Variable;

namespace {

// Snapshot the live param Variables (copies sharing storage AT CALL TIME). Re-taken per save because
// Variable::to() swaps in a new storage tensor — a snapshot from before a device move would be stale.
std::vector<Variable> snapshot(std::span<Variable* const> ptrs) {
    std::vector<Variable> v;
    v.reserve(ptrs.size());
    for (auto* p : ptrs) v.push_back(*p);
    return v;
}

void write_progress(const fs::path& path, const Progress& p, std::string_view code_sha,
                    std::uint64_t config_sha) {
    std::ofstream f(path);
    if (!f) return;
    std::string recent = "[";
    for (std::size_t i = 0; i < p.recent.size(); ++i)
        recent += std::format("{}{:.9g}", i ? ", " : "", p.recent[i]);
    recent += "]";
    f << std::format(
        "{{\n  \"step\": {},\n  \"best_nelbo\": {:.9g},\n  \"best_step\": {},\n"
        "  \"evals_since_best\": {},\n  \"recent\": {},\n  \"code_sha\": \"{}\",\n"
        "  \"config_sha\": \"{:016x}\",\n  \"updated_unix\": {}\n}}\n",
        p.step, p.best, p.best_step, p.stalls, recent, code_sha, config_sha,
        static_cast<std::int64_t>(std::time(nullptr)));
}

// Read + validate against the resumed step; have=false on absence/mismatch (a stale sidecar is ignored).
Progress read_progress(const fs::path& path, std::uint64_t expect_step) {
    Progress p;
    std::ifstream f(path, std::ios::binary);
    if (!f) return p;
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (body.empty()) return p;
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(body.data(), body.size());
    auto doc = parser.iterate(padded);
    if (doc.error()) return p;
    auto obj = doc.get_object();
    if (obj.error()) return p;
    for (auto field : obj) {
        auto k = field.unescaped_key(); if (k.error()) continue;
        auto v = field.value();         if (v.error()) continue;
        const std::string_view key = k.value_unsafe();
        if      (key == "step")             { std::uint64_t x; if (!v.get(x)) p.step = x; }
        else if (key == "best_nelbo")       { double x;        if (!v.get(x)) p.best = x; }
        else if (key == "best_step")        { std::uint64_t x; if (!v.get(x)) p.best_step = x; }
        else if (key == "evals_since_best") { std::uint64_t x; if (!v.get(x)) p.stalls = x; }
        else if (key == "recent") {
            if (auto arr = v.get_array(); !arr.error())
                for (auto e : arr.value_unsafe()) { double x; if (!e.get(x)) p.recent.push_back(x); }
        }
    }
    p.have = (p.step == expect_step);
    return p;
}

}  // namespace

double trend_slope(std::span<const double> ys) {
    const std::size_t n = ys.size();
    if (n < 2) return -1.0;   // <2 points → "still descending", never a plateau
    const double xm = static_cast<double>(n - 1) / 2.0;
    double ym = 0.0; for (double y : ys) ym += y; ym /= static_cast<double>(n);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = static_cast<double>(i) - xm;
        num += dx * (ys[i] - ym);
        den += dx * dx;
    }
    return den > 0.0 ? num / den : 0.0;
}

std::int64_t best_checkpoint_step(const std::string& ckpt_dir) {
    const Progress p = read_progress(fs::path(ckpt_dir) / "train_state.json", /*expect=*/0);
    return p.best_step > 0 ? static_cast<std::int64_t>(p.best_step) : -1;
}

Checkpointer::Checkpointer(Schedule sched, std::string ckpt_dir, std::string code_sha,
                           std::uint64_t config_sha, std::uint64_t plateau_window, double min_improve)
    : sched_(sched), dir_(std::move(ckpt_dir)), code_sha_(std::move(code_sha)),
      config_sha_(config_sha), plateau_window_(plateau_window), min_improve_(min_improve) {}

bool Checkpointer::due(std::uint64_t step) const noexcept {
    return sched_.eval_every != 0 && step % sched_.eval_every == 0;
}
std::uint64_t Checkpointer::steps_bound() const noexcept { return sched_.steps_bound; }
std::size_t   Checkpointer::eval_windows() const noexcept { return sched_.eval_nelbo_windows; }

std::uint64_t Checkpointer::load_weights(std::span<Variable* const> params) {
    resume_path_ = sub0llm::latest_checkpoint_path(dir_);
    if (resume_path_.empty()) return 0;
    auto view = snapshot(params);   // model on CPU here → view shares storage; load writes through
    const auto step = static_cast<std::uint64_t>(sub0llm::load_checkpoint(view, resume_path_));
    prog_.step = step;
    return step;
}

void Checkpointer::restore(sub0llm::nn::Optimizer& opt) {
    if (resume_path_.empty()) return;
    fs::path op = resume_path_; op.replace_extension(".opt");
    if (fs::exists(op)) (void)opt.load_state(op.string());
    const Progress p = read_progress(fs::path(dir_) / "train_state.json", prog_.step);
    if (p.have) prog_ = p;   // rehydrate best/stalls (honest early-stop continuation)
}

bool Checkpointer::record(std::uint64_t step, double metric,
                          std::span<Variable* const> params, sub0llm::nn::Optimizer& opt) {
    // best/best_step (with a min-improve debounce) drive WHICH checkpoint is served, not the stop.
    if (metric < prog_.best - min_improve_) { prog_.best = metric; prog_.stalls = 0; prog_.best_step = step; }
    else ++prog_.stalls;
    prog_.step = step;
    // Trend-line plateau detector: keep the last `plateau_window` metrics; stop once their fitted slope
    // is ≥ 0 (flat/rising held-out curve). The oscillating up/down deltas at a real plateau make the slope
    // cross 0 with no magnitude threshold needed.
    prog_.recent.push_back(metric);
    while (plateau_window_ != 0 && prog_.recent.size() > plateau_window_) prog_.recent.erase(prog_.recent.begin());
    const bool plateaued = plateau_window_ != 0 && prog_.recent.size() >= plateau_window_ &&
                           trend_slope(prog_.recent) >= 0.0;
    auto view = snapshot(params);   // current live params (on device) — save_checkpoint D2H's them
    sub0llm::save_checkpoint(view, dir_, static_cast<std::int64_t>(step));
    const std::string ck = sub0llm::latest_checkpoint_path(dir_);
    if (!ck.empty()) { fs::path op = ck; op.replace_extension(".opt"); opt.save_state(op.string()); }
    write_progress(fs::path(dir_) / "train_state.json", prog_, code_sha_, config_sha_);
    return plateaued;
}

void Checkpointer::save_safety(std::uint64_t step,
                               std::span<Variable* const> params, sub0llm::nn::Optimizer& opt) {
    auto view = snapshot(params);
    sub0llm::save_checkpoint(view, dir_, static_cast<std::int64_t>(step));
    const std::string ck = sub0llm::latest_checkpoint_path(dir_);   // the file we just wrote (highest step)
    if (!ck.empty()) { fs::path op = ck; op.replace_extension(".opt"); opt.save_state(op.string()); }
    prog_.step = step;   // best/best_step/stalls untouched — this is NOT an eval
    write_progress(fs::path(dir_) / "train_state.json", prog_, code_sha_, config_sha_);
    // Roll: delete the PREVIOUS safety checkpoint so only one extra file lingers — never the best (an
    // eval winner) nor a record() eval checkpoint (those self-prune later, separately).
    if (!safety_path_.empty() && safety_step_ != prog_.best_step) {
        std::error_code ec;
        fs::remove(safety_path_, ec);
        fs::path op = safety_path_; op.replace_extension(".opt"); fs::remove(op, ec);
    }
    safety_path_ = ck;
    safety_step_ = step;
}

}  // namespace sub0diff::train
