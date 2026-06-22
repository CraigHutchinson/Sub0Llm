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
    f << std::format(
        "{{\n  \"step\": {},\n  \"best_nelbo\": {:.9g},\n  \"best_step\": {},\n"
        "  \"evals_since_best\": {},\n  \"code_sha\": \"{}\",\n  \"config_sha\": \"{:016x}\",\n"
        "  \"updated_unix\": {}\n}}\n",
        p.step, p.best, p.best_step, p.stalls, code_sha, config_sha,
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
    }
    p.have = (p.step == expect_step);
    return p;
}

}  // namespace

std::int64_t best_checkpoint_step(const std::string& ckpt_dir) {
    const Progress p = read_progress(fs::path(ckpt_dir) / "train_state.json", /*expect=*/0);
    return p.best_step > 0 ? static_cast<std::int64_t>(p.best_step) : -1;
}

Checkpointer::Checkpointer(Schedule sched, std::string ckpt_dir, std::string code_sha,
                           std::uint64_t config_sha, std::uint64_t patience, double min_improve)
    : sched_(sched), dir_(std::move(ckpt_dir)), code_sha_(std::move(code_sha)),
      config_sha_(config_sha), patience_(patience), min_improve_(min_improve) {}

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
    if (metric < prog_.best - min_improve_) { prog_.best = metric; prog_.stalls = 0; prog_.best_step = step; }
    else ++prog_.stalls;
    prog_.step = step;
    auto view = snapshot(params);   // current live params (on device) — save_checkpoint D2H's them
    sub0llm::save_checkpoint(view, dir_, static_cast<std::int64_t>(step));
    const std::string ck = sub0llm::latest_checkpoint_path(dir_);
    if (!ck.empty()) { fs::path op = ck; op.replace_extension(".opt"); opt.save_state(op.string()); }
    write_progress(fs::path(dir_) / "train_state.json", prog_, code_sha_, config_sha_);
    return patience_ != 0 && prog_.stalls >= patience_;
}

}  // namespace sub0diff::train
