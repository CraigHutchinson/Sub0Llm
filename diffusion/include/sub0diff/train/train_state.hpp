#pragma once

// train_state.hpp — the dynamic-progress sidecar (train_state.txt) for an HONEST resume.
//
// Weights live in step_*.ckpt and Adam's moments in step_*.opt; this tiny key=value text
// file holds the REST of the run state — the curriculum position and the early-stop history
// — so a resume continues mid-climb instead of restarting both. It is the evolving per-step
// counterpart to the STATIC run configuration (run_config.json, sub0diff/config): config is
// what you launched with, this is where the run currently stands. `step` is matched against
// the resumed checkpoint so a stale sidecar is ignored. Reusable across training chapters.

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <string>

namespace sub0diff::train::trainstate {

struct State {
    bool          have = false;
    std::uint64_t step = 0;
    std::int64_t  curr_k = 1;
    float         curr_best = std::numeric_limits<float>::max();
    int           curr_stalls = 0;
    bool          curr_converged = false;
    float         best_nelbo = std::numeric_limits<float>::max();
    std::uint64_t evals_since_best = 0;
};

inline void save(const std::filesystem::path& path, const State& s) {
    std::ofstream f(path);
    if (!f) return;
    f << std::format("step={}\ncurriculum_k={}\ncurriculum_best={:.9g}\ncurriculum_stalls={}\n"
                     "curriculum_converged={}\nbest_nelbo={:.9g}\nevals_since_best={}\n",
                     s.step, s.curr_k, s.curr_best, s.curr_stalls,
                     s.curr_converged ? 1 : 0, s.best_nelbo, s.evals_since_best);
}

// Load and validate against the resumed step; returns have=false on any mismatch/absence.
[[nodiscard]] inline State load(const std::filesystem::path& path, std::uint64_t expect_step) {
    State s;
    std::ifstream f(path);
    if (!f) return s;
    std::string line;
    while (std::getline(f, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        try {
            if      (k == "step")                  s.step = std::stoull(v);
            else if (k == "curriculum_k")          s.curr_k = std::stoll(v);
            else if (k == "curriculum_best")       s.curr_best = std::stof(v);
            else if (k == "curriculum_stalls")     s.curr_stalls = std::stoi(v);
            else if (k == "curriculum_converged")  s.curr_converged = (v != "0");
            else if (k == "best_nelbo")            s.best_nelbo = std::stof(v);
            else if (k == "evals_since_best")      s.evals_since_best = std::stoull(v);
        } catch (...) { return State{}; }   // malformed → treat as absent
    }
    s.have = (s.step == expect_step);       // only trust a sidecar matching the resumed ckpt
    return s;
}

}  // namespace sub0diff::train::trainstate
