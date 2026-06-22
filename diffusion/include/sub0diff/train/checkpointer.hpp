#pragma once

// checkpointer.hpp — the reusable training-loop CHECKS in ONE place, so every trainer inherits the same
// discipline instead of reimplementing it (and drifting):
//   • the COVERAGE-RULE eval cadence (from make_schedule — never eval more often than ~½ epoch, else
//     consecutive evals differ only by noise and early-stop trips on noise),
//   • EARLY-STOP (best metric + patience, with a min-improvement deadband),
//   • the full HONEST-RESUME set: weights (step_*.ckpt) + Adam moments (step_*.opt) + dynamic progress
//     (train_state.json) — so a crash or a continuation resumes mid-climb, not from scratch.
//
// The model's device-order constraint stays the caller's (load weights on CPU → move to device → build
// the optimizer → restore), because only the caller knows when the model moves. This class owns the I/O
// and the rules. simdjson stays out of this header (the JSON read lives in checkpointer.cpp).
//
// Usage:
//   Checkpointer ck(sched, ckpt_dir, code_sha, config_sha, patience);
//   std::uint64_t start = ck.load_weights(param_ptrs);          // model on CPU
//   model.to(dev);  auto opt = make_optimizer(...);
//   ck.restore(*opt);                                           // Adam moments + progress history
//   for (auto s = start; s < ck.steps_bound(); ++s) {
//       train_step();
//       if (ck.due(s + 1) && ck.record(s + 1, eval_metric(ck.eval_windows()), param_ptrs, *opt)) break;
//   }

#include "sub0diff/train/schedule.hpp"

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace sub0diff::train {

// Dynamic training progress — the honest-resume state, persisted as train_state.json beside the weights.
struct Progress {
    bool          have      = false;   // a matching sidecar was loaded (step validated against the ckpt)
    std::uint64_t step      = 0;
    double        best      = std::numeric_limits<double>::max();  // best held-out metric (lower is better)
    std::uint64_t stalls    = 0;       // evals since `best` last improved
    std::uint64_t best_step = 0;       // the step whose checkpoint achieved `best` (the early-stop winner)
};

// The best (early-stop-winning) step recorded in `ckpt_dir/train_state.json`, or -1 if absent — so a
// loader can serve the BEST checkpoint, not merely the latest (which, after early-stop, is past the best).
[[nodiscard]] std::int64_t best_checkpoint_step(const std::string& ckpt_dir);

class Checkpointer {
public:
    // `sched` from make_schedule supplies the eval cadence + averaged eval-sample size + steps bound.
    // code_sha/config_sha tag the produced state (provenance). patience=0 disables early-stop.
    Checkpointer(Schedule sched, std::string ckpt_dir, std::string code_sha,
                 std::uint64_t config_sha, std::uint64_t patience, double min_improve = 0.01);

    // schedule-derived knobs the loop needs.
    [[nodiscard]] bool          due(std::uint64_t step) const noexcept;     // an eval/checkpoint is due
    [[nodiscard]] std::uint64_t steps_bound() const noexcept;               // safety / max steps
    [[nodiscard]] std::size_t   eval_windows() const noexcept;              // averaged eval-sample size

    // Resume step 1 (model still on CPU): load the latest weights into `params`. Returns the resumed step
    // (0 = fresh run, nothing loaded). Records which checkpoint we resumed from for restore().
    std::uint64_t load_weights(std::span<sub0llm::autograd::Variable* const> params);

    // Resume step 2 (after model.to(device) + optimizer built): restore Adam moments from the matching
    // step_*.opt and rehydrate the progress history (best/stalls) from train_state.json. No-op if fresh.
    void restore(sub0llm::nn::Optimizer& opt);

    // Record a held-out metric (LOWER IS BETTER) at `step`: save weights + opt + train_state.json,
    // update best/stalls, and return true iff early-stop should fire (stalls >= patience). The eval
    // cadence gate is the caller's (call only when due()).
    bool record(std::uint64_t step, double metric,
                std::span<sub0llm::autograd::Variable* const> params, sub0llm::nn::Optimizer& opt);

    // Time-based SAFETY checkpoint between evals: save weights + opt + train_state.json at `step`
    // WITHOUT running eval or touching best/stalls — pure crash insurance, since the coverage-rule eval
    // cadence is deliberately rare (≥½ epoch) and a crash mid-epoch would otherwise lose hours. Rolls a
    // SINGLE safety file (deletes the previous safety checkpoint, never the best/eval ones) so disk
    // doesn't bloat. Drive it from a wall-clock util::Heartbeat, independent of due().
    void save_safety(std::uint64_t step,
                     std::span<sub0llm::autograd::Variable* const> params, sub0llm::nn::Optimizer& opt);

    [[nodiscard]] const Progress&    progress() const noexcept { return prog_; }
    [[nodiscard]] const std::string& resumed_from() const noexcept { return resume_path_; }

private:
    Schedule      sched_;
    std::string   dir_, code_sha_, resume_path_;
    std::uint64_t config_sha_, patience_;
    double        min_improve_;
    Progress      prog_;
    std::string   safety_path_;        // last rolling safety .ckpt (deleted when the next one lands)
    std::uint64_t safety_step_ = 0;
};

}  // namespace sub0diff::train
