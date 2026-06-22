#pragma once

// heartbeat.hpp — a wall-clock interval gate for periodic side-effects (progress logs, safety
// checkpoints) decoupled from STEP count.
//
// Step-denominated cadence misreports on variable-throughput runs: "every 200 steps" is a torrent at
// 120 steps/s and silence at 1 step/s, and it couples crash-recovery granularity to the (deliberately
// rare) eval schedule. A time gate fixes both — a log every ~30 s reads the same whether fast or slow,
// and a safety checkpoint every ~few minutes bounds crash loss regardless of the half-epoch eval cadence.
// `due()` returns the elapsed seconds since it last fired (0 = not yet), so the caller can also compute
// the INSTANTANEOUS rate over the interval (Δsteps / Δt) instead of a misleading cumulative average.

#include <chrono>

namespace sub0diff::util {

class Heartbeat {
public:
    explicit Heartbeat(double period_seconds) noexcept
        : period_(period_seconds), last_(clock::now()) {}

    // If at least `period` has elapsed since the last fire: reset the timer and return the elapsed
    // seconds (always > 0). Otherwise return 0.0 (not yet due).
    [[nodiscard]] double due() noexcept {
        const auto now = clock::now();
        const double dt = std::chrono::duration<double>(now - last_).count();
        if (dt < period_) return 0.0;
        last_ = now;
        return dt;
    }

private:
    using clock = std::chrono::steady_clock;
    double            period_;
    clock::time_point last_;
};

}  // namespace sub0diff::util
