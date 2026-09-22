// sub0/phase_profile.hpp -- compile-time-gated EXCLUSIVE-time phase profiler for decode (forward_one).
//
// WHY THIS EXISTS. docs/OPTIMIZATION_PROCESS.md S5 requires re-deriving the O0/O1 profile after any
// change over ~20%. Until now that meant hand-writing a timing scaffold into decode.cpp (B27's atomics
// + destructor printer; again for the post-B35 profile) and reverting it afterwards -- boilerplate
// every time, and a profile nobody else could reproduce. This makes it a build axis: `sub0llm-configure
// --profile-phases 1` emits PROFILE_PHASES = true, and a default build compiles the scopes away to
// nothing (PhaseScope<false> is an empty type), so the default decode is unchanged byte for byte.
//
// EXCLUSIVE, NOT INCLUSIVE. Phases nest in decode: Gated Residual's write step runs INSIDE each mixer
// lambda, so inclusive timers would count it twice and the shares would sum past 100%. Entering a
// scope charges the time elapsed so far to the phase that was running and switches to the new one;
// leaving charges the scope and switches back. Every nanosecond is therefore charged to exactly one
// phase, and whatever no scope covers (embedding, final norm, loop overhead) lands in Phase::Other --
// the "unattributed" line B27's scaffold reported, which is the check that attribution is complete.
//
// ONE accumulator, owned by the engine library (sub0::phase_accumulator(), core.hpp). Not a header-local
// static: sub0_core is a DLL on Windows, where a function-local static in an inline header function gets
// a separate copy per module -- the tool would read its own empty counters while decode filled the DLL's.
//
// Single-threaded by design: every scope is opened on the thread running forward_one, never inside an
// OpenMP team (the routed-expert scope wraps ParallelExperts' whole team from the calling thread), so the
// counters are plain integers. A scope opened from a worker thread would race; none is.

#pragma once

#include "sub0/core.hpp"   // phase_accumulator(): the ONE accumulator, owned by the engine library

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace sub0::prof {

/// Decode's O1-level phases (docs/optimization/profile_post_b35.md's rows). Other = uncovered time.
/// MoeRouted nests inside Moe (one scope around the routed-expert team), so Moe is what is LEFT:
/// router, activation quantize, shared expert and the weighted combine.
/// Mixer is the full-attention mixer (QSA, or softmax attention); MixerGdn is the Gated DeltaNet mixer.
enum class Phase : std::uint8_t { Other, Mixer, MixerGdn, Moe, MoeRouted, GatedResidual, LmHead, Count };

inline constexpr std::array<std::string_view, static_cast<std::size_t>(Phase::Count)> kPhaseNames{
    "unattributed", "mixer: QSA/attention", "mixer: GDN", "MoE: router + shared + combine",
    "MoE: routed experts", "Gated Residual", "lm_head (+ final norm)"};

/** Per-phase accumulated nanoseconds and the phase currently being charged.
 *
 * @note One instance per process (decode is single-threaded at the phase level, see file header).
 *       reset() before the region to measure; read `ns` after.
 */
struct Accumulator {
    std::array<std::uint64_t, static_cast<std::size_t>(Phase::Count)> ns{};
    Phase current = Phase::Other;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();

    /// Charge the time since the last switch to the running phase, then make `next` the running one.
    /// @return the phase that was running, so a scope can restore it.
    Phase switch_to(Phase next) noexcept {
        const auto now = std::chrono::steady_clock::now();
        ns[static_cast<std::size_t>(current)] += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - since).count());
        since = now;
        const Phase prev = current;
        current = next;
        return prev;
    }
    void reset() noexcept {
        ns = {};
        current = Phase::Other;
        since = std::chrono::steady_clock::now();
    }
};

/// RAII exclusive-time scope. PhaseScope<false> is empty and compiles away entirely.
template <bool Enabled>
class PhaseScope {
public:
    explicit PhaseScope(Phase) noexcept {}
};

template <>
class PhaseScope<true> {
public:
    explicit PhaseScope(Phase p) noexcept : prev_(phase_accumulator().switch_to(p)) {}
    ~PhaseScope() { phase_accumulator().switch_to(prev_); }
    PhaseScope(const PhaseScope&) = delete;
    PhaseScope& operator=(const PhaseScope&) = delete;

private:
    Phase prev_;
};

}  // namespace sub0::prof
