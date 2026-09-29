// sub0/stage_probe.hpp -- the no-op stage probe the engine-free *_math.hpp headers default to.
//
// WHY. gdn::forward and gr::mix run several distinct kinds of work (projection GEMVs, conv, recurrence,
// norms) inside one call, and the decode phase profiler (phase_profile.hpp) needs to charge them
// separately. Those headers are deliberately engine-free, so they cannot include the profiler. Instead
// each takes a trailing `Probe` (a template parameter defaulting to this type) and calls
// `probe(Stage::X)` at the START of each stage. NoStageProbe's call operator is empty and constexpr, so
// a default build (and every non-decode caller) compiles the marks away entirely.
//
// A probe marks stage STARTS only: a stage runs until the next mark, and the caller's own scope
// restores the surrounding phase on exit. See prof::StageProbe for the profiling implementation.

#pragma once

namespace sub0 {

/// Probe that ignores every stage mark; the default for math-header entry points.
struct NoStageProbe {
    template <class Stage>
    constexpr void operator()(Stage) const noexcept {}
};

}  // namespace sub0
