// sub0/knob.hpp — a performance knob that is a compile-time constant in production but
// runtime-mutable while tuning.
//
// Many performance choices (CUDA TF32 math, block sizes, fusion toggles, ...) want a RUNTIME
// control while the autotuner sweeps them, but once a value is chosen it should become a
// COMPILE-TIME constant so the hot path pays nothing: `if (knob.get())` then folds away
// (dead-branch elimination) exactly like `if constexpr`. This one type expresses both, picked
// by the SUB0_TUNING build flag:
//
//   * production (SUB0_TUNING undefined): get() is constexpr (the baked default) and set() is
//     a no-op, so the optimizer removes the untaken branch in hot loops -- zero overhead.
//   * tuning     (SUB0_TUNING defined):   get() is a runtime read and set() mutates it, so the
//     autotuner can sweep the value; the runtime check is acceptable only while tuning.
//
// The baked default is normally a constant the configurator emits into sub0_config.hpp from
// the persisted tune cache (exactly like DEFAULT_THREADS), so the workflow
// "tune -> bake -> rebuild" promotes the chosen value to compile-time with NO source edit.
//
// Usage (the consistent pattern for every such knob):
//   using Tf32 = sub0::Knob<bool, CUDA_TF32>;   // CUDA_TF32 is baked in the generated config
//   if (Tf32::get()) { ... } else { ... }       // folds in production; runtime while tuning
//   Tf32::set(true);                            // sweep value (a no-op in a production build)

#pragma once

namespace sub0 {

// T: the knob's value type. BakedDefault: the configured/tuned value compiled in for production.
template <class T, T BakedDefault>
struct Knob {
#if defined(SUB0_TUNING)
    static inline T v = BakedDefault;            // runtime state the autotuner sweeps
    static T              get()    { return v; }
    static void           set(T nv) { v = nv; }
    static constexpr bool tunable = true;
#else
    static constexpr T    get()    { return BakedDefault; }  // baked: folds away in hot loops
    static constexpr void set(T)   {}                        // no-op in production
    static constexpr bool tunable = false;
#endif
};

}  // namespace sub0
