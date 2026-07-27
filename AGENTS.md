# Agent instructions for Sub0Llm

This file is a pre-flight checklist, not a tutorial. It exists because several rules below were
learned the hard way — a real near-miss or a real bug caught mid-implementation — and the goal is to
apply them BEFORE writing code, not discover them after. Every rule cites the concrete case that
established it, so a future edge case can be judged against the actual reasoning, not just the letter
of the rule.

If you're an agent (or a human) about to add a new op, optimizer, config knob, or file format change
to this engine: read this first.

## 1. No heap allocation in any per-step or per-call hot path

Every model dimension is a compile-time constant, so the engine's forward/backward/training-loop
allocates NOTHING at runtime: parameters, activations, and gradients all live in fixed-size arenas
sized once at startup (`backend_cpu.cpp`'s `ACT_CAP`/`MAX_NODES`, `g_param_*`). This is deliberate,
not incidental — it is why training throughput here is competitive.

**Negative example, caught 2026-07 (not before shipping):** `sub0::muon::newton_schulz5`
(`include/sub0/muon.hpp`) allocates fresh `std::vector`s on every call — it runs once per Muon-eligible
weight matrix, every optimizer step. Measured cost: ~11x slower than AdamW at d128 on a real corpus,
most of which is very likely this (unvectorized loops + repeated heap churn), not an inherent property
of the algorithm. This should have been caught during design, not after an A/B run exposed it.

**Rule**: any code that runs more than "once, at startup/configure time" — every training step, every
optimizer step, every generated token, every window sampled — must use pre-sized, reused scratch
storage (a static/thread_local buffer, an arena slot, a caller-owned buffer passed in) instead of
`std::vector`/`new`/`malloc` inside the call. If you can't size the scratch buffer at compile time
(dimensions vary at runtime), size it once outside the hot loop and reuse it across calls, not once per
call. If you're not sure whether a path counts as "hot enough" to matter, size it anyway — the cost of
being wrong the other way (a silent 10x regression discovered later by A/B, like above) is much higher
than the cost of a little extra scratch-buffer bookkeeping.

## 2. Bake every decision that's known upfront as `constexpr`/`if constexpr` — never a runtime branch

Model dims, precision, positional encoding, ternary weights, gated FFN — all baked by
`sub0llm-configure` into the generated header and dispatched with `if constexpr`, so the unused branch
compiles away entirely. This is the whole point of the staged configure→build architecture (see
`docs/WORKFLOW_ARCHITECTURE.md`).

**Rule**: a new capability that the user decides ONCE (not per-step, not something that needs to
change without a rebuild) should be a new configurator CLI flag emitting a `constexpr bool`/enum into
the generated header, gated with `if constexpr` — follow the `USE_TERNARY`/`USE_GATED_FFN`/
`POS_ENCODING` pattern exactly, don't invent a new mechanism.

**The one accepted exception, and how to take it**: a runtime flag is fine ONLY as a *temporary*
dev-iteration convenience (e.g. `--optimizer adamw|muon` — switching optimizers needs no rebuild
because it doesn't touch `PARAM_LAYOUT`/checkpoint shape, unlike a dims/architecture choice). If you
take this exception, you MUST leave an explicit `TODO(...)` comment at the runtime branch AND a memory
note explaining what compile-time form it should eventually take and what evidence would justify
converting it. Do not leave an undocumented runtime knob "temporarily" with no trail back to the
intended end state.

## 3. Checkpoint / binary-format changes are the highest-blast-radius category of change here

Model weight files and `.ckpt` optimizer state are fixed-size binary structs written directly to disk
(`engine_core.cpp`'s `Header`, `train_stage.cpp`'s `save_checkpoint`/`load_checkpoint`). ANY change to
field order, field count, or struct size makes every existing checkpoint fail to load — for a project
that runs multi-day training jobs, that can mean losing real, expensive, in-progress work.

**Rule, in order:**
1. Before adding a new field to detect a new kind of incompatibility, check whether an EXISTING field
   already discriminates it. (Gated-FFN needed no new `Header`/`.ckpt` field: `param_floats`/
   `trainable_floats()` already differs between a gated and non-gated build at identical dims, for
   this project's fixed `D_FF=4*D_MODEL` convention — confirmed by working through the actual
   arithmetic, not assumed.)
2. If a new field is genuinely unavoidable, prefer an ADDITIVE, gracefully-degrading format (the
   trailing optional tokenizer-fingerprint pattern in `save_model`/`load_model`: append at the end,
   and treat a short/missing read as "unknown, no guard" rather than a hard parse failure) over
   reshuffling existing fields.
3. Never make this call by assumption. Trace the actual byte layout and what happens when an
   old file meets new code before deciding a change is safe.

## 4. New capability = off by default, zero effect on existing builds until explicitly enabled

Every addition this project has shipped (ternary weights, gated FFN, Muon) defaults OFF/AdamW and is
verified to leave the default build's test suite byte-identical before being trusted. Concretely: after
adding a new op or optimizer path, rebuild the UNMODIFIED default configuration and confirm the full
test suite's pass/fail counts match exactly what they were before your change (not just "no new
failures" — check the actual assertion counts). If they don't match exactly, something leaked into the
default path that shouldn't have.

## 5. Verify precise algorithms against their actual reference source — never from recall alone

Newton-Schulz's exact coefficients, the momentum-EMA formula, and the Nesterov default were fetched
from the real Muon reference implementation before writing a single line, specifically because a
plausible-sounding recalled version would have been subtly wrong (momentum as a raw accumulation
instead of an EMA; the reference's `nesterov=True` default is easy to miss since it isn't exposed as a
class parameter). Separately, porting the reference's scale formula naively (`rows/cols`) would have
been silently backwards here, because this project's weight-matrix convention (`[rows=in,cols=out]`)
is the opposite axis order from the PyTorch code the reference is written against.

**Rule**: for any algorithm sourced from external research/a reference implementation, fetch and
quote the actual source (WebFetch the real code/paper, don't paraphrase from training data) before
implementing. Then explicitly re-derive how the reference's conventions (axis order, tensor layout,
sign conventions) map onto THIS project's own conventions — don't assume they match.

## 6. Every new op/optimizer needs a correctness check before it's "done" — not just "it compiles"

The standing pattern (`gelu_fast`/`dgelu_fast` kept mutually consistent so the FAST_MATH gradient
check still passes; `op_swiglu`'s backward verified via the engine's own finite-difference gradient
check at a real training config; `newton_schulz5`'s core mathematical property — off-diagonal collapse,
bounded diagonal — verified numerically before it was ever wired into the optimizer). See also
`verify-correctness-against-reference-before-perf` in memory: llama.cpp is the oracle for engine-level
correctness the same way a reference implementation is the oracle for a ported algorithm.

**Rule**: new forward math needs a numerical property check (gradient-check via finite differences,
or an independent from-scratch cross-check of the defining mathematical property) BEFORE it's
considered correct — passing compilation and "the loss goes down" are not sufficient on their own.

## 7. Test at more than one scale — bugs and behavior can be dims-dependent

LoopSplit's head/tail symmetry `static_assert` rejected every ODD-layer model even with the feature
switched OFF (`N_LAYERS - 0` is odd). It was invisible on the 8-layer toy config and failed instantly on
the 11-layer one — same code, different scale, different outcome. A correctness check that only ever
runs at one scale can miss a real, scale-dependent problem entirely.

**Rule**: when verifying new engine-level math, run the check at more than one model scale where
practical — at minimum a tiny fast-iteration config AND something closer to a real production config,
not just whichever is most convenient to compile.

**The tiny config must still be REALISTIC in every axis you are not deliberately varying — vocabulary
especially.** `engine_tests.cpp`'s gradient check reads ~3.8% low on a 493-token toy vocabulary and is
exact (0.9999+) at vocab 16535, at *every* size from d96 L8 H2 up to d448 L11 H7. That artefact was
misread for three weeks as "the gradient check fails at production dims", which sent a real investigation
in exactly the wrong direction and nearly triggered a needless redesign of the test. Prefer d96 L8 H2 at
a real vocabulary as the fast scale over a spike config; `factspike96` (vocab 493) must not gate
numerical gradient work. See memory `engine-tests-gradient-check-dims-dependent` for the measurements.

## 8. Only add the surface area actually consumed

Don't expose a CLI flag, CMake option, or config knob that nothing reads yet. (`SUB0_VOCAB`/
`SUB0_MIN_MERGE` were added speculatively once and reverted.) If you're building toward a feature in
stages, land the stage that's actually wired up; don't pre-add the interface for a later stage.

## 9. Validate against something real before calling an import/interop feature done

The GGUF reader was validated against a REAL downloaded model file (not just handcrafted fixtures) —
tensor names/shapes/dequantized values cross-checked against the real model's known HuggingFace config
and statistically sane weight distributions. Synthetic fixtures prove the parser logic; they don't
prove real-world byte-format fidelity. The same principle applied to the Muon A/B: a real corpus
(`data/tinystories.txt`), not just a toy file, before trusting the throughput/convergence numbers.

## 10. Changing a shared surface? Enumerate its CONSUMERS — the diff does not show you them

Three defects in the GQA/LoopSplit cycle were invisible in the diff and invisible to green test suites,
and all three were found the same way: by grepping for *consumers* of a surface the diff changed, not by
re-reading the diff.

- KV-cache rows narrowed from `D_MODEL` to `D_KV`. The engine side was updated; `tests/factspike_engine_tests.cpp`
  still copied `D_MODEL` floats out of them — an out-of-bounds read on every row, in the default test target.
  The `core.hpp` doc comment had already been updated to say `D_KV`; the callers of that doc were not.
- `memplan::Dims` gained two fields. Five sites aggregate-initialised it positionally; three were missed
  across two separate commits, silently mis-predicting VRAM and the *recommended* batch size.
- `LOOP_MIDDLE_LAYERS`/`LOOP_REPEATS`, and later `ROPE_THETA`, changed what the model computes without
  changing any tensor shape — so no `Header` field and not `PARAM_FLOATS` could discriminate them, and a
  checkpoint loaded silently into the wrong architecture.

**Rule**: when a change alters a shared *width*, *semantic*, *serialized format*, or *config axis*, list
every consumer repo-wide (`grep -rn <symbol> --include=*.cpp --include=*.hpp --include=*.cu`) and check
each one, including tests and tools. Then:

1. **Run the full suite with the feature ON at least once, unfiltered.** The GQA runs were CUDA-only and
   tag-filtered (`[cuda]`), so the file carrying the out-of-bounds read never executed under GQA at all.
   A filtered run is not evidence about the files it filtered out.
2. **Prove variants actually differ by a DERIVED quantity, not a banner.** All three arms of a `--kv-heads`
   A/B printed byte-identical config lines; only the parameter counts showed they were different builds.
3. **For a new compile-time axis, classify it**: shape-changing (`PARAM_FLOATS` discriminates it),
   computation-changing but shape-neutral (must join `ARCH_FINGERPRINT` in `layout.hpp`, or it loads
   silently and computes the wrong thing), or deliberately variable between train and inference (exclude,
   and record why). See `ARCH_FINGERPRINT`'s own comment.

Prefer removing the class over fixing the instance: `current_build_dims()` exists because five hand-listed
`Dims` initialisations produced three bugs, and `registry::describe_config()` exists because three
hand-listed config banners each omitted a different axis.

## 11. Spikes are temporary — retire them from the build once their finding has merged

A spike exists to answer a question. Once answered, the ANSWER belongs in the mainline and the spike's
scaffolding does not: it keeps costing build time on every configuration, keeps having to compile
through unrelated refactors, and keeps inviting the mistake of treating an exploratory toy config as a
correctness gate. A spike that is PARKED rather than successful does not need to keep being verified
at all.

**Objective membership test — so this is not re-litigated per spike**: a spike's tests may be gated once
NOTHING in `src/` or `tools/` includes its header. That is the mechanical signal that the finding was
merged and only scaffolding remains.

- `factspike` qualified: its finding (marker-inclusive span replay) had merged into `corpus_collapse.hpp`,
  which cites it, and nothing in `src/`/`tools/` includes `factspike.hpp`. Retired behind
  `SUB0_BUILD_SPIKE_TESTS` (OFF by default).
- `scratchspike` and `corpus_collapse` did NOT qualify: `train_stage.cpp` consumes both as blend sources,
  so their tests cover a production path and stay in the default build.

**Gate, don't delete.** Reviving parked work should be one flag, not archaeology. Verify the gate is a
true round trip — turning it back ON must restore the exact prior assertion counts, proving it changed
nothing but inclusion. And handle the fallout rather than leaving it dangling: any engine API whose only
consumer was the gated spike should say so at its declaration (keep it if it is the substrate that work
would resume from), and any recorded baseline that cited the spike's config must be re-recorded.

## Before you ship — quick checklist

- [ ] Any new per-step/per-call code path: zero heap allocation, scratch reused not reallocated (§1)
- [ ] Any new user-facing decision: baked constexpr via the configurator, `if constexpr`-gated, unless
      explicitly justified + TODO-flagged as a temporary runtime exception (§2)
- [ ] Any change touching `Header`/`.ckpt`/on-disk layout: traced byte-for-byte, existing-checkpoint
      compatibility explicitly reasoned through, not assumed (§3)
- [ ] New capability defaults off; default-build test suite assertion counts confirmed IDENTICAL
      before/after (§4)
- [ ] Any ported algorithm: fetched from its real source, axis/layout conventions explicitly re-mapped
      to this project's own (§5)
- [ ] New math has a numerical correctness check independent of "it compiles and the loss drops" (§6)
- [ ] Correctness/perf claims checked at more than one model scale (§7)
- [ ] No speculative, unconsumed CLI/config surface added (§8)
- [ ] Import/interop features validated against a real external file/corpus, not fixtures alone (§9)
- [ ] Any changed shared width/semantic/format/config axis: consumers enumerated repo-wide, full suite
      run unfiltered with the feature ON, new compile-time axis classified against `ARCH_FINGERPRINT` (§10)
- [ ] Spike whose finding has merged (no `src/`/`tools/` include of its header): gated out of the default
      build, gate verified as a round trip, orphaned API + baselines updated (§11)
