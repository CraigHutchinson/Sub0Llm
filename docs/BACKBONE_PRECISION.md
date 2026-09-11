# Backbone precision reduction — staged plan: BF16 first, native quant second

**Status: PLANNING. No code changed by this document.** Follows directly from
`docs/QWEN4_MEMORY_MAP.md` §7's dtype audit and §7d's correction (the backbone's real source model IS
quantized — `Q5_K`/`Q6_K`/`Q8_0` mixed per tensor role — and this project's own `sub0llm-transplant` tool
already fully dequantizes it to F32, once, offline, before `load_model` ever runs). User's own directive,
verbatim: "Lets make the BF16 approach first but we should support the model-native quantised data layout
as the next priority - its required for the model to actually fit in memory efficiently as bandwidth is
more vital for a large amount of the model than the compute side."

---

## 0. Why bandwidth, not just footprint — the argument, checked numerically

`QWEN4_MEMORY_MAP.md` §7c framed the BF16 lever mainly around §0's *residency ceiling* arithmetic (freeing
RAM for the sidecar). The user's point is a second, independent, and for decode probably **larger**
argument: decode is `SEQ_LEN` batch size **1** — one row through every weight, every token — which is the
textbook memory-bandwidth-bound shape (a GEMV's arithmetic intensity, ~2 FLOPs per 4 bytes read at F32, is
far below what this CPU's compute throughput could sustain if it weren't waiting on DRAM). So the backbone
isn't just *sitting* at 18.31 GiB — it is *read in full, from DRAM, every single token*, and every byte of
that read is on the critical path.

Estimated directly (weight-read term only, a lower bound — activation reads/writes and cache effects add
more, but the weight-read term dominates a GEMV):

| Backbone format | Size | Time to stream once, at 70 GB/s (typical dual-channel DDR5 mobile) |
|---|---:|---:|
| F32 (today) | 18.31 GiB | **~281 ms/token** |
| BF16 | 9.16 GiB | **~140 ms/token** |
| Native quant (~2.64–5.5 bit blend, format-dependent) | ~1.5–3.15 GiB | **~23–48 ms/token** |

Against today's measured ~5.5–5.9 s/token, 281 ms is ~5% — small *today*, because the MoE sidecar's own
disk-bound cost (B20/B21/B23) dwarfs it. **But that's exactly why this matters more, not less, as
Sub0MemPage's own work succeeds**: every millisecond B21/Sub0MemPage claw back from the sidecar's own
disk-bound cost makes the backbone's *own* 281 ms a larger fraction of what's left. This lever and
Sub0MemPage's are not competing for the same milliseconds — solving one makes the other more valuable, not
less.

### 0a. The prefill/decode arithmetic-intensity caveat — checked, not assumed

Raised directly, mid-plan: is this only true for decode, or does it help/hurt a compute-bound batched
prefill differently? **Checked against this gen tool's own real behavior, not assumed:**
`sub0llm-qwen4-gen.cpp:341` calls `sub0::forward_one` **once per prime position, in a loop** — there is no
genuine batched `T>1` `forward()` call anywhere in this tool's own prefill path. Measured: prefill's own
per-position time (5.91 s/token) is statistically indistinguishable from decode's per-token time
(5.60 s/token) — direct confirmation that *this build's* "prefill" is exactly as bandwidth-bound as decode
today, because it is mechanically the same T=1 call repeated, not a batched pass.

**This means the BF16/native-quant lever helps prefill exactly as much as decode, right now, in this
tool.** The caveat that would matter — a genuinely batched, compute-bound prefill (the B16 backlog item,
"make generation performance transitions and prefill measurable," names this direction) amortizing one
weight read across many rows' FLOPs, where a slower dequant-per-read format could plausibly cost more than
it saves — does **not yet apply**, because that code path doesn't exist in the gen tool today. **Named as
an open question to re-measure specifically if/when a batched-prefill path lands**, not before: at that
point, the right test is the same one WP6b already ran for the sidecar (`benchmarks/moe_expert_bench.cpp`'s
own decomposition style) — isolate the backbone GEMM's own dequant-vs-compute split at a real batch size,
rather than assume either direction.

---

## 1. Phase 1 — BF16 backbone storage

**STATUS: DONE, branch `feature/b24-bf16-backbone` (built across two passes -- an interrupted agent did
the bulk of the engine plumbing, a second agent found+fixed 4 real defects the interruption left
untested, then ran the correctness gate below for real on both the 4-layer sub-stack and the FULL
48-layer artifact).**

### 1a. Scope

1. **`sub0llm-transplant`** (`tools/sub0llm-transplant.cpp`): already calls `gguf::to_f32(slice, raw, out)`
   for every backbone tensor regardless of source format (§7d). Add an output-dtype option: after the
   existing F32 dequant, round each value to BF16 (truncate/round-to-nearest-even the mantissa — the
   standard BF16 conversion, and `gguf.hpp::bf16_to_f32` already exists as the read-side half of this pair;
   only `f32_to_bf16` needs writing) before the write. **No new decoder** — every source format
   (`Q5_K`/`Q6_K`/`Q8_0`/native `BF16`/`F32`) still goes through the exact same `gguf::to_f32` path it does
   today; only the final write step changes.
2. **Checkpoint format** (`model_file.hpp`'s `Header`, `save_model`/`load_model` in `engine_core.cpp`): the
   `S0L5` format needs a dtype tag (today implicitly always F32) so `load_model` knows how many bytes to
   read and what to do with them — a version bump, following this project's own established precedent for
   every prior format change (the tokenizer/architecture fingerprint trailers already gate exactly this
   class of "old file, new reader" mismatch).
3. **The engine's own storage and compute path** — the largest, most invasive part of this phase. Today
   `g_param_data` is `std::unique_ptr<float[]>` and every `Node::data`/`grad` is `std::span<float>`
   (`core.hpp`); every consuming op (`op_linear` and everything else that reads a parameter leaf) assumes
   F32 directly. Two shapes to choose between, not yet decided here:
   - **(a) Promote-on-load**: `g_param_data` stores BF16 bytes at rest, but a per-parameter-node accessor
     promotes to a small F32 scratch row at the point of use (or the whole tensor, if it fits a cache
     line's worth of reuse) — keeps every existing F32 compute kernel completely unchanged, isolates the
     BF16-awareness to the read side only. Simpler, but doesn't reduce ACTIVE cache/register pressure
     during the GEMV itself, only DRAM bandwidth (which, per §0, is the actual target — this is likely
     sufficient).
   - **(b) BF16-aware kernels**: `op_linear`'s inner loop reads BF16 directly and promotes per-element
     inline (a single shift, per §7c's own reasoning) — closer to what the CUDA backend's `ACT_DTYPE`
     path already does (`backend.cu:183`'s `act_t` conditional), and the more natural fit if this project
     wants the CPU and CUDA backends' dtype story to converge, but touches every SIMD kernel that reads a
     parameter span, not just the load path.
   **Recommendation for this plan, not yet built**: start with (a) — it's the smaller, more contained
   change, isolates risk to the load/access boundary, and directly targets the bandwidth argument in §0
   (a promote-per-read is one DRAM transaction of half the bytes, then pure-F32 compute exactly as today);
   revisit (b) only if profiling after (a) shows the promote step itself is a measurable new cost.
4. A new `PARAM_DTYPE` (or reuse the existing `Dtype` enum already defined for `GEMM_DTYPE`/`ACT_DTYPE`) —
   configurator-baked, following this project's own compile-time-over-runtime discipline (`AGENTS.md` §1) —
   analogous to how `MASTER_DTYPE`/`HEAD_DTYPE` are already named constants, just no longer hardcoded F32.

### 1b. Correctness gate

**Not bit-for-bit — cannot be, BF16 genuinely discards precision — gated by tolerance, following this
project's own already-measured, already-accepted precedent for exactly this class of change.**
`docs/WP4_SCOPE.md`'s own real cross-check (quoted, not re-derived) already established the expected noise
floor for a quantized-weight matmul in this exact model: an **F32**-weight projection agrees with a
high-precision reference to **~1e-7**; a **Q8_0**-weight projection (one of the source model's own real
formats) agrees to **~3.5e-3**. BF16 (8 mantissa bits, coarser than Q8_0's own effective precision for
values near its block's scale) should land in a comparable ~1e-3–1e-2 band — a real, already-precedented,
already-accepted noise floor in this codebase, not a novel risk. Concretely:

- Forward-pass parity: BF16-backbone `forward()`/`forward_one()` output vs. the current F32 reference, on
  the real 48-layer artifact and the WP5c determinism fixture prompt — report the actual relative
  difference (not a pass/fail against an arbitrary threshold picked before measuring), following this
  project's own "verify correctness against reference before performance" discipline.
- The neutral d196 suites (28,875,042/147, 120,889/244) re-run at BF16 to confirm nothing crashes/aborts
  and the SHAPE of every result is unchanged — these were never going to be bit-identical once weights
  change value, so this gate is about robustness, not equality.
- Generated text quality: a real decode run at BF16, same prompt/seed as every WP5c fixture run this
  session, judged qualitatively (still coherent English) as well as by the relative-diff numbers above —
  this project's own "the real transplanted model now generates real English" milestone is the bar a
  precision change must not regress.
- Measured, not assumed: the actual bandwidth win (§0's ~281→~140 ms/token estimate is a lower bound;
  measure the real per-token delta on a run with the MoE sidecar's own contribution held constant, e.g. by
  isolating just the backbone's own GEMV cost the way `moe_expert_bench.cpp` isolated the sidecar's).

### 1c. What was actually built, and the deviation from 1a's own recommendation

Built shape (a), promote-on-read, as recommended — but the interrupted agent found (and documented, in
`param_store.hpp`'s own header comment) that "promote-on-load" as literally worded in §1a is not buildable:
a materialized f32 VIEW of a bf16 tensor is exactly as large as the f32 tensor it replaces (no win at
rest), and materializing one per access pays 2N read + 4N write + 4N read-back where plain f32 pays 4N
read -- strictly worse. The actual shape is "promote INSIDE the innermost loop, in a register, never
touching memory": `param_t`/`ParamCPtr` (`param_store.hpp`) make `g_param_data` an array of `bf16` at
rest, and `Node::pdata` a read-only pointer PROXY (`Bf16CPtr`, `bf16.hpp`) whose `operator[]` widens on
read. Every `*_math.hpp` kernel that reads a parameter span was templated on its weight-pointer type
(`WP`) so the SAME kernel source compiles unchanged against `const float*` (F32) or `Bf16CPtr` (BF16) --
`gated_residual_math.hpp`, `gdn_math.hpp`, `moe_math.hpp`, `qsa_math.hpp`, `scratch_slots.hpp`. This is
functionally shape (a) (DRAM bandwidth is the only thing it targets, compute stays f32) but the
mechanism is closer to §1a's shape (b) description ("reads bf16 directly and promotes per-element
inline") than to (a)'s own "promoted F32 scratch row" -- the plan under-specified how (a) would actually
avoid re-materializing a whole tensor, and the real build resolved that the only way is per-element,
in-register promotion. Scope stayed inference-side as planned: `PARAM_DTYPE == BF16` is
`static_assert`-gated to imply `FORWARD_ONLY` (backend.cpp) -- a bf16 parameter arena has no honest f32
span for backward/AdamW to read or write, so `--prec-param 1` composes only with a forward-only build
(Gated Residual / MoE / QSA on), matching the real Qwen4-preview axes this feature exists for.

### 1d. Correctness gate — measured results

**Two real artifacts, both from the SAME `sub0llm-transplant`/`sub0llm-transplant-q48` run pair against
the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards** (`--param-dtype 1`, a new flag alongside `--param-dtype
0`/the untouched F32 default): the 4-layer real sub-stack (95 tensors, 1.581B params, 19.5s to transplant,
0 level-2 stat mismatches, 0 sidecar bit-for-bit mismatches, `--verify` bit-exact) and the FULL 48-layer
real model (1074 tensors, 4.915B params, 146.4s to transplant, same 0/0/bit-exact result). Both builds
compiled and ran through `sub0llm-qwen4-gen`/`sub0llm-qwen4-forward` at the real axes.

- **Forward-pass parity (§1b's own gate, `--dump-logits` on the WP4d 6-token fixture, L2-relative
  `||a-b||/||a||` over the full `[T x VOCAB]` logit array):**
  - 4-layer sub-stack: **~1.05** (105% -- essentially uncorrelated; 0/6 rows' argmax agreed). **Not
    representative of BF16 precision** -- this specific artifact has NO final norm layer by construction
    (§WP4d's own "LnF absent" finding) and its own F32-only diagnostic (`sub0llm-qwen4-forward`'s section
    5) already measures a **399.8%**-of-scale swing from removing ONE norm op on identical f32 weights, so
    a 4-layer truncation's un-normalized readout is inherently unstable territory, not a fair BF16 test.
  - **48-layer real model (the actual target): L2-relative diff ~0.199 (19.9%), 5 of 6 rows' argmax
    identical.** Still well above WP4_SCOPE's own Q8_0 precedent (~3.5e-3) and the ~1e-3-1e-2 band this
    section originally anticipated. The most likely explanation, not yet independently confirmed: MoE's
    top-k expert routing is discontinuous -- ~36 router decisions (3 MoE layers x 12 repeats) across the
    stack mean a small bf16 rounding of router logits can occasionally flip which experts get selected,
    and this model's own un-normalized final readout is independently measured (same tool, F32-only) to
    already be unusually sensitive to small architectural perturbations. Reported honestly rather than
    forced into the anticipated band.
  - `forward()` vs `forward_one()` parity stayed **bit-exact (0) under BF16** at both scales -- the decode
    path and the batched path read the bf16 arena identically.
- **Neutral suites:** `sub0_frontend_tests` (engine-free, unaffected by `PARAM_DTYPE`) -- **120,889
  assertions / 244 cases, all green**, under both an F32- and a BF16-tagged generated config, after fixing
  2 real template-deduction regressions the interrupted work introduced (see §1e). `sub0_tests` (the
  engine-linked, gradient/backward suite) **cannot run under BF16 at all, by design**: it requires a
  trainable (non-`FORWARD_ONLY`) build, and `PARAM_DTYPE == BF16` is gated to require `FORWARD_ONLY` --
  the two are mutually exclusive on purpose (§1c). Confirmed this is a pre-existing architectural fact,
  not a B24 regression: the SAME suite also fails to run against an F32 `FORWARD_ONLY` (MoE-on) config,
  crashing before Catch2 even starts. `sub0_tests` DOES pass fully at a small trainable F32 config with
  this branch's changes (**5,869,947 assertions / 147 cases, all green**) -- confirming the training path
  is untouched. The real substitute robustness gate -- `sub0llm-qwen4-gen`/`sub0llm-qwen4-forward`
  exercising embed / GDN / Gated-Residual / MoE-routing+experts / QSA / tied-head / save+load under BF16
  at BOTH real-axes scales -- ran to completion with zero crashes/aborts, which is what actually stands in
  for "the neutral suites, at BF16" here.
- **Generated text quality (WP5c fixture, `"The capital of France is"`, seed 1234, temp 0.8, topk 40,
  n=30, full 48-layer model):** BF16 produced coherent, on-topic, grammatically correct English: `" Paris.
  Is this correct?\n\n<think>\nThe user is asking a simple factual question: whether Paris is the capital
  of France. This is a well"` -- correctly identifies Paris, same as the F32 reference's own completion
  (`" Paris. How many countries have capitals with similar names?..."`). The two token sequences diverge
  after the shared first token (expected, given the measured ~0.2 relative logit difference feeding a
  temperature-0.8 sampler), but both sides are real, coherent English -- the gate this item asks for.
- **Bandwidth, measured on this host (§2c's own ~28-35 GB/s figure, not §0's superseded 70 GB/s):**
  full-48-layer decode went from **6.01 s/token (F32) to 5.48 s/token (BF16)**, prefill from 6.36 to 5.92
  s/token -- roughly **9% faster**, smaller than the backbone-alone estimate would suggest because the MoE
  sidecar's own disk-bound cost (B20/B21/B23) still dominates total decode time, exactly as this
  document's own §0 predicted it would. Model LOAD time roughly halved (17.9s -> 8.7s, tracking the
  9.16 GiB backbone-bytes-read halving almost exactly). Peak resident memory dropped **33.96 GiB -> 24.72
  GiB (-9.24 GiB)**, matching the theoretical 9.16 GiB backbone halving closely. This is a real, honest,
  measured win, not dominated entirely by the sidecar -- 9% at the token level, with load time and
  resident memory both moving by close to the theoretical maximum.

### 1e. Defects found and fixed while completing this phase (the interrupted agent's work, reviewed like an unfinished PR)

1. **`if constexpr` on a plain (non-template) or bool-NTTP function does not discard its untaken branch**
   -- the classic C++ gotcha: a discarded `if constexpr` branch is only exempt from full type-checking
   when it is itself dependent on an ENCLOSING TEMPLATE's parameter, not merely gated by a
   compile-time-constant condition. `param_store.hpp`'s `param_cptr`/`param_get`/`param_set` and
   `backend.cpp`'s `param_write_ptr`/`op_linear`'s ternary re-derivation were all written this way and
   failed to compile the instant a real (non-toy) generated config selected `PARAM_DTYPE`. Fixed by
   replacing each with a pair of ordinary overloads on the concrete pointee type (`bf16*`/`float*`),
   which sidesteps the whole issue via normal overload resolution rather than template dependence.
2. **`param_master_f32()` called unqualified from outside its enclosing `namespace cpu_detail`** (two call
   sites in `AdamW::step()`/`muon_step_one`, both physically after the namespace closes) -- a plain
   missing-qualification bug, fixed by qualifying the two call sites.
3. **`tests/scratch_embed_tests.cpp` and `tests/transplant_fixture_tests.cpp` broke under the new
   `WP`-templated `*_math.hpp` signatures** -- a bare `nullptr` no longer deduces a pointer type
   (`encode_slot`), and mixing a `const float*` (from a `const Applied` fixture struct) with a plain
   `float*` (a local `std::vector`) in the SAME `WP`-templated call fails template argument deduction
   ("conflicting types for WP") where the old non-template signature would have silently converted both
   to `const float*`. Both are pre-existing test files exposed by the templating this phase required, not
   BF16-specific; fixed with explicit `const float*` casts at the four affected call sites.
4. `tools/sub0llm-qwen4-forward.cpp` (a WP4d-era diagnostic tool, not itself in B24's scope) hard-`abort()`s
   under `PARAM_DTYPE == BF16` because several of its diagnostic sections read raw parameter bytes through
   `params_ptr()`, which is `[[noreturn]]`-refused under BF16 by design. Retrofitting those sections to
   read through `param_get()`/`ParamCPtr` was judged out of scope for this pass (it duplicates real,
   non-trivial math-core-replay logic); instead, sections 2/3b/5 (the ones touching raw `P`) are now
   skipped with a clear message under BF16, while sections 1/3/4 (`load_model`, `forward()`/`forward_one()`
   and their `--dump-logits`/parity checks -- exactly what B24's own correctness gate needed) are
   unaffected and were the tool used for the §1d measurements above.
5. **Cosmetic, fixed:** `backend.cpp`'s two host-memplan prints (`print_host_memplan`, `print_config`)
   computed the shared-arena estimate as `PARAM_FLOATS * sizeof(float)` unconditionally, so a BF16 build's
   own startup banner reported the F32-sized figure (measured: printed "shared 6032 MiB" for a build whose
   real resident arena is ~3162 MiB). The REAL allocation (`make_unique<param_t[]>(PARAM_FLOATS)`) was
   always correctly sized; only these two diagnostic prints were wrong. Both now use `PARAM_ELEM_BYTES`
   (`param_store.hpp`) instead of a hardcoded `sizeof(float)`.

---

## 2. Phase 2 — model-native quantized backbone (the bigger lever, the harder build)

### 2a. Why this is architecturally different from Phase 1, not just "more compression"

Phase 1 keeps the shape of every existing access (`Node::data` still points at *something* per-parameter,
promoted or not). A genuine native-quant backbone — reading `Q5_K`/`Q6_K`/`Q8_0` bytes directly, the same
formats the source model already ships and `gguf.hpp` already decodes for the sidecar — is architecturally
closer to what `moeq::Store`/`ExpertCache` already do for the routed experts than to Phase 1's promote-on-
read. But it is NOT a drop-in reuse of that machinery, because the backbone's access pattern is the
opposite of the experts':

| | MoE experts (today) | Backbone (Phase 2 target) |
|---|---|---|
| Access pattern | **sparse** — ~10 of 512 experts/layer/token | **dense** — every weight, every layer, every token |
| Resolve-pool hit rate | provably zero in decode (§3b of the memory map — every resolve is a genuine miss) | would be **100%** on every access if reused — a completely different caching question |
| What a "pool" buys | lets 10/512 stay compressed at rest, only ever materializing the ~2% actually needed | nothing to skip — everything is needed every time, so the question is purely "materialize once and keep, or materialize inline every read" |

So Phase 2's real design question is not "build an `ExpertCache` for the backbone" (there is no sparsity to
exploit) — it's the same (a) vs (b) fork Phase 1 already named, just with a real quantized format instead
of BF16's cheap promote:
- **(a) Dequantize once, into a persistent resident buffer** (BF16 or F32) at load time — this is
  literally Phase 1, so Phase 2 under this shape reduces to "pick a better on-disk format for
  `sub0llm-transplant` to read from, still ending at the same resident representation Phase 1 already
  built." Buys smaller checkpoint files and faster loads, NOT smaller resident memory or lower per-token
  DRAM bandwidth (the resident buffer is still BF16/F32-sized either way).
- **(b) Keep the quantized bytes resident, dequantize inline per read, every token** — the shape that
  actually delivers §0's ~23–48 ms/token bandwidth number, because the bytes read off the memory bus during
  the GEMV itself are the smaller, quantized ones. This is the real target, and it is a genuinely new
  per-op inline dequant path (format-specific, `Q5_K`/`Q6_K`/`Q8_0`'s own block-structured decode, already
  implemented in `gguf.hpp` for the offline transplant path, but never before called from inside a hot
  per-token compute kernel) — the CPU-side cost this project has so far only paid for ~10 sparse
  experts/token (§3c of the memory map) would now be paid for the WHOLE dense backbone, every token. This
  needs its own measurement, the same "measure before spending it" discipline `docs/MEMORY_AUDIT.md` §4
  already applies elsewhere: is the format-specific block-dequant cost, paid at 100% density instead of
  ~2%, still cheaper than the DRAM bandwidth it saves? WP6b's own single-expert decomposition
  (~46% of a cold resolve's cost is dequant, not I/O) is a real warning sign worth re-measuring here at
  a dense access pattern, not assuming the sparse case's ratio transfers.

### 2b. Sequencing and dependency

Phase 2 depends on Phase 1 landing first, not just by priority but structurally: Phase 1 builds the
`PARAM_DTYPE`-aware storage/access seam (`g_param_data` no longer assumed-F32, `Node`'s own accessor
promotes on read) that Phase 2's native-quant format would plug into as a second `PARAM_DTYPE` value,
rather than being a second, parallel storage mechanism built from scratch. Phase 2 should be scoped in
detail (a real WP-style design pass, format-by-format dequant cost measurement, a decision between 2a and
2b above) only once Phase 1 has real, measured numbers to build on — named here as the next priority per
the user's own directive, not designed in full in this pass. **B24 (below, §2c) supplies that
format-by-format measurement ahead of Phase 1 landing — it does not depend on Phase 1's engine work, only
on `gguf.hpp`'s existing dequantizers, so the 2a-vs-2b decision itself did not have to wait.**

---

### 2c. B24 measurement — the 2a-vs-2b fork, resolved

**Measured, not assumed, on the real Qwen3.8-Flash-Next UD-IQ1_S shards
(`D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S`), via a new engine-free benchmark,
`benchmarks/backbone_dequant_bench.cpp` (built as `sub0_backbone_dequant_bench`, following
`moe_expert_bench.cpp`'s own plain-`main()` precedent exactly, for the same reason: a cold/warm arm
distinction a Catch2 warm-up would destroy).** Branch `research/b24-native-quant-backbone`. Kept as a
permanent repo benchmark, not a scratch tool — the crossover math below is exactly the kind of thing a
future format addition (a different K-quant, a different source model) should be able to re-run against,
the same way `moe_expert_bench.cpp` stayed as the sidecar's own standing instrument.

**Method.** For each of the backbone's three real formats (`Q5_K`, `Q6_K`, `Q8_0`), picked the largest
real non-expert, non-PLE body-projection tensor of that type anywhere in the actual downloaded shards
(dims from the file's own tensor table, never guessed — `blk.3.attn_q.weight` [`Q5_K`, 31.46M elements],
`blk.2.attn_qkv.weight` [`Q6_K`, 26.21M elements], `output_hc_down.weight` [`Q8_0`, 3.28M elements]).
Measured two things directly, not estimated:

1. **Dequant cost** — `gguf::to_f32` (the exact function the offline transplant already calls) run
   warm, repeatedly, over each tensor's real decoded bytes.
2. **Achieved DRAM bandwidth per byte-width** — a replica-array streaming touch (one byte per 64B cache
   line, footprint forced past this host's 36 MiB L3, `docs/host-cpu-arrow-lake-hx.md`), at the
   quantized byte-width AND at F32 (today's resident format) and BF16 (Phase 1's).

**Result — decisive, not close, and the same direction for all three formats.** 2a (dequantize once,
keep a resident F32/BF16 buffer, read that every token) beats 2b (keep the quantized bytes resident,
dequantize every read) by roughly **5-30x**, not a coin-flip:

| Format | Measured dequant `D` | Break-even `D*` (F32-resident) | Break-even `D*` (BF16-resident) | Margin |
|---|---:|---:|---:|---:|
| `Q8_0` | ~0.54-0.70 ns/elem | ~0.10 ns/elem | ~0.03 ns/elem | 2a wins by ~5-7x |
| `Q5_K` | ~1.03-1.08 ns/elem | ~0.11 ns/elem | ~0.04 ns/elem | 2a wins by ~9-10x |
| `Q6_K` | ~1.10-1.23 ns/elem | ~0.10 ns/elem | ~0.04 ns/elem | 2a wins by ~11-12x |

(`D*`, the break-even dequant cost per element, is derived as shown in the benchmark's own header: 2b
wins iff `D < D* = (R - Q) / BW`, where `R` is the resident format's bytes/element, `Q` the quantized
format's bytes/element, and `BW` the quantized stream's own measured achieved bandwidth — repeated across
two independent runs, `D`/`D*` both stable to within their reported range.) **§0's own `~46%` warning
sign from WP6b's sparse-density case was right to flag — inline dequant IS the dominant cost here too —
but at 100% density the DRAM-bandwidth side of the ledger shrinks (quantized formats only save ~3-6x
bytes vs F32/BF16's ~2-4x, not the sidecar's ~12x blended ratio) while the dequant side stays roughly the
same per-element cost, so the net verdict flips decisively toward 2a, not 2b.** This is the opposite of
what §0's arithmetic implicitly hoped for (2b was "the shape that actually delivers the ~23-48 ms/token
estimate") — that estimate does not survive contact with the real per-element dequant cost.

**A second, independent correction this measurement surfaced**: §0's own bandwidth assumption (70 GB/s,
"typical dual-channel DDR5 mobile") does not match this host. The SAME benchmark's achieved streaming
bandwidth, measured directly rather than assumed, is **~28-33 GB/s** for F32/BF16-width reads on this
machine (`docs/host-cpu-arrow-lake-hx.md`'s Arrow Lake-HX, dual-channel), roughly **2.2x lower** than §0's
figure. Recomputing §0's own table with the measured number: F32 backbone stream ≈ **~630-910 ms/token**
(not ~281 ms), BF16 ≈ **~300-350 ms/token** (not ~140 ms). This does not change the 2a-vs-2b verdict
(both sides of the comparison use the same measured `BW`, so it cancels out of the crossover), but it
does change how big Phase 1's own win looks in absolute terms, and should replace §0's assumed figure
wherever this document or a related one cites it.

**Recommendation**: build 2a, not 2b. Concretely, this means Phase 2 should NOT build a new inline
per-token dequant path at all — the real leverage in "model-native quantized backbone" is entirely in
what `sub0llm-transplant` writes to disk (a smaller on-disk checkpoint, faster loads), landing at the
SAME resident representation Phase 1's `PARAM_DTYPE` seam already builds. Concretely: Phase 2 reduces to
"teach `sub0llm-transplant` an additional output mode that keeps blocks in something more compact than
BF16 if a further win is wanted there" — a decision about the RESIDENT format (BF16 vs. some other
fixed-width small type), never about doing block-decode inline in the hot per-token path. This also
collapses Phase 2's dependency on Phase 1 from "the format Phase 2 adds" to "there may be no separate
Phase 2 engine change at all" — once Phase 1's `PARAM_DTYPE` seam exists, whether the transplant tool
happens to read `Q5_K`/`Q6_K`/`Q8_0`/`F32`/`BF16` bytes on the way to writing a smaller resident format is
already exactly what it does today for BF16 output; there is no genuinely new *engine* mechanism 2b would
have required that 2a still needs.

**Honesty check not run, and why it's unlikely to matter here**: `docs/sub0mempage-research-empirical-
concurrency.md`'s §5b ballast-under-pressure discipline was not replicated in this pass (time-boxed) —
the numbers above are isolated-process measurements, not measured under the ~25 GiB resident memory
pressure the real engine runs under (`docs/QWEN4_MEMORY_MAP.md`). This is a real gap, flagged rather than
glossed over, but the margin here (5-30x, not a percent-level difference) is wide enough that a plausible
memory-pressure effect on EITHER side of the comparison (a slower achieved `BW`, a slower `D` from cache
contention) would have to be an order of magnitude larger than anything B21's own ballast test found
elsewhere in this codebase to flip the verdict. Re-running under ballast is still the right thing to do
before this is treated as fully closed, particularly because Q8_0's margin (5-7x) is the narrowest of the
three and the one worth re-checking first if this is revisited.

**Format-dependence, checked precisely as asked, not glossed**: all three formats land in the same
direction (2a wins), so this is NOT a case requiring a per-format split verdict — but the margins are not
identical, and `Q8_0` (the format with the fewest bits/element among the three real backbone formats,
1.0625 vs `Q5_K`'s 0.6875 and `Q6_K`'s 0.8203) has the narrowest margin, consistent with its dequant cost
`D` also being the LOWEST of the three (simpler block structure, per `gguf.hpp`'s own block-spec
comment) — the two effects partially offset rather than compound, which is why `Q8_0` isn't the clear
loser its higher bytes/element might suggest at a glance.

---

---

### 2d. B33 — what was actually built: FP8 (E4M3) as a third resident `PARAM_DTYPE`

**STATUS: DONE, branch `feature/b33-fp8-backbone`.** Per §2c's own recommendation ("Phase 2 reduces to a
resident-format choice for `sub0llm-transplant`'s output, not a new hot-path engine mechanism"), this adds
a SECOND resident format alongside BF16 -- not the block-scaled Q8/Q4 mechanism §2c decisively ruled out,
a genuinely different, flat, block-free 8-bit float. Mirrors Phase 1's own architecture exactly, one more
`param_t`.

**Encoding choice, stated precisely.** OCP/NVIDIA **E4M3**, specifically the **`e4m3fn`** convention (1
sign, 4 exponent bits/bias 7, 3 mantissa bits; NO infinities -- the exponent-all-ones/mantissa-all-ones
code point 0x7f/0xff is the ONE reserved NaN encoding, every other exponent-all-ones mantissa is an
ordinary finite value, pushing the true max magnitude to 1.75 x 2^8 = 448). This is the same convention
`torch.float8_e4m3fn` uses, and the reason it was chosen over the OCP variant WITH infinity: it is the
convention most weight-only FP8 quantization already targets, and "overflow saturates to NaN" is a
simpler, single failure mode than juggling a real (but rarely hit, at real weight magnitudes) infinity
encoding. `+-Infinity` and any overflowing finite input both saturate to the same NaN code point on
narrow.

**Files, new or changed:**
- `include/sub0/fp8.hpp` (NEW) -- `fp8` (1-byte struct), `fp8_widen`/`to_f32` (exact bit-decode, a real
  exponent remap unlike bf16's pure truncation -- e4m3's exponent range/bias genuinely differ from f32's),
  `fp8_narrow` (f32->fp8, round-to-nearest-even, one shift-based construction covering both the normal
  and subnormal output ranges plus their mutual overflow/carry cases), `Fp8CPtr` (the same minimal
  read-only proxy interface as `Bf16CPtr`). **Named `fp8_narrow`, not `f32_narrow`** -- `bf16.hpp`
  already defines `f32_narrow(float) -> bf16` in the same namespace, and C++ cannot overload on return
  type alone, so reusing that name is a hard redefinition error the moment both headers are included
  together (which `param_store.hpp` does) -- caught by the standalone syntax-check before it ever reached
  a real build.
- `include/sub0/param_store.hpp` -- `param_t`/`ParamCPtr` extended to a three-way `std::conditional_t`
  chain (F32/BF16/FP8); `param_cptr`/`param_get`/`param_set` gain a third overload each (`const fp8*`);
  the header's own `static_assert` now allows all three `Dtype` values.
- `include/sub0/model_file.hpp` -- `ParamDtype` gains `FP8 = 2`; `param_dtype_bytes`/`param_dtype_name`
  extended.
- `tools/configurator.cpp` -- generated `Dtype` enum gains `FP8` (not a repurposed `Q8`/`Q4` -- those
  names are reserved for the block-quant mechanism §2c ruled out, and reusing them would misleadingly
  imply that mechanism); `--prec-param` extended to `0=F32|1=BF16|2=FP8`.
- `tools/sub0llm-transplant.cpp` -- `--param-dtype 2` output mode: same dequant-to-f32-then-narrow shape
  as the existing BF16 branch, write/verify scratch buffers added in parallel.
- `src/backends/cpu/backend.cpp` -- **two dispatch-overload sites needed a third overload each**
  (`param_write_ptr_of(Fp8CPtr)`, `maybe_ternarize(Fp8CPtr&, ...)`) -- these are the same
  "ordinary-overloads-not-`if constexpr`" pattern `param_store.hpp`'s own comment documents, at the ONE
  other layer (backend.cpp's own dispatch functions, not the `*_math.hpp` kernels) that names `Bf16CPtr`
  by its concrete type rather than going through the generic `WP` template parameter. Both are trivial:
  `USE_TERNARY`/ternarization is a trainable-path feature, and `PARAM_DTYPE != F32` is FORWARD_ONLY by
  the same arena `static_assert` BF16 already required, so the FP8 overload is the identical no-op.
  Two `refuse_f32_params()` call sites (`params_ptr()`, `param_master_f32()`) generalized from
  `PARAM_DTYPE == Dtype::BF16` to `PARAM_DTYPE != Dtype::F32` -- previously a BF16-only guard that would
  have silently reinterpreted an FP8 arena as f32 had it been left unchanged.
- `src/engine_core.cpp`'s `load_model` -- the file-size-based dtype discriminator (the authoritative
  signal per B24's own established discipline; the on-disk tag is corroboration only, since it can be
  pre-B24 padding garbage) extended from a two-way (f32/bf16) to a three-way (f32/bf16/fp8) candidate-size
  comparison.
- **`include/sub0/*_math.hpp` kernels needed ZERO changes** -- confirmed by building and running the real
  48-layer artifact end to end: every kernel is already templated on its weight-pointer type (`WP`) from
  B24 Phase 1, so `Fp8CPtr` deduces into the exact same kernel source `Bf16CPtr`/`const float*` do, with no
  `is_same<WP,...>`-style dispatch anywhere in the codebase to break. The only real code beyond `fp8.hpp`
  itself was the two `backend.cpp` dispatch-overload sites above, which sit OUTSIDE the `WP`-templated
  kernels by construction.

**Round-trip correctness (`fp8_probe.cpp`, run standalone before any engine build):**
- All 256 fp8 code points, widened then narrowed, reproduce the identical bit pattern exactly (the 254
  finite/zero values) or a NaN (the 2 reserved NaN codes) -- a full, exhaustive self-consistency sweep of
  the entire domain, not a sample.
- Exact small values (0, -0, 1.0, -1.0, 2.0, 0.5, the max finite +-448) narrow to their hand-derived bit
  patterns and round-trip exactly.
- Overflow (1e30, +-infinity) and NaN-in all saturate to the reserved NaN code point.
- RNE tie-breaking verified both directions: 1.0625 (exact tie between mantissa 000/even and 001/odd)
  rounds DOWN to even (000); 1.1875 (tie between 001/odd and 010/even) rounds UP to even (010).
- Subnormal exactness: the smallest subnormal (2^-9) round-trips exactly; exactly half of it rounds to
  zero (RNE ties-to-even at zero).

**Real-model correctness gate, on the actual Qwen3.8-Flash-Next UD-IQ1_S GGUF shards, mirroring B24's own
gate exactly (quantized-resident-MoE variant, `--moe-quant-experts 1`, the same shape B24's own "95
tensors, 1.581B params" / "1074 tensors, 4.915B params" figures describe -- the first attempt at this gate
mistakenly used the DENSE, non-quantized-experts transplant target, which segfaults on THIS machine at
real MoE dims regardless of backbone dtype -- confirmed by reproducing the identical segfault on a freshly
transplanted BF16 artifact at the same dense-MoE shape; not an FP8 defect, a pre-existing gap in dense-MoE
support at these dims that nothing had exercised before, named here rather than silently worked around):**

- **4-layer sub-stack** (95 tensors, 1,581,285,280 floats, 22.6s transplant, 0/94 level-2 stat mismatches,
  0/95 `--verify` bit-for-bit mismatches). `forward`/`forward_one` parity **bit-exact (max diff 0)**.
  Logits vs the F32 reference: L2-relative ~1.05, 0/6 argmax agreement -- reproduces BF16's own documented
  4-layer figure (also ~1.05, 0/6, re-measured here as ~1.05/0/6 on a fresh BF16 transplant too) almost
  exactly. **Not representative of either format's real precision**, for the same reason §1d already
  gives: this artifact has no final norm layer (`LnF` absent by construction), and F32's own un-normalized
  readout is independently measured (400% here also present) to be unstable at this truncation -- both
  BF16 and FP8 land in the same "essentially uncorrelated" band at 4 layers, which is a property of the
  4-layer truncation, not of either reduced-precision format.
- **48-layer real model (the actual target)**: 1074 tensors, 4,915,107,200 floats, 169.2s transplant, 0
  stat mismatches, sidecar identical to BF16's own (73,728 expert tensors, 37.11 GiB, encoding is
  independent of the backbone's `PARAM_DTYPE`). `forward`/`forward_one` parity **bit-exact (max diff 0)**.
  **Logits vs the F32 reference: L2-relative diff 0.4299 (43.0%), 3/6 rows argmax-identical.** Measured on
  the SAME 6-token fixture, same machine, same session as a fresh BF16-vs-F32 re-measurement, which
  reproduced BF16's own documented ~0.199/5-6 figure closely (0.198895, 5/6, matching §1d's own number to
  four significant figures -- confirms the comparison methodology itself is sound). **FP8 is markedly
  worse than BF16 here, as physically expected** given e4m3's ~3 effective mantissa bits vs bf16's 7 --
  reported plainly rather than minimized, per this item's own instruction to be honest about a real
  precision/quality tradeoff.
- **Peak resident memory, measured on the SAME host in the SAME session (the directly comparable
  apples-to-apples pair, since the two builds' `PARAM_FILE_DTYPE`-driven arena size is the only axis that
  differs)**: BF16 19.31 GiB -> FP8 14.68 GiB, **-4.63 GiB**, closely matching the a-priori theoretical
  expectation (~4.5 GiB, three-quarters of the 6.14 GiB the FP8-vs-BF16 backbone-alone difference should
  be at these params, the remainder shared between both builds' identical MoE resolve-pool/activation
  arenas).
- **Decode throughput -- the honest, surprising result.** Two interleaved runs each, `sub0llm-qwen4-forward
  --tokens 6` (the same tool/methodology as B24's own §1d measurement), on the SAME host in the SAME
  session immediately after each other (thermal/cache-state as comparable as this session could make it
  without a reboot): **FP8 forward 8.21s/tok then 6.93s/tok (mean 7.57s/tok); BF16 forward 6.11s/tok then
  3.38s/tok (mean 4.74s/tok) -- FP8 measured ~60% SLOWER than BF16, not the modest few-percent win the
  session's own prior arithmetic anticipated.** `forward_one` shows the same direction (FP8 mean
  7.27s/tok vs BF16 mean 4.34s/tok). Both formats' SECOND run is markedly faster than their first (page
  cache warming the 37.11 GiB shared `.moeq` sidecar, the same effect B27 documented) -- a real confound
  this pass could only partially control for (two runs each, not the fully interleaved/repeated design
  B29's own A/B used, given this session's time budget), so the exact MAGNITUDE of the gap is not fully
  pinned down. But the DIRECTION held at both a cold and a warmer measurement for both formats, which is
  evidence against pure cache-state coincidence. **The most likely real explanation, not yet independently
  confirmed**: `Fp8CPtr::operator[]`'s widen is a genuine multi-branch exponent remap (subnormal check,
  NaN-code check, a 2-way ternary to locate a subnormal's leading bit) executed on EVERY element read at
  the backbone's 100%-density access pattern, where `Bf16CPtr::operator[]`'s widen is a single branchless
  16-bit shift -- i.e. e4m3's per-element CPU decode cost plausibly outweighs the DRAM-bandwidth bytes it
  saves over bf16, the same "measure the real per-element cost, don't assume the bandwidth side of the
  ledger wins" lesson §2c's own B24 measurement already taught for block-quant formats, now showing up
  again at a smaller but real scale for a flat format. **Named as the load-bearing open question for
  anyone reviving this format, not silently rationalized away**: if this hypothesis is right, a
  branchless/table-driven `fp8_widen` (e.g. the 256-entry lookup table this header's own comment names as
  an available alternative, not yet built) could plausibly close some or all of this gap -- untested here.
- **Generation-quality qualitative check: NOT run.** `sub0llm-qwen4-gen`'s real Qwen tokenizer files
  (`data/qwen_tokenizer/` or `$SUB0_QWEN_TOKENIZER_DIR`) are absent from this environment -- the same gap
  `sub0_frontend_tests`' own qwen-tokenizer test cases already report as "skipping" in this environment.
  The logit-level L2/argmax comparison above is the substitute quantitative signal; no qualitative
  "is the generated English still coherent" read was possible here. Flagged rather than glossed over: a
  0.43 L2-relative diff with only half the rows' argmax agreeing is a real, material risk to generation
  coherence that this pass could not directly verify one way or the other.

**Full suite (neutral small config, freshly reconfigured against `data/gsm8k.txt --dmodel 196` in this
worktree -- a DIFFERENT vocab/config than the session's own established `d196check` baseline, so the
absolute counts below do not match 28,969,623/147 and 120,889/244; the counts are compared BEFORE vs AFTER
this change at the SAME fresh config instead, which is the actual AGENTS.md S4 gate):**
- `sub0_frontend_tests`: **120,889 assertions / 244 cases**, all green -- exact match to the session's own
  established baseline (this suite is engine/config-independent, so it should and does match regardless of
  which engine config is active).
- `sub0_tests` (default F32/trainable, unmodified `PARAM_DTYPE`): **20,586,739 assertions / 147 cases**,
  all green, identical hashes (`arch_identity_tests`' own forward/grad/decode fingerprints) -- confirmed
  BYTE-IDENTICAL to a baseline run with this change's files reverted (via a tagged `git stash`) at the
  exact same fresh config, satisfying AGENTS.md S4 directly rather than by inference: the default F32
  build is provably unaffected by this change.

**Not run, and why**: `sub0_tests` cannot link against a FORWARD_ONLY (MoE/Gated-Residual/QSA-on) or
real-axes config at all, by the same pre-existing architectural fact §1d's own gate already documented
for BF16 -- `sub0llm-qwen4-forward`'s own `forward`/`forward_one` parity gate is what substitutes, exactly
as it did for Phase 1.

**Independently reverified, session owner (2026-09-11)**: reran the FP8-specific tests myself (288
assertions/6 cases, exact match) and the full frontend suite (121,177/250, exact match). Rebuilt and ran
my own quick interleaved A/B on the real 48-layer artifacts (`Sub0Llm-Qwen4-full48-bf16` vs
`Sub0Llm-Qwen4-full48-fp8`, 2 runs each, `sub0llm-qwen4-forward --tokens 3`): BF16 ~3.53-3.71 s/token vs
FP8 ~5.14-5.20 s/token — a real ~40-46% slowdown in my own sample, same direction and same order of
magnitude as the agent's own reported ~60%, confirming this is a genuine regression, not a fluke, a build
artifact, or noise. **Decision: NOT merged.** A real ~4.63 GiB memory win does not offset a real throughput
regression and a markedly worse quality floor than BF16 — the opposite trade a smaller/faster format is
supposed to offer. Kept on `feature/b33-fp8-backbone` (correctness-gated, reproducible) for whoever wants
to pursue a branchless/lookup-table `fp8_widen` — the likely fix for the throughput regression — as a
follow-up; not silently discarded, but not worth shipping as-is. See
`docs/INDEPENDENT_REVIEW_BACKLOG.md`'s B33 entry for the summary-table record.

**B37 (2026-09-11) — superseding update: integrated into `main` as a real, permanently-available
`--prec-param fp8`/`PARAM_DTYPE::FP8` build option, NOT as a default.** The "NOT merged" call above was
correct at the time (shipping FP8 AS THE DEFAULT, or dropping BF16, would have been a real regression),
but the user's later, separate direction — "integrate the unmerged branches to main... make them A/B
shootout compliant" — reframes the question: this is not "is FP8 worth replacing BF16 with" (no), it's
"should a correctness-gated, honestly-documented negative result live as a permanently buildable,
testable option instead of stranded on a branch nobody re-derives" (yes). Default `--prec-param`
(unset or `1`) still means BF16, bit-for-bit unchanged from `main` before this merge. See
`docs/INDEPENDENT_REVIEW_BACKLOG.md`'s B37 row for the integration's own verification detail.
---

## 3. What this document deliberately does not decide yet

- Whether BF16's promote-on-load (1a's option (a)) or BF16-aware kernels (option (b)) is the right shape —
  recommended (a) as the starting point, not committed.
- The exact `PARAM_DTYPE` enum shape/naming, and whether it reuses `Dtype`/`GEMM_DTYPE`'s existing
  machinery or is a new, CPU-backend-specific concept.
- ~~Phase 2's 2a-vs-2b fork — named as the real question, not resolved~~ **RESOLVED by B24 (§2c):
  measured on the real backbone tensors/formats, 2a wins decisively (5-30x margin) for all three real
  formats (`Q5_K`/`Q6_K`/`Q8_0`) — dequantize once into a resident buffer, do NOT dequantize inline per
  read. Phase 2 therefore reduces to a resident-format choice for `sub0llm-transplant`'s output, not a
  new hot-path engine mechanism. Not run under B21-style memory-pressure ballast — flagged as the one
  remaining honesty gap, judged unlikely to flip a margin this wide (§2c's own note).**
- Whether the CUDA backend's own existing `ACT_DTYPE`/`GEMM_DTYPE` BF16 path should be touched at all —
  out of scope; this plan is CPU-backend-only, matching `QWEN4_MEMORY_MAP.md`'s own scope.
