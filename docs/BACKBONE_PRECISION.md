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
the user's own directive, not designed in full in this pass.

---

## 3. What this document deliberately does not decide yet

- Whether BF16's promote-on-load (1a's option (a)) or BF16-aware kernels (option (b)) is the right shape —
  recommended (a) as the starting point, not committed.
- The exact `PARAM_DTYPE` enum shape/naming, and whether it reuses `Dtype`/`GEMM_DTYPE`'s existing
  machinery or is a new, CPU-backend-specific concept.
- Phase 2's 2a-vs-2b fork — named as the real question, not resolved; 2b is the one that delivers the
  user's own stated goal (bandwidth, not just footprint) but is the harder, riskier build and needs its own
  dense-access-pattern dequant-cost measurement before being committed to.
- Whether the CUDA backend's own existing `ACT_DTYPE`/`GEMM_DTYPE` BF16 path should be touched at all —
  out of scope; this plan is CPU-backend-only, matching `QWEN4_MEMORY_MAP.md`'s own scope.
