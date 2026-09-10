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
