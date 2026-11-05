---
name: qwen4-decode-perf-b24-b35
description: "Sub0Llm decode-throughput thread (B24-B35): BF16 backbone, disk-vs-compute-bound reframing, dequant fix, thread-count fix, transpose elimination, llama.cpp comparison, FP8/SIMD negative results"
metadata:
  node_type: memory
  type: project
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-11T14:52:01.874Z
---

**2026-09-10/11 session goal (`/goal`, blocks stopping until met)**: "actual token-generation near CPU
theoretical performance." Full chain below, every item independently reverified before merging (or
explicitly left unmerged with a real, honestly-reported negative result). Canonical record:
`docs/INDEPENDENT_REVIEW_BACKLOG.md` B24-B35, `docs/ACTIVE_WORK_LOG.md` for dispatch/close-out log
entries, `docs/BACKBONE_PRECISION.md` for the BF16/FP8 design detail.

- **B24** (done, merged `dda710b`): BF16 backbone storage. Phase 2 research separately proved
  dequantize-once-resident beats inline-per-read dequant by 5-30x for the real Q5_K/Q6_K/Q8_0 formats;
  real host DRAM bandwidth measured ~28-33 GB/s (not the assumed 70 GB/s).
- **B25** (done, bit-exact, throughput NULL ~1.4% worse — NOT merged): pipelined overlapped I/O for MoE
  resolve. Kept on `feature/b25-explicit-io-resolve` for a future cold-cache scenario.
- **B27** (pivotal): decode is NOT disk-bound any more — this session's own repeated access fully warmed
  the 37.11 GiB `.moeq` sidecar into the page cache; I/O is now ~0.02% of resolve cost, compute ~99.98%.
  Explains why B25 measured nothing; reframes B21 as a compute-side ceiling.
- **B28** (done, merged `8ed590d`): fixed `dequantize_iq2_xxs`'s ~6x-slower decode (real cause: a
  per-element branch on a sign bit, not just the tail-bound guard; branchless fix) — ~18-20% faster.
- **B29** (done, merged `b79c85c`, closes B21): `ParallelExperts`' 10 `MOE_DECODE_THREADS` were fully
  serialized by real DRAM-bandwidth contention (one resolve alone already ~44-52% of this host's
  bandwidth ceiling). Pinned to compile-time `1` — another ~2-9% faster. See
  [[b21-decode-concurrency-plateau]] for the full B21 root-cause story.
- **B30** (synthesis, since partially revised by B32): believed decode was near the DRAM-bandwidth floor
  (~3.57-3.67 s/token) — WRONG, see B32.
- **B32** (pivotal, user asked "how fast is llama.cpp latest with this model"): built+ran real
  `llama-bench` from `D:\Craig\llama.cpp-qwen4exp` (local fork) against the same real source GGUF,
  CPU-only, same host — llama.cpp generates at ~0.6-0.9 s/token, **~4-6x FASTER** than Sub0Llm at the
  time. Proved B30 wrong; directly validated B31's premise (llama.cpp's `ggml` kernels fuse dequant+matmul
  without a full intermediate DRAM round-trip).
- **B31** (done, merged `d75449b`): eliminated the MoE resolve's transpose stage entirely — provably
  bit-exact (`expert_ffn_row`'s accumulation order is exactly reproducible as a direct dot product against
  GGUF's untransposed source layout, since a transpose is a pure permutation; hand-verified against
  `transpose_out_in`'s real semantics before merging). ~7.9% faster — smaller than the ~40-50% theory
  implied (MoE resolve is only part of decode's cost) but real.
- **B33** (done, correctly built, real NEGATIVE result — NOT merged): FP8 (E4M3) backbone storage.
  Zero kernel changes needed, exhaustive round-trip-correct, but ~40-60% decode SLOWDOWN vs BF16
  (independently reproduced: ~3.5-3.7s BF16 vs ~5.1-5.2s FP8) — `Fp8CPtr`'s multi-branch exponent-remap
  widen costs more CPU than the bandwidth it saves. Kept on `feature/b33-fp8-backbone` for a future
  branchless-`fp8_widen` follow-up.
- **llama.cpp cross-check** (user-directed): read `ggml`'s real x86 quant kernels
  (`llama.cpp-qwen4exp/ggml/src/ggml-cpu/arch/x86/quants.c`). Core technique: quantized-activation ×
  quantized-weight integer SIMD dot products (`ggml_vec_dot_q5_K_q8_K` et al — never materializes a float
  weight array, converts to float once per block at the end). Filed as **B35, deliberately deferred**
  (new kernel family, doesn't fit the current float-`Node`/`WP` architecture, needs its own design pass).
  Separately confirmed Sub0Llm's hot reduction loops (`s += x[i]*w[i]`, every `*_math.hpp`) do NOT
  auto-vectorize at all regardless of AVX2 being enabled (`SUB0_NATIVE=ON`) — Clang: "cannot prove it is
  safe to reorder floating-point operations." User picked this smaller lever first over B35
  (`AskUserQuestion`).
- **B34** (done, correctly built, ANOTHER real NEGATIVE result — NOT merged): multi-accumulator SIMD
  restructuring (new `include/sub0/simd_reduce.hpp`). Found+fixed a real correctness bug during
  verification (a reordered dot product broke B31's own `expert_ffn_row`/`expert_ffn_row_source`
  bit-exactness invariant — fixed with a separate unreordered `dot_seq` primitive). But ~20% decode
  SLOWDOWN (independently reproduced, same direction) — vectorization landed on small/cheap reductions
  while the one dominant call site had to stay unreordered for correctness. First attempt hit an API
  rate limit mid-task, resumed in the same worktree per established precedent (same pattern as B24
  Phase 1). Kept on `feature/b34-simd-unlock`.
- **B26** filed, deliberately deferred: `Sub0Pipeline`'s `add_on_demand()`/`trigger()` stub matches B25's
  DAG shape — feed back only once B25 ships.

**Pattern across B28/B29/B31 (real wins) vs B25/B28-prefetch/B33/B34 (real nulls/negatives)**: every win
reduced actual DRAM bytes moved or memory contention; every null/negative was a compute-side-only change
on a workload B27 already proved is DRAM-bandwidth-bound. This is why **B35** (quantize both operands,
reduce bytes moved AND compute cost together) is judged the most promising remaining lever — it's the
only unstarted item that attacks bytes-moved, not just compute.

- **B35** (done, merged `341e37b`, independently reverified): **the first real throughput win of this
  thread — 2.4x.** Fused quantized MoE dot products (`--moe-quant-dot`, default OFF): keep the experts'
  native IQ1_S/IQ2_XXS/IQ4_NL bytes, quantize the activation row to int8 once, integer-SIMD dot directly
  against them — never materialising the 18.75 MiB f32 planes. Verified interleaved with zero competing
  processes: **3.633 → 1.511 s/token**. `forward()` stays bit-identical between arms (op_moe deliberately
  unfused). No raw intrinsics needed: integer addition has no reassociation barrier, so Clang vectorizes
  the portable loop unaided — the opposite of B34/B38's float-reduction experience.
  **BUT it buys speed with quality, and that must not be glossed**: fused-arm `forward`/`forward_one`
  parity goes from exactly 0 to max|diff| 4.94 on a ±15 logit range, L2 0.2938, argmax 4/6 — two of six
  rows pick a different top token, changing generated text under greedy decode. Precedents: BF16 0.199
  @5/6 (shipped default), FP8 0.43 @3/6 (default-off, not recommended). B35 sits between and ships
  default-off — but unlike FP8 (slower AND worse) it is a real choosable tradeoff.

**Session total**: baseline ~5.5-6.0 s/token → ~3.38 s/token (~40-45% faster, B27→B28→B29→B31), closing
roughly a third of the B32-measured gap to llama.cpp at the quality-preserving default.
**With `--moe-quant-dot` on: ~1.51 s/token**, which closes the llama.cpp gap from ~4-6x to ~1.7-2.5x —
but that figure is partly BOUGHT with quality, not purely earned. Goal materially closer, not met, and
the honest statement of where it stands must distinguish the two arms.

**2026-09-11, integration sweep (B36/B37/B38)**: user asked to integrate the three unmerged
correctness-clean-but-negative-result branches (B25/B33/B34) into `main` properly — "make them A/B
shootout compliant so they are visible improvements and cleanly maintained", not left stranded. Dispatched
three parallel integration packages, each adding a real, permanent, default-off, correctness-gated CLI
build option (matching the existing `--prec-param`/`PARAM_DTYPE` idiom) rather than a silent merge:
- **B36** (`d59fe30`): `--moe-io-mode reactive|pipelined` (B25). Required genuine re-targeting, not a
  diff replay — B25 predated both B31 (replaced `ExpertCache`+transpose with `ExpertCacheSource`) and B29
  (pinned `MOE_DECODE_THREADS=1`, invalidating B25's own per-thread staging design). New
  `ExpertCacheSource::resolve_from_bytes()` + a `MoeIoStage` indexed by selection order, not thread.
- **B37** (`183230b`): `--prec-param fp8` (B33). Near-clean cherry-pick (FP8 already fit the existing
  dtype-selection seam); `fp8_tests.cpp` now runs unconditionally as permanent regression coverage.
- **B38** (`1e161f4`): `--simd-reduce` (B34). Every kernel file's reduction call site threaded a `bool
  UseSimd` template param to shared `dot_choice`/`sumsq_choice`/`sum_choice` dispatchers — no duplicated
  kernel bodies; the `false` arm is a byte-for-byte copy of the original scalar loop, so default-off is
  bit-exact BY CONSTRUCTION, not just by testing.

Every default arm independently verified byte-for-byte/bit-exact unaffected (decode hash `816c4a54ad49b8cf`
unchanged across all three merges); every non-default arm's honest negative result documented prominently
at its own CLI option's help text, not just buried in the backlog. Hit the same "stale generated config"
issue three times running (rebuilt `sub0llm-configure` + regenerated each time, established B24-Phase-1
precedent). Two real merge conflicts resolved by hand (B36-vs-B37 docs; B36-vs-B38 both touching
`decode.cpp`'s shared MoE call site — combined B36's full lambda body with B38's added template arg).
All three source branches + worktrees deleted; content lives on `main` now.

## 2026-09-22 — O1 MERGED (539b9fd): decode 1.508 -> 0.705 s/token (2.14x), now inside llama.cpp's 0.6-0.9
- detail::gemv_avx2 in moe_quant_dot.hpp (vector accumulator per row, vpsignb+vpmaddubsw, F16C, IQ4_NL vpshufb);
  portable gemv kept as fallback + test ref. L2 IMPROVED 0.293822 -> 0.23252 (argmax 4/6 both). Suites exact.
- Microbench `sub0llm-bench-moeqd` (tools/) is the fast inner loop; its old-kernel numbers matched the real profile.
- Cold decode (fused, pre-O1): post-reboot 2.955, evicted 2.816/2.695, warm 1.651 s/token. Eviction ~5-9%
  optimistic. `run_perf_suite.py --cold` + `page_cache.py` (canary-verified, unelevated NO_BUFFERING purge —
  UNDOCUMENTED FS behaviour, re-verified every run). `--decode-only` tool flag needs kv_reset() first.
- Sandbox: priority mode default (affinity mode parked: self-confinement bug + 24-thread oversubscription).
  Post-PSU "baseline" row is marked invalid. `System` kernel load ~60-90% of a core persists even post-reboot.
- NEXT: re-profile at O1 point (MoE no longer 68.7%). Candidates: IQ1_S/IQ2_XXS grid lookups (~15% AVX2),
  IQ4_NL memory-side (retry prefetch — B28 was measured when compute-bound), mixer (was 21.2%).

## 2026-09-22 — O2 MERGED (470a894): decode 0.757 -> 0.313 s/token (2.42x), bit-identical
- include/sub0/gemv.hpp: one vectorized+threaded GEMV for all backbone projections (was 6 hand loops).
- B40: main thread pinned to ONE CPU confined every decode OpenMP team (B21/B29 "DRAM contention" was this).
- RECOMMENDED real-axes decode flags: `--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 10`
  (both thread axes default 1 in the configurator).
- Profile at 0.314: mixer 148 ms (47%), Gated Residual 79 (25%, serial/latency-side), routed 39, lm_head 25,
  router+shared 22. Bandwidth floor ~100 ms/token (8 GB @ 79 GB/s) -> ~3x left. Goal NOT yet met.
- Process backlog P1 (lean workflow as default dev workflow) recorded in docs/optimization/opportunities/README.md.

## 2026-09-22 late — O3 merged (1c0a6c2, 63d22ad, 4d164d9): decode ~0.21-0.25 s/token
- GR mix via gemv primitive (79 -> ~35-45 ms); --decode-omp-spin 1 (-20%, spread halved); narrow-output
  GEMV split (bench +38-48% on small shapes, end-to-end break-even within 16-30% thermal noise).
- Profile ~0.211: GDN 80 ms (floor 52), routed 36 (floor 10), GR ~34-54 (floor 16), QSA 29 (floor 15),
  lm_head 19 (at roof), router+shared 13. Total floor ~117 ms (9.3 GB/token @ 79 GB/s) -> ~2x left.
- MEASUREMENT: host thermal noise now 16-30% spread run-to-run; use microbenches + longer series (P1).

## 2026-09-22 PAUSED (user request) — resume from docs/ACTIVE_WORK_LOG.md's last entry
- O5 phase 1 MERGED (ef2e66c): native-quant backbone kernels correct but compute-bound (K-quant
  unpack 1-5 GB/s/thread; loses to bf16 gemv::axpy at 1 thread). Fit win real: 3.59 vs 9.15 GiB.
  Phase-2 gate: >= 10 GB/s compressed/thread, DRAM-streamed. docs/BACKBONE_NATIVE_QUANT.md S8a.
- O4 agent DONE, NOT merged (branch worktree-agent-ac92618df53668765 @ 42f82a0): GDN threaded over
  heads/channels (GDN 85->72 ms), MoE schedule(dynamic) (38->35 ms). Diff reviewed OK; needs rebuild +
  gates + A/B + O4 doc before merge. Lever 3 was reverted, not parked (S13 gap).
- O5 phase 2a Sonnet agent asked to commit WIP and stop; its hand-off names the branch.
- Local main NOT pushed (20+ ahead of origin).

## 2026-09-22 later — O4 MERGED (dafd846 + 425929c): decode 0.236 -> 0.211 s/token
- Lever 1 (GDN threaded over channels/k-heads/v-heads + b/a via gemv::axpy) is the whole win:
  GDN phase 90.0 -> 73.3 ms (-18.5%), reproduced in my own interleaved counterbalanced A/B.
- Lever 2 (MoE schedule(dynamic)) did NOT reproduce: -2.4% at 8 threads, +3.3% at 10, both inside
  the phase's spread. Kept (bit-exact, principled) but documented as weak evidence -- do not cite
  the agent's -8%.
- 8 vs 10 --moe-decode-threads now perform the same.
- Profile at 0.211: GDN 73, routed experts 46, QSA 32, GR 31, lm_head 21. Floor ~117 ms (bf16).
- Lesson reinforced: a delegated agent's phase-isolated numbers reproduced; its end-to-end and
  second-lever numbers did not. Re-measure each lever separately at the flags actually recommended.

## 2026-09-23 — O5 state; Codex is co-driving O5 on main
- Codex (commits authored as Craig) integrated phase 2b-1 sidecar (3442dbf+44dade3), kernel pass 4
  (1d0f752, per-256 ActSuper: faster but higher activation error, unmeasured end-to-end), and wired
  2b-2a (325e67b: TokEmb gather + Q4_K lm_head behind --backbone-quant-dot, .bbq.pair full-file hash
  pairing). Baselines on main after it: engine 29,510,661/147, frontend 148,922/285.
- Codex's "L2 0.162039" is forward-vs-forward_one with ONLY head+embd native — not comparable to the
  O-series 0.23252 (which is the MoE-quant-dot divergence). Always state which flags an L2 came from.
- BIG GAP: 2b-1 EXCLUDED the three GDN projections (vperm), which are 49.7% of backbone bytes/token
  (4.15 of 8.35 GB). Recoverable without re-blocking: in-proj = output-row permutation (copy rows at
  write); out-proj = input-axis permutation (gather the 6144-float activation before quantizing).
  Claimed as 2b-2b (row 950c7de), delegated with phases A (sidecar) / B (wire all roles) / C (gates+A/B).
- Live paired artifacts (.bin/.moeq/.bbq/.bbq.pair) are in agent-af27db7f3602819b1's worktree
  out/full48 — Codex's runs use them; do NOT delete that worktree until they are relocated.
- Codex commits to main concurrently: stage only your own hunks (git hash-object + update-index
  --cacheinfo trick for ACTIVE_WORK_LOG.md), never git add -A.

## 2026-09-23 — speculation / n-gram / MoE design (docs/SPECULATION_NGRAM_MOE_DESIGN.md, 0d5778a)
- READ FIRST before citing any Sub0Llm-vs-llama.cpp quality comparison: llama.cpp runs the n-gram/PLE
  table BY DEFAULT (lazy mmap, AUTO mode); our engine omits it. Unless PLE was disabled on the llama.cpp
  side, those comparisons compared different models.
- The engine's shipped n-gram stage (concat/additive) is NOT Qwen4's real PLE fusion (gated key/value +
  dilated causal conv, llama.cpp build_ple). Enabling PLE = building the gated variant.
- MTP: absent from the local GGUF; in the HF checkpoint it is a full QSA layer with its own 512-expert
  MoE; only vLLM's mtp.py implements its forward (transformers drops mtp.* on load).
- Order agreed in the doc: O5 -> measurements E1/E2/E3 -> PLE fidelity -> O7 expert kernels (compute-
  bound ~5 GB/s) -> batched verify path -> prompt-lookup -> MTP. Speculation ceiling ~1.2-1.5x on this
  CPU because routed-expert compute does not amortize across verified tokens.

## 2026-09-25 — O5 2b-2b merged (d2e6f80): FIRST valid native-backbone number
- READ FIRST: any decode-only (forward_one-only) data path that is memory-MAPPED gets first-touch
  soft faults INSIDE the timed decode, because the tool's forward() pre-warms only what forward() reads.
  The native .bbq sidecar hid its whole GDN gain this way (551k vs 82k decode faults). Fixed by
  bbq::Store::prefault() at load. Check [mem] page faults per arm before trusting any A/B that adds a
  new mapped resource.
- Valid A/B at the real flags (incl. --moe-quant-dot 1!): 0.218 -> 0.202 s/token (-7.3%, 6/6 rounds);
  GDN -25%, lm_head -46%; but every UNTOUCHED bf16 phase +5-13% (open: clocks? dual residency? TLB).
  L2 0.2325 -> 0.2788 (argmax 4/6 both). --backbone-quant-dot stays opt-in.
- Delegated agents have twice built A/B arms without --moe-quant-dot (arms ~1.0 s/token). Sanity-check
  that the baseline arm reproduces the known ~0.19-0.22 s/token before reading any delta.
- Host noise rose after a mains outage (default arm 0.189 on 09-23 vs ~0.218 on 09-25): compare within
  a run only.
- Integrate in a separate worktree when a long A/B is needed: Codex committed onto integrate/o5-2b-2b
  while it was checked out in the shared tree (harmless that time; ff delivered it to main).
- Hardlinked model dirs (same .bin/.moeq, different .bbq) let sidecar variants share one page cache;
  pointing an arm at a byte-identical COPY of the 37 GB .moeq thrashes RAM and voids the run.

## 2026-09-26 — O5 2b-3 merged (bccd72a): every native role wired, ~6.1 tok/s opt-in
- default 4.24 / head+GDN 5.43 / all roles 6.10 tok/s (clean rounds 6.1-6.3). GR -33%, QSA -17%,
  router+shared -32%. L2 0.2926, argmax 3/6 (first argmax drop) -> stays opt-in until llama.cpp oracle.
- Remaining native-path levers: QSA q|gate is 48 per-head gemv_plane<8> calls/layer (merge into one GEMV
  + scatter); drop bf16 copies of covered roles (memory); quality oracle.
- Delegated agents keep hitting the session rate limit mid-run: bank their worktree state with a WIP
  commit, then finish phase C (gates/A/B) myself -- faster than resuming them.

## 2026-09-26 — O8 row-split merged (0179e9e): +11% long-run, bit-exact
- Long-run (2,418-token --stage ppl) decode: 5.16 -> 5.71 tok/s, both arm orders. A 3-token perf run
  said -7% with 24-43% spread: short runs can't resolve ~10% effects; use long runs for tok/s claims.
- Open: thread sweep with row-split (12/16 threads incl. E-cores); libomp blocktime INT_MAX overflow warning.
- RESOLVED 2026-09-26: the libomp blocktime 'overflow' was a libomp message bug; INT_MAX clamps to INT_MAX/1000 ms (verified by kmp_get_blocktime). Now set directly, no warning (main).

## 2026-09-29 — O9 merged: per-256 ActSuper backbone kernels (`--backbone-act-super 1`)
G-PPL PASS (quality-neutral, dNLL -0.0098 CI -0.032..+0.012). Long-run 5.72 -> 6.53 tok/s (+14%, my passes; the
subagent's +28-33% did not reproduce — its super arm read ~7.1 vs my 6.15/6.91). Warm phase table: GDN 47 -> 36 ms.
Lesson: re-measure a subagent's speed-up yourself before quoting it; quality (deterministic) reproduced bit-for-bit.
Next levers: VNNI variant of the super kernels (parked, S14d); routed experts ~28 ms; GR 21 ms (Q8_0, untouched by O9).

## 2026-09-29 (late) — state at host reboot (main bd10548, all committed, not pushed)
- O10 profiling merged (18 sub-phases, compile away). O11 merged: threaded QSA indexer projection, 9.45 -> ~1 ms/token, bit-exact.
- O12 (fast Q8_0 kernel, GR + shared down) merged DEFAULT-OFF. G-PPL PASS on v2 (dNLL +0.0056, CI -0.0041..+0.0153).
  Only the agent's +6-7% long-run supports the speed. NEXT: my own long-run A/B, then flip `resolve(backbone_q8_fast, false)` -> `backbone_quant_dot != 0`.
- G-PPL fixture v2 (9,631 tokens) is now the default; v1's ~0.02 CI half-width was rounding noise (O12 control arm: ppl 14.70 -> 14.90 from summation order alone).
  O9's v1 "-0.0098" was likely noise too; its re-check on v2 (arm `--backbone-act-super 0`) was lost to the reboot, so rerun it.
- Next levers (O10 doc): thread the small serial ops (~4-5 ms); the K-quant plateau at ~53 GB/s vs the 79 GB/s roof.

## 2026-10-07 — O12 default ON; O9 re-verified on v2; convergence with the cache track (main c06e3f8, not pushed)
- 3-arm v2 run (9,631 tokens, reactive): base 11.2956 ppl; O12 on +0.0056 (CI -0.0041..+0.0153) PASS, ~+4% long-run,
  GR up 8.1 -> 6.8 ms, router+shared 9.0 -> 8.0; O9 on vs off +0.0019 (CI -0.0086..+0.0124) PASS, +18% (6.24 -> 7.35 tok/s).
  O12 promoted in `resolve_decode_defaults`. Frontend gate now 231,180 / 304.
- Suite: `--stage ppl` stages all arms first; `--ppl-speed-rounds N` (rotated long-run tok/s); `--ballast-room GIB`
  (regime 2: locked ballast + evict before every run; reports cache hit/waits/peak/residency).
- New `prof::Phase::MoeMissWait` row (0 under reactive). Fixed B38 test slot aliasing exposed by cbb89dd.
- O13 doc (`docs/optimization/opportunities/O13_storage_convergence.md`) ranks shared work; trace replay:
  17 GiB cache = 23.1 misses/token, 26 GiB = 7.4; recency predicts 0.0% of misses.
- IN FLIGHT at session end: regime-2 A/B (reactive vs cache17) queued behind another session's Sub0Pub compile
  benchmark; Sonnet agent on O14 (mapped parameter arena, `--param-arena mapped`, auto OFF until I re-measure).
- Other Claude sessions run timing benchmarks on this host: check for foreign clang/python jobs, not just this repo's log.
