---
name: wp4-handover-2026-09-07
description: "READ THIS FIRST for Sub0Llm/Qwen4-preview work -- current-state snapshot for a fresh session, pointing into the detailed narrative history rather than replacing it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-09T17:37:13.267Z
---

**Purpose of this file**: a fresh agent session (no access to the conversation that produced this) needs
to be productive on the Sub0Llm Qwen4-preview work immediately, without replaying the whole chronological
history in [[pivot-to-existing-model-support-qwen4-preview]] (still the source of truth for HOW and WHY
every decision was made — read it for depth, read this for orientation). Everything below was true as of
**2026-09-08, `main` at commit `b855451`**.

## Where things stand

**WP4a through f are ALL DONE and merged.** Real Qwen3.8-Flash-Next weights (a 4-layer sub-stack: 3 GDN +
1 QSA, the real repeating unit) are transplanted, loaded, and run through this engine's own
`Model::forward`. **The real cross-comparison against llama.cpp was run, found a genuine bug, and the bug
was found, fixed, and independently verified.** See [[pivot-to-existing-model-support-qwen4-preview]] for
the full chronological detail and `docs/WP4_SCOPE.md` in the repo for every stage's own "EXECUTED"
section with exact numbers — §6 WP4f's own section has the complete divergence/root-cause/fix writeup.

## The headline result (no longer a loose end — read this, don't re-derive it)

The first real cross-comparison (2026-09-07) found `blk.0.out` diverging at `rel_l2 = 0.996` — starting
at layer 0 itself, not merely at the PLE boundary the plan expected. Root-caused and fixed the same day:

**`include/sub0/transplant.hpp` was reading raw GGUF bytes as if they were the HF checkpoint verbatim.
They are not** — llama.cpp's own converter (`conversion/qwen.py`'s `Qwen3NextModel.modify_tensors`/
`_LinearAttentionVReorderBase.modify_tensors`, inherited by `conversion/qwen4exp.py`) rewrites values and
head order on the way in. Three specific rewrites, silently inherited by every WP4c/d/e artifact until
this fix: (1) every `Qwen4ExpTextRMSNorm` gain pre-folded to `1+w` (Sub0Llm's own norms ALSO add the 1,
so verbatim reading double-applied it); (2) `ssm_a` stored as the finished `-exp(A_log)`, not `A_log`
itself (`gdn_math.hpp` exponentiates again); (3) GDN's value heads reordered grouped→tiled (a pure
PERMUTATION, provably invisible to the existing statistical-comparison gate — the same blind spot
[[independent-reimplementation-catches-identity-swap-bugs]] names).

Fixed with explicit inverses in `transplant.hpp` (`fold_for`/`apply_fold`, `vperm_for`/
`ungroup_v_heads`), merged as `296f2a1`. Result: `blk.0.out`'s `rel_l2` improved **0.996 → 0.0221** (a
~45x reduction). **Still fails the strict `1e-4` gate — and this remainder was PROVEN, not assumed, to be
llama.cpp's own numerical behavior, not a Sub0Llm defect**: an independent NumPy float64 reimplementation
of the whole of layer 0, from the same dequantized weights, agrees with Sub0Llm to **1.1e-06–3.6e-06**
but with llama.cpp to only **1.2e-02–3.2e-02**. Isolated directly: an F32-weight projection from the same
input agrees with llama.cpp to 2.0e-07; a Q8_0-weight one to only 3.5e-03 — perfect correlation with
weight quantization tier, zero with mechanism. **llama.cpp's own kernels quantize activations, not just
weights** — a real, deliberate performance choice on its side. Layers 1-3 remain incomparable by design
(the PLE injection at layer 1) and this was also measured directly: llama.cpp's own PLE contribution is
68.0% of the layer-1 residual stream's norm.

**The actual headline, worth restating**: with the converter's conventions correctly undone, Sub0Llm
computes real decoder layer 0 to **2.6e-06 relative against an independent float64 reference** — a
stronger correctness statement than any comparison against llama.cpp itself could produce at this
quantization tier.

## 2026-09-08 update: new goal (interactive inference), WP5a/b dispatched, backend reorg by a third agent

**New governing goal, beyond the original WP1-4 plan**: user directed prioritizing actual interactive
inference with the real transplanted Qwen4 model (type a prompt, get generated text). Two parallel
prerequisites dispatched as WP5a (real byte-level BPE tokenizer, gated against `transformers`'s real
`vocab.json`/`merges.txt`, new file `include/sub0/qwen_tokenizer.hpp`, branch
`feature/wp5a-qwen-tokenizer`) and WP5b (scale the transplant to the full 48-layer model, `mmap` the MoE
sidecar since it's ~38GiB even quantized-resident, branch `feature/wp5b-full-scale-transplant`). A third
piece (sampling + a generation CLI loop) is deliberately deferred until both land.

Both agents hit the Opus rate limit mid-task, then were resumed once quota renewed. **Meanwhile a THIRD
concurrent agent appeared** ("GitHub Copilot", per `docs/ACTIVE_WORK_LOG.md`) doing a mechanical CPU/CUDA
backend reorganization: `src/backend_cpu.cpp` no longer exists — its content is now split across
`src/backends/cpu/api.cpp`, `api.hpp`, `backend.cpp` (`src/backends/cuda/backend.cu` similarly), with a
new `cmake/Backends.cmake` and per-backend CMakeLists.txt files, across 11 commits. **Verified this did
not lose Codex's in-flight uncommitted Muon (B14) work**: `src/backends/cpu/backend.cpp` already contains
the Muon scratch-reuse code (`g_muon_scratch`, `MUON_SCRATCH_FLOATS`) as committed content, and
`include/sub0/muon.hpp`'s own uncommitted diff is byte-identical to before the reorg — nothing dropped.
WP5a/b were briefed about the new file layout before resuming so their eventual merges don't assume the
old monolithic `src/backend_cpu.cpp`/top-level-only `CMakeLists.txt` paths. **User's workflow decision,
going forward: Codex is now used as a delegation agent for complex tasks** (currently on Intel iGPU
backend design), not just independent review — a role shift worth remembering for future dispatch
decisions.

**WP5b DONE, merged `e1bea23`.** Also fixed, same session, before the merge: `main` didn't compile at all
(`cb85d5a`) — three mechanical breaks left by the backend split (`PARENT_SCOPE` no-op on two source-
manifest variables since the whole include chain is `include()` not `add_subdirectory()`; an ADL
ambiguity at three internal `cpu_detail::` call sites in `backend.cpp` against the outer `SUB0_API sub0::`
declarations; `trainable_floats()` missing from the `api.hpp`/`api.cpp` facade). WP5b's own branch found
and fixed the SAME `PARENT_SCOPE` bug independently (`ad2c818`, byte-identical diagnosis) — a genuine
reassuring corroboration, resolved as a trivial merge conflict (kept the more detailed comment).

**WP5b headline**: the full 48-layer real Qwen4 model now loads and runs end-to-end in this engine.
`moeq::Store` was upgraded from an eager read to an `mmap` (new `include/sub0/file_map.hpp`, gated
bit-for-bit against an independently-reimplemented eager reader at three levels: raw bytes, dequantized
plane, and `moe::expert_ffn_row` output — the engine's only real consumer). Transplant produced
`D:\ModelWeights\Sub0Llm-Qwen4-full48\` (18.31 GiB backbone + 37.11 GiB sidecar = 55.42 GiB, 162.8s,
every gate green incl. 0/1,074 bit-for-bit `--verify`). Loads in 19.0s; `Model::forward` on 6 tokens in
50.35s, zero non-finite, `forward`/`forward_one` parity exactly 0; **peak working set 35.51 GiB of 63.43
GiB total** — the mmap was a real prerequisite, not a nicety: eager-reading the same run needs ≥69.45
GiB, more than total physical RAM, not just more than free RAM. **Verdict: memory/disk feasibility is
settled. Throughput is not** — ≈8.2 s/token (12× the 4-layer figure, linear in depth, on top of WP4e's
already-measured 3.7× dequantize-on-demand cost) — that's the next open question for real interactive
use, not correctness. Independently reverified before merge: rebuilt in an isolated worktree, hand-
checked `full48_totals.hpp`'s census arithmetic myself (NUM_PARAMS 74,802/PARAM_FLOATS 125,711,062,400
all-f32 → 1,074/4,915,107,200 quantized-resident, both matched my own computation exactly), reran the
test suites and reproduced every number. Full writeup: `docs/WP4_SCOPE.md` §6 "WP5b".

**WP5a DONE, merged `bb1fc70`.** A from-scratch reimplementation of Qwen3.8-Flash-Next's real byte-level
BPE tokenizer (`sub0::qwen_tok`, new `include/sub0/qwen_tokenizer.hpp`+`qwen_unicode.hpp`+
`qwen_unicode_tables.hpp`+`src/qwen_tokenizer.cpp`; deliberately does not touch this project's own
`tokenizer.hpp`/`casing.hpp`). Reads the real `vocab.json`/`merges.txt`/`tokenizer_config.json` at
runtime. The two hard obstacles: (1) `std::regex` supports neither Unicode property escapes nor the
negative lookahead the real pre-tokenization regex needs, solved by hand-transliterating each
alternative into its own leftmost-first matcher function, verified not asserted — character classes
swept over all 1,112,064 non-surrogate codepoints against the real reference engine, 0 disagreements;
(2) NFC (needed, confirmed three ways) implemented from UAX #15 (decompose/canonical-sort/compose),
verified against every codepoint's decomposition/composition/Hangul-syllable case.

**Headline finding, independently reproduced by me (not just the branch's own tests)**: transformers'
`AutoTokenizer` is the WRONG oracle for this model. `tokenizer_config.json`'s `tokenizer_class:
"Qwen2Tokenizer"` sends it through a class whose hardcoded `PRETOKENIZE_REGEX` lacks `\p{M}` (combining
marks), silently mis-tokenizing every Thai/Devanagari/vocalised-Arabic/pointed-Hebrew prompt with every
resulting id still valid — undetectable by inspection. I reproduced this myself directly against the
real HF-cached `tokenizer.json` and `AutoTokenizer.from_pretrained`: Thai `"กัน"` → `AutoTokenizer`
`[24400, 64020]` vs. the model's own `tokenizer.json` pipeline (this branch's real oracle) `[148783]` —
matched the branch's reported numbers exactly, both sides. The right oracle is `Tokenizer.from_file` on
the model's own `tokenizer.json`.

Test numbers, independently reproduced: 3041/9 with the real files present (I pointed
`SUB0_QWEN_TOKENIZER_DIR` at the local HF cache,
`C:\Users\craig\.cache\huggingface\hub\models--Qwen--Qwen3.8-Flash-Next\snapshots\...`), 2473/9 with
them absent (three file-gated cases correctly skip rather than silently passing), full suite
120,889/244 confirming the WP5b baseline (118,416/235) is untouched.

**WP5c DONE, merged `de7d803`. ALL THREE prerequisites for real interactive inference are now complete,
merged, and independently reverified.** `tools/sub0llm-qwen4-gen.cpp`: real tokenizer → real transplanted
48-layer weights → real sampling → real detokenization, wired end to end for the first time. Pure wiring
(no new math) over three already-verified pieces, so the review/gate shape was different — composition
checks (round trip, determinism, decode-prefix invariant, the stop condition) rather than another
numerical gate.

**THE HEADLINE, independently reproduced by me from a clean rebuild (own `sub0llm-configure` run at the
real 48-layer axes)**: prompt `"The capital of France is"` (ids `[760, 6511, 314, 9338, 369]`), seed
1234/temp 0.8/topk 40 → `" Paris. How many countries have capitals with"` (ids `[11751, 13, 2500, 1599,
5656, 599, 89798, 440]`) — matched the agent's report exactly, both the text and every id, on a binary I
built myself from the merged source. This is the first evidence in the entire work stream that the
transplanted weights are a *functioning* model, not just a well-formed pile of tensors — every prior
WP4/WP5a/WP5b gate checked shapes, statistics, or one layer's hidden state. Also independently confirmed:
determinism (same seed twice → identical ids) and that the seed is genuinely wired (seed 99 diverges to
`[11751, 13, 1061, 3254, 421, 11751, 369, 303]` after the first two tokens). Real cost: ~10s/token
prefill, ~9.65s/token generation, 37.11 GiB peak — throughput, not correctness, is what stands between
this and actual interactive use.

**A real, reported-not-fixed defect found along the way**: `sub0::sample_token` (`src/engine_core.cpp`)
holds two `std::array<float, VOCAB>` plus one `std::array<int, VOCAB>` as ordinary stack locals — 2.84
MiB in one frame at the real VOCAB=248,320, against Windows' 1 MiB default stack. Nothing had ever hit
this before because no real-axes consumer had ever *sampled* (WP4d/e/f and WP5b all stop at the logits).
Measured directly: relinked without the workaround, the tool dies at the first `sample_token` call with
`STATUS_STACK_OVERFLOW`. Worked around at the new tool's own link line (`/STACK:33554432`); the real fix
(reused `thread_local` scratch) is separate, cross-cutting engine work, not done here.

**Also found**: the shared working tree (`D:\Craig\GitHub\Sub0Llm`, not a `.claude/worktrees/*` one) is
currently checked out to `feature/wp5c-qwen-generation` (not `main`) with another agent's uncommitted
`AGENTS.md` edit sitting in it (a refinement of a "delegate research to Terra" note — benign, not mine,
deliberately untouched). `main`'s own branch ref was never at risk — every WP5c merge/build/verify this
session was done in throwaway worktrees specifically to avoid that dirty shared state. Logged in
`docs/ACTIVE_WORK_LOG.md` for whoever next uses that tree to notice and reconcile.

**Where this leaves the project**: the "prioritize interactive inference" goal from earlier this session
is now substantively met — a real prompt through the real transplanted Qwen3.8-Flash-Next model produces
real text, deterministically, with a working stop condition. What's left is performance (current
~10s/token is nowhere near interactive), and optionally a chat template layer (deliberately out of scope
for WP5c — raw prompts only, no `<|im_start|>` wrapping) and the `sample_token` stack-usage fix named
above.

## Cross-agent coordination (originally 2026-09-07, still in force)

**Read `docs/ACTIVE_WORK_LOG.md` before touching any shared file or running a CPU-heavy build/benchmark.**
As of 2026-09-07 the user decided both Claude Code and a separate agent (Codex, doing independent-review
work via `docs/INDEPENDENT_REVIEW_BACKLOG.md`) share ONE working tree (not separate git worktrees), for
efficient coordination. This makes a few things load-bearing, not just hygiene:

- **Never `git add -A`/`-u` or any blanket-stage operation.** Always stage by explicit path. Check
  `git status --short` before AND after every commit. The other agent's uncommitted edits can be present
  in the SAME working tree at any time.
- **Never `git checkout --`/`git stash`/`git reset` across a file you didn't personally add to the log.**
- Before merging a branch into this shared `main`, confirm via
  `git diff <merge-base> <branch> -- <the other agent's known-hot files>` that your branch's own commits
  never touch them — if empty, the merge is safe regardless of what's uncommitted in the working tree
  (git only applies the incoming branch's actual diff).
- `docs/INDEPENDENT_REVIEW_BACKLOG.md` (written by the other agent) is a genuinely thorough, well-scoped
  independent review of the whole repo (correctness, testing, build-hygiene, performance) — worth reading
  if you need work to do that isn't Qwen4/WP4. `docs/ACTIVE_WORK_LOG.md` already cross-checks its backlog
  items against WP4's own hot files and names which are safe to pick up right now.
- As of this handover: Codex has completed Muon optimizer work (B14, results in
  `docs/MUON_CPU_OPTIMIZATION.md`, uncommitted — do not disturb `include/sub0/muon.hpp`/
  `src/backend_cpu.cpp`/etc. until it's committed) and moved to Intel iGPU backend research (docs-only,
  no engine edits). Check `docs/ACTIVE_WORK_LOG.md`'s own Log table for the current, authoritative state
  — this paragraph will go stale before that file does.

## Other named follow-ups, not urgent, not forgotten

- **`D:\tools\llamacpp\Init.ps1`** is this ecosystem's own proper, maintained way to acquire/update/build
  llama.cpp. The ad-hoc clone at `D:\Craig\llama.cpp-qwen4exp` (used for all of WP4f, including the
  divergence fix's own re-verification) was legitimate work but should be consolidated through `Init.ps1`
  for any FUTURE llama.cpp work. Not done yet.
- **Whether the MoE sidecar (WP4e) should hold GGML's own quantized bytes verbatim (current) vs. a
  Sub0Llm-native re-encode** — reasoned through in `docs/WP4_SCOPE.md`'s WP4e section: re-encoding can't
  recover lost fidelity, only trade decode speed against storage size. Recommended order: profile the
  actual `resolve()` cost (VTune) before investing in a new format. Not started.
- **Layer-3 (QSA) end-to-end validation** — WP4f's dump DOES capture the QSA layer via the real engine
  path (unlike WP4d's own standalone replay, which can't reach `op_qsa`'s internal rope table), so this
  may already be partially closed as a side effect of the divergence fix — not explicitly re-checked.
- **`op_rmsnorm`'s eps is still `1e-5`** (real model value `1e-6`) for every NON-Gated-Residual build —
  the `LnF` fix only closed this gap for GR's own exposure.
- ~~Regenerating the transplant artifacts and re-running the real cross-comparison~~ — **DONE, 2026-09-08**:
  independently rebuilt `sub0llm-hidden-diff` and `sub0llm-transplant-q`, re-ran both against the actual
  files on disk, reproduced `blk.0.out`'s `rel_l2=2.211e-02`/cosine `0.999810` exactly and got 0 bit-for-
  bit mismatches on `--verify`. No longer an open item.

## Reproducing the cross-comparison yourself

Everything needed is on local disk, nothing needs re-downloading:

- Real GGUF source: `D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S\` (3 shards, 72.55GB total).
- Sub0Llm transplant artifacts: `D:\ModelWeights\Sub0Llm-Qwen4-sub4\` — **regenerate these post-fix**
  before trusting them for anything beyond what's already recorded in `docs/WP4_SCOPE.md` §6 WP4f, since
  the merging session did not personally regenerate the 43GB/9GB files (see follow-up above). Prefer the
  quantized-resident form (`qwen4_sub4_q.bin`+`.moeq`, ~9GB, ~14GB peak RAM) over the all-f32 one (~43GB,
  ~45GB peak RAM) — bitwise-identical output, far less memory pressure.
- llama.cpp build: `D:\Craig\llama.cpp-qwen4exp\build\bin\` (upstream commit `ccc3646`, CPU-only,
  clang++/Ninja) — `llama-qwen4-dump.exe` is the custom dump tool
  (`D:\Craig\llama.cpp-qwen4exp\examples\qwen4-dump\qwen4-dump.cpp`, ~250 lines, touches no llama.cpp
  model/architecture code) that accepts explicit token IDs and dumps named graph tensors.
- Canonical token array (both sides must use this exact one): `1543, 88123, 245000, 7, 99999, 156789` —
  6 raw ids, no BOS, no tokenizer.
- The diff tool: build `sub0llm-hidden-diff` from `tools/sub0llm-hidden-diff.cpp` (links no engine,
  builds in any existing build directory) and run
  `sub0llm-hidden-diff --a <sub0llm>.s0hd --b <llamacpp>.s0hd --gate 0.0001`.
- The ggml-dump-to-S0HD conversion script (llama.cpp's raw tensor dumps → S0HD) was a one-off Python
  script, deliberately not committed — rewrite it small if needed (~70 lines): read `manifest.tsv`, for
  each wanted ggml tensor name read its raw little-endian f32 file, reshape per the axis rule below,
  write S0HD (format documented in full in `include/sub0/hidden_dump.hpp`'s own header comment).
  **The axis rule, independently verified**: ggml's `ne[0]`-fastest-varying tensors of shape
  `[D_MODEL, HC_COUNT, T, 1]` (llama.cpp's `hc_init`/`l_last-N`), read as raw bytes in file order, are
  EXACTLY row-major `[T, HC_COUNT*D_MODEL]` with the same stream-major/channel-minor ordering this
  project's own `wide = hc_count*hidden_size` convention already uses — no transpose needed.
  `model.input_embed` is `[D_MODEL, T, 1, 1]` (2 real dims) — `rows=ne1(T), cols=ne0(D_MODEL)`, not the
  general rule.

## Repo/environment hygiene, as of this handover

- `main` at `b855451`, working tree has Codex's uncommitted Muon/Intel-iGPU-research files present (see
  the coordination section above) plus one PRE-EXISTING, unrelated, not-anyone's-active-work edit to
  `docs/TOKENIZER_V2_IDEAS.md` that has been present the whole thread — leave both alone.
- No live worktrees or branches from the divergence investigation remain — cleaned up after merging.
- `data/tokenizer_calibration.txt` (a git-LFS pointer) has been accidentally corrupted by build/test
  activity multiple times this whole thread — always caught via `git status --short` before committing
  and reverted with `git checkout -- data/tokenizer_calibration.txt`. Check this reflexively after any
  build/test session before committing anything.
- This machine: 63.4GB RAM, typically ~48GB free. Don't run multiple heavy processes concurrently, and
  now also check `docs/ACTIVE_WORK_LOG.md` before running anything CPU-heavy at all.

## Standing process, unchanged, still expected

Every WP4 stage was dispatched to an Opus subagent with a thorough brief, then **independently reviewed
and reverified before merging** — never merged on trust. That included: reading the actual diff,
hand-verifying the highest-risk algebra/logic (not just re-reading the agent's own claims), spot-checking
against known-correct references, and rebuilding+rerunning the regression suite myself before every
merge. Continue that discipline for whatever comes next.

## 2026-09-08/09 update: a real performance review, real VTune data, and the CPU decode path is now
## split cleanly for the fix that data points to. `main` at `db3594b` as of this update.

**All three WP5 prerequisites (tokenizer, full-scale model, generation loop) landed and were merged
earlier in this thread — see the section above this one for that detail.** What's new since:

- **A performance review artifact** ("The Ten-Second Token") was published analyzing the CPU roofline
  (this workload is memory-bandwidth-bound, not compute-bound — arithmetic intensity 0.5 FLOP/byte vs.
  an estimated 3-6 FLOP/byte ridge point) and a CPU/CUDA/Intel-iGPU synthetic-target matrix. Its
  non-obvious finding: the Intel iGPU shares the CPU's own DDR5 channel (UMA), so unlike a discrete GPU
  it does NOT open a faster memory pipe for this workload — any win there would have to come from access
  pattern or overhead, not bandwidth.
- **A real VTune session against the live decode loop** (never done before this) found the actual
  answer, not a guess: `transplant::transpose_out_in` — a naive read-contiguous/write-strided nested
  loop, written for the offline transplant tool's one-shot use and reused unmodified in the per-token
  MoE resolve hot path — is **52.5% of all sampled CPU time**, more than the three format decoders it
  wraps combined. Second finding in the same session: decode runs on **exactly 1 of 24 cores** the whole
  time; `backend.cpp` has zero `#pragma omp parallel` anywhere near `forward_one`. Both are now
  `docs/INDEPENDENT_REVIEW_BACKLOG.md` **B20**, with a real Work/Done-when contract. **B20's actual fix
  is NOT done yet** — this is the next concrete piece of work.
- **B19** (Copilot's implementation, reviewed and shipped by Claude Code — the session's first case of
  reviewing and merging another agent's own work rather than dispatching its own): `sub0llm-configure
  --vocab-exact N` skips the corpus-driven vocabulary learner entirely for an externally-defined
  vocabulary (WP5a's real tokenizer). Real measured win: **0.033s vs ~120s** for the real Qwen4 axes,
  confirmed byte-identical generated config to the corpus-driven path (bar the three intentionally-empty
  tokenizer paths). Merged `a2ce70b`.
- **I08 (decode/API slice)**: `src/backends/cpu/backend.cpp` (3,146 lines, mixed decode with the entire
  training/backward/optimizer machinery) split into `backend.cpp` (training/Node-graph, ~2,170 lines),
  new `decode.cpp` (607 lines — `forward_one`, the KV/GDN/QSA caches, the per-row kernels), and new
  `internal.hpp` (546 lines, backend-private — only what both TUs must share as ONE instance: the
  `Worker` pool, `moe_resolve` kept as a single shared inline definition, `Model`). **Zero behavior
  change, verified both by the dispatched agent and independently by Claude Code from a from-scratch
  rebuild**: the real 48-layer WP5c determinism fixture (prompt "The capital of France is", seed 1234)
  reproduces bit-for-bit, all 30 ids and the full continuation text, and the neutral-config regression
  suites show 0 failures with matching counts. A genuinely good piece of science along the way: an
  apparent 15-20% latency regression from a single before/after pair was NOT waved off — a proper
  counterbalanced interleaved A/B (12 runs, DLL swapped mid-session, order reversed for half) showed
  post-split is if anything marginally faster, and the "regression" range sits entirely inside the
  pre-split code's own measurement spread. Merged `cb9e2b1`. **This unblocks B20**: its fixes now land
  in a 607-line file, not the 3,146-line monolith.
- **B20 is now DONE too, merged `9c1c7e3`.** Part 1 (cache-blocked transpose, measured tile=16 not
  guessed): 52.9%→21.5% of CPU time, **9.14→5.68 s/token (1.61x)** — a real, substantial win,
  independently reproduced (I measured 5.57-6.00 s/token myself on a from-scratch rebuild). Part 2
  (parallelize per-expert resolve across threads, `Total Thread Count` 1→10): only ~2% net — because
  **it exposed that decode is now DISK-bound, not CPU-bound**. Measured: ~13,610 hard page faults/sec,
  ~56 MB/s matching physical disk reads, no warm-up trend over 30 tokens — the 37.11 GiB S0Q1 MoE
  sidecar mapping and the 18.31 GiB f32 backbone don't both stay resident in 63.4 GiB of RAM, so nearly
  every routed-expert read is a demand-paged disk read. **This is the new headline finding and the real
  next item** — not a CPU optimization opportunity, a memory-residency one (the sidecar's own mapping
  strategy, or reducing what needs to be resident at once). Both parts verified bit-for-bit throughout
  (transpose is provably lossless — a pure permutation, no arithmetic, verified by hand algebraically
  myself for all three call sites before trusting it; the parallel resolve's weighted-sum accumulation
  order is preserved exactly via a two-phase compute-then-combine restructuring, also hand-verified).
  A real defect found and fixed along the way: decode's new worker threads never called
  `set_flush_denormals` (FTZ/DAZ is per-thread MXCSR state) — would have made the bit-exactness claim
  silently false the first time an intermediate went subnormal on a thread that missed it.
  Independently reproduced before merging: `--verify` 0/1074, `forward`/`forward_one` parity 0 exactly,
  the WP5c determinism fixture byte-for-byte, neutral suites unchanged.
- **2026-09-09, same-day follow-up: a memory audit + isolated benchmark, done+merged `23ec79e`.** The
  user caught something real — after B20's "decode is disk-bound" finding, they pointed out 47GB was
  free at idle, which should be ample over the 37GB sidecar. Direct live measurement (Get-Counter
  during an actual run) showed why the idle figure was misleading: `Available MBytes` collapsed to
  ~6.8GB once the process's own working set reached its ~41GB peak — well short of the 55.42GB
  (backbone+full sidecar) needed to avoid any eviction. That led to a proper memory audit: `Worker::
  act_grad` (the per-ACTIVATION gradient arena, `src/backends/cpu/internal.hpp`) was STILL
  unconditionally sized at `ACT_CAP` regardless of `FORWARD_ONLY` — the exact same waste
  `WORKER_GRAD_FLOATS` already fixed for the per-PARAMETER gradient, just never applied to this
  separate array. **7.02 GiB of pure dead weight per Worker**, confirmed via a real
  `VirtualQueryEx`/`QueryWorkingSetEx` process walk cross-checked against a probe TU, both agreeing to
  the byte. Fixed with the identical `WORKER_GRAD_FLOATS` pattern. Real result: peak working set
  **40.98→33.96 GiB (-17.1%)**, and the freed pages went straight to the sidecar's own resident share
  (4.81→7.63 GiB at a comparable point in the same run) — confirming the hypothesis directly. A new
  standalone benchmark (`benchmarks/moe_expert_bench.cpp`) also decomposed one expert's resolve into
  its four real arms for the first time: **~38% page-in, ~46% dequantize, ~15% transpose, ~5% actual
  FFN compute** — directly answering WP4e's long-open "is a native re-encode worth it?" question (the
  arithmetic the model wants is under a twentieth of the cost). Also found (not fixed, logged as
  `docs/CPU_PERF_BACKLOG.md` 2d): `dequantize_iq2_xxs` is ~6x slower per element than its two
  neighbors — real, well-diagnosed (per-element table lookup blocking loop unrolling), material since
  14,336 of 73,728 routed-expert planes use that format. Independently reverified before merging: a
  GDN-only trainable build (act_grad must stay full-size there) reproduces its gradient-check tests
  exactly, the real 48-layer determinism fixture reproduces bit-for-bit with the exact memory numbers,
  and I independently reran the new benchmark and reproduced both headline findings myself.
- Repo hygiene: three agents have shared this working tree throughout (Codex, GitHub Copilot, Claude
  Code). The `docs/TOKENIZER_V2_IDEAS.md` pre-existing edit mentioned in the older hygiene note above is
  stale context — re-check current `git status` rather than trusting that paragraph. Check
  `docs/ACTIVE_WORK_LOG.md`'s Log table (not just the row table) for the latest coordination state
  before touching any shared file.
