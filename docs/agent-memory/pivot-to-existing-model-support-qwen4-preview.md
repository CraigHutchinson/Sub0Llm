---
name: pivot-to-existing-model-support-qwen4-preview
description: "Sub0Llm's from-scratch training research is CLOSED OUT as of 2026-08-30 (docs/ROADMAP.md status banner) -- new focus is supporting existing released models (weight import, distillation), first target Qwen3.8-Flash-Next (the Qwen4-preview architecture, released 2026-08-26). All 3 initial subagent workstreams (weight extraction, n-gram embeddings, Gated DeltaNet design) DONE, reviewed, merged to main as of 2026-08-31/09-01"
metadata: 
  node_type: memory
  type: project
  originSessionId: 84b5aaba-5d15-4c11-acba-06624e60ae97
  modified: 2026-09-08T09:14:31.415Z
---

**Decision**: no further from-scratch pretraining/architecture-sweep research is planned for Sub0Llm.
`docs/ROADMAP.md` carries the closing status banner (2026-08-30) summarizing what the from-scratch line
actually proved: the [[deterministic-mechanisms-thesis]] held up every time tested (arithspike, factspike)
and is the one result explicitly carried forward; GQA/LoopSplit/depth-attention/RoPE-scaling
([[nanbeige-features-progress]]) remain "implemented and verified correct, never accuracy/throughput
validated" — that gap is staying open, not being chased further. Tokenizer schemeV5
([[truecasing-collapse-unconditional]]) and the WSD cooldown ([[wsd-cooldown-validated-tinystories-smoke]])
are the two pieces with an actual measured win, and stay in active use.

**New direction**: support EXISTING released models — weight import, inference, distillation — rather
than growing this engine's own corpus/architecture sweep. First concrete target: **Qwen3.8-Flash-Next**
(Hugging Face `Qwen/Qwen3.8-Flash-Next`, released 2026-08-26), which Alibaba explicitly frames as an
early open-weight preview of the Qwen4 architecture (same pattern as Qwen3-Next ahead of Qwen3.5).
Full details + sourcing/confidence-flagged facts table: `docs/QWEN4_PREVIEW_REFERENCE.md`.

**Critical scoping decision, made mid-task after a user correction**: the model is 125B-total/6B-active
MoE + a 51B-param n-gram embedding table (~176B params, ~350GB bf16) — far outside what this engine can
train or even run (this machine: 63GB RAM, one ~8GB-VRAM laptop GPU, 589GB free disk). The engagement is
**NOT** "train toward this architecture" and **NOT** "build separate distillation tooling as an
independent workstream" — it is: **extract a few real weight tensors (one Gated-DeltaNet layer, the
n-gram embedding table) and use the model's own real published math as a correctness ORACLE, gating this
engine's own from-scratch implementations of the same mechanisms (built at Sub0Llm's own tiny scale) on
numeric agreement with real reference activations** — strictly stronger than the synthetic-only gradient
checks the engine has relied on so far ([[depth-attention-stage0-done]]'s "a gradient check cannot
validate this op" lesson is the direct motivation).

**Three subagents dispatched 2026-08-30, all on worktree branches off the new `main`**:
1. `research/qwen4-preview-extraction` — **DONE, reviewed, MERGED to main (802a3d0)**. Verified real
   `config.json`/`model.safetensors.index.json`/`tech_report.pdf`/`transformers==5.16.1` modeling code
   directly (everything in the doc before this was AI-summarized-webpage-sourced and flagged low/medium
   confidence — now corrected). **Real corrections found**: this is a VLM (`Qwen4ExpForConditionalGeneration`,
   `language_model_only: false`), not text-only as first assumed; the config's `"full_attention"` layers
   are actually remapped to QSA at load (`Qwen4ExpTextQSAIndexer`, DeepSeek-DSA-style) — there is no plain
   softmax attention anywhere in this model; the "~139 GGUF quantizations" claim didn't hold up (no
   `.gguf` files in the real repo at all). hidden_size=2560, n-gram table exactly 51,200,245,760 params,
   internal model_type is literally `qwen4_exp` in `transformers` already. Extracted real GDN layer-0
   weights + real n-gram-embedding rows via surgical HTTP range requests (~116MB total, never touching
   the ~360GB checkpoint or ~102GB n-gram table) and ran the model's own unmodified forward code on CPU to
   build fixtures at `tests/fixtures/qwen4_preview/` — these are the correctness gate for Stage 2.
   `.gitignore` gained a scoped `!tests/fixtures/**/*.bin` carve-out for them.
2. `feature/ngram-embeddings` — **DONE, reviewed, MERGED to main (32085c5)**. CPU-only Stage 0+1
   ("concat" fusion, single-layer input-embedding injection; multi-layer `NanbeigeNgramLayerFusion`
   gating + CUDA explicitly deferred, not dropped). Implementation reuses the block-matmul identity
   (`[x0|x1|..]@W == Σ x_e@W[e-slice,:]`) so "concat then linear" composes from already-verified
   `op_embed`/`op_linear`/`op_add` with zero new backward code or Node-fanout change — unlike depth
   attention, this genuinely didn't hit that wall. Correctly classified as a SHAPE-changing axis
   (PARAM_FLOATS already discriminates it, no new ARCH_FINGERPRINT bits needed, matching the GQA
   precedent). **Real bug found and fixed along the way**: the n-gram hash was reading raw token ids
   including persistent/scratch-slot ids (unbounded integers, no fixed identity) — caught by
   `persistent_slots_engine_tests.cpp`'s existing differential test, fixed by treating `id >= VOCAB` as
   "no signal" in both `forward()` and `forward_one()`. The affected pre-existing test was split
   `if constexpr` on `NGRAM_EMBED` (byte-identical at the default setting, a real compensating assertion
   only when the feature is on) rather than weakened. **Independently verified, not just taken on
   report**: read the full diff, then built+ran two FRESH configs myself (not reused build dirs) —
   production shape d196/L11/H7 with the feature OFF: 53,917,287 assertions/137 cases all green (exact
   134+3 delta); a small d128/L4/H4 config with the feature genuinely ON: 5,999,172/137 all green,
   including the presence test proving the ngram contribution is context-dependent, not a no-op. Also
   cross-checked structurally against the real Qwen weight fixtures from workstream 1
   (`tests/ngram_qwen4_fixture_tests.cpp`, engine-free, confirms this project's concat ordering matches
   the real module's `flatten(-2)` bit-for-bit; exact hash-formula match not expected, different scheme).
3. `feature/gated-deltanet-design` — **DONE, reviewed, MERGED to main (a97b075)**. Design + Stage 0
   skeleton only (no forward/backward op, no CUDA) — real recurrence formula re-derived from the actual
   `Qwen4ExpTextGatedDeltaNet` (superseded its first draft, sourced from Qwen3-Next as a proxy, once the
   real fixtures landed mid-task): `S=g*S; kv_mem=Sᵀk; delta=β*(v-kv_mem); S+=k⊗delta; o=Sᵀq`, decay
   `g=exp(-exp(A_log)*softplus(a+dt_bias))`, update `β=sigmoid(b)`. `ARCH_FINGERPRINT` had zero spare bits
   (worked through explicitly, not assumed) so this adds a new additive `ARCH_FINGERPRINT2` word +
   `CKPT_VERSION` 6→7, both per the existing hard-version-gate/additive-trailer contract. The new
   `--gdn-full-attn-stride` axis is double-gated at 0 (CLI::Range AND a compile-time static_assert) so
   Stage 1 (the actual op) cannot be enabled by any means yet. **Independently verified, not just taken on
   report**: read the full diff myself, then rebuilt+ran the real engine test suite (`sub0_tests`, not the
   engine-free frontend one) at production shape (d196/L11/H7) — 135,123,464 assertions / 134 cases, all
   green, case count exactly matches the expected +2 delta. Caught along the way: `out/build/d196check`'s
   cached generated header predated this (and apparently several prior) config-header changes and needed
   `sub0llm-configure` re-run to pick up the new field — a real, generalizable gotcha (a stale
   `out/build/*` cache silently builds a config-file version behind current `tools/configurator.cpp`,
   local/gitignored, harmless but re-run configure before trusting a build's test result post-config-change).

QSA (Qwen Sparse Attention, the other headline mechanism) is explicitly deprioritized/not delegated —
documented as open research in `docs/QWEN4_PREVIEW_REFERENCE.md`, least-verified of the three mechanisms
and least aligned with this engine's small-scale focus (its benefit is long-context latency).

**Next step when these land**: review each branch's diff before merging into `main` (none should be
merged unreviewed — these are new ops with real blast-radius per AGENTS.md §3, and this is the first
work landing on the just-created `main`, see [[main-branch-consolidation-2026-08-30]]).

## Follow-on 2026-09-01: n-gram table serving spun into its own project, Sub0Firn

The real Qwen n-gram table (51.2B params, ~102.4GB) can never be fully resident on this hardware
(63GB RAM, ~8GB VRAM) — a 4th subagent produced a design-only plan (`docs/NGRAM_TABLE_TIERED_STORAGE.md`,
merged to Sub0Llm main) for tiered serving (VRAM hot subset → RAM working-set → local disk/NVMe →
remote HTTP-Range source), with real cited prior art (Bandana/MLSys 2019 is the single most load-bearing
citation; also HugeCTR's Hierarchical Parameter Server, TT-Rec, DLRM/MoE-Infinity/FlashMoe frequency-aware
caching, LMDB-over-RocksDB reasoning for the disk tier) and reconciled against this engine's own
no-heap-in-hot-path rule (an explicit prefetch/resolve pass before the batched op) and the existing
mmap'd `corpus.tok` out-of-core precedent (a training corpus's n-gram working set is computable ahead of
time — a much stronger prefetch signal than reactive LRU).

User decided this should be a genuinely standalone, engine-agnostic sideline project rather than
Sub0Llm-internal — named **Sub0Firn** (glaciology term for the compacted intermediate snow layer between
fresh powder and solid ice; fits the "intermediate tier" concept directly). Repo now live:
**https://github.com/CraigHutchinson/Sub0Firn**, seeded 2026-09-01 with `README.md` (spec/requirements —
scope, non-goals, API surface `register_table`/`prefetch`/`wait`/`resolve_into`/`try_get`/`stats`/
`invalidate`, consistency/concurrency model) and `DESIGN_RATIONALE.md` (the full prior-art research,
ported from Sub0Llm's `docs/NGRAM_TABLE_TIERED_STORAGE.md`, which stays in Sub0Llm as the historical
record). No code in either repo yet — design-only. Recommended integration seam: vendored header-only
client first (mirrors the existing `gguf.hpp` precedent), graduating to a linked library or
out-of-process service only once real cross-process cache-sharing is needed.

**2026-09-01 follow-up, on request ("capture the prior art... ensure clear requirements... usable for
Sub0Llm")**: Sub0Firn's prior art moved out of the design narrative into its own findable
`PRIOR_ART.md`. Added a README §7 that traces the API against Sub0Llm's REAL merged n-gram code (not a
hypothetical sketch) — real call patterns (training: `resolve_into` before compute, per the
no-heap-in-hot-path rule; decode: single-row reactive lookups, no advance-known prompt to prefetch
against), a real concurrency number (~24 concurrent OMP threads on the dev machine, `DEFAULT_THREADS`),
the precomputed-working-set recipe made concrete (mirrors `corpus.tok`'s own out-of-core precompute),
and an explicit dtype-conversion requirement (bf16-on-disk → float32-to-caller is Sub0Firn's job, never
the consumer's). Two things flagged genuinely OPEN rather than silently resolved: decode-time
cache-miss policy (block vs. "no signal" fallback), and a reference offset-resolution callback for a
real sharded format (Qwen's 128-shard table). `docs/NGRAM_EMBEDDING.md` gained a "Stage 3" pointer to
this so a future implementer finds it from the actual consumer code, not just from the design doc.

**2026-09-01: Gated DeltaNet Stage 1 DONE, reviewed, MERGED to main (9f42e38).** CPU forward only
(sequential recurrence, `op_gdn` wired into both `Model::forward` and `forward_one`'s decode path,
per-layer param layout dispatch on `GDN_SCHEDULE`) -- the `GDN_FULL_ATTN_STRIDE` gate from Stage 0 is
now genuinely relaxed (was hard `{0}`-clamped by both CLI and `static_assert`). **Real correction found**
mid-task by re-fetching the actually-installed `transformers==5.16.1` source rather than trusting the
already-quoted doc snippet: query is scaled `1/sqrt(head_k_dim)` after L2-norm, one line before where
`docs/GATED_DELTANET.md`'s original quote began -- omitting it gives a plausible-but-wrong result.
**Independently verified, not just taken on report**: read the full diff (per-layer PARAM_FLOATS dispatch,
the `Op::GDN` backward case -- confirmed it's a genuine loud `std::abort()` with a clear message, not a
silent wrong-gradient path, since Stage 1 deliberately has no backward yet); rebuilt the fixture
correctness test myself -- matches the subagent's exact numbers (max abs diff 4.36557e-11 vs. the real
extracted Qwen weights, mutation tests for g=1/beta=1-degenerates-to-cumsum and beta=0-is-a-no-op both
exact); rebuilt the full production-shape suite at the NEUTRAL setting -- byte-identical hashes AND
assertion count (53,917,287/137) to the pre-merge baseline; built a genuinely GDN-ON small config
(d128/L4/H4, stride 2) and independently confirmed `forward_one` vs `forward` parity (worst rel diff
8.06e-07) -- a different shape/seed than the subagent used, same clean result. One real gotcha hit while
verifying: running the FULL default test suite (which exercises training/backward) against a GDN-ON
build correctly aborts loudly -- that is the deliberate safety behavior working as designed, not a bug;
the fix was to run only the forward/decode-path tests at that config, matching Stage 1's actual scope
(gen/eval/report work, train/tune do not). Next: Stage 2 (backward pass, recompute-based design per
`docs/GATED_DELTANET.md` S4) or Stage 3 (CUDA) -- neither started, no branch dispatched yet.

**2026-09-01: Gated DeltaNet Stage 2 DONE, reviewed, MERGED to main (fa41425).** CPU backward, recompute-
based (per §4's closing direction — rerun forward from retained inputs during backward rather than retain
every `S_t`, mirroring `docs/DEPTH_ATTENTION.md`'s CUDA-side recompute precedent). **Verification method
worth remembering**: hand-derived gradients checked against TWO independent oracles BEFORE any C++ —
(1) `transformers==5.16.1`'s own real autograd on the real extracted layer-0 weights (~3e-7 relative on
all 9 param grads + input), (2) an independent from-scratch PyTorch reimplementation at a different,
multi-key-head-group shape (~1e-15, machine precision) — a real bug (conflating `d(delta_t)` with `k_t`'s
own contribution, a dropped `-kv_mem_t` term in `d(beta_t)`) was caught by bisecting against saved
autograd intermediates before it ever reached C++. This "get real autograd gradients from the actual
installed reference library, not just a synthetic FD check" technique is a strong pattern worth reusing
for any future backward-pass port in this codebase. **Node-linkage question Stage 1 deferred is now
resolved**: `Op::GDN` backward needs 10 pointers (input + 9 params), more than `Node`'s 4-slot fanout —
solved via the exact `docs/DEPTH_ATTENTION.md` §5a side-table pattern (new `Node::gdn_link` int + a
thread_local `GdnLinkCache`), not a `Node` widening. **Independently verified, not just taken on
report**: read the full diff (the thread_local `GdnBwdScratch` design, the `+=`-accumulation into arena
grad spans, the LoopSplit-safety reasoning); rebuilt the real-oracle + FD + mutation tests myself, all
green; rebuilt the neutral-setting production suite — **byte-identical hashes AND assertion count
(53,917,287/137) across three consecutive merges now** (Stage 1, and now Stage 2, both leave the default
path completely untouched); built a genuinely GDN-on config and ran the FULL default suite (not just
forward/decode tests, since backward now exists) — **zero aborts**, 5,356,149/5,356,150 assertions pass,
the one "failure" being the same known non-bug from Stage 1 (a Stage-0-era test scoped to the neutral
build only). Stage 1's `Op::GDN` backward `std::abort()` is gone, replaced by the real implementation.
Next: Stage 3 (CUDA), gated on exact parity with this stage's sequential CPU reference — not started, no
branch dispatched yet.

**2026-09-02: Gated DeltaNet Stage 3 DONE, reviewed, MERGED to main (5e4832d) — plus a real bug found in
already-merged Stage 1/2 code.** CUDA forward only (chunked-parallel form, `torch_chunk_gated_delta_rule`,
re-verified against the real installed source rather than ported from the CPU sequential algorithm,
since the chunked form's parallelism across chunk positions is specifically what CUDA can exploit). CUDA
backward/decode deliberately unimplemented (loud aborts, mirroring Stage 1's own CPU precedent) — kept
the same forward-then-backward staging discipline the CPU port used.

**The important finding, worth internalizing as a general lesson, not just a bug report**: Stage 3's
independent, from-scratch CUDA reimplementation disagreed with the CPU engine at a real mixed-layer
model, which exposed a genuine bug that had been sitting in already-merged, already-"verified" Stage 1/2
code: `backend_cpu.cpp`'s `op_gdn` (both the batched-forward and `forward_one` decode call sites) passed
`dt_bias`/`a_log` in SWAPPED positions relative to `gdn::forward()`'s real declared signature, while
`gdn::backward()`'s own call already had them correct — a real forward/backward mismatch. **Every one of
Stage 1/2's own extensive correctness gates was structurally blind to this specific bug**: the real-Qwen
fixture test and the mutation tests call `gdn::forward()`/`recurrence_step()` directly with their own
correctly-ordered local variables, bypassing `op_gdn`'s buggy glue entirely; the `forward_one`-vs-`forward`
parity check stayed consistent because BOTH call sites had the identical swap; and the whole-model
finite-difference check compares a NAMED tensor's numeric and analytic gradients, which stays
self-consistent under a symmetric relabeling of two structurally-similar same-shaped parameters (a
whole-vector directional-cosine aggregate check is also just less sensitive to an error localized to 1-2
of many parameter tensors than a per-tensor isolated check would be — worth remembering for any future
op with multiple similar-shaped parameters). **The general, transferable lesson**: a real, independent,
from-scratch reimplementation of the same math (not just more tests against the same code, and not just
more scales/shapes of the same code path) is uniquely powerful for catching identity/labeling bugs that
every self-referential check — fixture tests bypassing the buggy glue, symmetric parity checks, aggregate
gradient checks — will structurally miss. This is a stronger, more specific instance of
[[verify-correctness-against-reference-before-perf]] and the depth-attention "a gradient check alone
cannot validate this op" lesson ([[depth-attention-stage0-done]]) — extending it to: even a real-weight
fixture match and a real-oracle gradient check can both be blind to a bug in the glue code that feeds
them, if that bug is a consistent relabeling rather than a wrong formula.

**Independently verified on real hardware, not just taken on report**: confirmed the swap bug is real by
reading `gdn::forward()`'s actual declared parameter order against the buggy call sites myself; got a
real CUDA build working independently (`Import-VcVars`-equivalent PowerShell setup, nvcc 13.2/sm_120,
confirmed via `where.exe cl.exe`/`nvcc.exe` in the same process) at a config I configured myself
(d384/L2/H4/kv2, stride 2 — different from the subagent's own test shapes); reproduced the CUDA-vs-CPU
sequential-reference parity (rel diff 8.57e-07 to 3.67e-06 across 3 shapes) and the real-Qwen-fixture
match (3.638e-11, matching the subagent's 3.64e-11) myself; reproduced "CUDA forward matches CPU engine
logits" (max abs diff 0.000516, well inside tolerance) on my own independently-configured shape; ran the
full unfiltered suite at a real GDN-on CUDA build and got the EXACT same 166/186 pass, 20 fail pattern
the subagent reported, and confirmed every one of the 20 failures by name: 1 known non-bug
(`ARCH_FINGERPRINT2` off-premise), 18 correct `sub0_cuda_backward` refusals, 1 pre-existing
`kTestDims`/`trainable_floats` shape mismatch already flagged as unrelated to this branch.

Next: Stage 4 (CUDA backward), if ever pursued — not started, no branch dispatched. This closes out the
originally-planned staged GDN implementation (Stage 0 config skeleton → Stage 1 CPU forward → Stage 2 CPU
backward → Stage 3 CUDA forward) end to end.

**2026-09-02: deployment feasibility DONE, reviewed, MERGED to main (246a52e).**
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` — real GGUF sizing (byte-exact, via the same HTTP-range-header-parse
technique the earlier safetensors extraction used, now proven to work on GGUF too), MoE-offload prior art
survey, naive n-gram mmap PoC (design-only). **Key verified facts**: every GGUF quant tier from
UD-IQ1_S (72.55GB) to BF16 (354GB) exceeds this machine's 63GB RAM — nothing fits fully resident at any
quant level. The 51.2B-param n-gram/PLE table is clamped to a fixed floor (`Q8_0` ~54.4GB or `IQ4_NL`
~28.8GB) regardless of the model's overall quant label — NOT shrunk proportionally — and for several
tiers (BF16/Q8_0/UD-Q5_K_XL/UD-Q6_K_XL) is the entire contents of one skippable GGUF shard file, no byte
surgery needed. Real gap found: `include/sub0/gguf.hpp`'s `TensorType` only recognizes F32/F16/Q8_0 —
cannot yet size/decode BF16/IQ4_NL/K-quants — named as the actual next concrete unit of work regardless
of which cache design eventually sits on top of it (useful to both Sub0Firn and any future naive
shortcut). MoE-offload prior art (real, independently confirmed): `llama.cpp`'s `-ot`/`--n-cpu-moe` are
real, shipped, static/no-prediction, usable TODAY zero-new-code against these exact files; MoE-Infinity
(real, active, "activation-aware expert caching") is the most directly reusable research system if this
model is ever actually served, though not confirmed for this exact `model_type` (only Qwen3-MoE/
Qwen3.5-MoE listed). **Recommendation, endorsed after independent spot-checks**: do NOT bulk-download
any quant tier — even the smallest is a 72.5GB download for a merely "technically launchable" result;
this project's real leverage stays the already-shipped GDN/n-gram validation-oracle work, more surgical
extraction if a specific new question arises, not deploying the full model. Naive n-gram mmap PoC stayed
design-only as instructed: `Node::data` being a non-owning `std::span` means `op_embed` needs zero
changes to read from mmap (a genuine positive finding, reusable later), but the external Qwen table has
no existing consumption seam in `Model::forward` at all, and needs the `gguf.hpp` dtype-decode gap
closed first regardless. **Independently verified before merging** (not taken on report): fetched the HF
blobs API myself and confirmed the `Q8_0-00003-of-00006.gguf` shard size byte-for-byte; confirmed
MoE-Infinity is real/active and its own README hedge matches the doc's (lists Qwen3-MoE/Qwen3.5-MoE, not
this exact architecture). One tiny (~2.17MB, 0.003%) discrepancy in the UD-IQ1_S tier total found and
noted, doesn't change any conclusion, not chased further.

**Housekeeping note, generalizing from the pre-dispatch cleanup**: 6 stale worktrees for already-merged
agent branches had accumulated across this whole GDN effort before being cleaned up this session (see
below) — from now on, remove a subagent's worktree + delete its branch (local + origin) right after
merging, not batched later. Did this immediately after merging this doc, confirmed clean.

**Also, before that dispatch: found and cleaned up significant worktree/branch cruft** — 6 stale
worktrees for already-merged agent branches (never cleaned up after their merges: GDN stages 1-3,
n-gram-embeddings, gated-deltanet-design, the qwen4-preview extraction, AND a `research/ngram-tiered-
storage` worktree from the Sub0Firn thread) plus one broken new-agent worktree hit a `core.worktree
redirect` error on dispatch. All confirmed as ancestors of `main` first, then worktrees force-removed and
branches deleted locally + on `origin`. **Lesson for future sessions**: worktree cleanup after a merge
was being skipped in this whole GDN effort — should happen as part of each merge, not accumulate until a
dispatch fails.

**2026-09-02: Phase 0 (memory/data orchestration architecture) DONE, reviewed, MERGED to main (bb5c350).**
`docs/QWEN4_MEMORY_ORCHESTRATION.md` v1 — real live measurements this session (GPU 8151/7891 MiB
total/free, PCIe Gen5 x8; RAM 63.44GiB total but only ~43.8GiB free under normal desktop load — the key
corrective to "63GB RAM" used loosely so far; disk 584-628GB free), per-component param counts freshly
derived from real `config.json` fields nested under `text_config` (`moe_intermediate_size=640`,
`shared_expert_intermediate_size=640`, `vocab_size=248320`, `tie_word_embeddings=false`), and a strong
cross-check: the GDN per-layer byte count independently re-derived from config arithmetic exactly
reproduces Stage 1's own real fetched value (115,917,248 B). **Key finding**: the backbone alone
(~3.9-4.1B params, ~7.6-8.2GiB bf16) is already at or past this GPU's entire realistic VRAM ceiling
before any MoE expert or runtime state is added. **Chosen v1 MoE placement policy**: static layer-range
CPU/GPU split (llama.cpp `-ot`/`--n-cpu-moe` style), not an activation-aware cache — the arithmetic
leaves near-zero spare GPU budget for caching sophistication to matter yet. **Compile-time/runtime
dividing line for the next MoE work package**: a compile-time-SIZED, fixed-capacity GPU-resident
expert-slot table per layer (capacity plausibly 0 for a first build on this hardware), populated/evicted
entirely at runtime. **Independently verified before merging**: re-ran every hardware measurement myself
(nvidia-smi/PowerShell, all matched closely), re-fetched the real `config.json` myself and found the
fields nested under `text_config` (my first naive top-level check came back empty, which could have
looked like a real discrepancy — was not, just the wrong query path), confirmed the per-expert arithmetic
by hand.

**A real process gotcha hit merging this one, worth remembering**: after this subagent's first attempt
died to a session rate limit before writing anything (worktree auto-cleaned, nothing lost), the RESUMED
agent's session ended up operating directly in the PRIMARY repo working directory instead of getting a
fresh isolated worktree — `git worktree list` showed only one entry, the main repo path itself checked
out on the research branch. Nearly caused a confused merge (a `git merge` while unknowingly checked out
ON the feature branch reported "already up to date" against itself, and `git push origin main` was a
silent no-op since local `main` hadn't actually moved) — caught by checking `git branch -vv`/
`git status --short --branch` before trusting the merge result, no data lost since the branch content
was intact throughout. **Lesson: after resuming an agent whose original worktree was auto-cleaned
(nothing had been written yet), verify with `git worktree list` / `git status --branch` which branch the
PRIMARY repo directory is actually on before running merge/push commands — don't assume it's still on
`main`.**

**2026-09-02: WP1 (Gated Residual / hyper-connections) Stage 0+1 DONE, reviewed, MERGED to main (6998460).**
Real formula verified against `transformers==5.16.1`'s actual `Qwen4ExpTextGatedResidual` + the real
decoder-layer wiring (load-bearing — the module alone doesn't show how its outputs get consumed). Grouped
RMSNorm with a zero-centered `1+w` gain (a real, documented divergence from this engine's existing
Ln1/Ln2/QNorm/KNorm convention, which uses `w` directly) feeds a low-rank sigmoid read-gate (MEAN-pooled
across `hc_count=4` streams, not summed) and a separate write-gate (`injection_weights ∈ (0,2)`).
**Deliberate, well-justified simplification**: kept the engine's existing Ln1/Ln2 rather than removing
them (the real model has no separate pre-block norm — GR's own `hc_norm` does double duty) — correctly
scoped since this stage's correctness gate is the isolated GR module, not full-layer parity; flagged
explicitly as a later stage's work, not silently glossed over. **Checkpoint design, cleaner than GDN's
own**: PARAM_FLOATS delta is a pure sum of strictly-positive products (unconditionally monotonic in
`hc_count`/`hc_lowrank`/`N_LAYERS`, unlike GDN's mixed-sign delta) — PARAM_FLOATS alone discriminates it,
no new `ARCH_FINGERPRINT` word needed, matching the n-gram-embeddings precedent. No Node-fanout side
table needed (Stage 1 has no backward yet). **Independently verified before merging**: rebuilt the
fixture test myself — reproduced the exact bit-for-bit match (`max|out−expected|=0` for both `mixed_input`
and `injection_weights`) and the presence test (each of the 4 streams' removal produces a distinct
nonzero perturbation, and a mean-vs-sum discriminator confirms MEAN specifically); rebuilt the neutral
setting at production shape — **byte-identical hashes across FOUR consecutive merges now** (Stage1/2/3 of
GDN, and now this), assertion delta exactly +12/137→139 as reported; built a GR-ON config at `hc_count=3`
(deliberately different from the real model's 4, to confirm genuine generality) and independently
confirmed `forward_one`/`forward` parity (6.75e-07) and that the backward-abort safety guard fires
correctly on the full training-touching suite.

**Same process gotcha as Phase 0, worth reinforcing**: hit the exact same "worktree redirect" dispatch
error as before on first attempt (cleaned up via `git worktree remove --force --force` + branch delete,
then retried successfully) — this is now a recurring failure mode when stale worktrees accumulate, not a
one-off; check `git worktree list` before every dispatch, not just after a resume.

Next per the approved plan: **WP2 — MoE routing**, Stage 0+1 CPU forward, built against Phase 0's
placement architecture (compile-time-sized expert-slot table, per `docs/QWEN4_MEMORY_ORCHESTRATION.md`
§5c). Not yet dispatched.

**2026-09-03: WP2 (MoE routing) Stage 0+1 DONE, reviewed, MERGED to main (b3ae26d).** The hardest mechanism
yet — real router/combination formula verified against `transformers==5.16.1`'s actual
`Qwen4ExpTextSparseMoeBlock`/`TopKRouter`/`Experts`: softmax over all 512 experts → top-10 selected →
weights renormalized to sum to 1 (`norm_topk_prob` defaults `True`, confirmed absent from the real
`config.json`); per-expert 3-matrix SwiGLU weighted-summed across the 10 selected; a fully separate
always-on shared expert scaled by its OWN independent sigmoid gate, not re-weighted by the router at all.
**Checkpoint design genuinely diverges from GDN/GR's precedent, correctly worked out rather than copied**:
`NUM_EXPERTS` is shape-changing (PARAM_FLOATS alone discriminates it) but `EXPERTS_PER_TOK` is
shape-NEUTRAL yet computation-changing (two builds differing only in top-k cutoff produce byte-identical
checkpoints that compute different things) — joins the EXISTING `ARCH_FINGERPRINT2` word's already-reserved
byte 1 rather than a new trailer. **A real SIGSEGV was found and fixed during the agent's own pre-commit
verification**: `op_moe` was wired into the GDN branch's FFN dispatch but not the softmax-attention
branch's (the one every layer actually runs at the default `GDN_FULL_ATTN_STRIDE=0`) — root-caused via
stderr instrumentation, not guessed. **Independently verified before merging**: rebuilt the fixture test
myself (exact match, `1.09e-11` — a different tiny number than the agent's own `1.45e-11`, expected since
MoE's weighted-sum accumulation is more summation-order-sensitive than GDN/GR's single projections, both
consistent with genuine correctness); confirmed the router genuinely selects different experts for
different inputs with correctly-renormalized weights; rebuilt the neutral setting at production scale —
**byte-identical hashes across FIVE consecutive merges now**, case count (142) matching exactly; built a
real MoE-ON config at `num_experts=6, experts_per_tok=3` (deliberately different from the agent's own
8/2) and confirmed `forward_one`/`forward` parity (7.18e-07) and the backward-abort safety guard firing
correctly.

**Process note**: this dispatch's first-layer agent immediately sub-delegated to a "Fable" model agent
per the (now-corrected) standing policy — see [[fable-subagent-for-big-risky-tasks]], user corrected this
same day to use `model="opus"` since Fable isn't available on this account. The sub-delegate itself also
hit a rate limit mid-work once and was resumed cleanly (real progress was already committed+pushed each
time, nothing lost either time).

Next per the approved plan: **WP3 — QSA** (Qwen Sparse Attention + lightning indexer), the last missing
mechanism, lowest priority (only 12/48 real layers use it). Not yet dispatched.

**2026-09-01, further follow-up**: brought Sub0Firn up to the same requirements rigor as the sibling
`Sub0Log` project (github.com/CraigHutchinson/Sub0Log, reviewed at pinned SHA `b992f1a3` — see
[[sub0log-adoption-backlog]]). Sub0Firn now has: `REQUIREMENTS.md` (normative R1-R10, each a testable
sentence + concrete justification, mirroring Sub0Log's exact style), `AGENTS.md` (pre-flight checklist,
each rule grounded in a real design decision already made, not boilerplate), `STYLE_GUIDE.md`,
`LICENSE.md` (MIT, matching Sub0Log), `.gitignore`, a buildable `CMakeLists.txt` (interface-library
skeleton, C++23, subproject-aware), and `include/sub0firn/sub0firn.hpp` (honest skeleton, no
implementation yet). `docs/` restructured to match Sub0Log's granularity: `docs/README.md` (index),
`docs/prior-art.md` (moved out of the design narrative), `docs/tiered-storage-design.md` (the full
design), `docs/reference-consumer-sub0llm.md` (the concrete Sub0Llm acceptance-test trace). Verified the
CMake skeleton actually configures before pushing. Sub0Llm's own `docs/NGRAM_TABLE_TIERED_STORAGE.md`
pointer note updated to match (was pointing at the old pre-restructure filenames).

**2026-09-01, design refinement: is "frozen" a hard requirement or an optional mode?** User's question,
worked through and resolved, not just tweaked: R4 was conflating two guarantees — a row's VALUE never
silently changing (load-bearing, stays hard) vs. a table's addressable SIZE being fixed forever (not
actually required by anything else). Resolution: added `REQUIREMENTS.md` DR1, explicitly deferred past
v1 but named now so the row-addressing model doesn't foreclose it — a logical table may later resolve
through a frozen `base` + a small, genuinely-writable `overlay` source (single administrative writer,
append-only), so e.g. a project-specific vocabulary extension can layer on top of the huge frozen Qwen
n-gram table without the base ever changing. Verified two new prior-art citations before writing this
down: HugeCTR's dynamic embedding table (`embedding_vec_size=-1`) genuinely grows at TRAINING time by
inserting a fresh row on first sight of a new key — closest real precedent for growth itself, though
out of Sub0Firn's own scope (gradient-trained). Its own Hierarchical Parameter Server (the
INFERENCE-time counterpart) deliberately does NOT auto-insert a missing key, returns a configured
default instead — the closer analog to Sub0Firn's actual serving role, and the reason DR1 should stay
opt-in, not default. OverlayFS-style layered composition (read-only lower + writable upper) is the
actual architectural fit, not the training-time analog.

**2026-09-03, WP3 (QSA — Qwen Sparse Attention + lightning indexer) Stage 0+1 merged (`15f7dbf`), the
third and last missing mechanism.** Real `Qwen4ExpTextQSAIndexer`+`Qwen4ExpTextAttention` formulas
verified against installed `transformers==5.16.1` (`docs/QSA.md`), surfacing two load-bearing facts
`docs/QWEN4_PREVIEW_REFERENCE.md` never recorded: `q_proj` is double-width and chunked PER HEAD (query |
gate live inside each head's own slab, not a flat `[queries|gates]` split — an identity-swap-class hazard
per [[independent-reimplementation-catches-identity-swap-bugs]]), and the attention output is gated by
`sigmoid(gate)` before `o_proj`. Indexer: ReLU-then-SUM score (no softmax anywhere), raw keys cached
completely unnormed/unrotated and only normed+rotated AFTER mean-pooling into blocks (at the block's
first-token position), block top-k independent of the causal mask, incomplete tail ALWAYS visible
regardless of score. Verified `config.json`'s `layer_types` is bit-for-bit `gdn_schedule_for(4)` at
`l%4==3` — no new per-layer schedule axis needed; derived a three-way `MIXER_SCHEDULE`
static_assert-pinned to the existing `GDN_SCHEDULE` so they can't drift apart.

Fingerprint classification: `num_attention_heads`/`head_dim`/`num_key_value_heads` reuse the EXISTING
`N_HEADS`/`D_HEAD`/`N_KV_HEADS` (no duplicate knobs — same call `docs/MOE.md` made for `D_FF`);
`indexer_n_heads`/`kv_heads`/`head_dim` are shape-changing (`qsa_param_delta()` strictly positive);
`indexer_budget`/`compress_ratio` are shape-neutral-but-computation-changing, join `ARCH_FINGERPRINT2`
(budget 16 bits since 2048 doesn't fit 8, ratio 8 bits, 24 spare) — no new trailer, no `CKPT_VERSION`
bump.

**Bug found by the agent itself, unprompted** (explicitly briefed to hunt for the MoE SIGSEGV bug class
before declaring done): `rope_apply_row()` rotated `rotary_dim` channels of whatever vector it was handed
— an out-of-bounds write whenever `indexer_head_dim < rotary_dim`. Surfaced as a `forward()`-vs-
`forward_one()` parity failure, root-caused with a standalone harness (ruled out the plausible "top-k tie
amplifies float noise" hypothesis by checking real score margins first), fixed by passing the vector's
own width plus a `static_assert(!USE_QSA || QSA_INDEXER_HEAD_DIM >= D_HEAD)` guard. Separately, proactively
built a GDN(3)+QSA(1) hybrid config specifically to prove `op_qsa` is wired into BOTH the GDN-on and
GDN-off `forward_one` dispatch branches via one shared lambda — the exact MoE-precedent bug class, closed
off before merge rather than found after.

CPU forward only; backward is a loud `std::abort()` (Stage 2 not built, real subtleties named in
`docs/QSA.md` S6 — notably the OPPOSITE gradient-flow conclusion from MoE's router: QSA's ReLU-sum score
has no softmax coupling, so an unselected block gets exactly zero gradient, not a small one); CUDA build
refuses via `static_assert(!USE_QSA)`.

**My own independent re-verification before merging** (same standard as every prior mechanism): rebuilt
`sub0_frontend_tests` myself from a clean checkout of the agent's worktree, confirmed the real-weight
fixture (layer-3 QSA weights, 198,976 bytes via surgical HTTP-Range extraction) matches exactly — 349
assertions/4 cases, identical to the agent's own count. Built a genuinely different QSA-ON config the
agent never tested (`D_MODEL=48, N_LAYERS=13` — odd/ragged tail under stride 4, `H=8/kv=4`, indexer
`3/1/8`, `budget=6/ratio=3`, `SUB0_COMPUTE=CPU` to satisfy the CUDA-refusal guard): built clean,
`[layout]` 398,462/16 and `[qsa]` 100/3 all green. Rebuilt `sub0_tests` with `--clean-first` at production
shape (d196 L11 H7 seq256) with QSA left neutral/off: byte-identical hashes to the agent's own report —
`acfb10b7d0fa03ad`/`5f16f677739cecc9`/`470870e4c0867de5`, 53,919,635 assertions/145 cases, all green —
now six consecutive merges at exactly this neutral baseline (GDN×3 stages, n-gram, GR, MoE, QSA all
byte-identical when off).

Cleaned up: merged `origin/feature/qsa-stage1` (`75f36ab`) into `main` as `15f7dbf`, pushed, removed the
worktree, deleted the branch (local + origin).

**All three missing mechanisms (Gated Residual, MoE, QSA) are now merged.** Per the plan
(`C:\Users\craig\.claude\plans\jiggly-baking-bear.md`), WP1→WP2→WP3 are done.

**2026-09-03: post-merge performance review of all three, with fixes, both merged.** User's explicit
instruction after WP3 landed: review WP1-3 with an eye for performance/aliasing/memory (unnecessary copy,
heap allocation). I did the review myself by reading the merged math cores directly (no subagent for the
review pass itself). Findings:

- **`gr::mix()` (Gated Residual) and `moe::expert_ffn_row()` (MoE)**: both had an OUTPUT-major GEMV loop
  (outer over the output dim, inner striding the weight matrix by the output width) instead of this
  codebase's own established INPUT-major/contiguous convention (`linear_row`'s idiom — outer over the
  input/contraction dim, inner contiguous over output, matching `[rows=in,cols=out]` weight layout).
  GR's version was materially worse (a `wide = hc_count * hidden_size` stride, ~40KB at real scale,
  and GR wraps EVERY layer) than MoE's (`d_ff`-wide stride, one FFN per selected expert). No aliasing
  bugs, no heap allocation in either file otherwise — MoE's row-independent design and GR's
  `hc_norm`/`gate`/`combine`/`tile` were already clean.
- **QSA's `indexer_select_row()`**: a genuine ALGORITHMIC complexity defect, not just cache-locality —
  it recomputed every complete block's pooled+`k_layernorm`'d+RoPE'd key from raw tokens on EVERY call
  regardless of `kv_len`, costing `sum_t (t+1)/ratio ~ T^2/(2*ratio)` over a T-row prefill for an
  `O(T/ratio)` quantity (both the batched `qsa::forward()` and the decode `do_qsa_mixer` paths).
- Secondary, cross-cutting, NOT fixed (flagged only): `arena_alloc` (backend_cpu.cpp) always zero-fills
  both the returned activation AND grad spans unconditionally, even when the grad span goes unused for
  Stage-1 forward-only ops with no backward implemented.

**GR+MoE fix**: did this one myself (Sonnet), inline, not delegated — Opus was rate-limited at the time
and the user explicitly chose "I do it myself now (Sonnet)" via AskUserQuestion, a one-time deviation from
the usual delegate-to-Opus-then-review split since both roles would have been me anyway. Reordered both
loops to input-major/contiguous, matching `linear_row`; verified as a pure summation-reorder safe against
each fixture's own tolerance (GR 5e-5, MoE 5e-7, MoE fixture dims tiny: `hidden_size=16, d_ff=8`).
Independently verified: fixture tests unchanged (117/5), full frontend suite unchanged (115,097/196),
production-shape neutral hashes byte-identical before/after (`cc328cff8ce732b3`/`7c892472dee9c262`/
`63fe6a47d3e4f704`, 28,504,807/145 at d196 L11 H7 seq256 vocab1270), GR+MoE-ON structural checks identical
before/after (19/5). Merged to `main` as **`b980638`**.

**Third pillar (performance) closed out after the fact**: the merge above verified correctness but never
actually measured wall-clock speed — completed per the standing three-pillar policy by extracting the
pre-fix (`b980638^`) and post-fix `gated_residual_math.hpp`/`moe_math.hpp` into namespaced copies
(`before_ns`/`after_ns`) and calling `mix()`/`expert_ffn_row()` directly from a standalone benchmark
(clang++ -O3 -march=native, no repo build system involved) at REAL Qwen3.8-Flash-Next dims
(`hidden_size=2560`; GR `hc_count=4, hc_lowrank=320`, T=32; MoE `d_ff=640`). Interleaved before/after
rounds (5 rounds × 20/200 reps) per [[thermal-confounds-ab-wallclock-testing]], stable across 3 full runs:
**GR `mix()`: ~5.0-5.1x speedup** (106ms → 20-21ms per call at this scale); **MoE `expert_ffn_row()`:
~14.3-14.9x speedup** (6.2-6.6ms → 0.44ms per call) — MoE's win is larger because its per-call weight
matrix is proportionally smaller relative to cache-line/prefetch overhead of the strided access pattern
being removed. Output bitwise identical before/after in both cases (`max|before-after| = 0.0`), confirming
the speedup is a pure reorder with zero numerical cost. No heap allocation in either version; the `after`
scratch requirement is roughly 2x (GR: `+wide` floats for the `up_val` array; MoE: `+d_ff` floats for the
new `g_scratch`), a deliberate, bounded, caller-sized trade of a bit more scratch for the vectorizable
access pattern.

**QSA fix**: dispatched to an Opus subagent (branch `perf/qsa-block-key-cache`) — hit a rate limit
immediately on first dispatch, then on resume restated its plan without executing (twice), needed an
explicit "stop describing, implement now" nudge before it actually wrote code. Once it did: split
`indexer_select_row()` into `pool_block_key()` (computes ONE block's key — cacheable because every input,
including the RoPE rotation position, is a function of the block alone, never the querying position) plus
score/topk/tail logic that reads from a caller-owned cache extended by only the blocks completed since the
last call (at most one, since `kv_len` grows one row at a time on both paths). Both `qsa::forward()`
(batched, cache lives in its own scratch) and `do_qsa_mixer` (decode, cache lives in `QsaCache`'s
per-execution slot) go through the SAME primitive — the seam that produced both MoE's Stage-1 SIGSEGV and
QSA's own earlier RoPE bug. `QsaCache::n_cached` is zeroed UNCONDITIONALLY on reset (unlike `raw_k`'s
assign-on-size-change) — caught by mutation: a `reset()` that resized instead of zeroed `n_cached` passes
a same-ids-twice decode test (identical inputs re-derive identical cache) but fails a decoy-primed one
(0.164501 vs the 1e-3 gate), so `engine_tests.cpp`'s parity test was extended with a decoy priming pass.
New invocation-count regression test proves the complexity change directly, since a cached and uncached
implementation produce identical OUTPUT by construction: `pool_block_key()` calls 45→5 at T=20,
compress_ratio=4 (naive `sum_t (t+1)/4` vs `T/ratio`), decode path pays the identical 5, and a
deliberately cold-cache run costs exactly the naive 45 AND matches the cached output bitwise. No new
flag, no shape/checkpoint change (`PARAM_LAYOUT`/`ARCH_FINGERPRINT2` untouched) — pure scratch-population
fix. `docs/QSA.md` §11 has the full writeup.

**My own independent re-verification before merging** (rebuilt everything myself, not on trust): built a
genuinely fresh from-scratch QSA-ON config via the configurator (`D_MODEL=16, N_LAYERS=2, N_HEADS=2,
N_KV_HEADS=2, SEQ_LEN=64`, indexer `2/1/8`, `budget=8/ratio=4`) in a new build dir — `[qsa]` 676
assertions/5 cases (was 349/4), matching the agent's report exactly; full `sub0_frontend_tests` 115,424
assertions/197 cases (was 115,097/196, +1 case +327 assertions — exactly the new regression test, nothing
else), and the WARN line itself printed "5 calls (was 45...)" confirming the invocation count directly.
Reran the existing `d196check` build (production-shape, QSA left neutral/off) and got hashes
byte-identical to both the agent's report and my own earlier GR/MoE verification: `cc328cff8ce732b3`/
`7c892472dee9c262`/`63fe6a47d3e4f704`, 28,504,807/145 — zero drift across two separate merges. Merged to
`main` as **`c6db692`**. Caught and reverted an accidental `data/tokenizer_calibration.txt` (git-LFS
pointer) corruption during this session's build experiments, again — see
[[caps-bit-nothing-reads-is-not-a-guard]]-adjacent lesson, this is now the SECOND time this exact file has
been accidentally touched by build/configure activity in this thread; worth a closer look if it recurs a
third time.

**2026-09-03: VTune-guided follow-on pass found a fourth real defect the code-reading review missed.**
User's explicit ask: use VTune to capture further metrics and guide further optimization, beyond what
code-reading alone found. Built a standalone (non-repo-build) benchmark exercising GR+MoE+QSA forward
math together at REAL Qwen3.8-Flash-Next dims (hidden_size=2560; MoE d_ff=640/512 experts/top-10 with a
16-buffer expert pool to keep memory bounded while still exercising the real 512-wide router scan; QSA at
its full real dims incl. `n_heads=24, head_dim=256, idx_head_dim=128, budget=2048, ratio=4`), compiled
with `-g` so VTune could resolve symbols (per [[cpu-profiling-tooling-backlog]]'s own earlier lesson about
missing `.pdb`s). `vtune -collect hotspots -target-pid <pid> -duration <n>` (unelevated, user-mode
sampling — same no-friction path as that memory's prior finding) attached to the running benchmark process
launched via PowerShell `Start-Process` (bash `&`-backgrounding was unreliable — the process finished
before the attach happened at least twice; launching via PowerShell and attaching in the same command
with a short `Start-Sleep` worked reliably).

**Result**: `moe::expert_ffn_row` 30.1% (expected, already-fixed code — genuine work), but
**`moe::router_topk_row` 17.1%** — second-biggest hotspot, and NOT touched by the earlier WP1-3 review
(that pass fixed `gr::mix()`/`expert_ffn_row()` but nobody had looked at the router). Checked the code:
same defect class exactly — `router_w` is `[hidden_size, num_experts]` row-major (this project's own
[in,out] convention) but the logit loop was output-major (outer over `num_experts`, inner over
`hidden_size` striding `router_w` by `num_experts` floats per step). Fixed to input-major/contiguous
matching `linear_row`'s convention — pure summation reorder, same tolerance argument as before. `[moe]`
fixture unchanged (44/2), full `sub0_frontend_tests` unchanged (115,424/197), neutral d196 hashes
byte-identical (`cc328cff8ce732b3`/`7c892472dee9c262`/`63fe6a47d3e4f704`, 28,504,807/145), `[gr],[moe]`
structural checks unchanged (1004/5). Standalone benchmark: MoE per-token cost (T=64, 10 experts +
shared) **~650-790ms/iter → ~452ms/iter**. Merged to `main` as **`46c54b1`**.

The remaining top hotspots (`qsa::linear_row`, 3 separate call sites totalling ~40%) were checked and are
ALREADY input-major/contiguous — genuine, unavoidable attention-projection compute at real dims (q/gate/
k/v/o projections, 2560×6144-scale GEMVs), not a further bug. Further gains there would mean SIMD/BLAS or
threading work, out of scope for this pass — noted as a real, identified opportunity for whenever WP4's
performance needs revisiting, not chased further now.

**Lesson for next time a "code-reading review is done" claim gets made**: this is the SECOND time in one
day that a claimed-complete review missed a real defect the same *pattern* would have caught if applied
exhaustively (WP1-3's own review missed the MoE router; VTune caught what code-reading missed once). A
profiler run over realistic call patterns is a genuinely different, complementary check to a
targeted code-reading sweep — worth doing routinely after any "we reviewed X for performance" claim,
not just when explicitly asked.

**`docs/CPU_PERF_BACKLOG.md` created (`604cceb`)** — a living repo doc (not just memory) consolidating all
four fixed defects above plus three OPEN, checked-but-not-fixed items for whoever picks up perf work next:
(1) `arena_alloc`'s unconditional grad-scratch zero-fill for Stage-1 forward-only ops (smallest/safest,
do first), (2) the 2026-07-16 training-side VTune findings (P-core affinity, gradient-reduction DRAM
bandwidth, vector capacity headroom — different subsystem, own pass), (3) QSA's `linear_row` GEMVs
(confirmed genuine real compute at real dims, real SIMD/BLAS/threading headroom, deliberately NOT started
opportunistically — scope it when WP4's real-scale numbers make it the actual bottleneck). Also has the
reusable standalone-benchmark-+-VTune-attach methodology written up self-containedly (doesn't depend on
this session's own memory files being available to a future reader).

**2026-09-03: WP4 actually started.** Dispatched the memory-orchestration revisit + WP4 scoping to Opus
(branch `research/wp4-scope`, background, worktree) per the plan's own pre-condition. Independently
spot-checked its two most load-bearing structural claims directly against the repo (`gguf.hpp`'s
`TensorType` enum has zero consumers in `src/`/`tools/`; `layout.hpp`'s `GDN_DIMS{D_MODEL, N_KV_HEADS,
N_HEADS, D_HEAD, D_HEAD, GDN_CONV_KERNEL}` really does alias GDN's head geometry onto the attention axes)
— both confirmed exactly as claimed. Merged to `main` as `d11052f` (docs-only, `docs/QWEN4_MEMORY_ORCHESTRATION.md`
v2 + new `docs/WP4_SCOPE.md`).

**The single most consequential finding**: evaluating this repo's own `make_param_layout()` at Qwen4's
real `RunConfig` axes gives 74,899 tensors / 124,027,786,776 floats against the real model's
125,711,064,960 — a 1.68B-param gap, entirely in GDN (−1.335B) and QSA (−0.349B), both because
`D_HEAD = D_MODEL/N_HEADS` is baked into the engine and `layout.hpp`'s `GDN_DIMS` aliases GDN's key/value
head counts and dims onto the attention axes. **A real weight transplant is impossible until this is
fixed** — `gdn::Dims`/`qsa::Dims` (the math cores) already take every geometry field independently and
are fixture-verified correct at the real shapes (`head_dim=256`, `head_k_dim=128`); only the
`RunConfig`→`Layer` engine binding is wrong. This is `docs/WP4_SCOPE.md`'s §2 "four expressibility
blockers" (A: `--head-dim` axis, widest blast radius; B: independent GDN head-count/dim axes; C:
`--rotary-dim` for `partial_rotary_factor`; D: remove `Ln1`/`Ln2` under GR, since the real decoder layer
has no separate pre-norm — GR's own `hc_norm` IS the pre-norm, a correctness requirement not a tidiness
one).

Also corrected: Gated Residual is 640.6M params, not "single-digit millions/layer, negligible" (v1's
estimate before the mechanism existed); QSA is 46% larger per layer than v1 assumed (v1 never saw
`q_proj`'s doubled query|gate width, found via `docs/QSA.md` §1b's full `Qwen4ExpTextAttention.__init__`
quote); MoE's router is a **dense per-token read over all 512 experts** (softmax-before-topk), so it must
stay always-resident backbone, never offloaded with its layer's experts; v1's "two independent routes
agree" reconciliation against the published 125B parameter count is **retracted** (was an artifact of the
QSA/GR undercounts) — a real 0.71B discrepancy remains unexplained, cheaply resolvable via a 131-header
shard census that has not yet been run (§6c item 7, new).

**User decisions on the agent's 8 open questions** (via AskUserQuestion, same session): (1) run
synthetic-shape validation AND the real acquisition **in parallel, not gated on each other** — explicit
user reasoning: "getting real weights from disk will also drive the how/when/where to effectively
integrate sooner," i.e. don't treat the real file as merely a later swap-in; (2) transplant format is
**GGUF/`UD-IQ1_S`**, not the agent's own safetensors/bf16 recommendation — consequence: `gguf.hpp` needs
real `IQ4_NL`/K-quant dequant, a bigger lift than the `BF16`-only addition the safetensors path would have
needed; (3) first real-run scope is a **4-layer real sub-stack** (layers 0-3, the real 3-GDN+1-QSA
repeating unit), not the full 48; (4) Blocker A (`--head-dim`) gets the **full faithful fix**, not
deferred — real attention weights must be transplantable. Recorded in `docs/WP4_SCOPE.md` §7 (`81c5320`).

**Real download completed this session**: all three `UD-IQ1_S` GGUF shards (10.9MB + 49.99GB + 22.54GB =
72.55GB exactly, byte-verified against the HF API's own reported sizes) fetched via resumable `curl -C -`
to `D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S\` (outside the git repo). Explicitly a full-tier
download, NOT the surgical byte-range extraction this project used five times before — a deliberate
override of `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §4b's own "do not acquire any GGUF quant tier
wholesale... as this project's next action" conclusion, justified because WP4 (unlike every prior
mechanism-building pass) actually needs resident weights to run a forward pass, not fixture-sized byte
ranges. Includes the n-gram/PLE table (not stripped — surgically excluding it means a byte-range-aware
skip inside a shard where it's interleaved with other tensors, which duplicated the scoping agent's
in-flight design work, so simplicity won here; having it locally isn't wasted regardless, per the plan's
own Sub0Firn-seam framing).

Also found (not yet actioned): no built `llama.cpp` binary on this machine, but an UNBUILT source checkout
exists at `D:\Craig\diffusiongemma\llama.cpp` — WP4f is "build it," not "acquire it from scratch."

**2026-09-04: WP4b DONE, reviewed, MERGED to main (`fe39cca`).** Dispatched to Opus (branch
`feature/wp4b-shape-blockers`); hit a rate limit once mid-task (resumed cleanly via SendMessage, actually
continued the interrupted work rather than restating a plan — contrast with the QSA perf-fix agent's
earlier two false starts). All four blockers landed as four separate commits (B, C, D, A in that order).

**Headline result**: `make_param_layout()` at Qwen4's real `RunConfig` axes now produces
`PARAM_FLOATS = 125,711,064,960` — **exactly** the real model's own total (was `124,027,786,776` before
this stage; the full 1,683,278,184-float gap is closed to zero). `NUM_PARAMS = 74,803`, not the `74,899`
originally estimated in §2h — the difference is exactly `2 × 48`, the `Ln1`/`Ln2` pairs blocker D
correctly removes (the agent explicitly did NOT fudge the target to match a wrong prior number — both
totals are cross-checked against an independently hand-written per-tensor census in the same test file,
[[independent-reimplementation-catches-identity-swap-bugs]] applied to shape rather than math).

**Two real bugs found by the consumer sweep**, both invisible at every neutral shape: (1) `forward_one`'s
decode path spelled the head width `C / H` instead of `D_HEAD` — caught immediately by the existing
forward-vs-`forward_one` parity check the moment a genuinely non-square build existed; (2) the
`QSA_INDEXER_HEAD_DIM >= D_HEAD` static_assert would have rejected the REAL model's own configuration
outright (128 ≥ 256 is false) — relaxed to `>= ROTARY_DIM` (128 ≥ 64), the actually-tight bound
`rope_apply_row` indexes.

**My own independent re-verification before merging** (same standard as every prior mechanism, and this
one mattered more than usual given the size): rebuilt the new `sub0_qwen4_shape_tests` binary myself and
reproduced `NUM_PARAMS=74803`/`PARAM_FLOATS=125711064960` exactly. Rebuilt all three standard shapes and
got hash-identical results to the agent's own report — then went one step further than usual and
independently reconstructed the "before" state myself (checked out the pre-WP4b file versions into the
same worktree/corpus/script rather than trusting the agent's own before/after comparison) and confirmed
byte-identical neutral hashes at d196 (`a4d2041b5e3128d4`/`550cdbd1222f2ddb`/`d9eda44f32a060f9`, 145→147
cases from the 2 new layout tests, nothing else). Confirmed all four real-weight fixture tests
(GDN/GR/MoE/QSA) byte-unchanged at 115,424 assertions/197 cases. Spot-checked both bug fixes and the new
`ARCH_FINGERPRINT2` bit-packing directly in the diff — correct, well-documented, consistent with the
established rule-#1-vs-rule-#2 classification discipline.

**Honestly self-flagged tradeoffs, checked and accepted rather than blocking merge**: `GDN_KEY_HEADS`
joining `ARCH_FINGERPRINT2` exhausts its 56 spare bits — the next shape-neutral/computation-changing axis
needs a third additive word (a known, named future cost, not a bug); the `rms_norm_eps` gap is only HALF
closed (GR's own exposure to `op_rmsnorm`'s `1e-5` is gone, but `LnF`/`op_qknorm` still use it — scoped
out as its own future change, recorded in `docs/GATED_RESIDUAL.md` §2, not silently left inconsistent);
`memplan::Dims::head_dim` is appended at the END of the struct rather than semantically next to
`n_kv_heads`, specifically because 5 call sites aggregate-initialize `Dims` POSITIONALLY (AGENTS.md §10's
own named bug class) — a legitimate safety-first tradeoff, though those 5 call sites are a good future
boy-scout target; gate item 4 ("memplan's `param_floats()` equals `PARAM_FLOATS` at the real axes") was
literally unsatisfiable since memplan has no GDN/GR/MoE/QSA term at all, so it was rescoped to a `d_q`/
`d_kv`/`qkv_stride`/`qk_pre_stride` identity check plus a NEW always-on (not CUDA-only) `static_assert`
guarding the dense-architecture identity — a real improvement, not a dodge. CUDA: a new
`static_assert(N_HEADS*D_HEAD==D_MODEL)` guard added mirroring the GR/MoE/QSA precedent; GDN's own
existing CUDA path is unaffected (it already takes runtime `gdn::Dims`).

Cleaned up: merged `feature/wp4b-shape-blockers` into `main` as `fe39cca`, pushed, removed the worktree,
deleted the branch (local + origin).

**2026-09-04: real GGUF census (before dispatching WP4c) found the dequant scope is much bigger than
assumed.** With all three `UD-IQ1_S` shards fully downloaded locally, I wrote a throwaway tool against
this project's own `gguf::Reader` to read the ACTUAL per-tensor type breakdown directly (no more guessing
from an HTTP header probe). Real result, recorded in `docs/WP4_SCOPE.md` §3a-bis (`d2314fb`): **seven
GGML formats beyond `gguf.hpp`'s existing F32/F16/Q8_0** — `Q4_K`(embeddings), `Q5_K`(attention/GDN
projections), `Q6_K`(GDN out_proj), `Q8_0`(shared-expert/hyper-connection down/up), `IQ1_S`/`IQ2_XXS`
(MoE routed gate/up, **varying PER LAYER** — unsloth's own "Dynamic" mixed quantization, e.g. layers
0/3/5 use `IQ1_S` while 1/2/4 use `IQ2_XXS` for the identical logical tensor), `IQ4_NL`(MoE routed down +
the n-gram table), `BF16`(QSA indexer). Two structural findings beyond the format checklist: (1) GGUF's
tensor granularity ≠ the HF/safetensors granularity every mechanism doc's transpose table was built
against — the indexer's `index_qk_proj` and MoE's `gate_up_proj` both arrive as TWO separate GGUF tensors
each, needing concatenation, not just transpose; (2) dequant dispatch must be driven by each tensor's own
`type_raw` per-instance, never assumed from its name/role, because of the per-layer quantization mixing.

**Surfaced this to the user before committing to the larger implementation** (via AskUserQuestion, since
it materially changed the size of the engineering commitment from "add IQ4_NL" to "seven formats + a
different tensor-pair reconstruction design") — **user re-confirmed the full GGUF path** with the real
cost now explicit, rather than switching to the agent's original safetensors/bf16 recommendation.

**2026-09-04: WP4c DONE, reviewed, MERGED to main (`4659eb5`).** Dispatched to Opus (branch
`feature/wp4c-transplant`); completed in one pass, no rate-limit interruption this time.

**What exists now**: `include/sub0/gguf.hpp` decodes all seven real formats (`Q4_K`/`Q5_K`/`Q6_K`/
`IQ1_S`/`IQ2_XXS`/`IQ4_NL`/`BF16`), transcribed from ggml's own `dequantize_row_*` reference (fetched
from the local `llama.cpp` checkout at `D:\Craig\diffusiongemma\llama.cpp`, per AGENTS.md §5), unified
under one `block_spec()` size-rule table. `include/sub0/transplant.hpp` is the engine-free GGUF-name→
`PARAM_LAYOUT` mapping. `tools/sub0llm-transplant.cpp` is the offline tool (compiled against the real
Qwen4 axes with `N_LAYERS` forced to 4, links no engine library — same ODR-avoidance shape as WP4b's
`sub0_qwen4_shape_tests`). `include/sub0/model_file.hpp` extracts the `S0L5` header out of
`engine_core.cpp` (pure move, size pinned, zero byte-layout change) since the transplant tool is now a
second writer of that format.

**Produced a real artifact**: `D:\ModelWeights\Sub0Llm-Qwen4-sub4\qwen4_sub4.bin`, **46,590,469,832
bytes** (11,647,617,440 f32 floats — a 10.5x expansion from 4,418,181,760 encoded GGUF bytes; MoE is
97.3% of every layer) — the real 4-layer sub-stack (3 GDN + 1 QSA, all 512 MoE experts per layer since
routing is a runtime decision), transplanted from the already-downloaded `UD-IQ1_S` file.

**Four-level gate, all green**: (i) total-count reconciliation — 6,240/6,240 destinations filled, 2
synthesized (`ln_f`/`lm_bias` — **the real GGUF has no `output_norm` tensor under any name**, a genuine
open question flagged for WP4d, not smoothed over), 7 unmatched sources (all PLE, correctly excluded);
(ii) per-tensor statistics — 6,238 checked, 0 mismatches; (iii) layer-0 GDN fixture replay —
**4.37e-11**, matching the mechanism's own already-established fixture precision; (iv) layer-3 QSA
fixture replay — **1.86e-09**, same. Three mutation probes (alpha/beta identity swap, naive
down-the-middle q|gate split, swapped indexer concat order) each produced a real, detectable divergence
— the gate isn't vacuous.

**My own independent re-verification before merging**: rebuilt `sub0_frontend_tests` from a clean
worktree checkout and reproduced 116,990 assertions/216 cases exactly. Rebuilt the transplant tool
myself and re-ran `--verify` against the EXISTING artifact (not a fresh transplant) — reproduced every
gate-level number exactly (6240/6240, 2 synthesized, 7 unmatched, 6238/0 mismatches, 0 bit-for-bit
mismatches of 6240). Spot-checked the K-quant sub-block scale/min unpacking (`get_scale_min_k4`, the
classically fiddly part of these formats) against my own knowledge of ggml's real implementation —
exact transcription. Verified the alpha/beta GDN tensor-name mapping is sourced from a real
`llama.cpp`/`gguf-py` `tensor_mapping.py` table (checked in the diff), not guessed. Confirmed the
`model_file.hpp` extraction is a byte-identical pure move: rebuilt `sub0_tests` at the neutral d96
shape and got hashes identical to WP4b's own recorded baseline (`05053d4289968fa1`/`bc2a4a880031821e`/
`7172b328827c0196`, 17,827,371/147). Confirmed the artifact file's actual size on disk matches the
claimed byte count exactly.

**Real file contradicted `docs/WP4_SCOPE.md`'s own §3a-bis estimates in several ways** — recorded in the
doc rather than silently reconciled: MoE needs per-expert 3-D slicing (not the fused-tensor split §3a-bis
assumed); QSA's `attn_q` is a per-head split (§3a-bis only flagged the indexer as a granularity
mismatch); `output_norm.weight` doesn't exist under any name; the PLE is six tensors at layer 1, not
just the one big table; two 2-D tensors (`token_embd`, `ssm_conv1d`) do NOT transpose despite the "every
2-D weight transposes" rule — because GGUF's `ne` axis order already reads like this project's own
`[in,out]` convention for exactly those two, while the underlying bytes are the PyTorch transpose. This
last one is worth remembering: **no shape assertion can catch a missed transpose there**, since both
orientations declare identical dims.

**Housekeeping**: the `data/tokenizer_calibration.txt` corruption (git-LFS pointer, same file as before)
recurred a THIRD time — this time from the agent's own test run, not mine. It self-reverted rather than
committing it, correctly, but flagged it as worth a root-cause look given the repeat count. Neither the
agent's final push nor my own independent verification run left it dirty.

Cleaned up: merged `feature/wp4c-transplant` into `main` as `4659eb5`, pushed, unlocked+removed the
worktree (it reported "locked" on completion — needed `git worktree unlock` before `remove --force`),
deleted the branch (local + origin).

**2026-09-04: WP4d dispatched** (Opus, branch `feature/wp4d-real-forward`, background). This is the first
stage to actually LINK the engine at the real Qwen4 axes (4 layers) and call `Model::forward` — every
prior real-axes target (WP4b's shape test, WP4c's transplant tool) deliberately avoided linking
`sub0_core` to dodge an ODR violation on `PARAM_LAYOUT`. Task: build a real configurator-generated build
at these axes, load `qwen4_sub4.bin` through the REAL `load_model()`, run `Model::forward` on the
existing fixture's own known input and diff against the known real output (a STRONGER check than WP4c's
own math-core-level replay, since it now exercises the Node graph and GR's real wrapping too), run
`forward`-vs-`forward_one` parity at real dims (named in the plan as "the check that has caught a real
bug in every one of WP1-3"), and resolve the two open items WP4c's report flagged: whether the
synthesized `LnF` (the real GGUF has no `output_norm` tensor under any name) is actually correct, and
whether `op_rmsnorm`'s still-1e-5 eps ever gets hit on a GR-wrapped GDN/QSA layer's real path (WP4b's
Blocker D only closed this for GR's OWN exposure, not the general op).

**Real resource constraint flagged explicitly in the brief**: this machine has 63.4GB RAM, ~49GB free at
dispatch time; the artifact alone is ~43.4GB resident with no offload (WP4e not built yet) — tight, not
comfortable. Agent was told to use the smallest feasible `T`, not run other heavy processes
concurrently, and to report a genuine infeasibility-on-this-hardware finding honestly rather than
force a fit by shrinking the real model's shape (which would stop being a real run).

**2026-09-04: WP4d DONE, reviewed, MERGED to main (`3cca1d4`) — THE REAL QWEN4 SUB-STACK RUNS.** Hit a
rate limit once mid-task, resumed cleanly (continued the actual work, no false restart). This is the
first stage to compile+link `sub0_core` at the real Qwen4 axes at all — every prior real-axes target
(WP4b's shape test, WP4c's transplant tool) deliberately avoided linking the engine to dodge an ODR
violation, so this is also the first stage that could have found a bug the isolated math-core checks
structurally could not see.

**It found five**, every one invisible to WP4b/WP4c's own gates for exactly that reason:
1. **A real, exploitable stack-buffer-overflow segfault in `gdn_math.hpp`** — `forward()`'s and
   `backward()`'s per-row `gated`/`d_gated_row` buffers were `float[4096]` stack arrays with a comment
   calling that "generous" for `value_dim`. The real model's `value_dim` is 48×128=6144 — a 2048-float
   stack overrun, a hard crash on the very first real GDN layer. No existing fixture (all sliced,
   value_dim 384) could ever have reached it. Fixed by carving both rows from caller-owned scratch, sized
   by `Dims` — applied to BOTH forward and backward even though backward is unreachable in the specific
   build that found it (GR/MoE/QSA abort in `backward_node`; only GDN's own backward is real) — correct
   "fix the class, not the instance" discipline.
2. `core.hpp`'s `static_assert(D_MODEL % N_HEADS == 0)` — a leftover invariant from when `D_HEAD` was
   always derived; false at the real axes now that WP4b's `--head-dim` exists. Relaxed to the quantity
   that actually holds by construction (`D_Q % N_HEADS == 0`).
3. **The configurator had no `--d-ff` at all** — FFN/MoE-expert width was always derived from `D_MODEL`
   (6848 vs. the real `moe_intermediate_size` 640, a per-expert weight SHAPE under MoE, not a tuning
   knob). Added, 0=derive.
4. **The configurator's `d_model % n_heads` check ran BEFORE `--head-dim`'s own resolution**, refusing
   exactly the real Qwen4 shape (2560/24 heads) the flag exists to express — WP4b's blocker A, half-landed.
5. **`ensure_shared_params()` eagerly allocated ~217 GiB** (4× `PARAM_FLOATS`-sized arenas + a 5th per
   Worker) on a 63GB machine, before a single token embeds. New derived `FORWARD_ONLY = USE_GATED_RESIDUAL
   || USE_MOE || USE_QSA` (reads the SAME `USE_*` constants the `backward_node` abort()s are written
   against, so it cannot drift out of agreement) skips the three dead arenas; GDN correctly excluded
   (its backward is real). Every access point refuses loudly at its own seam
   (`grad_ptr`/`adam_m_ptr`/`adam_v_ptr`/`reduce_gradients`/`AdamW::step`/`mk_param`'s per-leaf grad span/
   the n-gram grad alias) rather than reaching a null pointer. No default build turns any of the three
   mechanisms on, so this is bit-identical everywhere that exists today.

**Real numbers, on real weights, for the first time**: `load_model()` accepted the artifact first try
(~41s, 43.39 GiB). `Model::forward` vs. an independent math-core replay: layers 0-2 (GDN) agree to
1.8-3.3e-08 relative — float32 noise, no structural disagreement (layer 3/QSA excluded — `op_qsa` needs
the backend's internal rope table, not exercisable from a standalone replay). `forward`-vs-`forward_one`
parity: max|diff| = 5.96e-06 over 6×248,320 logits — **no bug found this time** (unlike WP4b's real
`C/H` bug at this exact same check). Peak working set 45.29 GiB against ~48GB free — tight but real,
and only possible because of the `FORWARD_ONLY` fix (without it: ~217 GiB, i.e. this run could not have
started at all).

**The two open questions from WP4c, answered with measurements, not assumptions**: eps is reachable at
exactly ONE site (`h = op_rmsnorm(h, ln_f)` — GR's own norms and QSA/GDN's own already hard-code the
correct 1e-6, `op_qknorm` compiles out under `USE_QSA`); moving it 1e-5→1e-6 shifts logits by 1.67e-05,
argmax unchanged — real but small. **`LnF`'s synthesized 1.0 does NOT reproduce the real model** —
RMSNorm is a normalizing OPERATION, not just a gain; measured `max|with ln_f − without any final norm| =
0.174` against logit rms 0.64 (~27% of scale), reaching 4.90 at position 0. The real GGUF has no
`output_norm` tensor under any name (confirmed independently in WP4c), so the structurally correct fix
is REMOVING the site entirely under `USE_GATED_RESIDUAL` (exactly Blocker D's own precedent) —
deliberately NOT done in this stage, since it's shape-changing and would invalidate the existing 46.6GB
artifact, needing its own WP4c-shaped transplant re-run and gate. **Until that lands, this build's
logits are not the real model's — WP4f must compare hidden states, not final logits.**

**My own independent re-verification before merging went further than any prior stage, given the
significance**: rebuilt `sub0_tests`/`sub0_frontend_tests` from a clean checkout, reproduced both counts
and hashes exactly (17,827,371/147 at d96; 116,990/216 for the fixtures). Reviewed the `gdn_math.hpp`
fix's scratch-sizing arithmetic and confirmed it's correctly threaded through both `forward()` and
`backward()`. Confirmed every `FORWARD_ONLY` access point refuses at its own seam. **Then, given this
stage's significance, went all the way and reproduced the ENTIRE real-axes run myself from scratch** —
ran the real `sub0llm-configure` invocation to completion (vocab pinning `--vocab 248202` → `248320`
confirmed exact), built `sub0_core` + the new `sub0llm-qwen4-forward` tool at the real axes myself, and
ran it against the existing artifact: **every single number reproduced exactly** — load time, memory,
all three layer diffs to the exact exponent, forward/forward_one parity, both eps/LnF measurements. Not
approximately — exactly, digit for digit.

Cleaned up: merged `feature/wp4d-real-forward` into `main` as `3cca1d4`, pushed, removed the worktree,
deleted the branch (local + origin). No repo corruption this time (checked `data/tokenizer_calibration.txt`
explicitly — clean).

**Remaining before WP4f (llama.cpp comparison)**: the `LnF` removal (shape-changing, needs its own
transplant re-run — the natural next small stage); WP4e (now rescoped per the user's own catch to
dequantize-on-demand rather than f32 placement — see the separate 2026-09-04 entry above); layer-3 (QSA)
end-to-end validation through the full engine path (only layers 0-2/GDN were checked this stage, since
the standalone replay can't reach `op_qsa`'s internal rope table — needs a different validation approach,
not yet designed).

**2026-09-04: `LnF` question DEFINITIVELY resolved, fix dispatched.** User asked to fix it immediately
while fresh. Before assuming "the real model has no final norm" (my first instinct), checked whether a
common GGUF conversion technique — folding a final RMSNorm's gain into the following `lm_head` matrix
(mathematically exact, since the gain is an elementwise scale that commutes with the matmul contraction)
— might mean the synthesized `gain=1.0` was already CORRECT rather than wrong. The local `llama.cpp`
checkout (`D:\Craig\diffusiongemma\llama.cpp`, dated 2026-06-17) predates Qwen4Exp support entirely (zero
hits for qwen4/hc_norm/hyper_connection), so its real converter source couldn't be read directly. Fetched
the REAL `model.safetensors.index.json` directly from HuggingFace instead (170,726 bytes, the actual
source checkpoint's own tensor list) and searched every non-per-layer "norm" tensor name: the ONLY
language-model-side one is `model.language_model.hyper_connection_mixer.hc_norm.weight` — the GR
model-level EXIT instance's own `hc_norm` (already a real `GrHcNorm` `PARAM_LAYOUT` entry, already
correctly transplanted, confirmed present in the GGUF too as `output_hc_norm.weight`). **No separate
final-norm tensor exists anywhere — not in the GGUF, not in the real safetensors checkpoint.** Two
independent real sources agree. `docs/GATED_RESIDUAL.md`'s own earlier citation of a
`Qwen4ExpTextModel.norm` (asserted, never actually checked against the real checkpoint) is **wrong** —
flagged for correction. Conclusion, not a hypothesis: the real model has no separate final RMSNorm at
all; the GR exit's own `hc_norm` (computed as part of that instance's `mixed_input`) IS the model's
actual final normalization before `lm_head`.

**DONE, reviewed, MERGED to main (`ec98ba9`).** Dispatched to Opus (branch `fix/gr-remove-lnf`), completed
in one pass. `make_param_layout()` stops emitting `LnF` under `USE_GATED_RESIDUAL` (exactly Blocker D's
own precedent — `gr_param_delta()` gains a further `-d_model`, once, at the model level); `Model::forward`/
`forward_one` feed `lm_head` straight from the GR exit's own `mixed_input`; `transplant.hpp` drops the
`Dest::LnF` enumerator entirely (no slot, no source, nothing to synthesize); `docs/GATED_RESIDUAL.md`'s
wrong citation corrected, kept auditable.

**Shape**: `PARAM_FLOATS` 125,711,064,960 → **125,711,062,400** (exactly `-D_MODEL`=2560),
`NUM_PARAMS` 74,803 → **74,802** (exactly -1). Regenerated the WP4c artifact (old one preserved as
`qwen4_sub4.bin.pre-lnf-fix`): 6,239 tensors, four-level gate green (level 3 GDN layer-0 replay 5.09e-11,
level 4 QSA layer-3 replay 1.40e-09, 0 bit-for-bit mismatches). **`forward`-vs-`forward_one` parity is now
EXACTLY 0** (was 5.96e-06) — the two paths' previously slightly-different normalization implementations
are simply gone, a clean structural confirmation this was the actual source of that residual disagreement.
Logits now span a real dynamic range (rms 0.61-2.31, was artificially flattened to 0.60-0.83 by the
redundant per-position renormalization) — structurally what removing a norm the real model lacks must
look like, though not yet checked against a real reference (no real-dims oracle exists in this repo yet;
WP4f's llama.cpp comparison is what would settle it, and per this same finding must compare hidden states
rather than final logits regardless, since layer-boundary exclusions elsewhere already required that).

**My own independent re-verification**: reviewed the layout.hpp/backend_cpu.cpp/transplant.hpp diffs
directly (the GR-only guard correctly mirrors Blocker D's own idiom; `forward()`/`forward_one()`'s
branches are correctly made mutually exclusive, explaining the parity improvement). Rebuilt
`sub0_tests`/`sub0_frontend_tests` from a clean checkout and reproduced every number exactly: neutral d96
hash byte-identical with the expected +1 assertion, frontend suite exactly -1. Built and ran the
real-axes shape test myself and reproduced `NUM_PARAMS=74802`/`PARAM_FLOATS=125,711,062,400` exactly.
Did not personally re-run the full 43GB artifact regeneration + WP4d re-verification (given the
extreme resource/time cost relative to marginal value once the code-level fix was thoroughly confirmed
correct and the reported artifact-level numbers were already strongly self-consistent — e.g. reproducing
WP4d's own original 0.174/27% perturbation figure exactly as the predicted delta).

Cleaned up: merged `fix/gr-remove-lnf` into `main` as `ec98ba9`, pushed, removed the worktree, deleted
the branch (local + origin).

**2026-09-04: post-WP4b-d review pass (user request, "standing project methodology") — clean, no
findings.** Reviewed the newly-landed GGUF dequant decoders, `transplant.hpp`, `gdn_math.hpp`'s scratch
fix, and `backend_cpu.cpp`'s `FORWARD_ONLY` change for optimization/efficiency/DRY. Conclusion: no
actionable issues. The Q4_K/Q5_K decoders share an obvious-looking sub-block pattern but were
deliberately kept as separate, faithful transcriptions of ggml's own reference decoders (per AGENTS.md
§5) rather than artificially unified — the right call for bit-level format code, where "DRY" would trade
verifiable correctness for a modest line-count win. The offline transplant tool already reuses its
buffers across all 6,239 destination tensors; `transplant.hpp` has zero dynamic allocation; the engine
fixes are `if constexpr`-gated (zero runtime cost). This is a case where the review's honest output is
clean, not a forced finding.

**DONE, reviewed, MERGED to main (`7a301be`).** Dispatched to Opus (branch `feature/wp4e-dequant-on-demand`),
completed in one pass. New `S0Q1` sidecar format (`include/sub0/moe_quant.hpp`) holds each routed
expert's gate/up/down planes in their native GGML encoding — per-layer mixed quantization carried
per-tensor, never assumed. Main model file drops `MoeGate`/`MoeUp`/`MoeDown` entirely under a new
`MOE_QUANT_EXPERTS` axis (shape-changing, `PARAM_FLOATS` discriminates it, no `ARCH_FINGERPRINT2` bit
needed — none was free after WP4b anyway).

**The structural key to why the bitwise gate is trustworthy, not just measured**: `moe_math.hpp`'s
`forward_row`/`forward` are now thin wrappers over new `forward_row_via`/`forward_via` functions taking a
`resolve(expert_idx)` callback — the SAME arithmetic in the SAME order runs for both the f32-resident
path (a lambda indexing pointer arrays) and the new quantized-resident path (a lambda that dequantizes on
demand into a compile-time-sized pool, `MOE_RESOLVE_SLOTS=8`, though correctness needs exactly 1 — every
extra slot is a pure cache). One code path, not two that happen to agree.

**Measured, same 4 real layers**: `NUM_PARAMS`/`PARAM_FLOATS` 6,239/11,647,614,880 (all-f32) →
**95/1,581,285,280** (quantized-resident). On-disk: 43.39 GiB → **5.89 GiB blob + 3.17 GiB sidecar = 9.06
GiB total (4.79x)**; the 6,144 routed expert tensors alone go from 37.50 GiB f32 to 3.17 GiB encoded
(**11.82x**). Peak working set for a real forward pass: 45.29 GiB → **14.32 GiB (3.16x)**. Real,
honestly-reported cost: `Model::forward` on 6 tokens goes from 1.43s to **5.27s (3.7x slower)** — the
right trade only because the f32 alternative doesn't fit past 4 layers at all.

**A real defect found and fixed, same class as every prior WP4 stage**: the sidecar's descriptor table
was built walking `(layer, plane, expert)` while a running-cursor offset was assigned in that SAME walk
order, but the payload is written in descriptor-index order `(layer, expert, plane)` — every descriptor
pointed at the position its tensor would have had in the OTHER order. Caught by the tool's own bit-for-bit
sample check: **92 of 96 planes mismatched** before the fix, **0 of 96** after. Fixed by decoupling
offset assignment into its own pass over the finished descriptor array, in the exact order the payload
write uses. `tests/moe_quant_tests.cpp` pins the ordering as a permanent regression.

**Gate result**: `Model::forward`'s full `[6×248320]` logits, dumped from both the existing all-f32 build
and the new quantized-resident build, SHA-256-identical — 1,489,920 float32 values, 0 differing bits.

**My own independent re-verification before merging**: reviewed the `moe_math.hpp` refactor directly and
confirmed `forward_row` is genuinely a thin wrapper, not a parallel implementation — the actual structural
basis for trusting the bitwise gate. Reviewed the descriptor-ordering bug fix and confirmed the
offset-assignment and payload-write passes now walk the array in identical order. Rebuilt
`sub0_tests`/`sub0_frontend_tests` from a clean checkout and reproduced every number exactly (neutral d96
hash byte-identical; frontend suite exactly +369 assertions/+6 cases, all passing including the
ordering-regression test). Confirmed all artifact files on disk match their exact claimed byte sizes, and
**independently re-hashed the two saved logit dumps myself** (`sha256sum`) — confirmed identical,
verifying the headline gate directly from artifact evidence rather than the agent's self-report alone.

**Honestly-flagged open items** (not correctness gaps — the bitwise gate already proves numerical
equivalence — real follow-up work): CUDA wasn't rebuilt this pass (low risk, `MOE_QUANT_EXPERTS` is
compile-time false in every neutral build); the 3.7x slowdown is unattributed to a specific cause
(suspected but not profiled); the resolve pool's hit rate is uninstrumented, so `MOE_RESOLVE_SLOTS=8` was
chosen by argument not measurement; `print_host_memplan` under-reports by the sidecar's 3.17 GiB since
that's a runtime, not compile-time, quantity.

Cleaned up: merged `feature/wp4e-dequant-on-demand` into `main` as `7a301be`, pushed, removed the
worktree, deleted the branch (local + origin).

**All of WP4a-e are now done.** Remaining: WP4f (the llama.cpp comparison harness — must compare hidden
states, not final logits, per the earlier PLE-exclusion and LnF findings), the layer-3/QSA end-to-end
validation gap from WP4d, and the honestly-flagged WP4e follow-ups above (CUDA build, perf profiling,
cache instrumentation) whenever real-scale work returns to them.

**2026-09-04: extra review pass before starting WP4f** (user flagged quota running low, asked for extra
checks). Found and closed one thing WP4e had left open: `layout.hpp` already has
`static_assert(!MOE_QUANT_EXPERTS || USE_MOE)`, so the existing CUDA-side `static_assert(!USE_MOE)` in
`backend_cuda.cu` transitively refuses `MOE_QUANT_EXPERTS` too — the "CUDA wasn't rebuilt" gap WP4e
flagged is provably zero-risk, no rebuild needed to confirm it. Also ran a genuinely fresh, full-suite
regression at the production-shaped d196 config (not the d96/d132 shapes used at each individual merge)
— hit the same "stale generated header" class of issue seen before (this build dir predated WP4b's new
GDN axes), regenerated through the real configurator, and got a clean pass (28,755,032/147,
117,358/222 — exactly the counts expected from current `main`), confirming the whole WP4b→c→d→LnF→e
sequence composes with zero accumulated regressions at a shape none of the individual merges checked.
Repo hygiene also confirmed clean (no stray worktrees/branches, no large files tracked, `out/` gitignored).

**WP4f started, parallelized across two agents** (a genuine change from WP1-3's forced-sequential
pattern, since WP1-3 all touched the same shared `layout.hpp`/`backend_cpu.cpp`/`configurator.cpp` while
WP4f's two halves touch completely disjoint things): Agent A (Sub0Llm repo) built the hidden-state-dump
harness; Agent B (no repo, entirely external) acquired/built llama.cpp. Both completed successfully.

**Agent A — DONE, reviewed, MERGED to main (`7e453a7`).** `sub0::forward_capture` (a named-intermediate
sink over the REAL `forward()` loop, matching the existing `loop_pass_stats` precedent exactly — armed
for one call, disarmed after, every emit() call a predictable never-taken branch in production) + a new
`S0HD` exchange format (`include/sub0/hidden_dump.hpp`, deliberately foreign-producer-friendly: named
tensors, the input tokens carried IN the file so the differ can VERIFY "same tokens in" rather than
assume it, no padding/compression so a 30-line foreign snippet can emit one) + `tools/sub0llm-hidden-diff.cpp`
(the per-layer `rel_l2` divergence table against a stated `1e-4` gate, links no engine so "nothing in
`src/` links llama.cpp" stays true by construction). **Canonical input fixed**:
`[1543, 88123, 245000, 7, 99999, 156789]` — 6 raw ids, no BOS, no tokenizer. Produced
`D:\ModelWeights\Sub0Llm-Qwen4-sub4\wp4f_sub0llm.s0hd` (10,261,285 bytes, 32 tensors, from the WP4e
quantized-resident artifact — 14.32 GiB peak, not 45 GiB). **Named the actual comparability bound**: the
real model injects PLE at decoder layer 1 (0-indexed), so only `tok_embd`/`gr_tile`/every `blk.0.*`
tensor is strictly comparable — `blk.0.out` is the bounding tensor for any pass/fail claim; everything
from `blk.1.res_in` onward is not the same function. My own independent re-verification: reviewed the
`emit()`/`emit_layer()` call sites directly (confirmed they match the established `pass_delta`/
`pass_hnorm` precedent, not a new invasive pattern), reviewed `hidden_dump.hpp`'s reader/writer in full
(clean, every foreign-input failure returns an error rather than asserting), rebuilt both test suites
from a clean checkout and reproduced every number exactly (neutral d96 hash byte-identical, frontend
suite exactly +73/+4), and confirmed the produced `.s0hd` artifact's file size matches exactly.

**Agent B — the llama.cpp acquisition, complete, real, and separately valuable beyond WP4f itself.**
Confirmed (not assumed) the existing `D:\Craig\diffusiongemma\llama.cpp` checkout (2026-06-17) and the
one at `D:\tools\llamacpp\vendor\llama.cpp` (found only AFTER the agent finished, via a Init.ps1-managed
project at `D:\tools\llamacpp\` the user pointed out — even older, 2026-05-29) both predate `qwen4exp`
support. Found real upstream support: PR #27742 (danielhanchen), merged into `ggml-org/llama.cpp` master
2026-08-27. Shallow-cloned `ccc3646` fresh to `D:\Craig\llama.cpp-qwen4exp`, built CPU-only via
clang++/Ninja (no vcvars needed), and got a REAL model load: the actual `UD-IQ1_S` GGUF loads (~45 GiB
peak, fits in the ~48GB typically free) and generates coherent English text. Wrote a minimal custom dump
tool (`examples/qwen4-dump/qwen4-dump.cpp`, ~250 lines, touches no llama.cpp model/architecture code) to
get explicit-token-array forward passes with full tensor dumps (nothing upstream accepted a raw token
array). Dumped a real forward pass — **but on a DIFFERENT token array** (`[760, 6511, 314, 9338, 369,
11751, 13]`, " The capital of France is Paris.") than Agent A's canonical one, since the two agents were
dispatched before either's exact input was fixed — **reconciling this (re-running llama.cpp's dump tool
with Agent A's exact 6-token array, or converting the existing dump) is the actual next, well-scoped step
to complete WP4f's cross-comparison, not yet done.**

**Two independently valuable findings from Agent B, beyond just "it builds"**:
1. **llama.cpp itself is non-deterministic at depth from MoE routing flips across different CPU repack
   settings** (same commit/weights/tokens, `--no-repack` vs default): layer-0 intermediates bit-identical,
   but 111 of 330 (layer,token) expert-set decisions differ starting at layer 6, growing to 88%-of-scale
   divergence by layer 47. Layers 0-5 route identically in both modes. **This sets how WP4f's own gate
   should actually be interpreted**: tight numeric gate only where routing is provably stable (layers
   0-3, which is exactly the sub-stack this project has); a discrete expert-set comparison, not a strict
   numeric one, for anything deeper — a second, independent reason for the same layer-boundary
   restriction `docs/WP4_SCOPE.md` §5's PLE exclusion already requires.
2. **llama.cpp's own source comment independently corroborates the `LnF` finding**: *"there is no
   output_norm: the final hyper-connection mixer carries it"* — confirming, from a completely different
   angle (reading the real inference engine's own code, not the checkpoint's tensor index), what this
   project found by fetching the safetensors index directly.

**Safety-checked Agent B's one git-internals touch**: it moved `origin/master` and added a shallow-graft
line in `D:\Craig\diffusiongemma\llama.cpp` while checking whether that checkout could be updated in
place (it could not — no local commits ahead, but a `diffpr` branch with unrelated diffusion-gemma work
was checked out). Verified independently: `git status --short` clean, still on `diffpr`, `git fsck` clean
— the repo is genuinely undamaged, the shallow-graft lines are a harmless byproduct of the fetch.

**User's own catch, recorded rather than acted on immediately (quota)**: `D:\tools\llamacpp\Init.ps1` is
this ecosystem's own proper, maintained way to acquire/update/build llama.cpp (GPU backend
auto-detection CUDA→Vulkan→CPU, model management, server/bench/tune scripts) — Agent B's ad-hoc fresh
clone to `D:\Craig\llama.cpp-qwen4exp` was real, necessary, correctly-scoped work given it didn't know
this tool existed, but going forward any further llama.cpp work should go through `Init.ps1` (which would
need `-ForceRebuild`/a `git pull` to get past its own stale `vendor/llama.cpp` checkout) rather than
continuing to accumulate ad-hoc clones. Not consolidated this session — flagged as a named follow-up.

**Quantization-format question, user's own catch, recorded not chased (quota)**: whether the S0Q1
sidecar should hold GGML's own K-quant/IQ-quant bytes verbatim (current, WP4e) vs. a Sub0Llm-native
re-encode better suited to this engine's actual consumption pattern (`docs/WP4_SCOPE.md`'s WP4e section,
2026-09-04 addendum). Key reasoning worked through: GGML's formats are co-designed with FUSED
quantized-matmul kernels this engine doesn't have (it fully dequantizes to f32 first, so the sub-block
bit-unpacking complexity buys nothing here); re-encoding cannot recover fidelity already lost by
unsloth's own quantization (only speed/size trade against each other — e.g. `Q8_0`, already the
simplest decoder in `gguf.hpp`, is markedly faster to decode but 2-5x more bytes than the current mixed
`IQ1_S`/`IQ2_XXS`). Recommended order: profile the actual `resolve()` cost (VTune) BEFORE investing in a
new format — the payoff is currently a guess, not a measurement, and the cheaper suspect (the reused
`std::vector::assign` re-zeroing per resolve, already flagged by WP4e) should be ruled in/out first.

**2026-09-07: ran the ACTUAL WP4f cross-comparison for the first time — a real, localized divergence
found, root cause dispatched.** With both agents' work merged, reconciled the token-array mismatch
myself: re-ran `llama-qwen4-dump` with Sub0Llm's own canonical array
(`1543,88123,245000,7,99999,156789`), wrote a small conversion script (kept in scratchpad, not
committed, per policy) turning llama.cpp's raw ggml tensor dumps into `S0HD`, and ran
`sub0llm-hidden-diff` for real. **Independently confirmed the conversion's own axis semantics before
trusting the result**: checked `hc_init`'s raw floats directly and confirmed all 4 hyper-connection
streams hold an identical copy of the embedding per token — exactly what a tile op must produce,
confirming ggml's `[D_MODEL,HC_COUNT,T]` layout really does map to Sub0Llm's own `[T,HC_WIDE]`
stream-major convention with no transpose needed (not an assumption).

**Result**: `tok_embd`/`gr_tile` (everything before any decoder layer runs) match EXACTLY, bit for bit
(`rel_l2=0.0`, cosine 1.0). **`blk.0.out` through `blk.3.out` all fail the `1e-4` gate**, starting at
layer 0 itself (`rel_l2=0.996`, cosine 0.594) — far beyond llama.cpp's own measured internal noise floor
at this depth (~6.7e-04 from CPU repack settings alone), so this is a real disagreement, not noise.
Growing divergence at deeper layers is consistent with layer 0's error propagating forward, not
independent new errors per layer. **This IS WP4f's actual deliverable** per the plan's own framing — a
localized divergence is as valid and useful an outcome as agreement. Recorded in `docs/WP4_SCOPE.md`
(`2e1149a`).

**DONE, reviewed, MERGED to main (`296f2a1`) — ROOT CAUSE FOUND, FIXED, AND THE RESIDUAL GAP PROVEN NOT
TO BE A SUB0LLM DEFECT.** Dispatched to Opus, hit a rate limit mid-fix, resumed once (a real, substantive
resume — it picked up exactly where it left off, unlike the QSA agent's earlier false starts).

**Root cause, one wrong assumption with three instances**: `transplant.hpp` treated a GGUF tensor as the
HF checkpoint tensor with only its axes declared backwards. **It is not — llama.cpp's own converter
(`conversion/qwen.py`'s `Qwen3NextModel.modify_tensors`/`_LinearAttentionVReorderBase.modify_tensors`,
inherited by `conversion/qwen4exp.py`) rewrites VALUES and HEAD ORDER on the way in**, so its own graph
can be simpler — reading the file verbatim silently loads a different model, with every shape and every
per-tensor statistic correct (which is exactly why nothing caught it before). Three specific rewrites,
each confirmed against the real `UD-IQ1_S` bytes:
1. **Every `Qwen4ExpTextRMSNorm` gain is pre-folded to `1+w`** (the file's own converter adds the 1),
   except GDN's `ssm_norm` (deliberately excluded by the converter) — Sub0Llm's own `gr::hc_norm`/
   `qsa::rms_norm_row` ALSO add the 1, so a verbatim read applied `2+w`.
2. **`ssm_a` is stored as the FINISHED `-exp(A_log)`, not `A_log` itself** — `gdn_math.hpp` computes
   `-exp(a_log)` itself and so exponentiated a second time.
3. **GDN's value heads are reordered grouped→tiled** by the converter (so ggml's own tiled broadcast
   reproduces HF's `repeat_interleave`) — `gdn_math.hpp` implements the HF/grouped association directly,
   so the file's order needed undoing.

**Why every prior gate missed all three**: #3 is a PERMUTATION, so the existing level-2 per-tensor
statistics (mean/std/min/max) are PROVABLY blind to it —
[[independent-reimplementation-catches-identity-swap-bugs]]'s exact blind spot, now hit a second time in
this same project by a different mechanism. Levels 3/4's fixture replay re-encoded real HF values
VERBATIM into synthetic GGUF (not through the real conversion pipeline), so #1/#2's two omissions
cancelled each other exactly; and the level-3 GDN fixture happens to use `num_k_heads==1`, where #3's
permutation degenerates to the identity — untestable there even in principle. **Every prior WP4c/d/e
artifact silently inherited all three defects** — this wasn't a WP4f-only bug, it retroactively affected
everything built on `transplant.hpp` since WP4c, though it was invisible because nothing before WP4f
compared against an independent, real, full-precision reference.

**Fix**: `fold_for`/`apply_fold` (the two value rewrites) and `vperm_for`/`ungroup_v_heads` (the head
reorder) in `transplant.hpp`, applied to the destination AFTER the existing level-2 stats check (keeping
that check a test of placement alone). `NegExpALog`'s inverse refuses loudly (not a silent NaN) if a
source value isn't negative. New `tests/transplant_tests.cpp` case uses a genuine `n_k=2, n_v=6` shape
specifically because the existing fixture's `n_k=1` can't exercise the permutation at all.

**Verified numbers**: level 3 (GDN layer-0 replay) unchanged at 5.09e-11; level 4 (QSA layer-3 replay)
improved 1.86e-09→1.40e-09. New mutations: verbatim `ssm_a` moves layer 0 by 9.5e-05 (~1.9e6× the
achieved agreement); verbatim gammas move layer 3 by 5.7e-03 — real, detectable, not vacuous.

**Re-ran the actual cross-comparison**: `blk.0.out`'s `rel_l2` improved from **0.996 → 0.0221** (cosine
0.594 → 0.999810) — a ~45x reduction. **Still fails the strict `1e-4` gate — and this remaining gap was
PROVEN, not assumed, to be llama.cpp's own numerical behavior, not a Sub0Llm defect**: an independent
NumPy float64 reimplementation of the ENTIRE layer 0 (GR read, GDN, GR write, GR read, MoE+shared
expert, GR write), computed from the SAME dequantized weights, agrees with **Sub0Llm to 1.1e-06-3.6e-06**
at every intermediate (cosine 1.00000000) but with **llama.cpp to only 1.2e-02-3.2e-02** — and it
reproduces the differ's own `blk.0.out` number against llama.cpp to three significant figures (2.2105e-02
vs the differ's measured 2.211e-02). **Mechanism isolated directly, not inferred**: from the identical
input tensor, an F32-weight projection agrees with llama.cpp to 2.0e-07 while a Q8_0-weight one agrees to
only 3.5e-03 — perfect correlation with weight quantization tier, zero correlation with which mechanism
(GDN/GR/MoE) is involved. **llama.cpp's own matmul kernels quantize ACTIVATIONS, not just weights** — a
real, deliberate performance choice on llama.cpp's side, confirmed not a bug, and NOT bounded by the
earlier-measured 6.7e-04 repack/no-repack noise floor (both those runs share the same activation
quantization, so that number only ever measured summation order). Also measured directly (not assumed):
llama.cpp's own PLE contribution at layer 1 is 68.0% of the residual stream's norm — confirms layers 1-3
really are comparing two different functions, exactly as §5's exclusion predicted.

**The headline finding, worth restating plainly**: with the converter's conventions correctly undone,
Sub0Llm computes the real Qwen3.8-Flash-Next decoder layer 0 to **2.6e-06 relative against an
independent float64 reference** — a stronger correctness statement than a comparison against llama.cpp
itself could ever produce at this quantization tier, since llama.cpp's own quantized-activation kernels
are themselves ~1-2 orders of magnitude less precise than that reference.

**My own independent re-verification before merging**: reviewed `fold_for`/`apply_fold` and
`vperm_for`/`ungroup_v_heads` directly, hand-verified the RMSNorm-gain and `A_log` algebra and the
V-head grouped↔tiled index-mapping algebra myself (not just trusting the report). Confirmed the wiring
in `sub0llm-transplant.cpp` applies both inverses in the documented order with the documented loud-
refusal behavior. Reviewed the new unit tests — round-trips through the actual consumer formula, explicit
"this really moved, not vacuous" checks, explicit checks that only the V block reorders (Q/K untouched).
Rebuilt `sub0_tests`/`sub0_frontend_tests` from a clean checkout in an isolated worktree and reproduced
every number exactly (neutral d96 hash byte-identical, frontend suite exactly +233 assertions/+4 cases,
all 12 `[transplant]`-tagged cases passing).

**2026-09-08: NEW GOAL beyond the original plan — real interactive inference with the transplanted
model.** The original plan (Phase 0, WP1-4) ends at WP4; this is a genuinely new work package, named
WP5 in its own commits/docs. User's framing: "tackle newly identified issues as we find them and
proceed" — less design-doc-first ceremony than WP4's own staging, more direct forward progress, same
verification discipline once code lands.

**Three things are missing for real interactive inference, identified this pass**: (1) a real Qwen
tokenizer — Sub0Llm's own from-scratch BPE tokenizer has zero relationship to the transplanted model's
vocabulary; (2) the full 48-layer model — WP4's own artifacts are deliberately a 4-layer validation
sub-stack; (3) an actual interactive generation loop (sampling + CLI), not yet started, deferred until
(1) and (2) land.

**WP5a dispatched** (Opus, background, branch `feature/wp5a-qwen-tokenizer`): the real tokenizer is
confirmed `Qwen2Tokenizer` (byte-level BPE, GPT-2/GPT-4 lineage) — verified directly, not assumed, by
fetching the real `tokenizer_config.json` from `Qwen/Qwen3.8-Flash-Next` and loading it through the
locally-available `transformers==5.16.1`/`tokenizers==0.23.1` packages. Real facts gathered before
dispatch: `vocab.json`(6.7MB)/`merges.txt`(3.4MB) available (simpler than the unified 12.8MB
`tokenizer.json`); base vocab 248,044 + 276 special/added tokens = the real `VOCAB=248,320` axis
already used throughout WP4b-f; the real pre-tokenization regex uses Unicode property escapes
(`\p{L}`/`\p{N}`/`\p{M}`) that C++'s standard `<regex>` does NOT support — flagged explicitly as a real
technical obstacle to solve properly (hand-rolled UTF-8 codepoint classification, or another real
working approach), not paper over. **A genuinely surprising real fact found before dispatch, worth
remembering**: real Qwen tokenizes digit SEQUENCES one digit at a time (`"123456"` → 6 separate
single-digit tokens `[16,17,18,19,20,21]`), confirmed directly against the real tokenizer, included in
the dispatch brief so the agent's fixture gate covers it. Gated against the real
`AutoTokenizer.from_pretrained` reference (both `transformers`/`tokenizers` confirmed locally available),
same "verify against real reference" discipline as every prior WP4 stage. Deliberately scoped to NOT
touch the existing `tokenizer.hpp`/`tokenizer.cpp` (Sub0Llm's own from-scratch scheme, used by every
non-Qwen build) — a new, separate, engine-free component.

**WP5b dispatched** (Opus, background, branch `feature/wp5b-full-scale-transplant`): scale WP4's own
4-layer validation artifacts to the REAL full 48-layer model. Real, previously-identified risk carried
forward from WP4e's own "still open" list: the quantized-resident MoE sidecar is currently read fully
into memory (`moeq::Store`), fine at 3.17GiB for 4 layers but would be ~38GiB at 48 layers — genuinely
uncertain to fit on this 63GB machine even before the backbone's own (also-scaling) footprint.
Instructed to implement the `mmap` upgrade FIRST (a real prerequisite, not optional), verify it
bitwise-identical to the existing eager-read path at the already-validated 4-layer scale before trusting
it at 48, then attempt the real transplant and report HONEST measured peak RSS — explicitly told not to
force a fit by shrinking any real model axis, matching WP4d's own established precedent that "it doesn't
fit" is a valid, useful outcome, not a failure to route around.

**2026-09-08, once Codex freed the machine (full system, no contention): closed the one remaining gap —
personally reproduced the real artifact-level result end to end, not just the fixture-level numbers.**
The dispatched agent had already regenerated `qwen4_sub4_q_wp4f.bin`+`.moeq` (quantized-resident,
post-fix) and a fresh `wp4f_sub0llm_fixed.s0hd` dump — reused those rather than redoing the expensive
regeneration, but independently RE-RAN both tools myself against the files on disk: (1) built
`sub0llm-hidden-diff` fresh and diffed `wp4f_sub0llm_fixed.s0hd` against `wp4f_llamacpp_canonical.s0hd`
myself — reproduced `blk.0.out`'s `rel_l2 = 2.211e-02`, cosine `0.999810` EXACTLY, and every other
layer's numbers exactly, directly from the bytes, not the agent's self-report; (2) built the dedicated
`sub0llm-transplant-q` target and ran `--verify` against the same artifact from scratch — **0 of 95
bit-for-bit mismatches**, and the fold/reorder counts matched exactly (13 zero-centred gammas, 3
`ssm_a`→`A_log` corrections, 24 v-head reorders — identical to the agent's own reported counts). This is
now genuinely triple-verified: code review, test-suite reproduction, AND full artifact-level
reproduction from the raw GGUF file through to the final cross-comparison numbers, with zero step taken
on trust.

**Coordination note**: this entire investigation and merge happened WHILE a separate agent (Codex,
working on Muon optimizer changes per `docs/INDEPENDENT_REVIEW_BACKLOG.md`'s B14) had live, uncommitted
edits to `muon.hpp`/`backend_cpu.cpp`/etc. in the SAME shared working tree (a deliberate 2026-09-07
decision — see `docs/ACTIVE_WORK_LOG.md`, a new coordination file created this session). Confirmed via
`git diff <merge-base> <fix-branch>` that the fix branch's own commits never touched any of Codex's
files before merging, used explicit `git add <path>` (never `-A`/`-u`) throughout, and verified
`git status --short` before and after every commit — Codex's uncommitted work was never disturbed.

**2026-09-04: user caught a real architectural gap from the 500GB-at-48-layers number** — "that's why we
need iq4." WP4e as originally scoped (`docs/QWEN4_MEMORY_ORCHESTRATION.md` §3b/`docs/WP4_SCOPE.md`'s own
first WP4e draft) was a PLACEMENT policy: where do an expert's f32 bytes live (arena vs. mmap'd/staging
region). That solves the wrong problem — confirmed by checking `backend_cpu.cpp` directly:
`op_linear`/`op_moe` read a plain `std::span<float>`, no quantized-tensor concept exists anywhere in the
engine, so "offloading" f32 bytes still needs a 500GB f32 FORM to offload from in the first place. The
whole point of downloading a quantized tier was to keep the resident footprint small; a placement policy
over an already-materialized 500GB f32 blob has already lost that. **Rescoped WP4e in
`docs/WP4_SCOPE.md` (`938f53f`)**: routed experts must stay resident in their NATIVE quantized bytes
(`IQ1_S`/`IQ2_XXS`/`IQ4_NL`, per-layer-mixed per §3a-bis), dequantized only for the `experts_per_tok`
(10) actually selected per token into a small reusable scratch buffer, immediately discarded — exactly
what `llama.cpp` itself does, and exactly what the `gguf::` decoders WP4c just built are the right
primitive for. Confirmed this doesn't retroactively invalidate WP4c/WP4d: the f32 transplant was a
deliberate, correct, SCOPE-LIMITED choice for a byte-exact correctness gate at 4-layer scale (43GB,
which fits) — never intended as the full-scale storage design. Also confirmed the earlier
memory-orchestration doc's own §3b arithmetic ("2.64 MiB/expert IQ4_NL-class") was ALREADY using the
quantized size correctly — the mismatch was only ever in WP4c's own f32 transplant choice, not in that
doc's own math. New WP4e gate: bitwise-identical output between the existing all-f32 4-layer path and a
dequantize-on-demand path over the SAME 4 layers — checkable without needing the full 48-layer scale to
be meaningful.
