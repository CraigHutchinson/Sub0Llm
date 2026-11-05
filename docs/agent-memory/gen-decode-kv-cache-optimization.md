---
name: gen-decode-kv-cache-optimization
description: "Autoregressive gen KV-cache incremental decode (forward_one) — CPU+GPU DONE, shipped in gen_stage.cpp AND (2026-07-09) unified into preview_at/gen_self_stats via new sub0/decode.hpp -- autotemp went from ~1760s to 344s (~5x) on the real d448 model, GPU decode confirmed active. DKV register-split still pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Gen was O(n·T²): it re-ran the FULL forward over the whole context every token AND computed the
full-vocab LM head on all T rows (only the last used). Fixed with a KV-cache incremental decode.

**CPU forward_one — DONE, committed (44b74ed), SHIPPED in native.** `sub0::forward_one(id,pos)` +
`kv_reset` in backend_cpu.cpp: per-layer K/V cache, runs one row per token (row helpers
linear_row/rmsnorm_row/rope_row replicate the dense op_* math in the SAME order). Bit-parity unit
test (`forward_one (KV-cache) matches the full forward per position`, rel<1e-3). gen_stage fast-paths
it when ctx+n ≤ SEQ_LEN and !USE_TERNARY, else the old full-forward loop. **~27x/token at d192
(1.9 vs 51.7 ms), 13.5 ms/tok at d448, output coherent.** The [[scoring-use-batched-forward-not-forward-one]]
note predates this — forward_one now exists (for GEN/decode; scoring still wants batched forward).

**GPU forward_one — COMMITTED (fab30ef) but EXPERIMENTAL, NOT wired to gen.** backend_cuda.cu:
resident device K/V cache (g_kv_k/g_kv_v) + forward_one_device reusing the M=1 dense launches +
3 new kernels (embed_one, rope_one, attn_decode_kernel = one block/head, single query vs cached keys).
API sub0_cuda_kv_reset + sub0_cuda_forward_one; guard `sub0-cuda-selftest decode`. **Fast: ~2 ms/tok
d448 (~6.5x the CPU), 0.75 ms d192.** Parity vs GPU full forward: near-exact d192 (rel 1.8e-4) but
DIVERGES at d448 under RANDOM weights (top-1 agree 82.8%, rel 2.15) — pos 0-3 fine, blows up later.
DIAGNOSIS (not yet proven a bug): the two-pass decode softmax vs the full forward's flash online-
softmax differ ~FP-eps/step and amplify over 11 layers of near-flat random-weight logits (argmax
noise-fragile). RULED OUT: BF16 (whole inference path is FP32 — g_dev_params, FP32 cuBLAS,
attn_fwd_tiled_kernel<float>; act_t=BF16 is training-only) and a multi-key threading bug (pos=128 is
clean; diffs are SCATTERED/position-dependent, not a pos≥128 jump). CONFIRMED amplification: making
the random model more contractive (weight scale 0.05→0.02) lifts top-1 82.8%→94.9%. A residual
single-position rel spike (~2.6) persists though → maybe a rare numerical edge to chase. Almost
certainly BENIGN for a TRAINED (sharp/stable) model — the CPU forward_one uses
the same two-pass softmax and gives coherent d448 gen. **NEXT: validate GPU forward_one decode-gen on
the TRAINED fineweb model (top-1 vs full ~100%?), then wire sub0::forward_one → GPU dispatch on a GPU
build.** If d448 still diverges on a trained model, hunt an HD=64/multi-key-per-thread bug in
attn_decode_kernel (blockDim=128, pos≥128 = multi-key stride).

**STILL PENDING (user asked "GPU forward_one next and the dkv"): DKV register-split.**
attn_bwd_dkv_kernel holds 4×HD float arrays (kr,vr,dka,dva)=256 regs at HD=64 → spills. Split into
DV (kr+dva=2HD) + DK (kr+vr+dka=3HD) key-parallel kernels, or shared-stage q/dout, to reclaim
occupancy. Correct + 45x as-is; measure the split at HD=64 (native), keep only if faster.

**Fineweb_smoke train COMPLETE 2026-07-01: plateaued step 18212 (2.90 ep), val_nelbo 3.26→2.9601,
grammatical output.** User next runs a full fineweb_EDU (46GB) train in the background — benefits
from this session's flash attention (~10x train) + progress-named checkpoint retention (--keep).
Session shipped 7 commits (flash fwd+bwd, checkpoint retention, gen KV-cache, attn_bwd knob cleanup,
GPU decode). See [[attention-kernel-throughput-bottleneck]].

**E2E SMOKE PASSED 2026-07-01 (ts_gpu, v2 corpus.tok):** configure(v2 uint16)→train(flash attn,
~320k tok/s, val_nelbo 2.1173 @ step 1500)→GEN(CPU KV-cache forward_one) gives coherent TinyStories
from a cold prompt (correct name/pronoun coreference). So the CPU decode path is proven on a real
trained model. **A trained tinystories model now exists as the oracle for the GPU-decode validation**
(the pending next step below): run sub0-cuda-selftest decode / a top-1 GPU-vs-full compare on a TRAINED
model (expect ~100%, unlike the 82.8% random-weight case) — if it holds, wire sub0::forward_one → GPU
dispatch on a GPU build. Retrain a small ts model if the scratchpad copy is gone.

**NEXT-UP (user /compact note 2026-07-01):** (1) GPU decode trained-model validation → wire into gen
(above); (2) DKV register-split (above); (3) reader/writer tokenize IO-overlap — see
[[configurator-ingest-optimization]].

**BOTH #1 and #2 DONE 2026-07-02 (commits 4762486, 3779fd7):** (1) GPU decode validated on a real
trained model via a new hidden-tag test (150-step CPU overfit to sharpen weights, since random-init
logits are the known false-divergence case) — 100% top-1, ~1e-6 rel diff at d768/HD=96 (the config
that diverged under random weights). Wired `gpu_decode_enable()` into gen_stage.cpp mirroring
GpuTrainer's pattern (upload once, CPU fallback on any failure); confirmed end-to-end on the real
e2e tinystories model via sub0llm-gen (coherent output, "gen: decode backend GPU" logged). (2) DKV
split done — see [[attention-kernel-throughput-bottleneck]] for the full writeup (dv: 0 spill; dk:
1172B→560B spill; both correctness- and speed-verified at native d768). (3) reader/writer IO-overlap
remains open.

**UNIFIED into preview_at/gen_self_stats too, 2026-07-09 (commit 8076663), via a Sonnet 5 subagent
dispatch.** Found live while building the model-comparison-table feature: running `autotemp` showed
0% GPU utilization and ~4% CPU (one core) -- `gen_self_stats` (autotemp's generation sweep) and
`preview_at` (training/report previews) were STILL using the old O(n·T²) `sub0::forward()` loop this
memory's own CPU-forward_one section fixed for `gen` back on 2026-07-01/02 -- they'd just never been
migrated. New `include/sub0/decode.hpp`: `kv_decode_generate()` (the shared GPU-first/CPU-fallback
per-token loop, with an optional `on_token(logits_row, tok)` hook so `gen_self_stats`' own NLL-under-
true-distribution scoring gets the logits row it needs without a second forward pass) + `DecodeSession`
(RAII, brings up one GPU session reused across a whole report sample battery or autotemp temperature
grid, since re-enabling per generation would re-pay CUDA init/upload/graph-recapture 6-110x over).
`gen_stage.cpp` itself was refactored to call the SAME shared functions (pure extraction, no behavior
change -- diff-verified). Deliberately did NOT touch `evaluate()`/`mean_entropy()` -- those score a
FIXED, already-known span in one batched forward() call, which [[scoring-use-batched-forward-not-forward-one]]
already established is the efficient choice for scoring (KV-cache decode would be a regression there,
not a win). EOS-stop (`a016d17`) preserved exactly in the shared loop; a new regression test
(`tests/engine_tests.cpp`, "kv_decode_generate stops before pushing EOS") overfits a trivial
deterministic pattern so a real trained-then-primed context reliably predicts eos_id next, then asserts
nothing gets appended and the literal `<|endoftext|>` string never appears.

**Traced and confirmed safe, not just assumed:** `preview()` (the end-of-training sample print) now
opens its own `DecodeSession` while a live GPU training session is nominally still "active" at that
call site. Verified this is genuinely safe by reading the actual call order in `sub0_train_stage`
(`gpu.sync_to_host()` → the FINAL retry-guarded `save_model()` → `write_meta()` → `preview()` →
`gpu.shutdown()` -- the authoritative save to disk already happened before `preview()` touches the
device at all) AND the real implementations of `sub0_cuda_init`/`upload_params`/`shutdown` in
backend_cuda.cu (all three are genuinely idempotent -- `init` no-ops if `g_dev_params` already exists,
`shutdown` null-checks every free), not just trusting the agent's comment claiming this.

**Independently re-verified end to end, not just trusted the subagent's report:** rebuilt myself,
reran the new regression test directly (passes), ran the full non-slow/non-bench `sub0_tests` suite
myself and found 3 failures on unrelated CPU-vs-GPU batched-forward numeric-tolerance tests (NOT the
decode path this touched) -- reran those 3 three times with different random seeds, all passed cleanly
each time, confirming pre-existing flakiness (this project's own code comments already document
random-weight-init CPU/GPU divergence as a known, benign characteristic), not a regression. Ran
`sub0_frontend_tests` clean (13019 assertions/81 cases). Independently timed a live `autotemp` run
against the real `d448l11h7...rgwq_20260709-032056` model myself: 344s wall-clock (vs the agent's
claimed ~374s, and the ~1760s pre-change baseline) -- output was BYTE-IDENTICAL to my own earlier
pre-refactor measurement of the same model/seed (same recommended temp 0.33, same sample text
verbatim). Confirmed GPU decode is genuinely active on this system via `sub0llm-gen`'s explicit
`"gen: decode backend GPU (device KV-cache)"` announcement (autotemp/preview_at don't print this
themselves, but they now call the literal same shared functions gen_stage.cpp does, so this transfers
structurally, not just by inference).

This closes the loop this memory's own "STILL PENDING" section from 2026-07-01/02 left open (GPU
decode existed but only `gen` used it) -- `report`, `autotemp`, and the training loop's own preview
print all get the fast path now too. Directly unblocks `sub0llm models --refresh` (a comparison-table
feature being built the same session) from being impractically slow across a 5-7 model sweep.
