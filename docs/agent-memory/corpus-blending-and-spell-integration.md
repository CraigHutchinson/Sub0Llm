---
name: corpus-blending-and-spell-integration
description: "IMPLEMENTED (CPU, not yet committed): general weighted per-window corpus blender (include/sub0/blend.hpp) + the uncombine/combine curriculum wired as its first non-base source via --spell-mix, persisted in config.json, re-applied on resume; gen's decode interceptor folded into kv_decode_generate and auto-enabled for spell-trained models. CPU end-to-end green; GPU blended-training gated on the pending device ignore-index CE kernel."
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Productionizes the combine/uncombine spike ([[spellspike-poc-results]]) AND builds the general
corpus-blending capability the user asked for ("integrate the spellspike into the production model
and consider further how we train the corpus, so supporting corpus blending/interleaving"). Confirmed
design via AskUserQuestion: **on-the-fly weighted, per-window** blending; **general N-source blender,
base+spell first**. Plan: smooth-noodling-kurzweil.md. 2026-07-10.

**5 components, all implemented + CPU-validated:**
1. `include/sub0/blend.hpp` (engine-free, 4 unit tests in tests/blend_tests.cpp): `BlendSource{TokView
   view; span docs; span mask; double weight}` + `pick_source` (weighted) + `sample_blend`. KEY
   invariant: a SINGLE source consumes NO rng draw (pick_source short-circuits) -> a one-source run's
   window-sampling stream (hence resume point) is byte-identical to pre-blend. A source is a mmap
   corpus OR an in-memory generated buffer; the blender only borrows spans.
2. train_stage.cpp loop: `std::vector<BlendSource> sources` (base = source 0). Per window, `sample_blend`
   picks a source; CPU path copies that source's window into cpu_win + a parallel cpu_mask, calls the
   existing `train_batch(..., loss_mask)` (the LOSS_IGNORE_INDEX foundation). GpuTrainer::step
   generalized to materialize from `sources[src_idx[b]]` (mask->LOSS_IGNORE_INDEX in targets). A masked
   blend on the GPU is REFUSED (`gpu_train && any_masked` -> error) until the device CE ignore-index
   kernel lands. Single unmasked source == pre-blend behavior exactly.
3. The uncombine curriculum as a generated source: `spellspike::build_dataset` over the production
   tokenizer (deserialized from tokenizer.tok), drilled_frac 1.0 (full vocab coverage), seeded off the
   run seed (deterministic + reproduced on resume). `sub0_train` now links `sub0_frontend` directly
   (CMake) for deserialize + the generator. Held in a function-scope `spell_ds` that outlives the loop.
4. `--spell-mix <f>` CLI (threaded through sub0_train_stage's new param) -> base weight 1-f, spell
   weight f. Persisted as `spell_mix` (a new `X(double, spell_mix, 0.0)` row in SUB0_RUN_CONFIG_FIELDS,
   registry.hpp) so config.json records it and RESUME re-applies it (same discipline as the optimizer
   fix; verified: "config.json overrides this invocation's --spell-mix: 0.000 -> 0.300"). NOTE: did NOT
   add a general `--blend corpus:w,...` CLI -- [[only-add-arguments-we-need]]: no second corpus consumes
   it yet, and blend.hpp is already N-source, so a future corpus drops in trivially.
5. Folded the old standalone `kv_decode_ops` INTO `kv_decode_generate` as optional expand/combine
   callbacks (null => byte-identical to today; the DRY the user wanted). Interception is CPU-KV-cache
   for now (device-decode injection = follow-on). gen_stage.cpp enables it ONLY for models whose
   config.json has spell_mix>0 (reads read_config_json, deserializes the raw Tokenizer for callbacks;
   `sub0_gen` also links sub0_frontend) -- a normal model keeps GPU decode untouched. Spike engine test
   migrated to the folded path (greedy via topk=1) and REPRODUCES the ceiling (held-out nth 0.966, rt
   0.900 at 3000 steps).

**Validation (CPU spike_tied build, d768 untouched):** full engine suite 72 cases/4.9M assertions
green (loss-mask + RunConfig round-trip now covering spell_mix); blend_tests 4/4; spellspike engine
test green; a real `train --spell-mix 0.3` ran ("blend: base 70% + uncombine curriculum 30% (486 task
words, 5832 traces, 72808 tokens)"), config.json persisted, resume re-applied it, gen enabled the
interceptor for the spell model and stayed silent for a spell_mix=0 control. **A `-DSUB0_BUILD_CUDA
-fsyntax-only` pass on train_stage.cpp + gen_stage.cpp CAUGHT A REAL BUG** (unqualified
LOSS_IGNORE_INDEX in GpuTrainer, which is outside namespace sub0 -- would have broken the GPU build);
fixed to sub0::LOSS_IGNORE_INDEX. Lesson: `#if SUB0_BUILD_CUDA` host code is invisible to a CPU build
-- syntax-check it before assuming the GPU build is safe.

**Files (CPU half):** blend.hpp (new), blend_tests.cpp (new), train_stage.cpp, gen_stage.cpp,
decode.hpp, registry.hpp, cli_stages.hpp, run_config (auto via X-macro), spellspike_engine_tests.cpp,
CMakeLists.txt (sub0_train + sub0_gen link sub0_frontend). COMMITTED 8b2a4f5.

**GPU half -- DONE, COMMITTED c4654f3 (2026-07-11, after d768 plateaued and freed the device):**
- `ce_backward_kernel` (backend_cuda.cu): a target < 0 row is inert (zero grad, no loss) alongside the
  existing padding case; per-window loss/grad normalizes over ACTIVE (non-ignored) count via a new
  optional `active[]` param. When unmasked, active==len -> the dense path is BIT-IDENTICAL (dense
  gradient-parity gate unchanged). `active[]` computed host-side in run_fwd_bwd (from the targets it
  already holds), uploaded only when a target is masked; threaded through backward_device/head_ce_chunked
  + new g_tr.active device buffer (also added to memplan train_scratch_bytes).
- Lifted the `any_masked` GPU refusal in train_stage.cpp -- masked blend runs on either backend.
- New parity test "CUDA masked (ignore-index) backward..." PASSES: grad cos 0.9999, GPU/CPU mean loss
  agree (10.0085 vs 10.0095) -> active[] normalizer matches CPU ce_active. Full [cuda] suite 37/37 green.
- Real GPU d768 `--spell-mix 0.2` run: clean, GPU backend, "base 80% + curriculum 20% (14413 task
  words, 172956 traces)", no refusal.
- LESSON: `-fsyntax-only -DSUB0_BUILD_CUDA` only checks train_stage/gen_stage (host TUs); it does NOT
  cover backend_cuda.cu (nvcc) -- the first real ts_cuda rebuild caught 2 extra ce_backward_kernel
  selftest call sites the syntax-check couldn't see. Rebuild the CUDA backend for real before claiming
  a kernel-signature change compiles.
- Fixed 3 PRE-EXISTING [cuda] failures surfaced by running the suite on this BF16 d768 build (NOT from
  the CE change): dense-linear + tied-head GEMM checks used FP32-era tolerances -> made BF16-aware
  (rel-L2 / relaxed reldiff gated on GEMM_DTYPE); memplan footprint over-predicts a steady ~140MB at
  d768 -> asymmetric band (tight UNDER = OOM-risk, wide OVER = safe; FOOTPRINT_OVERPREDICT_TOLERANCE_MB)
  in both test + runtime message. See [[memplan-vram-prediction-gap-at-scale]].

Whole feature (CPU+GPU) now COMPLETE + committed (8b2a4f5 + c4654f3, local branch, not pushed).
