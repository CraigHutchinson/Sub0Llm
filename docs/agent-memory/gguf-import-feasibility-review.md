---
name: gguf-import-feasibility-review
description: Feasibility review for importing open-weight GGUF model weights into Sub0Llm; verdict staged-viable via JackFram/llama-160m; step1 (reader) + step2 (SwiGLU FFN, CPU) + step3a (tensor data + Q8_0 dequant) DONE; step3b (actual S0L5 weight transplant) + steps4-5 (vocab, e2e parity) + CUDA kernel still open
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**User's framing (2026-07-03):** could `configure` import an existing open-weight model's weights instead
of only training from scratch. Qwen 3.6 / Gemma 4 named as notable but with an explicit steer toward a
simpler matching model for a first PoC; explicit instruction that a viable spike gets built now, and if it
proves out, the REMAINING work becomes a follow-up task list for a later session (not built now).

**This engine's actual architecture (verified from code, not assumed):** pre-norm decoder, all dims
compile-time constexpr. RMSNorm (gamma-only, eps 1e-5). Plain MHA -- no `n_kv_heads` concept anywhere in
CPU or CUDA backends. No Q/K/V/O biases. RoPE full-head-dim, theta 10000, **interleaved adjacent-pair**
convention (rotates `x[2m],x[2m+1]`) -- NOT the NeoX half-split convention (`x[m],x[m+d/2]`) most GGUF/Llama
weights assume; an importer needs a row-permutation step for this, not a reinterpretation. FFN is a plain
**2-matrix GELU(tanh) WITH biases** b1/b2, D_FF hardwired to `4*D_MODEL`. Untied LM head (separate matrix +
bias). Checkpoint is a custom `'S0L5'` binary format that hard-rejects any dim mismatch. Tokenizer is a
custom corpus-learned unigram/JOIN scheme, vocab per-corpus, fingerprint-guarded -- not any standard
BPE/SentencePiece vocabulary.

**Verdict: viable, but staged -- not a drop-in importer.**

1. **Qwen 3.6 and Gemma 4 are ruled out flat.** Qwen 3.6 is hybrid ~75% Gated-DeltaNet linear-attention +
   GQA + MoE. Gemma 4 interleaves sliding-window/global attention with MoE variants. Importing either means
   rebuilding the attention stack, not writing an importer -- exactly the failure mode the "simpler model"
   steer was trying to avoid.
2. **Every realistic modern candidate (Llama/Mistral/TinyLlama/SmolLM2) needs the same one change**: SwiGLU
   gated FFN (this engine has no gate-matrix slot in `PKind`/`PARAM_LAYOUT` today) -- most also add GQA on
   top, which is a second, independent gap.
3. **Best-matching realistic target: a plain Llama-1-style MHA model**, e.g. `JackFram/llama-160m`
   (d768/L12/H12 -- same d768 class as this project's own production config). Matches on RMSNorm, no
   biases, RoPE theta 10000, plain MHA, untied head -- needs ONLY the SwiGLU FFN op, nothing else.
   GPT-2/Pythia-style alternatives match the FFN shape (2-mat+bias) but need LayerNorm (a new op on BOTH
   backends) + bias tensors throughout -- MORE total work despite looking closer on paper.
4. **The tokenizer is the real blocker, not tensor shapes.** Token-remapping between vocabularies is
   confirmed infeasible without shared retraining (embedding rows encode meaning under a SPECIFIC
   segmentation). The only real path: vendor the source model's own vocab wholesale for that specific build,
   bypassing the JOIN/casing FSM. The tokenizer's existing unigram-Viterbi mode is shape-compatible with a
   SentencePiece-style vendored vocab, which is a lucky fit, not a coincidence worth over-trusting without
   verifying against a real file.

**Step 1 -- SHIPPED this session** (commit 89ceac9): `include/sub0/gguf.hpp`, an engine-free (no generated
config, no engine link) GGUF header/KV-metadata/tensor-table reader, deliberately NOT reading tensor data
payloads or decoding quantized formats. Plus `check_llama_mha_compat()`: compares parsed tensor names/shapes
against a plain-MHA Llama-style expectation, flagging gated/SwiGLU FFN (`ffn_gate.weight` presence --
learned the hard way that `ffn_up.weight` alone is NOT a gating signal, it's used in both plain and gated
conventions), GQA (attn_k/v row count < attn_q), attention/FFN biases, and quantized dtypes. Tested against
13 handcrafted in-memory GGUF fixtures (64 assertions) in `tests/gguf_tests.cpp`, part of
`sub0_frontend_tests`. No real GGUF file was available/needed for this stage -- the compat-report LOGIC is
tested, not validated against a real download yet.

**Step 1b -- validated against a REAL downloaded file (2026-07-04):** `Felladrin/gguf-llama-160m`
(Q8_0 quant, 166 MiB, HF-hosted GGUF conversion of `JackFram/llama-160m`, the exact candidate named
above) downloaded to a scratch dir (not committed -- large binary fixture, not a source file) and read
with an ad-hoc validation harness (`sub0::gguf::Reader` + `check_llama_mha_compat`, also scratch, not
committed). **Parsed cleanly**: version 3, 111 tensors, 22 metadata KV pairs, all string-array (32000
vocab tokens) and other-array (32000 scores, correctly count-only not materialized) fields read
correctly. Findings:
- `has_gated_ffn: true`, `matches_plain_mha: true`, 0 missing tensors across all 12 layers --
  CONFIRMS the earlier code-only assessment exactly: this file needs ONLY the SwiGLU FFN op, nothing
  else structurally.
- `any_quantized: true` (Q8_0, `type_raw=8`) correctly flagged as not-plain-float, not misread as
  raw floats -- the "quantized, dequantize first" signal works on a real quantized file, not just a
  synthetic one.
- Metadata cross-checked against `JackFram/llama-160m`'s known HF config (hidden=768, layers=12,
  heads=12, ffn=3072) -- exact match on `llama.embedding_length`/`block_count`/`attention.head_count`/
  `feed_forward_length`.
- **New finding, not covered by the handcrafted fixtures**: `llama.rope.freq_base` is ABSENT from this
  file's metadata (llama.cpp defaults it to 10000.0 when missing, matching this engine's own default
  theta -- lucky, but the importer must replicate this default-when-absent behavior explicitly, not
  assume the key is always present).
- STEP 1 IS NOW FULLY DONE (both the handcrafted-fixture logic tests AND real-file validation).

**Step 2 -- SwiGLU-gated FFN, CPU DONE (2026-07-04); CUDA kernel still deferred.** New
`--gated-ffn 1` configurator flag bakes `constexpr bool USE_GATED_FFN` into the corpus header
(default false, zero effect on existing builds). Touch points, all `if constexpr`-gated so the
non-gated path compiles to identical code:
- `include/sub0/core.hpp`: new `Op::SwiGLU`.
- `include/sub0/layout.hpp`: new `PKind::Wg`; `NUM_PARAMS`/`make_param_layout()` branch to 9
  params/layer (Ln1,Ln2,Wq,Wk,Wv,Wo,Wg,W1,W2 -- W1/W2 reused as up/down, NO FFN bias, matching the
  GGUF/Llama convention) vs the existing 10 (plain, GELU+bias).
- `src/backend_cpu.cpp`: `fast_sigmoid`/`silu_fast`/`dsilu_fast` (mirrors the existing GELU
  fast/exact dual-path pattern exactly, mutually consistent so gradcheck passes in fast mode);
  `op_swiglu(gate_pre, up_pre)` forward builder (`y = silu(gate)*up`, two-input shape like `op_add`)
  + its `Op::SwiGLU` backward case (`d(gate) = dy*up*dsilu(gate)`, `d(up) = dy*silu(gate)`); `Layer`
  struct gained a `Wg` field (always present, nullptr when unused -- thread_local zero-init makes
  this safe); `build_layout()`/`init_weights()`/`forward()`/`forward_one()` (the KV-cache row path,
  needed its own `g1[D_FF]` buffer + `silu_row`) all branch on `USE_GATED_FFN`; `calc_act_cap()`
  (gated FFN needs 3 T*F activation nodes not 2 -- one extra `op_linear` for the gate branch) and
  `MAX_NODES` updated.
- **Checkpoint format: deliberately NOT touched**, and this was the important design call. The
  obvious move (add `gated_ffn` to `engine_core.cpp`'s binary `Header` struct and/or
  `train_stage.cpp`'s `.ckpt` compat-stamp list) would change `sizeof(Header)`/the ckpt byte layout
  for EVERY build, gated or not -- breaking resume for the then-83-hour-and-counting live production
  run (`models/fineweb_edu_workflow`) the moment its build dir got rebuilt against the new source.
  Caught this before making the change, not after. Verified unnecessary anyway: `param_floats`
  (Header) / `nfloat` (`.ckpt`, from `sub0::trainable_floats()`) already differ between a gated and
  non-gated build at IDENTICAL d_model/n_layers/n_heads/seq_len/vocab (gated: `3*C*F`/layer; plain:
  `2*C*F+F+C`; never equal for this project's fixed `D_FF=4*D_MODEL`), so the EXISTING checks already
  reject a cross-load correctly -- zero risk to the live checkpoint, zero new fields needed.
- `include/sub0/registry.hpp` (the `models`/`models --prune` diagnostic, NOT the load-time safety
  gate) DID get a `gated_ffn` field added -- safe because it's a plain text `meta.txt` key=value
  format, not a fixed binary struct, so old files parse fine with the field defaulting to 0. New 'g'
  dir-name tag (mirrors 't'/ternary, 'r'/rope).
- `src/backend_cuda.cu`: `static_assert(!USE_GATED_FFN, ...)` (mirrors the existing ternary guard) --
  no CUDA kernel exists yet, so a gated+GPU build hard-fails at compile time rather than silently
  computing the wrong (plain-GELU) forward. The configurator also refuses `--gated-ffn 1` with a GPU
  compute backend directly (clearer, earlier error than waiting for the nvcc compile to fail).
- **Verification**: (1) default (non-gated) `smoke_gpu` full rebuild + full suite:
  124,694,438/124,694,439 assertions pass -- the ONE failure is the pre-existing, already-flagged,
  seed-dependent "analytic gradients match finite differences" flake, unrelated (same failure exists
  on the pre-gated-FFN code too, confirmed both before and after). (2) A dedicated tiny (d32 L2 H4
  seq32 vocab756) CPU-only `--gated-ffn 1` build: 363,450/363,450 assertions pass, INCLUDING that same
  gradient-check test -- passed cleanly there (and across 3 more reruns at different random seeds),
  which is the direct numerical proof the new `op_swiglu` forward/backward math is correct. (3) A real
  end-to-end smoke run on that build: train 30 steps (loss 6.42->6.13 monotonically) + `gen` (KV-cache
  forward_one path) + `models` listing (correctly tagged `...v756rg`, shown loadable; correctly showed
  the unrelated production model as incompatible) + resume-from-checkpoint another 10 steps (6.15->6.03,
  continued correctly) -- all clean, no crashes. `tests/layout_tests.cpp`'s two layout assertions and
  `tests/frontend_tests.cpp`'s registry test were also made gating-aware (both previously hardcoded the
  non-gated formula/arg count) rather than just left to fail on a gated build.
- **CUDA kernel still deferred** -- op_swiglu + the Wg param slot need a device implementation before
  a gated model can train/infer on GPU; tracked as `TODO(gated-ffn-gpu)` at both guard sites.

**Step 3a -- GGUF tensor DATA access + Q8_0 dequantization, DONE (2026-07-04); the actual weight
transplant (writing an S0L5 checkpoint) is still open.** Extended `include/sub0/gguf.hpp` (still
engine-free, no generated config needed) beyond step 1's metadata-only reader:
- `Reader::data_offset()` -- absolute byte offset where tensor data starts: the tensor-info table
  end, rounded UP to `general.alignment` (u32 KV, default 32 when absent, per the GGUF spec). Every
  `TensorInfo::offset` is relative to this, not the file start -- step 1 never needed it since it
  stopped at the tensor table.
- `Reader::tensor_byte_size()`/`tensor_bytes()` -- bounds-checked raw byte span for one tensor
  (F32: n*4, F16: n*2, Q8_0: `ceil(n/32)*34` for the 2-byte-f16-scale + 32-int8-quant block layout;
  0/empty for any other type, so a caller can't accidentally misread an unsupported quant format).
- `f16_to_f32()` -- standard IEEE-754 binary16->binary32 bit-manipulation conversion (needed for both
  F16 tensors and Q8_0's per-block f16 scale; no `<stdfloat>` on this toolchain yet). Verified against
  12 known bit patterns including both subnormal boundaries, +-0, +-inf -- all exact.
- `dequantize_q8_0()` -- decodes a `block_q8_0` buffer (`value[i] = qs[i] * f16_to_f32(block_scale)`).
- `to_f32()` -- dispatches on `TensorInfo::type_raw` (F32 passthrough / F16 widen / Q8_0 dequant);
  returns false for anything else, deliberately (no silent misdecoding of an unimplemented quant type).
- **Verification**: 9 new Catch2 cases (`tests/gguf_tests.cpp`, now 22 cases / 242 assertions for the
  whole gguf suite, 1487/1487 for the full engine-free frontend suite) covering f16 bit patterns,
  alignment computation, exact F32 round-trip, a hand-built Q8_0 block (exact int8*scale check), a
  partial final block (n not a multiple of 32), a too-short buffer (rejected, not partially decoded),
  bounds-checking (truncated data section -> empty span, not OOB read), and refusing an unimplemented
  quant type. PLUS validated against the SAME real 166 MiB llama-160m file from step 1b (ad-hoc
  scratch harness, not committed): decoded `token_embd.weight` (Q8_0, 24.5M elements), `attn_q.weight`,
  `ffn_gate.weight` (all Q8_0) and both norm weights (F32) -- every decode landed at statistically sane
  values for trained NN weights (near-zero mean, small std for GEMM matrices, e.g.
  `attn_q.weight: mean=-0.00007 std=0.07615`; positive learned-scale norm gammas, e.g.
  `output_norm.weight: mean=2.78`) with byte counts matching `tensor_byte_size()` exactly in every case.
- **Not yet done (the actual importer)**: mapping decoded GGUF tensors into `PARAM_LAYOUT` order,
  the RoPE interleaved-pair row permutation on Wq/Wk, dims resolution (GGUF metadata -> a matching
  `sub0llm-configure --gated-ffn 1 ...` invocation), and writing a real `S0L5` checkpoint. This needs
  its own session -- it is inherently two-stage (emit config -> rebuild at those dims -> THEN
  transplant weights) because of this project's compile-time-dims architecture, and needs the
  vendored-vocab tokenizer (step 4) to actually decode/encode text afterward, not just to hold weights.

**Deferred to future follow-up tasks:**
3. **Importer**: reads a validated GGUF file's tensor data (not just metadata) and writes an `S0L5`
   checkpoint -- including the RoPE interleaved-pair row permutation on Wq/Wk, and zeroed `lm_bias`/`pos_emb`
   slots this engine's layout always carries.
4. **Vendored-vocab tokenizer mode**: bypass the JOIN/casing FSM, load GGUF's `tokenizer.ggml.tokens`/
   `scores` directly into the existing unigram-Viterbi decode path; decode + ASCII-prompt encode is enough
   for a PoC.
5. **End-to-end greedy-parity test against llama.cpp** on the imported model -- the same oracle-based
   verification this project already uses for its own kernels (see
   [[verify-correctness-against-reference-before-perf]]).
