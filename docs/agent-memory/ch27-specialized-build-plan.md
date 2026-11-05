---
name: ch27-specialized-build-plan
description: "Ch27 Specialized Model Compilation — compile-time monomorphization stack; P1+P2-scaffold done, Gemma path + llama.cpp baseline pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: fb632281-974b-437a-89ff-47e56b5ae70c
---

New chapter **ch27_specialized_build**: "move every runtime-determined model axis
to compile time" so one model + one CPU compiles into maximally-specialized code.
Motivating target **Gemma 4 12B** (baseline vs llama.cpp); proven on the local
**Qwen3-0.6B-Q8_0.gguf** (the only model file present).

**Done & verified (480 tests pass, debug build clean):**
- GGUF metadata extended: `GGUFModelConfig` now parses `norm_eps`
  (`.attention.layer_norm_rms_epsilon`) and `sliding_window`
  (`.attention.sliding_window`) — needed for Gemma. `src/nn/gguf_loader.cpp`,
  `include/sub0llm/nn/gguf_loader.hpp`.
- `tools/specialize/main.cpp` → `sub0llm-specialize`: reads a GGUF's METADATA ONLY
  and emits `generated/<stem>_spec.hpp` (a struct where every axis is
  `static constexpr` + feature flags + static_asserts) and `<stem>_arch.json`.
  Features detected from arch string + tensor presence: use_qk_norm
  (blk.0.attn_q_norm.weight), tied_embeddings (!output.weight), attn_qkv_bias
  (blk.0.attn_q.bias), and Gemma-family → norm_plus_one/GeGLU/embed_scale=sqrt(D)/
  local_global_stride=6. Gotcha fixed: `std::format("{}f", 1.0f)`→"1f" is invalid
  C++; use `float_lit()` helper that ensures a '.'/exponent before the 'f'.
- `include/sub0llm/nn/static_spec.hpp`: `StaticSpec` concept (the generated header's
  contract) + `SpecDerived<S>` (kv_group, q/kv_proj_dim, constexpr inv_sqrt_head_dim,
  kv-cache floats/token).
- Chapter scaffold: `chapters/ch27_specialized_build/{main.cpp,CMakeLists.txt,README.md}`
  + committed `generated/qwen3_0_6b_q8_0_spec.hpp`. The demo binary is monomorphized
  to the spec via CMake-injected `SUB0LLM_CH27_SPEC_*` defines; builds with NO model
  file present (committed spec). `-DCH27_SPEC=<stem> -DCH27_SPEC_TYPE=<...>` to retarget;
  `ch27-regen-spec` target re-runs the tool. Registered in chapters/CMakeLists.txt.

**Pending:**
- P2: monomorphized RMSNorm+attn+FFN forward `template<StaticSpec S>` with std::array
  buffers, microbench vs dynamic Tensor path (the actual perf proof).
- P3: Gemma-family arch in the model, manifest-gated — GeGLU (vs SwiGLU), embed×sqrt(D),
  (1+w) RMSNorm, interleaved local/global attention w/ dual RoPE base. Qwen3 unaffected.
- P4: gemma4-cli/server + harness tabulating load/tok-s/RSS vs llama.cpp + writeup.

**Resources (provided by user 2026-06-04):**
- Gemma 4 12B GGUF on local disk: `D:\Craig\GitHub\Sub0Llm\models\gemma-4-12b-it-Q8_0.gguf`
  (12.67 GB, Q8_0). Smaller Qwen3-0.6B is the prototyping model (less RAM).
- llama.cpp (baseline) local project: `D:\tools\llamacpp` (source + prebuilt).
  Prebuilt CPU binaries: `D:\tools\llamacpp\vendor\llama.cpp-prebuilt\b9334\cpu\`
  — has `llama-cli.exe`, `llama-bench.exe`, `llama-perplexity.exe`,
  `llama-completion.exe`, `llama-gemma3-cli.exe` (+ per-uarch ggml-cpu-*.dll,
  and cuda/ sycl/ vulkan/ variants). Run with that dir on PATH for the DLLs.
- Gemma 4 info: blog https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/
  model card https://huggingface.co/google/gemma-4-12B-it
  GGUF (unsloth) https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/gemma-4-12b-it-Q8_0.gguf
  README https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/README.md
- User directive: ensure memory footprint is respectable/expected (report RSS;
  watch loading two ~13GB models at once).

**Gemma 4 12B architecture — EXTRACTED by our own sub0llm-specialize from the GGUF**
(generated/gemma_4_12b_it_q8_0_spec.hpp; family detection worked end-to-end):
  arch=gemma4, V=262144, D=3840, n_heads=16, n_kv_heads=16 (NO GQA), head_dim=512,
  n_layers=48, d_ff=15360, context_len=131072, rope_base=1e6, norm_eps=1e-6,
  embed_scale=sqrt(3840)=61.97, qk_norm=true, tied_emb=true, (1+w)norm=true,
  GeGLU, sliding_window=1024 with global every 6 layers.
  NOTE TO VERIFY: head_dim=512 (from .attention.key_length) is unusually large
  (Gemma 3 used 256) and q_proj=16*512=8192 >> D=3840 — cross-check against
  llama.cpp's load log before trusting for the specialized forward.

**GEMMA 4 12B GROUND TRUTH (from GGUF `--dump-meta`, 2026-06-04) — it is a HARD,
heterogeneous, multimodal arch; correctness gate already paid off.**
First-pass GGUF extraction was WRONG (head_dim 512, kv 16) — two bugs found:
(1) my `ends_with` matched vision-tower keys in the multimodal GGUF → fixed by
capture-all-metadata + `resolve_config` with EXACT `<arch>.` prefix (gguf_loader.cpp
rewritten: read_kv_pair now captures typed maps + stringified dump; new `metadata()`
accessor; `sub0llm-specialize --dump-meta`). (2) Even after the fix, the real GGUF
keys show Gemma 4 is NOT uniform:
  - `gemma4.attention.head_count` = 16 (scalar)
  - `gemma4.attention.head_count_kv` = ARRAY[48]  ← PER-LAYER kv heads (our scalar
    n_kv_heads is meaningless; it currently falls back to n_heads=16 = WRONG)
  - `key_length`=512 / `value_length`=512  (GLOBAL layers head_dim)
    `key_length_swa`=256 / `value_length_swa`=256  (SLIDING layers head_dim)
    → head_dim VARIES BY LAYER (512 global vs 256 local)
  - `rope.freq_base`=1e6 (global) / `rope.freq_base_swa`=10000 (sliding) → dual RoPE
  - `rope.dimension_count`=512 / `_swa`=256 (partial-rotary related)
  - `attention.sliding_window`=1024 + `sliding_window_pattern`=ARRAY[48] bools (per-layer
    local/global mask — NOT a simple every-6 stride)
  - `final_logit_softcapping`=30, rms_eps≈1e-6, gelu_pytorch_tanh, tied emb,
    embed_scale=sqrt(3840)=61.97, (1+w) norm.
**RESOLVED BY TENSOR SHAPES (2026-06-04, authoritative):** Gemma4 GGUF tensor dims
(logical out,in): token_embd (262144,3840); attn_q (4096,3840); attn_q_norm [256];
attn_k blk.0/1 (2048,3840)=8kv but blk.5 (512,3840)=**2kv** → head_dim=256 (NOT 512;
512 was VISION pollution), n_heads=16, and **n_kv_heads VARIES PER LAYER (8 or 2)**.
d_ff=15360. So Gemma4 IS per-layer heterogeneous — confirmed from the file, no llama.cpp
source needed. Our uniform ModernGPT can't represent it. PLAN (README, dedicated effort):
per-layer LayerSpec (kv count, window local/global from pattern[48], rope 1e6/1e4) +
GeGLU(gelu-tanh) + (1+w)RMSNorm + embed×sqrt(3840) + final softcap30 + partial rotary;
gate logit/greedy parity vs llama-cli; text-only. Q8 loader generalizes per-layer kv.
DEFERRED: large multi-feature build, do as its own session, correctness-gated.
**CROSS-VERIFIED (2026-06-04) vs official dev guide + ai.google.dev model card:** those
pages are HIGH-LEVEL (confirm only: 48 layers, 1024 window, 262K vocab, 256K context,
~11.95B params — all match GGUF/HF, no conflict). Internals from GGUF tensors + HF
config (agree). PER-LAYER KV PATTERN (from tensors, all 48): global layers = index%6==5
(5,11,17,23,29,35,41,47 = 8 layers) have **2 kv heads**; other 40 local layers have 8.
Uniform 5-local:1-global. Global layers use fewer kv heads = KV-cache optimization for
full-context attention. head_dim 256 uniform, n_heads 16. This is the authoritative
per-layer LayerSpec for the Gemma forward.

**CONFLICT (now resolved, kept for history):** HF original config.json (https://huggingface.co/google/gemma-4-12B,
bf16) says UNIFORM num_key_value_heads=8, head_dim=256, partial_rotary 0.25 — does
NOT match the local unsloth Q8 GGUF's per-layer encoding. Since we LOAD the GGUF,
**llama.cpp's gemma4 graph code (local source D:\tools\llamacpp) is the authoritative
interpreter of THIS file**, not the HF config. Must read it before implementing the
Gemma forward — guessing = "performant but incorrect."
**Implication:** our uniform ModernGPT cannot represent Gemma 4 (per-layer kv +
per-layer head_dim + per-layer window + dual rope + softcap). A faithful Gemma 4
text forward is a MAJOR effort. Recommended sequencing: prove the full pipeline
end-to-end on Qwen3 first (specialize → monomorphized runner → logit/greedy/ppl
PARITY vs llama.cpp on Qwen3-0.6B → perf), since Qwen3 is uniform GQA and we already
load+run it; then take on Gemma 4 grounded in llama.cpp's gemma4 source.

**CORRECTNESS HARNESS — `tools/verify` → `sub0llm-verify` (modes: tokenize/greedy/
logits/nll) loads a GGUF, reuses BPETokenizer::from_vocab + load_gguf_model +
forward/generate.** Oracle = llama.cpp prebuilt (`D:\tools\llamacpp\...\b9334\cpu\`:
llama-tokenize, llama-cli, llama-perplexity).
Results on Qwen3-0.6B (prompt "The capital of France is"):
- TOKENIZER PARITY: EXACT — ours == llama-tokenize = 785 6722 315 9625 374.
- Our greedy: " Paris. The capital of Italy is Rome..." (argmax ' Paris' logit 16.56) —
  coherent & correct.
- Our perplexity on a 24-tok passage: mean_nll 2.886, ppl 17.93.
- GOTCHA: llama-cli defaults to CONVERSATION mode (applies chat template) → compare
  with `-no-cnv` for raw-completion parity (else it answers "Okay, the user is asking
  about..."). Qwen3-0.6B is an instruct model.

**QWEN3 PARITY + PERF (established 2026-06-04):** tokenizer EXACT vs llama-tokenize;
greedy temp-0 gives all-correct world facts (" Paris. The capital of Italy is Rome.
The capital of Spain is Madrid...") — Qwen3 forward CORRECT. Perf: OUR dynamic engine
= 11.6 tok/s gen vs llama.cpp 16.98 tok/s (~68%); llama prompt 196 t/s; model 604 MiB
Q8 (we dequant Q8->f32 = ~4x RAM, a cost to fix). So real ~1.46x gap to close just to
MATCH llama; their CPU kernels are heavily tuned — specialization thesis tested hard.
Exact perplexity-vs-llama match DEFERRED (methodology differs: llama strided windows;
our 512-tok full-vocab-head pass is minute-scale ~80 GFLOP). Strong evidence already
sufficient. Memory note: keep RSS in check; we hold f32 weights.

**WEIGHT-REPRESENTATION PROFILING (`backends/cpu/quant.{hpp,cpp}` + `tools/qbench`
+ `test_quant.cpp`, 2026-06-04).** User: "profile both, don't assume; varies per
model." Built 5 GEMV paths, all SIMD (AVX2/F16C native), correctness-gated:
f32 / f16(2B) / Q8 deq->f32 / Q8 deq->f16 / Q8 quantized-int8(1.06B).
Result ordering (Qwen3 & Gemma, speedup vs f32):
- **Q8 quantized int8 = WINNER**: 1.38-1.74x Qwen, 1.65-2.50x Gemma; 3.76x less RAM;
  ~0.4% relRMS. KEY FIX: first version LOST on big/memory-bound layers (lm head
  0.82x) due to per-block scalar f16 conversion + per-block int hsum. Fixed with
  F16C hardware f16->f32 (`_mm256_cvtph_ps`) + accumulate-as-float (one hsum/row).
  Flipped lm head 0.82x->1.77x. Quantize ACTIVATION once, maddubs int8 dots.
- **f16 = near-lossless middle**: 1.0-1.71x, 2x less RAM, relRMS ~1.5e-4 (~10x more
  accurate than Q8). Best for accuracy-critical / f16-native models. "Dequant to f16
  AT LOAD" == f16-store; beats f32-store.
- **Q8 deq->f32**: ~parity (0.85-1.20x), dominated by quantized (same RAM, slower).
- **Q8 deq->f16 (per-op f16 intermediate) = WORST** for single-token GEMV (0.68-0.90x):
  extra round-trip, no reuse, no accuracy gain. Only helps BATCHED/TILED matmul
  (T>1 prompt) where dequant tile is reused — separate experiment, not built.
Direction: store Q8 + quantized int8 matmul (default) with f16 as accuracy option;
makes Gemma-4-12B loadable (~13GB vs ~50GB f32). Body uses int8; consider F16C/tiling
specifics per layer. PERF CONTEXT: our dynamic engine was 11.6 tok/s = 68% of llama
17 t/s BEFORE these kernels; quantized GEMVs are the lever to close/beat that.

**Q8 WIRED INTO GENERATION — WE BEAT LLAMA.CPP (2026-06-04).**
`Linear::quantize_weights()` (Q8 weight + int8 apply_one for single-token),
`SwiGLUFeedForward/ModernTransformerBlock::quantize_weights()`,
`ModernGPT::quantize_for_inference()` (FFN + LM head; attention still f32 — its
(in,out) layout needs a transpose to quantize). `forward_one` LM head uses int8 when
quantized. `sub0llm-verify --q8` toggles. RESULT on Qwen3-0.6B: f32 12.2 tok/s ->
**Q8 19.5 tok/s (1.60x), = 115% of llama.cpp's 17 tok/s — we beat it**, with
greedy output BYTE-IDENTICAL to f32. Test: "Linear Q8 apply_one matches f32" (487->
unit added). CAVEAT: keeps BOTH f32+Q8 resident (f32 for autograd) -> RAM higher not
lower; pure-inference mode dropping f32 reclaims 3.76x RAM (follow-up). More headroom:
quantize attention projections too. Next: (a-cont) attention Q8 + pure-inference RAM
mode; (b) Gemma 4 forward from DOCS not llama.cpp source (user pref).

**A-CONT DONE — attention Q8 + RAM reclaim (2026-06-04).** Quantized attention Q/K/V/O
too (GQA::quantize_weights transposes (in,out)->(out,in) row-major Q8; q8_matvec helper;
forward_one uses int8 for q/k/v/o). Added free_f32_weights() cascade (Linear caches
in_/out_ so Q8 survives free; GQA frees W_*; block/swiglu cascade) and
ModernGPT::quantize_for_inference(drop_f32): frees attn+FFN f32 AND the tied embed table
(embedding lookup then dequant_row from lm_head_q8_; cached_vocab_/cached_embed_ added so
accessors+forward_one survive the free). `sub0llm-verify --q8 / --q8-only`; RSS via
K32GetProcessMemoryInfo (NOMINMAX needed). RESULTS Qwen3-0.6B:
- f32 12.2 t/s @ 2323 MiB; Q8(f32 kept) 24.3 t/s (143% llama, 2.0x f32);
  Q8-only(f32 dropped) 21.1 t/s (124% llama) @ **854 MiB (2.72x less RAM)**.
- greedy BYTE-IDENTICAL across all. 488 tests pass. Peak-RSS-at-load still 2323
  (materialize f32 then quantize) -> quantize-on-load is a follow-up.
**QUANTIZE-ON-LOAD DONE (2026-06-04).** `load_gguf_model_q8(reader)` builds an
inference model with f32 ELIDED (threaded `alloc_weights=false` through Embedding/
Linear/SwiGLU/GQA/block/ModernGPT ctors → placeholders) and copies RAW Q8_0 blocks
straight into int8 buffers (our BlockQ8_0 == GGUF Q8_0 block, 34B; attention sliced
per-head: q/k/v contiguous (heads·Dh,D), o gathered from (D,H·Dh) since Dh=128 mult of
32; ffn direct (out,in); tied embed = token_embd raw, dequant 1 row on lookup). Friend
of those classes (fwd-declared in gpt.hpp). New: GGUFReader::load_tensor_q8/tensor_type.
`sub0llm-verify --q8-load`. RESULT: peak RSS **651 MiB (3.57x less than f32 2323)** with
NO f32 ever materialized, ~23 t/s (134% llama), greedy BYTE-IDENTICAL. 488 tests pass.
Best mode overall (beats q8-only's 853 MiB since f32 never allocated). THIS makes Gemma
12B loadable (~13GB not ~50GB). NOTE: block ctor hardcodes rope_base=10000 but Qwen3
GGUF says 1e6 — yet output correct; possible latent long-context issue, pre-existing,
flagged not fixed.
**BATCHED TILING DONE + DEFAULTS (2026-06-04).** `matmul_q8_0_q8_0(W,Xq,Y,M,K,T)`
rows-outer/cols-inner reuses each W row across T cols (W streamed once vs T×). qbench
`--batch T`: T=16 row-reuse vs per-column = ffn 1.34x, ffn-down 1.86x, **lm head 1.91x**
(biggest where memory-bound). Same FLOPs, only loop order — pure locality. CLI + server
now DEFAULT to Q8 quantize-on-load (load_gguf_model_q8); `sub0llm-cli --f32` opts out;
episodic stays f32 (needs training). n_params computed from reader.tensors() (parameters()
elided under q8). 489 tests pass.

**TILING/LAYOUT (analysis, README):** single-token GEMV is bandwidth-bound w/ ZERO reuse
→ tiling weights can't help (Q8 byte-shrink was the locality win); tiling pays off only
for T>1 prompt/batched GEMM (weight tile reused across T cols) — next experiment, qbench
has the shape harness. Layout already cache-friendly (row-major (out,in)).

**SPECIALIZATION VERDICT (honest):** representation/kernel specialization (Q8 int8 +
drop f32, per model+CPU) is DECISIVE (2x speed, 2.7x RAM, beats llama). Compile-time
DIMENSION monomorphization (constexpr spec) is MARGINAL (~1.02x) for transformer GEMVs
(memory/compute-bound, not loop-overhead-bound). State this plainly. See [[verify-correctness-against-reference-before-perf]].

**KEEP-PRIMITIVES RULE (user feedback, in CLAUDE.md):** correct reusable SIMD helpers
not currently on a hot path are marked `[[maybe_unused]]` with a comment, NOT deleted
(e.g. `hsum_epi32_avx`). "We worked hard on those." See [[keep-simd-kernel-primitives]].

**P2 microbench honest finding:** single SwiGLU FFN GEMV (D=1024,F=3072), static
constexpr dims vs runtime dims, both -march=native → ~1.01x, ~3.7 GFLOP/s. A
reduction/bandwidth-bound GEMV is shape-AGNOSTIC: compile-time bounds don't change
the inner loop, and the naive C++ float reduction doesn't vectorize without
fast-math reassociation. Lesson: monomorphization's win is NOT raw matmul throughput
(BLAS-class, same either way) — it's end-to-end (op-dispatch/alloc elimination,
small-fixed-dim unroll + remainder elimination, fusion). Measure it end-to-end vs
llama.cpp, not per-GEMV.

**TOKENIZER GATE FAILS for Gemma (2026-06-04):** tokenizer.ggml.model="gemma4" =
SentencePiece (▁ marker, add_bos id 2, eos 106). Our BPETokenizer (GPT-2 byte-level)
mismatches: "The capital of France is" → llama `<bos> 818 5279 529 7001 563` vs ours
`818 245237 41626 245237...` (▁ leaks as token 245237). So a SentencePiece-compatible
tokenizer is a HARD PREREQUISITE before any Gemma forward parity. (Qwen3 gpt2 tokenizer
matched exactly — that's why it worked.) Gemma plan order now: SentencePiece tokenizer
(parity-gated) → per-layer heterogeneous forward.
**SOLVED (2026-06-04):** built SPTokenizer (tokenizer/sentencepiece.{hpp,cpp}):
score-based PQ merge, ▁(U+2581) marker, byte fallback <0xNN>, BOS prepend = llama
llm_tokenizer_spm. GGUFVocab now captures scores/token_type/bos/eos/add_bos; read_kv_pair
extended. verify auto-selects SPM for gemma*/llama. PARITY vs llama-tokenize EXACT:
"The capital of France is"→2 818 5279 529 7001 563; "the café costs $5"(utf8 file)→
2 1437 33443 5384 609 236810. (café argv divergence was Windows console UTF-8 mangling,
clean via --file.) test_sentencepiece.cpp; 493 tests pass. NOTE: Unicode in TEST_CASE
names breaks ctest name-matching on Windows codepage (mojibake) — keep test names ASCII
(same class as the em-dash issue). NEXT: per-layer heterogeneous Gemma forward.

**GEMMA 4 ARCH CORRECTED via per-layer tensor dump (2026-06-04) — head_dim VARIES:**
earlier "2 kv heads on every 6th" was WRONG (divided by 256). Truth from attn_q_norm
dims (=head_dim per layer):
- LOCAL layers (40, index%6 != 5): head_dim 256, 16 q heads, 8 kv heads, sliding
  window 1024, RoPE theta 1e4.
- GLOBAL layers (8, index%6 == 5): head_dim **512**, 16 q heads, **1 kv head (MQA)**,
  FULL attention, RoPE theta 1e6. (Big heads + 1 KV head = cheap KV over 256K ctx.)
Per layer tensors: attn_norm (input), post_attention_norm, ffn_norm (pre-ffn),
post_ffw_norm, layer_output_scale[1] scalar, attn_q/k/v/output, attn_q_norm/k_norm,
ffn_gate/up/down. Block = x + post_attn_norm(attn(attn_norm(x))); h + post_ffw_norm(
ffn(ffn_norm(h))); (1+w) RMSNorm; GeGLU(gelu-tanh); embed×sqrt(3840); final softcap 30;
tied. UNKNOWNS needing parity loop: layer_output_scale semantics, query_pre_attn_scalar,
whether RoPE partial. D=3840, V=262144, d_ff=15360, 48 layers, eps 1e-6.
FORWARD IS A LARGE HETEROGENEOUS BUILD — needs iterative parity vs llama (slow 12.7GB
ref). Reading TENSORS not metadata/config was essential (config said uniform).
**TOKENIZER PERF NOTE (user, 2026-06-04):** SIMD string parsing + fixing the O(n²) BPE
merge (CLAUDE.md bottleneck) are real STAGED wins but OFF the generation hot path
(tokenize = one-time prompt cost, negligible vs per-token gen). Matter for long-prompt/
document/high-throughput serving. Sequence AFTER SentencePiece correctness. Staged
improvements compound end-to-end.

**GEMMA4 FORWARD — AUTHORITATIVE, READ FROM llama.cpp SOURCE (2026-06-05).** Read
`D:\tools\llamacpp\vendor\llama.cpp\src\models\gemma4.cpp` (the graph) + build_norm/
build_attn in llama-graph.cpp + convert_hf_to_gguf.py. THIS is ground truth, resolved
every unknown. 12B is DENSE (no MoE — no ffn_gate_inp), n_embd_per_layer=0 (no per-layer
embeddings). Per-layer forward (inpL = residual stream):
  cur = rmsnorm(inpL, attn_norm)                      # plain x_normed*w
  Q   = wq·cur → reshape(head_dim,n_head) → rmsnorm(q_norm) per head → RoPE
  K   = wk·cur → reshape(head_dim,n_kv)   → rmsnorm(k_norm) per head → RoPE
  V   = wv·cur → reshape(head_dim,n_kv)   → rmsnorm(NO WEIGHT, plain unit-RMS per head)
  attn= softmax(Q·Kᵀ * 1.0)·V  [SCALE=1.0, NOT 1/sqrt(dh)!] → wo·  → D
  cur = rmsnorm(attn, post_attention_norm); attn_out = cur + inpL
  cur = rmsnorm(attn_out, ffn_norm); cur = down·(gelu_tanh(gate·cur) ⊙ (up·cur))  # GeGLU
  cur = rmsnorm(cur, post_ffw_norm); cur = cur + attn_out
  inpL= cur * layer_output_scale[l]                   # SCALAR ~0.05-0.36, scales WHOLE stream
Final: rmsnorm(inpL, output_norm); logits = token_embd·(tied); logits = 30*tanh(logits/30).
Embedding: inpL = dequant(token_embd[id]) * sqrt(3840)=61.9677.
RESOLVED UNKNOWNS (all were guesses before, now CONFIRMED from source):
- (1+w) norm: BAKED INTO GGUF at convert time (data_torch+1 "Gemma3RMSNorm adds 1.0").
  So use PLAIN rmsnorm(x)*w; weights are large (attn_norm ~17-30, q_norm ~1.02). Do NOT add 1.
- attention scale = hparams.f_attention_scale = **1.0** (gemma4.cpp:11 "self.scaling=1.0,
  no pre-attn scaling"). QK-norm controls magnitude; NO 1/sqrt(head_dim). query_pre_attn
  question MOOT.
- V-norm: ggml_rms_norm with NO weight (gemma4.cpp:227) — easy to miss, parity-critical.
- layer_output_scale: real per-layer scalar, multiplies full residual stream (gemma4.cpp:362).
- RoPE = NEOX (half-split, same as Qwen3) — pairs (x[i],x[i+dh/2]). theta_i=pos*base^(-2i/dh).
  GLOBAL layers use freq_factors = rope_freqs.weight[256] (=dh_global/2): values are
  [0:64]=1.0, [64:256]=1e30 → theta/ff kills rotation on pairs 64..255 (only first 64
  freq-pairs rotate on global head_dim=512). LOCAL layers: no freq_factors, full rotation,
  base=1e4. dimension_count==head_dim both → full rotary (not partial).
Per-layer from tensor shapes: q[16*dh,D] k/v[n_kv*dh,D] o[D,16*dh]; LOCAL dh=256 n_kv=8
base=1e4 window=1024; GLOBAL(idx%6==5) dh=512 n_kv=1(MQA) base=1e6 full. eps=1e-6.
PLAN: self-contained GemmaModel (gemma.{hpp,cpp}) forward_one + KV cache; store raw Q8 per
tensor in (out,in) GGUF layout, use backend::cpu::matvec_q8_0_q8_0 directly (M=out,K=in,
no per-head weight slicing — slice the f32 q/k/v results per head instead). gelu via
backend::cpu::gelu_f32 (it's gelu-tanh). Tied LM head = token_embd matvec.

**GEMMA4 12B DONE — PARITY + PERF (2026-06-05).** GemmaModel (nn/gemma.{hpp,cpp}) +
sub0llm-gemma tool (tokenize/logits/greedy, -t N). Q8 quantize-on-load, 11.91B params,
~12.1 GiB RSS. PARITY GATE PASSED vs llama.cpp b9334: tokenize exact; greedy
BYTE-IDENTICAL across multiple clean prompts (24-48 tok) incl. the model's own
<|channel>thought tokens (both engines emit them identically — NOT a llama mode).
Adversarial/confused prompts can flip a near-tied token under Q8 activation quant (llama
itself is nonsensical there) but every clean prompt matches. Committed 3 commits on
branch. CPU = Intel Core Ultra 9 275HX = Arrow Lake-HX HYBRID 8 P-core + 16 E-core, 24
threads, NO SMT.
PERF (self-timed decode loop = llama's own eval-time metric, NOT llama-bench; same
workload, byte-identical tokens):
- single-core: OURS 1.60 t/s vs llama 1.37 = 117% — OUR per-core int8 GEMV BEATS ggml
  (which even has AVX-VNNI + weight repack). Kernel is good; gap was only thread SCALING.
- threading: naive per-GEMV fork-join (CV, fresh threads ~340/tok) scaled 1.2x (1.96).
  Fixed by adopting ggml's threadpool design → 3.5x (5.59 t/s @ 20t = 93% of llama 6.00):
  (1) persistent pool, (2) SPIN-WAIT barrier (atomic gen counter + _mm_pause; CV's ~10us
  wake × 340 sync/tok killed scaling), (3) thread AFFINITY (SetThreadAffinityMask, 1
  thread/logical CPU; first 8 = P-cores). Off-by-one: caller participates so -t N = N
  total (was N+1, oversubscribed 24).
SCALING IS DRAM-BANDWIDTH-BOUND (zero weight reuse at T=1: tok/s = BW / 12.65 GB):
1c=20GB/s, 8c(P-only)=56, 16c=70, 20c=71 (best), 24c REGRESSES to 59 (spin workers
starve OS — leave cores free). P-cores ~4x more BW-efficient/core than E-cores. Physical
ceiling ~70-76 GB/s (~5.5-6 t/s) — where BOTH engines land. 2c≠3.2 because (a) ~25-30%
serial main-thread work (norms/rope/QKnorm/attn-softmax/gelu/residual/act-quant) doesn't
parallelize [Amdahl], (b) BW saturates by 8-16c. Dynamic 32-row tile cursor lets fast P +
slow E cores share work w/o barrier straggler. Default = hw-4; -t 8 = P-only, -t 1 = base.
Last ~7% vs llama = its extra BW extraction (VNNI more outstanding loads + weight repack)
— a kernel/prefetch refinement, not threading. NEXT: that prefetch/VNNI kernel lever, or
move on (correctness+near-parity-perf achieved).
TOOL: tools/scripts/gguf_dump.py (meta/tensors/val modes) — handy GGUF inspector.

**PUSHED + BRANCH RECONCILED (2026-06-08).** Branch had diverged: origin had c85f398
"add GGUF model support to CLI and server" (a DUPLICATE of our d30cae9 — both added GGUF
to cli/server + BPETokenizer::from_vocab). Hand-merged (user chose "keep best of both"):
kept OUR Q8-quantize-on-load default + --f32, folded in c85f398's .gguf extension
auto-detection on --model-dir (now --model F.gguf AND --model-dir F.gguf both work) +
clearer help; from_vocab kept ours (byte_level_=true is parity-critical) + their <pad>
detection. Merge commit 4e38d32 pushed (non-force, c85f398 preserved). 493 tests pass.
**AVX-VNNI + bench_membw added** (commit bebe9d2): dot_q8_0_q8_0 now dispatches vpdpbusd
(__AVXVNNI__/__AVX512VNNI__) + software-prefetches weights 8 blocks ahead;
SUB0LLM_DISABLE_VNNI for A/B. benchmarks/bench_membw.cpp = STREAM-style DRAM probe
(read/copy/triad, thread-swept, core-pinned).
**OPEN PERF TARGET (user framing):** bench_membw measured ~104 GB/s EXERCISED bus max,
but real decode only hits ~71 GB/s = ~30% gap NOT explained by the bus → our access
pattern / serial fraction / per-core load-issue limits. Strategy user proposed: extend
bench_membw with a representative test ladder (pure-read→strided→gather/embedding→real
GEMV shapes), divide-and-conquer to find WHERE we fall off 104 GB/s; tournament shootout
w/ subagents for kernel/threading options, iterate on round winners. Consider: shared
silicon (Arrow Lake E-cores share L2 in 4-clusters; ring/LLC + mem-controller shared, so
the "wall" may be ring/LLC not raw DRAM); use 2/4-core theoretical-max as a SIMPLER
baseline than 20+ cores; simplify multi-core before scaling N-axis. **THERMAL DRIFT: must
INTERLEAVE llama.cpp and sub0llm runs (A/B/A/B short) — Arrow Lake-HX throttles on
sustained load, back-to-back long runs bias results.**
**GROUNDWORK DONE — GAP LOCALIZED (2026-06-08, commit bdd1910).** Extended bench_membw
into a Q8 decode LADDER (links lib): read (pure f32) → q8_stream (Q8 bytes, no math) →
q8_gemv (real dot_q8_0_q8_0 over 512MiB buffer, activation reused = decode hot path);
thread-swept, core-pinned, best-of-2/cell + cooldowns + thermal-drift recheck. FINDING:
q8_gemv ≈ q8_stream ≈ read at scale (94 vs 84 vs 108 GB/s @24t) → **the int8 kernel is
MEMORY-bound not compute-bound, saturates ~94 GB/s (87% of read ceiling) IN ISOLATION.**
Real decode only 71 GB/s = 75% of that → **the ~25% gap is ORCHESTRATION, NOT the kernel**:
~25-30% serial main-thread work (norms/rope/QKnorm/attn-softmax/gelu/residual) + 340
barriers/token + small per-GEMV M (2048..262144 vs ladder's single 131586-row dispatch).
So the NEXT real lever is NOT kernel micro-opt (it's bus-saturating) — it's reducing
per-token orchestration: FUSE/BATCH GEMVs to cut barrier count + OVERLAP the serial
fraction (worth ~20 GB/s, more than the kernel has left). vs llama the residual ~6% is its
VNNI weight-repack. True DRAM peak = LPDDR5x ~120-136 (102 guess was low); exercised read
max ~108 GB/s; thermal drift -18% cool-vs-hot confirmed (interleave A/B/A/B mandatory).
**ORCHESTRATION + VECTORIZATION PASS (2026-06-08, commits 22e59c3..b8b9982, pushed).**
User: keep optimizing OURS toward the achievable kernel ceiling (~94 GB/s≈7.4 t/s),
compare to llama only at the end. Levers (all parity-gated byte-identical to llama, all
drift-free A/B via new `sub0llm-gemma --mode bench` which ROTATES config order each round
+ interleaves in ONE process):
- GEMV FUSION: pool dispatches a small GemvJob list sharing one activation under ONE
  barrier — Q/K/V (from attn_norm) + gate/up (from ffn_norm) fuse; ~337→~193 barriers/tok.
  +4.6% (reproduced twice). --no-fuse toggles (set_gemma_fuse).
- SOFTCAP SKIP: final 30*tanh(z/30) is monotonic → greedy/argmax skips 262K tanh/tok
  (forward_one(...,apply_softcap=false)). Below noise (~<1%) but correct+free.
- ATTENTION PARALLELIZED: added generic run_generic/parallel_for to the spin pool; per-head
  Q/K/V-norm+rope+cache-store and score/softmax/weightedV run on the pool. ~neutral short-
  context (GEMV-dominated) but essential for LONG context (attn grows O(ctx)).
- SIMD f32 PRIMITIVES (the big "thousand cuts" win, +~5% on the serial path): clang
  -Rpass-analysis=loop-vectorize proved native (-ffp-contract=fast, NO -ffast-math) leaves
  every float REDUCTION scalar (rmsnorm Σx², attention QKᵀ dot + weighted-V sum, argmax).
  Added sum_squares_f32/dot_f32/axpy_f32/argmax_f32 to PUBLIC quant.{hpp,cpp} (8-wide AVX
  bulk + scalar remainder; argmax keeps first-max). Wired into rmsnorm + attention + tool
  argmax. Lifted no-fuse baseline ~5.54→5.83 median tok/s. test_quant.cpp +4 tests (497).
  Remaining auto-vec gap = softmax exp+sum (needs vectorized exp; parallelized already).
NET: ~5.59 → best ~6.1-6.2 tok/s, all parity-clean. KEY LEARNING: bench_membw ladder
showed the GEMV kernel ALREADY saturates the bus in isolation (94 GB/s); the gap is
orchestration (serial fraction + sync + small-M) — each lever is single-digit % (death by
a thousand cuts), confirming we're near the practical limit for the per-layer-sequential
fork-join structure. Bigger jump would need ggml-style whole-forward-in-threadpool (workers
run norms/attn too, no fork-join boundaries) — high effort, deferred. Branch had also
diverged from origin (c85f398 dup GGUF-to-CLI feature) → hand-merged (kept our Q8-on-load +
their .gguf extension detect), pushed (merge 4e38d32).

**PROMPT-PROCESSING (PP) OPTIMIZED (2026-06-08, user: PP is priority, TG done).** User added
bench_sweep_threads.ps1 (PP+TG sweep, Latin-square thermal balancing). PP was ~3x slower
than llama single-thread (1.42 vs 4.28). ROOT CAUSE: PP was sequential forward_one per
prompt token = T independent GEMVs re-streaming all 12.6GB each (bandwidth-bound at decode
speed) + recomputing the 262K LM head every token. FIXES (all BYTE-IDENTICAL greedy):
- forward_prefill (gemma.{hpp,cpp}): batched prefill, one pass/layer over all T tokens via
  pooled row-reuse GEMM; causal attention over the batch; LM head last-token only. forward_one
  gained compute_logits=false (fill KV, skip head) for the T==1 fallback. (out,T) GEMM outputs
  transposed to (T,out) for contiguous per-head norm/RoPE. PP 1.42→~14.7 t/s @16t (~10x).
- gemm_row_q8_0 (quant.{hpp,cpp}): 4-COLUMN REGISTER-BLOCKED int8 GEMM — load weight block,
  fold |w|, decode f16 scale ONCE, reuse across 4 activation columns (q8_block_dot/q8_accum_col
  helpers; vpdpbusd VNNI). matmul_q8_0_q8_0 + pmatmul use it. Same dots same order → bitwise
  identical (unit-tested). PP 14.7→19.1 t/s @16t = **92% of llama (20.87)** (was 70%); 1-thread
  2.49 vs 3.76 = 66%. Remaining = wider blocking/scheduling + forward_prefill transposes.
**E-CORE FINDING (user hypothesis CONFIRMED 2026-06-08):** TG is bandwidth-bound so it
PREFERS E-cores. set_gemma_affinity_base(b) / --cpu-base N pins pool to CPUs [b..b+threads).
Arrow Lake P=0-7, E=8-23. Measured (stable ×3): 8 E-cores 5.47 t/s > 8 P-cores 4.92 (+11%);
16 E-cores = 8P+8E = 6.63 t/s — full TG throughput on E-cores with ALL 8 P-CORES FREE.
P-cores throttle under all-core memory load + gain nothing from IPC on a zero-reuse stream.
**OVERLAP STRATEGY (user: critical for wall-time):** hybrid CPU maps PP→P-cores (compute-bound
GEMM) || TG→E-cores (bandwidth-bound) concurrently. Within ONE request PP→TG is causally
serial; real overlap = continuous batching (seq A decodes on E-cores while seq B prefills on
P-cores) — server-architecture effort, the next big lever. Default unchanged (16t=6.63).
**DEEP RESEARCH LAUNCHED (2026-06-08):** Workflow deep-research on novel cross-discipline CPU
LLM inference optims beyond llama's tiled GEMM (LUT/T-MAC mpGEMM, sub-8bit/BitNet, sparsity/
PowerInfer, speculative/Medusa/EAGLE, entropy-coded weights). Flagged: most big wins are LOSSY
(break byte-exact parity gate) — separate lossless (tiling/sched/entropy-Q8) from lossy
(Q4/ternary/sparsity). Report pending.

Related: [[scoring-use-batched-forward-not-forward-one]], [[verify-correctness-against-reference-before-perf]].
