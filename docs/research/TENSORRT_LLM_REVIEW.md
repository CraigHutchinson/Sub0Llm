# TensorRT-LLM review — what, if anything, Sub0Llm should take from it

Research date: 2026-09-23. Bounded research review, no installs/builds/benchmarks performed, per the
task that produced this doc. Primary sources are `github.com/NVIDIA/TensorRT-LLM` (`main` branch unless a
tag is named) and `nvidia.github.io/TensorRT-LLM`, fetched directly and quoted below — see AGENTS.md §5.

## Verdict

**Not a library to link against, on this host, now or soon — but a real source of ideas, with one
idea worth prototyping immediately and independent of TensorRT-LLM itself.** TensorRT-LLM is a
Linux-only, PyTorch+TensorRT+Triton-dependent serving stack; this project's engine is a dependency-light,
compile-time-baked, single-user Windows C++ binary. The two do not compose as library+consumer. What
transfers is *engineering knowledge*, mostly gated behind the dormant CUDA backend's revival, plus one
algorithmic idea (speculative/lookahead decoding) that is backend-agnostic and attacks the exact wall
Sub0Llm's own optimization docs already named: no token has ever been able to reuse another token's
backbone-weight stream.

## Ranked recommendations

| # | Recommendation | When | Why |
|---|---|---|---|
| 1 | **Prototype speculative/lookahead decoding in Sub0Llm's own CPU engine** (prompt-lookup / n-gram draft first, since it needs no extra trained head; a small draft head later) — verify K drafted tokens in one batched forward pass of the real model instead of K sequential `forward_one` calls. | **Now** | The only technique here that changes Sub0Llm's actual bottleneck: `BACKBONE_NATIVE_QUANT.md` §1 states plainly "every backbone weight is read exactly once per token, with no cross-token amortization... decode is `SEQ_LEN`=1." Batched verification reads the backbone weight stream once for K candidate tokens. Purely algorithmic — TRT-LLM's EAGLE/Medusa/lookahead code is CUDA/Python and not reusable, but the technique needs no TensorRT-LLM dependency at all. Report gains up to 3.6x token throughput (NVIDIA's own blog, cited below) for a mechanism that composes with, not competes against, O1-O5's per-token speedups. Gate per AGENTS.md §2 (configurator flag, `if constexpr`) and §6 (needs its own correctness check: accepted-token distribution vs. baseline, not just "faster").
| 2 | **Read the Qwen3-Next FLA/GDN kernels and MoE grouped-GEMM kernel shapes as CUDA design references** once `src/backends/cuda/backend.cu` is actually extended for Qwen4-preview inference (it is currently a training-era skeleton — forward/backward/AdamW — not wired to this model at all). Re-derive rather than depend: CUTLASS and FlashAttention pieces are BSD-3-Clause, so copying kernel *shapes* is fine; depending on the TensorRT-LLM runtime is not. | **Later, tied to CUDA-backend revival** | Confirms this engine's own GDN math (already sourced from the real `transformers` `modeling_qwen4_exp.py`, per `QWEN4_PREVIEW_REFERENCE.md`) against a second real implementation of the same delta-rule recurrence — an "independent reimplementation" cross-check in the spirit of the `independent-reimplementation-catches-identity-swap-bugs` memory note, but secondary since `transformers` is already the primary source both projects port from.
| 3 | **Decline linking against, building, or running TensorRT-LLM on this host**, as a library or as a benchmark oracle, indefinitely unless the hardware/OS situation changes. | **Decline** | Windows support was deprecated at v0.18.0 and is gone from `main` (no `windows/` directory); this host is Windows 11. Even Linux/WSL2 support for this host's actual GPU class (consumer Blackwell, sm_120/121) was incomplete for large parts of 2026 (issue #11799, opened 2026-02-28). The model this project actually runs (Qwen3.8-Flash-Next / `Qwen4ExpForConditionalGeneration`, with Gated Residual hyper-connections and a 51B-param n-gram/PLE embedding table) has no found TensorRT-LLM support — the closest is plain Qwen3-Next (GDN+MoE only, no hyper-connections, no PLE table), a materially different model. llama.cpp remains the only practical oracle because it already runs the exact GGUF file this project uses.

---

## 1. What TensorRT-LLM is today

**Architecture.** The README (fetched from `github.com/NVIDIA/TensorRT-LLM`, `main`) states: *"Architected
on PyTorch, TensorRT LLM provides a high-level Python LLM API that supports a wide range of inference
setups — from single-GPU to multi-GPU or multi-node deployments,"* and *"TensorRT LLM is designed to be
modular and easy to modify. Its PyTorch-native architecture allows developers to experiment with the
runtime or extend functionality."* The current mainline flow is `tensorrt_llm/_torch/...` (PyTorch-native
execution, e.g. `tensorrt_llm/_torch/models/modeling_qwen3_next.py`), not the older TensorRT-engine
ahead-of-time-compiled flow the project's early history (and the `windows/README.md` at `release/0.5.0`)
was built around. It integrates with NVIDIA Dynamo and Triton Inference Server for serving.

**License.** `LICENSE` at the repo root (fetched raw): *"Copyright (c) 2011-2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved. This project is licensed under the Apache 2.0 license, whose full
license text is available below."* Bundled third-party components, quoted from the same file's later
sections: CUTLASS — *"Original Source: https://github.com/NVIDIA/cutlass ... Licensed under the BSD
3-Clause License"*; FlashInfer (Apache 2.0); Flash-Attention (BSD-3-Clause); `causal-conv1d`
(BSD-3-Clause); Mamba (Apache 2.0); plus an `LTX-2 Community License Agreement` for one bundled
diffusion-model component. No legal blocker to reading or re-deriving from any of this — the practical
blocker is platform and dependency-stack, covered below.

**Platform support — Windows and consumer Blackwell, verified directly.**
- `docs/source/reference/support-matrix.md` (fetched): *"TensorRT-LLM requires Linux x86_64 or Linux
  aarch64."* No Windows or WSL is listed as a supported OS in that doc.
- Release notes (`nvidia.github.io/TensorRT-LLM/release-notes.html`, fetched): *"[BREAKING CHANGE]
  Windows platform support is deprecated as of v0.18.0. All Windows-related code and functionality will
  be completely removed in future releases."* Confirmed by direct repo-tree inspection: the current
  `main` branch top-level listing (fetched) has no `windows/` directory (it exists only in old tags such
  as `release/0.5.0`).
- Consumer Blackwell (sm_120/sm_121 — this host's own class, an RTX 5070 Laptop GPU per
  `docs/INTEL_IGPU_BACKEND_DESIGN.md`'s registry read): release notes record RTX 50-series support
  arriving at v0.17.0 *"via Windows Subsystem for Linux (limited models)"* — i.e. even that was never
  native-Windows, and predates the v0.18.0 Windows deprecation. More recent evidence:
  `github.com/NVIDIA/TensorRT-LLM/issues/11799`, *"[Feature]: Compile trtllm-gen FMHA cubins for
  SM120/SM121 (consumer Blackwell)"*, opened 2026-02-28, describing that *"trtllm-gen FMHA kernels used
  by flashinfer lack pre-compiled cubins for SM120 and SM121"* affecting RTX PRO 6000 (SM120), RTX 5090
  (SM121) and DGX Spark GB10, because *"SM120 does not have tcgen05 instructions or TMEM"* — a genuine
  architecture difference from datacenter Blackwell (SM100), not just a missing recompile. The issue is
  now closed, but issue #10241 (*"Add NVFP4 KV cache support for SM120 in trtllm-gen"*) shows consumer
  Blackwell continuing to trail datacenter Blackwell in feature coverage. **UNVERIFIED**: whether full
  attention-kernel parity for sm_120/121 exists on the exact `main` commit as of 2026-09-23 — the closed
  issue establishes the FMHA gap was real and was fixed at some point, not that every kernel path is now
  at parity.
- Current version observed via the GitHub Releases page: `v1.3.0rc28` topmost, with `v1.3.0rc27` and
  `v1.3.0rc26` immediately below (fetched 2026-09-23). **UNVERIFIED**: the exact publish dates the fetch
  returned read "September 2024," which is inconsistent with the issue-tracker evidence above (issue
  #11799 opened 2026-02-28) and is very likely a date-extraction artifact of the fetch, not a real
  timestamp — treat the dates as unverified, the version numbers as reliable.

**Net for this project**: this host cannot run TensorRT-LLM at all without a separate Linux machine or a
WSL2 install whose own docs describe engine builds as *"slow due to filesystem"* — and even then, this
host's specific GPU class (consumer Blackwell, sm_120) has a documented history of second-tier kernel
support behind datacenter Blackwell.

## 2. Model coverage: does it support Qwen3.8-Flash-Next / Qwen4-Exp?

Not fully, and the gap matters. `github.com/NVIDIA/TensorRT-LLM/pull/7892`, *"[None][feat] Support Qwen3
next"* (fetched), adds:
- `tensorrt_llm/_torch/models/modeling_qwen3_next.py` and
  `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`;
- *"Gated delta-net linear attention, MoE blocks, Triton fused QKVZBA and gating"*, plus *"Flash Linear
  Attention and fused recurrent gating paths for faster inference"* and *"KV cache management tailored
  for Qwen3-Next"*, under `tensorrt_llm/_torch/modules/fla/`.

That is real, working native support for the Gated-DeltaNet + MoE combination Sub0Llm's own `gdn_math.hpp`
and `moe_math.hpp` implement (per `QWEN4_PREVIEW_REFERENCE.md`'s verified config: `linear_num_key_heads`,
`linear_num_value_heads`, 512 routed experts / top-10). **What the PR summary does not mention, and I
could not find elsewhere**: Gated Residual / hyper-connections (`hc_count=4` low-rank residual streams,
this project's `gated_residual_math.hpp`), the 51.2B-param n-gram/PLE embedding table
(`per_layer_token_embd.weight`), or the QSA sparse-attention indexer that replaces 12 of the 48 layers'
labelled-`full_attention` blocks. Those are Qwen4-Exp-specific additions on top of plain Qwen3-Next, and
none surfaced in the search. **Marked UNVERIFIED rather than "absent"**: TensorRT-LLM's model registry
moves fast (Qwen3 MoE/dense landed at v1.0, EAGLE3+Qwen3-disaggregated at v0.21 per the release notes) and
a targeted search of `tensorrt_llm/_torch/models/` on `main` for `qwen4` or `gated_residual` would give a
definitive answer; that search was not run (kept in scope of "bounded review," not exhaustive repo grep).
For the 512-expert/top-10 MoE configuration specifically: not directly confirmed for Qwen3-Next, but
TensorRT-LLM's MoE kernels are proven at a comparable scale — DeepSeek-V3's 256-expert/top-8 MoE is a
shipped, documented configuration (NVIDIA's own DeepSeek-R1/V3 B200 tech blog, linked from the README) —
so the *kernel* is very likely expert-count-parametric even if the *model file* for Qwen4-Exp specifically
does not exist yet.

## 3. Techniques worth learning from, per bottleneck

Current profile for context (`docs/optimization/opportunities/README.md`, `profile_post_o1.md`): MoE
68.7%, mixer (GDN+QSA) 21.2%, lm_head 5.4%, gated-residual 4.6% of a ~0.21s/token CPU decode.

**a) Weight-only quantized GEMV/GEMM.** TensorRT-LLM's own quantization docs (fetched via search,
`nvidia.github.io/TensorRT-LLM/features/quantization.html`): *"The INT4 and INT8 Weight-Only techniques
consist in quantizing the weights of a model and dequantizing those weights on-the-fly in linear layers
(Matmuls). The activations are encoded using floating-point values (FP16 or BF16)."* A separate `W*A8`
(SmoothQuant-style) mode quantizes activations too. The same doc states the applicability condition
directly: *"in the context of small-batch inference scenarios (batch size ≤ 4), the key consideration is
memory bandwidth, making weight-only quantization methods the preferred choice"* — an independent
confirmation of the bandwidth-bound diagnosis this project's own O1-O5 work already reached. **Important
correction to avoid overclaiming a match**: TensorRT-LLM's baseline "weight-only" mode keeps activations
in FP16/BF16, unlike Sub0Llm's own `moe_quant_dot.hpp`/`backbone_quant_dot.hpp` (B35, O5), which fuses an
**int8-quantized activation** against native-quant weight bytes — that is closer to TensorRT-LLM's
separate `W*A8` mode than to its plain weight-only mode. Net: Sub0Llm is not behind the reference idea
here, it independently arrived at the more aggressive of TensorRT-LLM's two quantization modes, already
measured (O1: +2.14x decode). **Transfers to**: nothing actionable for the CPU path beyond what's already
built; the GPU-kernel implementation (`cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/`, `marlin/`) is a
CUDA-backend-revival reference only, for tile/warp-shape ideas.

**b) MoE kernels (grouped GEMM, expert scheduling).** `cpp/tensorrt_llm/kernels/moe/` implements
grouped-GEMM routing (repo tree listing, fetched) — the standard high-batch MoE serving pattern: collect
tokens routed to each expert across a *large batch*, then run one big GEMM per expert. This is exactly
the regime where it pays off, and exactly the regime Sub0Llm's decode is not in: `BACKBONE_NATIVE_QUANT.md`
states plainly that at `SEQ_LEN=1` decode, *"every backbone weight is read exactly once per token, with no
cross-token amortization... each token pays a full weight stream either way."* At batch=1, grouped-GEMM
MoE degrades to per-expert GEMV — the exact problem Sub0Llm's own B35 (`MOE_QUANT_DOT.md`, fused
quantized MoE dot product, +2.4x) already solved independently. **Transfers to**: nothing new for
single-user CPU decode; would matter only if Sub0Llm ever served *concurrent* requests at real batch, which
is not this project's shape today.

**c) Linear-attention/GDN kernels.** The most directly relevant kernel family — same Gated-DeltaNet math
this project's `gdn_math.hpp` implements. Worth reading `tensorrt_llm/_torch/modules/fla/*` (Triton,
Python) purely as source, as a second real-world port of `torch_chunk_gated_delta_rule` to cross-check
against. Secondary in priority only because Sub0Llm's own reference (`QWEN4_PREVIEW_REFERENCE.md`) was
already pulled from the primary `transformers==5.16.1` source both projects port from, not from
TensorRT-LLM's derived version. **Transfers to**: CUDA-backend revival, as noted in recommendation #2 —
Triton kernels themselves don't run outside CUDA, but the recurrence structure and chunking strategy are
portable knowledge.

**d) Decode-phase attention (XQA) and paged KV cache.** `decoderMaskedMultiheadAttention/`,
`contextFusedMultiHeadAttention/`, `kvCacheUtils.h`, `fmhaDispatcher.cpp` (repo tree, fetched) are
CUDA-only decode-step attention kernels with paged KV-cache management. Relevant only to the 12/48 QSA
attention layers in Qwen4-Exp (a minority of the 21.2% mixer-phase budget, GDN being the other ~36/48
layers) and only once/if the CUDA backend runs real inference. **Transfers to**: CUDA-backend revival
only; no CPU analog worth porting (Sub0Llm's own `qsa_math.hpp` already implements the indexer/block
top-k selection from the real reference).

**e) Speculative decoding (Medusa, EAGLE, lookahead/n-gram).** Already covered as recommendation #1 —
repeated here because it is the one item explicitly worth elevating. NVIDIA's own technical blog
(`developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/`,
title fetched via search) reports throughput gains up to 3.6x. TensorRT-LLM ships Medusa (extra
prediction heads), EAGLE-1/2/3 (single-layer draft transformer over hidden states, draft-generation and
verification both inside the engine), and lookahead/n-gram-style approaches (per the docs page title
`Speculative Sampling — TensorRT-LLM` and multiple independent write-ups). **This is the one item in this
whole review that does not require anything from TensorRT-LLM's code** — the mechanism (propose K
candidate tokens cheaply, verify all K in one batched forward pass of the real model, accept the matching
prefix) is architecture- and backend-agnostic. It is the only technique surveyed here that changes
Sub0Llm's actual constraint (no cross-token weight-stream reuse at batch=1), rather than optimizing
within it. Concrete starting point for this project: prompt-lookup / n-gram-match drafting needs no
trained draft head and can reuse this project's own tokenizer/corpus tooling for a first cut; a trained
draft head (Medusa/EAGLE-style) is a natural follow-up once the batched-verify plumbing exists. **Transfers
to**: the CPU engine directly, today — no CUDA dependency at all. Must be designed as a new
configurator-baked capability per AGENTS.md §2 (the batched-verify shape is a real architectural choice,
not a runtime toggle) and needs a correctness gate per AGENTS.md §6 (verify accepted-token distribution
matches the non-speculative baseline's greedy/sampled output before trusting any throughput number, per
AGENTS.md §9's "validate against real output" discipline).

## 4. Is linking against it realistic?

No, on multiple independent grounds, each sufficient alone:
- **Platform**: Linux x86_64/aarch64 only, confirmed above; this host is Windows 11. No native path.
- **Dependency stack**: PyTorch + TensorRT + Triton (JIT) + CUTLASS + (for multi-GPU) NCCL/MPI, a
  multi-gigabyte Python/C++/CUDA install, versus this project's explicit design goal of a dependency-light
  C++23 engine with zero runtime allocation and every dimension baked at configure time (AGENTS.md §1, §2).
  Running the stack at all reintroduces exactly the kind of runtime flexibility/indirection this project
  has spent its whole history removing.
- **Separability of components**: the "separable-looking" pieces (`cpp/tensorrt_llm/kernels/moe/`,
  `weightOnlyBatchedGemv/`, `cutlass_kernels/`) are written against TensorRT-LLM's own runtime context —
  its memory allocator, its tensor/engine plumbing, or (in the newer `_torch` flow) PyTorch ATen tensor
  types. Extracting one kernel means re-deriving its calling convention and buffer-ownership model, not a
  header-only drop-in; CUTLASS itself (BSD-3-Clause, genuinely standalone) is the one piece that is
  actually separable, and Sub0Llm doesn't currently target CUDA tensor-core GEMM/GEMM-like kernels closely
  enough for that to be a near-term win — it's a CUDA-backend-revival-stage consideration, not now.
- **Licensing cost**: effectively none — Apache 2.0 core, BSD-3-Clause CUTLASS/FlashAttention/
  causal-conv1d, MIT elements. Not the blocker; noted for completeness since the task asked for it
  explicitly.

## 5. As a benchmark reference

Not realistic soon. Structurally, yes — it could play the same "second independent oracle" role
llama.cpp already plays, for the day the CUDA backend runs real Qwen4-preview inference. Practically,
four separate blockers stack: (1) no native Windows path exists on this host; (2) even Linux/WSL2 support
for this host's own GPU class (sm_120/121 consumer Blackwell) has a documented history of trailing
datacenter Blackwell (issue #11799, #10241); (3) `src/backends/cuda/backend.cu` is presently a
**training-era skeleton** (forward/backward/train_batch/AdamW device kernels, per its own header comment)
with no Qwen4-preview inference path to benchmark against at all — TensorRT-LLM would be racing nothing
today; (4) the model itself is not confirmed supported — the closest TensorRT-LLM coverage (plain
Qwen3-Next) omits Gated Residual hyper-connections and the 51B-param n-gram/PLE table that make this
project's actual target model (Qwen3.8-Flash-Next / Qwen4-Exp) architecturally distinct, so even a
successful run would be benchmarking a different model. Revisit only once all three are true: the CUDA
backend runs real Qwen4-preview inference, a Linux/WSL2 path is available for this host's GPU, and
TensorRT-LLM ships confirmed Qwen4-Exp (hyper-connections + PLE) support.

---

## Sources

- `https://github.com/NVIDIA/TensorRT-LLM` (README, `main`)
- `https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE`
- `https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/reference/support-matrix.md`
- `https://nvidia.github.io/TensorRT-LLM/release-notes.html`
- `https://github.com/NVIDIA/TensorRT-LLM/issues/11799`
- `https://github.com/NVIDIA/TensorRT-LLM/issues/10241`
- `https://github.com/NVIDIA/TensorRT-LLM/pull/7892`
- `https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels`
- `https://github.com/NVIDIA/TensorRT-LLM/releases`
- `https://nvidia.github.io/TensorRT-LLM/features/quantization.html`
- `https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/`
- `https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/windows/README.md` (historical, for the
  "Windows support existed, then was deprecated" timeline)
- In-repo: `AGENTS.md`, `docs/ROADMAP.md`, `docs/QWEN4_PREVIEW_REFERENCE.md`,
  `docs/optimization/opportunities/README.md`, `docs/BACKBONE_NATIVE_QUANT.md`,
  `docs/INTEL_IGPU_BACKEND_DESIGN.md`, `src/backends/cuda/backend.cu`
