# Dense models used as-is: quantised-resident weights

Status: scalar mapped-projection baseline implemented; engine integration pending. Tracking: issue #15. Facts below were read from the real
`unsloth/Qwen3.8-27B-GGUF` `UD-IQ1_S` file (5.77 GiB) on 2026-11-06 (AGENTS section 9), not recalled.

## Goal

Run a quantised model from its own weights, as downloaded, and spend the memory that saves on context and
speed. The dev box has 29 GiB RAM. The 27B UD-IQ1_S weighs 5.76 GiB; today's path would expand it to a 51.4 GiB
bf16 blob (`sub0llm memplan`, d5120 L64 ff17408 v248320), which is why a model that fits cannot run.

## What is in the file (verified)

851 tensors, 5.76 GiB. Quant type is chosen PER TENSOR (the unsloth "UD" scheme), so one role mixes several types.

| type | share of bytes | tensors | in `gguf.hpp` today |
|---|---|---|---|
| IQ1_S | 58.0% | 264 | yes (dequantise) |
| IQ2_XXS | 11.9% | 59 | yes |
| Q3_K | 8.8% | 1 (`output.weight`) | **no** |
| Q2_K | 7.8% | 8 (incl. `token_embd.weight`) | **no** |
| IQ1_M | 4.3% | 25 | **no** |
| IQ2_XS | 4.2% | 21 | **no** |
| IQ2_S | 3.1% | 9 | **no** |
| IQ3_XXS | 1.1% | 12 | **no** |
| Q8_0 / F32 | 0.6% | 96 / 353 | yes |
| IQ4_XS, IQ3_S | <0.1% | 2 / 1 | **no** |

70.5% of the bytes are in formats the reader can already decode; 29.5% (79 tensors) are not. By role the bulk
is FFN gate/up/down (3.4 GiB, 59%), then `attn_qkv` (479 MiB), `ssm_out`, `attn_gate`, `attn_q`, `attn_output`;
`output.weight` (1.27G elements) is the largest single per-token GEMV. `ssm_alpha/beta` are Q8_0 and norms/conv are F32.

## What exists to build on

- `.moeq`: routed experts kept quantised-resident, dotted in place (`moe_quant_dot.hpp`), keyed `(layer, expert, which)`.
- `.bbq` (S0B1): per-role backbone store for LM head, token embedding, gated-residual and shared-expert-down
  (`docs/BACKBONE_NATIVE_QUANT.md`). Not the dense FFN or the attention/GDN projections.
- `gemv.hpp` column-split threading; bit-exact across thread counts.
- `decode.cpp` has `if constexpr (USE_MOE)` branches, so a dense path exists; whether it covers GDN + gated FFN at
  these dims is unverified.

## Design

1. **Map the GGUF, do not copy it.** Weights are mmap'd from the downloaded file; tensor offsets come from its table.
   No transplant blob for quantised tensors. Only what the engine must hold in another form (small F32/bf16 items) is loaded.
2. **A dense quant store keyed `(Role, layer)`** like S0B1, per-tensor `type_raw` (a role mixes types), covering FFN
   gate/up/down, `attn_qkv`, `attn_gate`, `attn_q/k/v/output`, `ssm_out`, `output.weight`, `token_embd` (row gather).
3. **Dot kernels per type, in priority order by bytes**: P1 IQ1_S + IQ2_XXS (70%); P2 IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS,
   IQ4_XS, IQ3_S (13%); P3 Q2_K, Q3_K (16%, and `output.weight` is the hottest GEMV, so it matters for speed).
   Until a type has a kernel, its tensors may be dequantised to bf16 at load as an explicit, reported fallback
   (cost per tensor to be computed; `token_embd` and `output` alone are about 2.4 GiB each at bf16).
4. **Worker sizing**: single-stream decode must not plan 32 x 9 GiB arenas (`memplan` today).
5. Defaults follow AGENTS section 4: off while evaluated, auto-on for the class it was proven on.

## Memory budget (to confirm by measurement)

Initial proving ground (2026-11-07): `sub0llm-dense-probe MODEL.gguf TENSOR` maps the GGUF and
uses `denseq::project` to decode one block at a time into caller-reserved scratch, without expanding
the matrix. The real Q3_K `output.weight` (5120 x 248320, 546,304,000 encoded bytes) completed with
256 floats of scratch and peak RSS 568,128 KiB. First/middle/last outputs match a whole-row
dequantize-then-dot check exactly. Real IQ1_S `blk.0.ffn_gate.weight` and IQ1_M
`blk.1.attn_gate.weight` also pass those row checks. These checks establish projection ordering and
bounded scratch; they share the scalar decoders and do not establish independent ggml parity.
This baseline is not yet connected to engine parameter loading or token generation.

Weights 5.76 GiB + KV/GDN state + activations for one stream. Target well under 29 GiB, leaving the rest for context
and page cache. Per token the engine reads roughly the file size, so memory bandwidth (about 79 GB/s on this
CPU, `docs/HANDOVER.md`) bounds decode; not yet measured.

## Gates

- Per kernel: bit-for-bit against a scalar dequantise-then-dot reference on real tensors from this file (AGENTS 6).
- End to end: logits against llama.cpp on the same GGUF (the project's oracle), at more than one scale (AGENTS 7).
- Then tokens/s on this machine, with the three-pass rule (AGENTS 13) before any "doesn't pay" verdict.
- A transplant/profile check: `sub0llm-configure --list-profiles` already compares the profile with the GGUF header (#13).

## Open questions

- Does the dense decode path support Qwen3.5-style GDN + gated FFN at d5120 (the `qwen35` arch)? Run it at the
  smallest scale first.
- Int8-activation versus float-activation dots for the I-quants (the project's MoE precedent is in `MOE_QUANT_DOT.md`).
- MTP and vision tensors in the repo's other files are out of scope.
