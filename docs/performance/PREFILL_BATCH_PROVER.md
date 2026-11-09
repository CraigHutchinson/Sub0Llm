# Batched prefill: preliminary prover (#74)

Question: if prompt tokens are projected in a batch (each weight block decoded once and reused for every token in a tile)
instead of one at a time, how much faster is the projection work on the real 27B weights, and is every output identical?
Design: `docs/design/PREFILL_BATCHING.md`. Tool: `tools/sub0llm-prefill-prover.cpp` (target `sub0llm-prefill-prover`).
Library: `denseq::project_batch<Tile>` (`include/sub0/dense_quant.hpp`), `IQ1S::dot_row_batch` (`include/sub0/quant/iq1_s.hpp`).

## Method and limits

- For the largest 2D weight tensor of each quant type in `Qwen3.8-27B-UD-IQ1_S.gguf` (SHA256 `3895b6ea...`), run the serial
  path (one `denseq::project` per token, rows split over 8 OpenMP workers exactly as `Layer::project_impl` does) and the batched
  path at token tiles 8, 16 and 32, interleaved, median of 5 rounds (3 at 128 tokens) after a warm-up.
- Every output of every variant is compared bitwise (`memcmp`) with the serial output.
- **Projections only, synthetic activations**: GDN recurrence, attention scores, norms and the real prompt are not included, so
  this proves the mechanism and sizes the opportunity; it is not a model-level prefill result. The serial baseline is `main`'s
  `native-all` kernels (not the draft fused IQ2_XXS of PR #67).
- Machine: AMD Ryzen 9 8945HX (16C/32T), Linux 7.0.0-42, clang 21.1.8, `-O3 -march=native`, pinned with `taskset -c 0-7`, machine idle
  (load about 0.8), model file warm. Build `sub0llm-prefill-prover` from this branch. 2026-11-09.
- The whole-model estimate weights each format's measured ms per MiB by that format's bytes in the file, excluding the LM head
  (evaluated for the last prompt token only) and the embedding (a gather).

## Results (13 tokens, 8 workers, `native-all`)

| format | tensor | serial ms | best batched ms | speedup | bit-identical |
|---|---|---:|---:|---:|---|
| Q2_K | ffn_down | 30.6 | 10.8 | 2.8x | yes |
| IQ2_XXS | ffn_down | 74.1 | 13.7 | 5.4x | yes |
| IQ2_XS | ffn_down | 111.9 | 16.3 | 6.9x | yes |
| IQ3_XXS | ffn_up | 109.8 | 15.3 | 7.2x | yes |
| IQ2_S | ffn_down | 111.8 | 16.5 | 6.8x | yes |
| IQ1_M | ffn_down | 122.8 | 17.9 | 6.9x | yes |
| IQ3_S, IQ4_XS | attn_v | 7.1, 4.5 | 0.9, 0.8 | 7.7x, 5.5x | yes |
| Q3_K | output (head, not used by prefill) | 1445 | 215 | 6.7x | yes |
| **IQ1_S** | ffn_down | 12.0 | 11.0 | **1.1x** | yes |

Projected projection-time speedup for the model (13 tokens): **2.61x** (serial 8,277 ms to 3,174 ms). At 128 tokens, FFN
tensors, tile 32: formats on the generic dequantise-then-dot path reach 5.7x to 8.7x, IQ1_S 1.38x, whole-model projection
estimate **3.08x**. The gain grows with the batch because the decode is amortised over more tokens.

## What it shows

1. **The mechanism is proven and exact.** Batching changes no output bit on any of the 11 tested tensor types, because every token keeps
   its own accumulator and adds in the serial path's order.
2. **Most formats gain 3x to 9x**: their serial cost is dominated by decoding each block per token; the batch decodes once.
3. **IQ1_S is the lever.** It is 58% of the file bytes and 67% of the projected batched time. Its serial native kernel already decodes in
   registers and streams about 17 GB/s of weights per token (roughly half of the 32.9 GB/s probe), so batching gains only 1.1x to 1.4x.
   The batched kernel runs at roughly 16-20% of the 8-core FP32 roof: each output row re-streams the whole activation tile from L2.
   If IQ1_S reached 4x, the projected whole-model projection gain would be about 5x (derived from the table's ms per MiB).
4. Pass 1 (decode-reuse with 16+ live accumulators) made IQ1_S slower (0.77x, register spills); pass 2 (decode a block to a buffer,
   advance four tokens' accumulators together) reached 1.09-1.38x. A third pass is the next step (three-pass rule).

## Next

- IQ1_S: a row x token register micro-kernel (for example 4 rows x 4 tokens) so each activation load feeds several rows and the
  activation tile stays in L1; keep per-token order bit-identical. Re-measure alone and in combination (AGENTS section 13).
- Generic path: the same token blocking; try AVX-512 (`__AVX512F__` is available; the code uses AVX2 only).
- Integrate in the tool (`--prefill-chunk`): FFN and attention first, then the GDN `project_batch` hook, then the corpus gate before any default.
- Compare with and without PR #67's fused IQ2_XXS and PR #71's IQ1_S kernel once they land.
