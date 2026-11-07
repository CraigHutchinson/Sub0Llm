# Independent Qwen reference checks

These standalone harnesses link an external llama.cpp checkout; production Sub0Llm does not link it.
Verified revision: `cf659f240b232e15700b1721bbb43c1d973c52e9` (2026-11-07).

Build llama.cpp with shared libraries, CPU backend and target `llama`. From this project's root,
compile the decoder checker using `-Iinclude -I<reference>/ggml/include`, link `-lggml-base` from
`<reference>/build/bin` and set that directory as the runtime library path. Pass the actual GGUF
filename. It compares the first/middle/last block of every registered quantized tensor bit for bit.
On the downloaded 27B: 1,494 blocks, zero differing values. This samples three blocks per tensor,
not every weight in the file.

Compile `capture_llama_layer0.cpp` using `<reference>/include` and `<reference>/ggml/include`,
linking `-lllama -lggml-base`. Pass `MODEL.gguf HIDDEN.f32`. It feeds raw token ID 1543, requests
the first-layer output callback and stops graph evaluation after capture. The output is 5120 f32
values. The standalone Sub0Llm `sub0llm-dense-layer MODEL.gguf HIDDEN.f32` dumps the corresponding
vector using its actual generated 27B SiLU profile.

Observed comparison: Sub0Llm RMS 0.2252388235, llama.cpp RMS 0.2254017014; maximum absolute difference
0.0178432465, relative L2 error 0.00796917017, cosine 0.99996848447. This is a measured discrepancy,
not a passing whole-layer parity gate. llama.cpp uses quantized-activation matrix products whereas
the scalar Sub0Llm proving ground uses float activations. Stage-by-stage comparison is needed to
distinguish that expected numeric difference from a layout or math defect. All sampled scalar
weight decoders agree exactly with ggml, narrowing the investigation.

Stage investigation (2026-11-07): both capture tools now also dump attention normalization, GDN
output, attention residual, FFN normalization and FFN output as `<output>.<stage>-0`. Unmodified
reference relative L2 errors at those stages are respectively 2.59317e-6, 0.00836615, 0.00832557,
0.0420741 and 0.0300869. The first large divergence is therefore within the GDN mixer.

`llama-float-activations.patch` is an **external diagnostic patch**, applicable to the pinned
llama.cpp checkout. It replaces registered quantized CPU dot functions with float-activation scalar
dots using official decoders; it is not a production Sub0Llm dependency or an unmodified-reference
result. Rebuild target `llama`, then compile `capture_llama_layer0_float.cpp` with the same flags
as the ordinary capture plus `-lggml-cpu`. The explicit CPU buffer override alone does NOT disable repacking: llama.cpp interprets it as permission to select CPU_REPACK.
The float diagnostic still differs: GDN output relative L2 0.00836949, final output 0.00782898.
An isolated run feeding the exact diagnostic reference normalized input into Sub0Llm still gave
GDN relative L2 0.00836924. Thus activation quantization alone and initial normalization rounding
have not explained the discrepancy. Next compare internal QKV/Z/B/A projections, convolution,
head permutation and recurrence. No whole-layer correctness claim is supported yet.


Resolved first-token discrepancy (2026-11-07): rebuild the diagnostic reference with
`-DGGML_CPU_REPACK=OFF` as well as applying the float-activation patch. Confirm the run log has no
CPU_REPACK tensors. The earlier float runs still used a repacked Q2_K kernel for `ssm_out`, which
bypassed the diagnostic dots. With repacking disabled, the independent first-layer comparison passes:

| Stage | relative L2 |
|---|---:|
| raw QKV projection | 2.78010e-6 |
| raw Z projection | 2.85571e-6 |
| convolution + SiLU | 2.91136e-6 |
| recurrence + gated norm, raw head order | 1.10609e-6 |
| GDN output projection | 7.10384e-7 |
| FFN output | 1.81880e-6 |
| complete layer output | 8.05504e-7 |

Final maximum absolute error is 1.04904e-5 across 5120 values. Reproduce the final comparison with
`python3 tools/reference/compare_hidden.py SUB0.f32 LLAMA_FLOAT_NO_REPACK.f32 --max-relative-l2 0.00001`.
The ordinary, unmodified reference still differs by 0.797% because its activation quantization differs;
that remains the expected measured numerical limit of this float-activation proving ground.
This verifies token 1543 at position zero through one layer, not multi-token recurrence or later layers.


## Full-sequence F32 attention gate (2026-11-07)

The pinned llama.cpp `src/llama-graph.cpp::build_attn_mha` casts F32 K/V inputs to F16 when flash
attention is enabled. F32 cache settings alone do not prevent this. The all-layer/token harness now
sets `cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED`; combine it with the scalar diagnostic patch,
`GGML_CPU_REPACK=OFF`, and F32 KV caches. Confirm the run log says `flash_attn = disabled`.
The existing GDN-only first-layer checks do not execute ordinary attention and are unaffected.

A nine-position plain ChatML diagnostic now passes unchanged Sub0Llm outputs at every position, with
worst relative-logit L2 3.44317e-5 at the unchanged 0.001 threshold. Automatic flash attention had
produced three failures above 0.001; those were cross-precision comparisons, not evidence of state
corruption. Complete results and scope are in `docs/QWEN27B_NATIVE_TOKEN.md`.

```sh
python3 tools/reference/compare_sequence.py ACTUAL_PREFIX REFERENCE_PREFIX \
  --positions 9 --expected-values 248320 --max-relative-l2 0.001
```

This command returns 0 only when every requested complete finite vector passes, 1 on numerical
failure, and 2 on invalid/missing data. Use it as a declared workload correctness command; checking
only the first position is insufficient. The expected vocabulary size is explicit. Tests in
`tools/reference/tests` exercise late failures, absent/nonfinite/mismatched dumps and invalid limits.


The token capture harness accepts `MODEL.gguf PREFIX [COMMA_SEPARATED_TOKENS [THREADS]]`; workers
are 1..8 (default two). Invalid counts are rejected before loading a model. For the chat PoC,
`check_chat_template.py TOKENIZER_DIR/chat_template.jinja` renders the actual exported template in
Jinja2's sandbox with `enable_thinking=False,preserve_thinking=False` and checks the supported
one/three-message and Unicode/whitespace cases against the wrapper. Jinja2 is a dev-only dependency;
the runtime wrapper remains stdlib-only. The legacy format-detection API is not a full Jinja oracle.
