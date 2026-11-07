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
