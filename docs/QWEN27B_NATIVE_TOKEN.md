# Native mapped Qwen 27B: one-token inference

`sub0llm-dense-token` executes the complete downloaded model for one raw token at position zero:
embedding, all 64 hybrid GDN/full-attention layers, dense SwiGLU FFNs, final RMSNorm and Q3_K LM head.
It keeps the original quantized weights mapped. It uses Sub0Llm's GDN math, GGUF decoders and head
conversion helpers. It is a standalone proving ground; the general engine loader does not yet load
these mapped tensor views. No tokenizer, sequence cache, context window or autoregressive loop exists
in this tool. At one position, attention softmax is exactly one, so Q/K normalization and RoPE cannot
affect attention output; the tool computes V, the fused projection's gate, and output projection.
This shortcut is valid only for this position-zero, single-token interface.

A local build is preserved under `out/build/qwen35-native/` in the Qwen worktree. Run:

```sh
out/build/qwen35-native/sub0llm-dense-token /run/media/craig-hutchinson/54D4C5011B8AF586/Models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf 1543
```

Rebuild after generating `qwen3.8-27b` configuration headers in a build directory:

```sh
sub0llm-configure --profile-dir profiles --profile qwen3.8-27b -o BUILD/generated/sub0_config.hpp
clang++ -std=gnu++26 -O2 -Wall -Wextra -Werror -fconstexpr-steps=1000000000 -IBUILD/generated -Iinclude tools/sub0llm-dense-token.cpp -o BUILD/sub0llm-dense-token
```

Or build the explicit CMake target `sub0llm-dense-token` in that configured build. An optional fourth
argument is a dump prefix; it writes `.layer0` through `.layer63`, `.norm` and `.logits` as little-endian
f32 vectors. Input token IDs must be integers in the configured vocabulary range.

Observed on this machine, token 1543: 248320 finite logits, top token ID 13, top logit 9.2862072;
peak RSS 5643620 KiB (5.38 GiB). The single run took 29.90 seconds; this is a proving-ground observation,
not a controlled performance benchmark.

## Independent numerical check

Against llama.cpp revision `cf659f240b232e15700b1721bbb43c1d973c52e9`, with the preserved scalar
float-activation diagnostic patch and `GGML_CPU_REPACK=OFF`:

- All 64 hidden vectors are captured; worst layer relative L2 is 0.000290549 (layer 52).
- Last-layer hidden relative L2: 0.000213061.
- All 248320 logits: relative L2 0.000107051, maximum absolute difference 0.00251627.
- Both implementations select top token ID 13.
- The refactored first-layer tool retains relative L2 8.05504e-7.

`tools/reference/capture_llama_token_float.cpp` captures the independent all-layer/logit reference
for raw token 1543; compile/link like the first-layer float harness. The reference patch is diagnostic,
not part of production Sub0Llm. Stock llama.cpp quantizes activations and is a different numerical path.
Use `compare_hidden.py` for vector checks; the measured full-logit check passes relative tolerance 0.001.
This verifies one input, not vocabulary-wide behavior, multiple positions or a context window. Those
remain required before chat/backend work (#43), along with independent sequence-state validation,
full suites and review before merging PR #44. Tracking: #15 and #29.
