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

## Stateful sequence proving ground (2026-11-07)

The tool now supersedes the original position-zero limitation. It keeps each GDN layer's recurrent
state and convolution history, and each attention layer's float32 K/V cache, for the sequence.
Full attention computes Q/K normalization (epsilon 1e-6), raw-GGUF split-half partial RoPE,
causal softmax, sigmoid gating and the output projection at each position.

Pass comma-separated raw token IDs. To run one greedy feedback step after a prompt, provide the
optional continuation count after a dump prefix (use `-` to disable dumps):

```sh
out/build/qwen35-native/sub0llm-dense-token MODEL.gguf 1543,88123
out/build/qwen35-native/sub0llm-dense-token MODEL.gguf 1543 - 1
```

Prompt plus continuation positions cannot exceed the generated `SEQ_LEN` (128 in the 27B profile).
The tool rejects overflow and malformed IDs before mapping weights. Context resets on each process
invocation; there is no truncation/sliding-window policy. Continuation count 1 feeds the first argmax
back once and computes the following logits. No special-token stopping, tokenizer or chat template
is provided. This remains a proving ground with per-layer temporary allocations, not an optimized
long-running chat service. Subsequent-position dumps use `.posN` before the existing suffixes.

Independent float-activation/F32-cache reference results (all 248320 logits per position):

| Input sequence | Position | top token (both) | relative L2 | max absolute |
|---|---:|---:|---:|---:|
| 1543,88123 | 1 | 198 | 6.10538e-5 | 0.00117445 |
| 1543 then greedy 13 | 1 | 220 | 3.76653e-5 | 0.00120127 |

Observed peak RSS is 5834604 KiB (~5.56 GiB). Both relative-logit checks pass tolerance 0.001.
The reference token harness accepts an optional comma-separated prompt after its dump prefix and
uses F32 KV caches to match this path. These are two-position checks, not long-context validation;
context-limit rejection is checked separately. Chat/backend integration (#43), longer sequence and
reset tests, normal engine-loader integration and broad merge gates remain.

Broad frontend regression gate: native Clang full frontend suite passes 220407 assertions in 326
cases after updating the recipe-count expectation for ordinary-attention query/gate halves. Some
optional external-fixture checks skip when their model/tokenizer artifacts are absent; actual 27B
checks above are separate. Default-engine regression is tracked in the active PR before merge.

## Embedded tokenizer preparation

`sub0llm-export-tokenizer MODEL.gguf DIRECTORY [TEXT]` exports the downloaded model's own vocabulary,
merge ranks, token-type classifications and EOS/pad IDs into the existing Qwen tokenizer loader's
three-file format. It omits unused padding rows. The verified qwen35 model exports 248077 usable
entries, EOS 248046 and pad 248055. No tokenizer download is required. The optional text argument
prints comma-separated IDs usable by `sub0llm-dense-token`, then a decoded round trip.

A durable export and executable are under `out/build/qwen35-native/tokenizer` and
`out/build/qwen35-native/sub0llm-export-tokenizer`. The CMake target links the existing frontend library.
Independent llama.cpp tokenization agrees on the four checked samples: `Hello, world!` (IDs
9419,11,1814,0), contractions/digits/newlines, multilingual text with emoji, and chat special tokens.
The reference harness is `tools/reference/tokenize_llama.cpp` (compile against the pinned llama.cpp
headers/library, no diagnostic patch needed for tokenization). This is sample coverage, not exhaustive
vocabulary or chat-template validation. The mapped runner still accepts raw IDs; direct text/chat
integration is the next step. Signed GGUF metadata arrays are retained with length validation; the
full frontend suite passes 220412 assertions / 327 cases after this change.

## Literal text input and decoded output

After exporting this model's tokenizer, the mapped runner accepts `--text` and decodes greedy
predictions. For the durable build, tokenizer defaults to the sibling `tokenizer/` directory:

```sh
out/build/qwen35-native/sub0llm-dense-token MODEL.gguf --text 'Hi'
out/build/qwen35-native/sub0llm-dense-token MODEL.gguf --text 'Hi' --continuation-steps 1
```

Use `--tokenizer-dir DIRECTORY` for another export location, and `--dump-prefix PREFIX` for numeric
dumps. Raw positional IDs/dump/continuation arguments remain supported. Text and raw IDs are mutually
exclusive; empty prompts and context overflow are rejected. EOS predictions stop text generation and
are not included in decoded continuation. Vocabulary count and EOS are checked against GGUF metadata;
use tokenizer files exported from the same model. No chat template or implicit BOS is added.

Verified `Hi` -> ID 12675 -> next ID 11 -> decoded comma. All 248320 logits match the independent
float-activation reference at relative L2 8.83061e-5 and max absolute 0.00204515. Existing raw-sequence
checks remain documented above. The runner now links the existing frontend and CLI11 libraries;
prefer the explicit CMake target for rebuilding. Direct compile requires the same frontend/simdjson
libraries and CLI11 headers as the project. Structured chat templating/service and broader EOS/context
behavior remain follow-ups under #43.
