# Native mapped Qwen 27B: one-token inference

`sub0llm-dense-token` executes the complete downloaded model for one raw token at position zero:
embedding, all 64 hybrid GDN/full-attention layers, dense SwiGLU FFNs, final RMSNorm and Q3_K LM head.
It keeps the original quantized weights mapped. It uses Sub0Llm's GDN math, GGUF decoders and head
conversion helpers. It is a standalone proving ground; the general engine loader does not yet load
these mapped tensor views. It supports bounded raw-token/text sequences with persistent GDN and
F32 K/V state, greedy continuation, exported embedded tokenization and text EOS stopping. The console
chat proving ground is tracked in #43. This tool still allocates temporary vectors during evaluation;
it is a correctness proving ground rather than a resident service.

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

Against llama.cpp revision `cf659f240b232e15700b1721bbb43c1d973c52e9`, use all three controls:
scalar float-activation diagnostic patch, `GGML_CPU_REPACK=OFF`, and
`LLAMA_FLASH_ATTN_TYPE_DISABLED` with F32 K/V caches. F32 cache configuration alone does not ensure
F32 attention: the pinned automatic flash path casts K/V to F16 in `build_attn_mha`.

The earlier one/two-position and literal-text logit figures used automatic flash attention. Those
cross-precision comparisons are superseded by the nine-position F32 attention check below. No
production math changed. The layer-0 GDN check (relative L2 8.05504e-7) is unaffected by this attention
setting. Stock llama.cpp quantizes matrix activations and remains a different numerical path.

`tools/reference/capture_llama_token_float.cpp` explicitly disables flash attention. Compile/link
like the first-layer float harness. The reference patch is external diagnostic code, not production
Sub0Llm. `compare_hidden.py` checks one vector; `compare_sequence.py` checks every requested position
and the vocabulary count and refuses missing/nonfinite/truncated or mismatched dumps. Numerical
coverage remains finite; a passing sample is not vocabulary-wide or maximum-context validation.
Tracking: #54, #29, #43.

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

Corrected F32 attention reference, 2026-11-07 on CrogBuntu1, all 64 layers and 248320 finite logits
per position. Raw prompt: `248045,846,198,12675,248046,198,248045,74455,198`.
This is a plain ChatML diagnostic token sequence; it is not a claim of complete embedded-Jinja rendering.
Unchanged Sub0Llm outputs pass the existing 0.001 relative-L2 threshold:

| Position | top token (both) | relative L2 | max absolute |
|---|---:|---:|---:|
| 0 | 8678 | 1.7000983e-06 | 3.194809e-05 |
| 1 | 198 | 6.0709968e-06 | 9.9658966e-05 |
| 2 | 3710 | 4.1833545e-06 | 5.0067902e-05 |
| 3 | 248046 | 3.8836839e-06 | 8.5830688e-05 |
| 4 | 198 | 6.7872393e-06 | 0.00015199184 |
| 5 | 248045 | 9.9881983e-06 | 0.00013542175 |
| 6 | 248068 | 5.2306388e-06 | 5.5789948e-05 |
| 7 | 198 | 3.4431713e-05 | 0.00057816505 |
| 8 | 248068 | 2.9207262e-05 | 0.00067281723 |

Worst relative L2: 3.44317e-5. Previously positions 5/7/8 exceeded 0.001 against automatic flash
attention; the mismatch was in oracle precision, not demonstrated engine-state drift. A wider RMS
accumulation experiment barely changed that cross-precision error and was not retained. No threshold
was widened, no production arithmetic changed. Reference was rerun from the checked-in warning-clean harness at two workers; all logit files
are byte-identical to the initial eight-worker reference capture. Sub0Llm's
saved sequence used the reviewed row-partitioned projection proving ground in draft #47.

```sh
python3 tools/reference/compare_sequence.py ACTUAL_PREFIX REFERENCE_PREFIX \
  --positions 9 --expected-values 248320 --max-relative-l2 0.001
```

The reference harness accepts comma-separated raw prompt IDs after its dump prefix. Independent
longer-context/greedy sequence coverage, complete EOS examples and general engine-loader integration
remain separate work. This resolves the recorded nine-position precision discrepancy, not every task
in #29 or #43.

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
vocabulary or chat-template validation. The mapped runner accepts raw IDs and literal text; the bounded chat wrapper uses the verified
non-thinking template subset in `docs/QWEN_CHAT_POC.md`. Signed GGUF metadata arrays are retained with length validation; the
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

Verified `Hi` -> ID 12675 -> next ID 11 -> decoded comma. The prior literal-text oracle figures used auto-flash and are superseded by the corrected sequence
check above. Existing raw-sequence behavior remains available. The runner now links the existing frontend and CLI11 libraries;
prefer the explicit CMake target for rebuilding. Direct compile requires the same frontend/simdjson
libraries and CLI11 headers as the project. The console chat/context proving ground and actual EOS examples are documented in
`docs/QWEN_CHAT_POC.md`; resident service and general engine-loader integration remain separate.
