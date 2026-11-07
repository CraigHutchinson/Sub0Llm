# Local Qwen 27B chat and context proving ground

`scripts/qwen_chat_poc.py` drives the mapped 64-layer backend with the downloaded model's embedded
byte-BPE tokenizer. It retains user/assistant text, replays the accepted transcript each turn, checks
the 128-position budget before inference, and supports `/context`, `/reset` and `/quit`. It starts a
new process per turn; it is not a resident HTTP service or a cross-request incremental KV session.

```sh
python3 scripts/qwen_chat_poc.py
# Or select a different file explicitly:
python3 scripts/qwen_chat_poc.py MODEL.gguf
```

The default model is the sibling `../../Models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf` download
relative to the repository root. Defaults use `out/build/qwen35-native/sub0llm-dense-token` and sibling `tokenizer/`, eight independent
output-row workers and up to 32 new tokens. Override `--backend`, `--tokenizer-dir`, `--threads` and
`--max-new-tokens` as needed. More than one worker requires OpenMP. The backend's default remains one
worker. Its pre-reserved per-worker scratch and disjoint output rows preserve the scalar projection
order; the checked eight-worker literal `Hi` logits are byte-identical to scalar logits.

## Verified template and completion

The wrapper implements the GGUF template's **text-only, non-thinking subset**, with
`enable_thinking=False` and `preserve_thinking=False`: trimmed user/assistant messages and the final
`<|im_start|>assistant\n<think>\n\n</think>\n\n` prefix. No custom system/developer messages, tools, vision,
thinking toggle or partial assistant prefix is supported.

`sub0llm-export-tokenizer` also exports the actual `chat_template.jinja` when present. The dev-only
`tools/reference/check_chat_template.py` uses Jinja2's sandbox to compare this embedded source with
the wrapper; one-message, three-message, Unicode/whitespace cases match byte for byte (71, 142 and
155 UTF-8 bytes). Jinja2 is required only for this independent QA script; the chat wrapper uses the
Python standard library. The previous legacy llama.cpp template-format API simplified ChatML instead
of executing the complete embedded Jinja, so its earlier 52/123-byte comparisons did not establish
full-template parity and are superseded by this check.

The backend returns decoded text through `--response-file` and a final `TEXT_RESULT` line containing
the generated-token count and `finish=eos|length`. The wrapper accepts only the final diagnostic and
reports `finish_reason` in its JSON. EOS is not appended to the assistant text. A length-limited reply
is explicitly labelled. Invalid UTF-8 at a capped byte-BPE boundary is refused rather than silently
changing replayed content. A failed turn never commits a new user/assistant pair.

## Context accounting

`--inspect-context` counts and checks a prompt without inference. The reservation is
`max-new-tokens-1` additional evaluated positions: the final prediction does not need feeding back.
Prompt plus reservation must fit 128; next turn is retokenized with the assistant/end-of-turn markers
and checked afresh. There is no silent truncation. `/context` reports the last accepted prompt budget
and message count; `/reset` clears transcript and last-result/usage state. The compiled cache capacity
is fixed in this proving ground; these diagnostics do not imply dynamic cache-memory planning.

## Actual-model verification, 2026-11-07, CrogBuntu1

The actual model answered `Hi` with **`Hi there!`** and stopped at EOS after three generated tokens.
The prompt used 13 tokens. All 16 evaluated positions, including the EOS prediction, matched the
independent F32 reference at the unchanged relative-logit L2 gate 0.001: worst 7.87653e-5, worst
absolute logit difference 0.00124872; all 16 top IDs agree. Use
`compare_sequence.py ACTUAL_PREFIX REFERENCE_PREFIX --positions 16 --expected-values 248320
--max-relative-l2 0.001` to check the full trajectory.

The reference requires the pinned scalar float-activation diagnostic patch, `GGML_CPU_REPACK=OFF`,
F32 KV caches and **explicitly disabled flash attention**. Automatic flash attention casts F32 K/V
to F16 and produced the old nine-position failures; #54/PR #55 corrected that oracle mismatch. No
production arithmetic or numerical tolerance was changed. The reference harness accepts optional
worker count 1..8 after its comma-separated token list (default two); this trajectory used eight.

One separate complete `Hi` turn observed 5,869,828 KiB peak RSS (5.60 GiB), below the declared
8,192 MiB budget, and again produced `Hi there!` at EOS. This is a single memory observation, not a
controlled speed benchmark or maximum-context memory bound. Checked backend SHA256:
`c89cdcc9756e100a84eed5511c6def7ed2daa439f3749e30c802ad86632d4d34`.

The real two-turn wrapper run answered `Hi` with `Hi there!` (three tokens, EOS), then answered
`Repeat my first message.` with `Hi` (one token, EOS). Prompt usage grew from 13 to 31 positions,
accepted message count from two to four; `/reset` then reported zero messages and null context.
Actual tokenizer-based overflow was refused before inference and preserved accepted history.

The wrapper's unit checks cover template structure, completion/cap labels, reset, failed context,
missing diagnostics and partial UTF-8 rollback. The scripted run and reference checks are recorded in the issue-linked PR. This remains finite sample coverage, not a maximum-context quality
sweep or proof of normal engine-loader integration. Per-layer temporary allocations, prompt replay,
scalar block decoding and absent native phase timers still limit practical performance.

Tracking: #43 (chat/context proving ground), #29 (broader runtime integration), #18 (quant modules).
