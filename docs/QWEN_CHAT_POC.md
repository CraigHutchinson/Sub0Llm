# Local chat and context-window proving ground

The console wrapper `scripts/qwen_chat_poc.py` uses the mapped 27B backend and exported embedded
Qwen tokenizer. It retains user/assistant transcript text between turns, renders the verified plain
ChatML subset, reports token budget, refuses overflow, and supports `/context`, `/reset` and `/quit`.
Each accepted turn starts a fresh backend process and replays the full transcript; this is a console
proof of concept, not a resident server or incremental cross-request KV session.

```sh
python3 scripts/qwen_chat_poc.py MODEL.gguf --max-new-tokens 8
```

Defaults point to the preserved `out/build/qwen35-native/sub0llm-dense-token` and sibling `tokenizer/`
in the worktree. Override `--backend` and `--tokenizer-dir` when needed. `--threads` defaults to 8;
OpenMP is required for more than one worker. Raw model/tool defaults retain one worker. Output rows
are partitioned among pre-reserved independent scratch buffers. A real `Hi` run at eight workers was
bit-identical to scalar logits and observed at 4.31 seconds (not a controlled benchmark).

The backend's `--inspect-context` validates/counts a prompt without inference, and `--response-file`
returns decoded text without requiring log parsing. The wrapper reserves at most `max-new-tokens-1`
additional evaluated positions because the final predicted token need not be fed back. A turn is
accepted only if prompt plus this reservation fits the generated capacity (128). Next turn is
re-tokenized and checked again with the additional assistant/end-of-turn markers included. There is
no silent truncation. Failed turns preserve accepted history; reset clears transcript and usage.
Capped output ending inside a UTF-8 character is rejected rather than silently changing replayed text.

## Verification and limits

- Plain single-user and three-message history templates match the downloaded model's llama.cpp
  template renderer byte for byte (52 and 123 bytes in the checks).
- The actual 27B two-turn scripted run (`Hi`, `/context`, `Hi`, `/context`, `/reset`, `/context`),
  capped at one response token per turn, used 9 then 21 prompt tokens, retained 2 then 4 messages,
  and reset to zero messages/null context. Both capped responses were `<think>`: the default model
  template enables thinking. This demonstrates state/accounting, not useful complete answers.
- Actual tokenizer-based overflow is refused before inference; accepted history remains unchanged.
- All nine prompt positions select the same top token as the pinned float-activation reference.
  However relative-logit L2 errors range from 5.28e-5 to 0.001668; the last is 0.001353. Positions 5,
  7 and 8 fail the existing 0.001 relative gate. This longer-sequence drift is unresolved; no threshold
  has been widened. The earlier one/two-position parity results remain valid for those tested inputs.
- No tools, images, custom system policy, thinking toggle, sliding context or HTTP interface is
  implemented. The wrapper uses the model's default thinking behavior. EOS stopping exists in the
  backend but has no real-model full-response termination example yet.

Tracking: issue #43; numerical sequence drift remains part of #29. The independent template harness
is `tools/reference/chat_template_llama.cpp`. Broad merged regression results are in
`docs/QWEN27B_NATIVE_TOKEN.md`; this follow-up does not modify engine math.
