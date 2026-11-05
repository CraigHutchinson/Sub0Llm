---
name: wordspike-natural-prose-collapse
description: DONE+committed (09fa775) - scratch-slot collapse works in real natural-language sentences (not just synthetic control-symbol protocol); live KV-cache splice needed no new engine primitive
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User asked for scratch-token binding to work in ordinary sentences ("Rumplestiltskin has 10 eggs...") and
explicitly directed: "Scope a real fix. Also remember scratch token replacement can be performed during
prefill/generation when compound words are found or generated." Investigating first surfaced that the
*existing* trained scratch mechanism ([[meanpool-alternatives-prior-art-and-math]]-era scratchspike.hpp)
only worked over a synthetic control-symbol protocol (`# n <slot>`, literal control bytes) — a hand-written
natural sentence would never have exercised real binding behavior. This session built the real thing.

**The mechanism** (both reuse `encode_join`'s existing `TOK_SPELL_START/END`/`TOK_JOIN` markers as the
compound-word signal, not re-derived word boundaries — a model trained on any real text already emits these):
- **Prefill** (`sub0::prefill_collapse`, scratch.hpp): pure preprocessing pass over an already-encoded
  prompt, no engine change — collapses a recurring word's later mentions before generation starts.
- **Generation** (`kv_decode_generate`'s new `word_collapse` param, decode.hpp): retroactively splices the
  KV-cache when the model finishes writing a compound word live (`ctx.resize()` back to before the word
  started, re-`feed()` the replacement slot). **Needed no new low-level engine primitive** — verified
  directly against `backend_cpu.cpp`'s `KVCache`: it's a flat, position-indexed buffer with no separate
  length counter, so a smaller re-fed position is just a normal overwrite; RoPE (driven by the same
  explicit `pos` arg, not persistent state) is automatically correct after. This closes a gap
  [[repeatspike-harness-collapse-result]] and decode.hpp's own former comment both flagged as unbuilt.

New `"wordspike"` blend-schedule generator (`include/sub0/wordspike.hpp`), wired into gen_stage.cpp and
train_stage.cpp alongside scratchspike/op_curriculum. Composes with op_curriculum's arithmetic-result
collapse in the SAME document, sharing the one 6-slot pool (no Phase-3 sentinel-pair dependency needed —
see [[persistent-slot-selection-problem-backlog]] — a natural document rarely names more than a handful of
entities). Proven live via real CLI train+gen ("Rumplestiltskin" correctly collapses and re-expands for
display).

**A real methodology bug worth remembering for future capstone designs**: the `[.wordspike]` capstone
initially scored 0.0 across the board. Root cause was NOT a mechanism bug — `eval_arm` asked *free
generation* to produce the collapse point itself (either emit the slot, or byte-exactly re-spell an
arbitrary invented name), but `emit_doc` masks (`mask=0`, harness-injected) that exact position in BOTH the
COLLAPSE and FUZZY arms during training. Masked positions carry **zero gradient signal about what to
predict there** — so neither arm was ever trained to produce anything at that spot, and the eval was testing
an unsupported claim. Rereading `repeatspike.hpp`'s actual design confirmed the pattern: its proven 0.95-vs-
0.13 result never asks generation to produce the collapse either — mentions 2/3 are fed as already-resolved
slot tokens, and only the QUERY *after* the collapse point is graded. **Fix**: resolve the collapse via the
harness *before* generation starts in eval too (matching training's teacher-forced shape exactly), and
grade only what training actually taught (the downstream op-delegation continuation). After the fix,
op-delegation reached 1.00 accuracy in both arms from step 300 onward.

**How to apply**: when designing a capstone A/B where a token/span is `mask=0` in training, never build an
eval that asks free generation to spontaneously produce that exact span — check what's actually graded
(mask=1) first, and eval only that. This generalizes beyond wordspike: any spike curriculum using
harness-injected (masked) content should have its eval mirror the SAME teacher-forced shape the training
data uses up to the graded portion, not attempt "prove the model can also generate the ungraded part."

Full regression suite green on both CPU-only (tok_cpu, 119 cases) and CUDA (gsm8k, 158 cases) build configs
after the fix. Committed 09fa775.
