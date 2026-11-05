---
name: op-marker-dedicated-token-backlog
description: "Backlog decision - mint dedicated TOK_OP_START/END (retire the literal \"op\" word discriminator), grounded in Qwen/Gemma4 tool-calling prior art; blocked on zero free reserved-id headroom"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Op-calls today are `TOK_TURN_START op <name> <operands> TOK_TURN_END` (reusing the chat-turn markers +
literal word "op" as the tool-call discriminator, per [[deterministic-mechanisms-thesis]] §1). User pointed
out this recurring per-call text cost ("op " = ~2-3 tokens on the byte-level Unigram tokenizer, since "op"
is unlikely to be a learned merge) sounds like it should instead be a dedicated `TOK_OP_START`/`TOK_OP_END`
marker pair, and asked for a prior-art check against Qwen/Gemma tool-calling.

**Prior art confirmed the instinct (checked 2026-07-15):** Qwen (Hermes-style, Qwen2.5/Qwen3) reuses
`<|im_start|>`/`<|im_end|>` for the outer turn but wraps the call itself in its OWN dedicated added tokens,
`<tool_call>...</tool_call>` (result: `<tool_response>...</tool_response>`). Gemma 4 / FunctionGemma goes
further — `<|tool_call>`/`<tool_call|>` and `<|tool_response>`/`<tool_response|>` are dedicated special
tokens explicitly distinct from `<start_of_turn>`/`<end_of_turn>`, not nested in them. Neither production
scheme reuses the generic turn pair + a magic word the way Sub0Llm's op frame does today.

**Why not implemented yet:** zero free reserved-id headroom. All 6 `TOK_RESERVED_4..9` ids (casing.hpp) are
already claimed by the scratch-slot pool (`scratch_slots.hpp`: `SCRATCH_SLOT_COUNT = TOK_MARKER_COUNT -
TOK_RESERVED_4` = 6, no slack). Minting `TOK_OP_START`/`TOK_OP_END` means either shrinking the scratch pool
6→4 or bumping `TOK_MARKER_COUNT` 32→34 — either way a tokenizer-FORMAT bump (n_base/VOCAB shift, every
learned piece id shifts) that invalidates every currently-trained model (fineweb d448, the new `--gsm8k`
model, …).

**Decision (user, 2026-07-15): record as backlog only, don't implement now.** Given the choice between
implementing immediately (steal 2 ids from the scratch pool, or grow total marker count) vs. deferring, user
picked deferring. Do this as ONE deliberate, batched format bump when the scratch-pool-growth backlog item
(see [[deterministic-mechanisms-thesis]] §C "named regions vs reserved ids") is also actually acted on, so
the retrain cost is paid once for both instead of twice. Documented in full in
docs/DETERMINISTIC_MECHANISMS.md (§1, right after the "op is load-bearing" note).

**How to apply:** don't propose implementing this marker change in isolation — check whether the scratch-pool
growth work is also queued first, and batch them. If a future format bump is already happening for any other
reason, this is the moment to fold this change in too.
