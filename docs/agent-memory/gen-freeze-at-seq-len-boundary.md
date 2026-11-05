---
name: gen-freeze-at-seq-len-boundary
description: "ROOT-CAUSED + FIXED 2026-07-29 — nothing hangs at --n >= SEQ_LEN; the run falls off the KV-cache decode path onto a full-forward-per-token loop (~100x slower). Also made every `report` sample glacial"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T17:31:17.067Z
---

Reported 2026-07-28 as "sub0llm-gen appears to freeze at `--n 512` while `--n 500` is fine" on a
seq512 model. **Not a hang. A performance cliff, and an entirely mundane one.**

## Root cause

Both `preview_at` (train_stage.cpp) and `gen_stage.cpp` guard the fast path with:

```cpp
if (static_cast<int>(ctx.size()) + n <= SEQ_LEN) { kv_decode_generate(...); return; }
```

KV-cache decode is O(T) per token. Miss that precondition and control falls into a
**full forward per token** sliding-window loop. With a 4-token prompt, `--n 500` fits (504 <= 512) and
returns in about a second; `--n 512` misses by exactly the prompt's length and grinds through 512 full
forwards. The boundary looked mysterious because it is the prompt length, not `n`, that decides it.

## The much bigger blast radius: every `report` sample

`report`'s sample battery called `preview_at(prompt, SEQ_LEN, ...)`, which **can never** satisfy
`ctx.size() + n <= SEQ_LEN` with a non-empty prompt. So every sample of every report had always taken
the slow path. Measured on ls_A_deep16_s1 (d192 L16): a full report went from **>25 minutes (killed,
never completed)** to **5 seconds**. This is most of why `report` had a reputation for being unusably
slow -- more than the CPU-bound NELBO eval it was blamed on.

## Fixes (different, deliberately)

- **`preview_at` CLAMPS**: `n = min(n, SEQ_LEN - ctx.size())`. Its contract becomes "up to n more
  tokens, bounded by the trained window". Every caller only ever wanted a sample that fits, so this
  costs nothing and keeps them all on the fast path. The slow loop stays -- it is the only path under
  `USE_TERNARY`.
- **`sub0llm-gen` WARNS instead**: sliding past the trained window is a real capability (it is what
  `--attn-sinks` exists for), so clamping there would remove function. It now prints that it left the
  fast path, why, and what `--n` would stay on it.

## Lesson worth keeping

A silent fall-off-the-fast-path is indistinguishable from a hang from the outside, and the guard
condition mentioned a variable (`n`) that was only half the story. When a fast path has a
precondition, say so when it is missed -- see also [[check-captured-logs-before-process-archaeology]].
