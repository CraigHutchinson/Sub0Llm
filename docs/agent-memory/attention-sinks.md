---
name: attention-sinks
description: StreamingLLM-style --attn-sinks gen flag (2026-07-04), scoped down to the existing past-SEQ_LEN fallback path only; item 5 of the architecture backlog
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Implemented `--attn-sinks N` (item 5 of [[architecture-ideas-backlog]]) as a `gen`-time RUNTIME flag
(like `--temp`/`--topk`/`--seed`), NOT a configure-time constexpr like the other 4 features this
session -- correctly so: this is a decoding POLICY choice (how to handle context overflow), not a
model-shape/weight-meaning decision, so it doesn't belong under the "bake known-upfront decisions as
constexpr" preference (see [[compile-time-decisions-preferred]]) -- that preference is about the
model's own architecture, not per-invocation generation behavior.

**Deliberately scoped DOWN from the original backlog idea.** The original framing (a ring-buffer
KV-cache with RoPE position rebasing, retrofit into `forward_one`/`KVCache`) would have touched a
correctness-critical, heavily-validated hot path shared with training/eval, on both CPU and CUDA
backends. Found a much smaller, equally-correct entry point instead: `gen_stage.cpp` ALREADY HAD a
"past SEQ_LEN" fallback (`sub0::forward()` full-recompute per step, CPU-only, triggered whenever
`ctx.size() + n > SEQ_LEN`) that just did a plain last-SEQ_LEN sliding window with no sink retention.
Since that fallback already recomputes the ENTIRE window from scratch every step (no incremental
cache to manage), adding sinks there is just changing which tokens go into the window: keep the first
`n_sink` tokens of `ctx` + the most recent `SEQ_LEN - n_sink`, instead of a plain last-SEQ_LEN slice.

**Why no RoPE changes were needed:** `op_rope`/`forward()` assign position = ROW INDEX within the
current forward() call, not the token's true absolute generation step. RoPE only encodes RELATIVE
position for the attention score, so reassigning sinks to rows `[0,n_sink)` and the recent tail to
`[n_sink,SEQ_LEN)` lands everything in the SAME in-distribution position range the model was trained
on, with no artificial position gap between the two halves (recent tail starts exactly at `n_sink`,
contiguous with the sink block) -- zero changes needed to `op_rope`, `forward()`, or the KV-cache path
at all. This is the standard StreamingLLM "position remapping" trick, just free of charge here because
of how `forward()` already assigns positions.

`n_sink` is clamped to `SEQ_LEN/2` so an unreasonable `--attn-sinks` value can't eat the whole window.
The CPU/GPU incremental KV-cache fast paths (`forward_one`/`sub0_cuda_forward_one`, used whenever
`ctx.size()+n <= SEQ_LEN`) are completely untouched -- sinks only affect the slow fallback that already
existed for the overflow case, so this doesn't change performance characteristics, only quality once
that fallback triggers.

**Verification:** the untouched-code-path branch (`ctx.size() <= SEQ_LEN`, i.e. `n_sink==0` or no
overflow) is the exact same expression as before, just moved under a conditional -- zero regression
risk there, confirmed by frontend/engine suites staying at their baseline assertion counts (gen_stage.cpp
isn't linked into the test suite; verified by full rebuild across all targets instead, all clean).
The new windowing logic itself is pure index arithmetic (no floats, no autograd), traced by hand for
correctness (including the "boundary" case where ctx.size() is just barely past SEQ_LEN) rather than
needing a gradient-check-style numerical proof -- judged proportionate given the lack of numerical
subtlety, unlike the session's other additions (SwiGLU/Muon/tied-embeddings/QK-norm) which all touch
autograd math and got the full finite-difference treatment. Also ran a real generation past SEQ_LEN=32
with and without `--attn-sinks`, confirming the flag visibly changes output once triggered (proving the
branch fires) with no crash. Did NOT do a rigorous quality (perplexity/coherence) A/B -- the test model
was tiny/undertrained (word-salad either way at that scale), so no meaningful signal would come from
eyeballing it; a real quality claim needs a properly-trained model, out of scope for this pass.
