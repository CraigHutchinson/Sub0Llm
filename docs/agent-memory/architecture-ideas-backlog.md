---
name: architecture-ideas-backlog
description: 6-item architecture backlog from a "what's next in LLM research, and our own ideas" discussion (2026-07-04) -- ALL 6 DONE (corpus-driven sizing closed 2026-07-07); CUDA parity for the 4 constexpr-baked features deferred, see [[cuda-parity-backlog-deferred]]
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User asked (2026-07-04) whether SwiGLU (already added CPU-only for GGUF import, see
[[gguf-import-feasibility-review]]) could be a core benefit beyond just import compatibility, and to
survey other recent-research and original ideas. Response, ranked by expected value AT THIS PROJECT'S
SCALE (small from-scratch models, custom tokenizer, short-ish context) rather than frontier-lab scale
where a lot of "standard" tricks (GQA, MLA, sliding-window attention) mainly pay off on KV-cache
memory that barely matters here yet. User's own framing: "task up all of these, muon is notable too
so prioritise" -- Muon done first, see [[muon-optimizer]]; this file tracks the rest.

**1. Muon optimizer -- DONE (2026-07-04).** See [[muon-optimizer]] for the full writeup: hybrid
Muon(hidden 2D matrices)/AdamW(embeddings/head/norms/biases), `--optimizer muon`, CPU-only,
Newton-Schulz verified against the reference algorithm + a real training run, zero checkpoint-format
risk, zero regressions to the existing test suite.

**2. SwiGLU D_FF rebalance -- DONE (2026-07-04); flipping the DEFAULT to gated is a separate,
still-open decision.** Added `sub0::config::d_ff_for(d_model, gated_ffn)` (`config_util.hpp`, pure +
unit-tested): plain FFN keeps `4*D_MODEL` unchanged exactly; gated (SwiGLU) uses
`ceil_to_64((8/3)*D_MODEL)`, derived directly from THIS project's own params equation (plain =
`2*D*F`, gated = `3*D*F`, equal FFN param count when `F_gated = (2/3)*F_plain = (2/3)*4*D = (8/3)*D`)
-- not copied from the literature, though it happens to match the common "SwiGLU ~8/3" convention for
exactly the same reason (that convention was derived the same way against ITS OWN 4x baseline).
`tools/configurator.cpp` computes `d_ff` once (right after the `d_model % n_heads` validation) and
uses it consistently in BOTH the generated header (`D_FF = <value>`, was a hardcoded
`4 * D_MODEL` expression string) and the VRAM-clamp's `memplan::Dims` construction (was a second,
independently-hardcoded `4 * d_model` -- fixed the DRY violation, not just the new gated case).
Verified: a unit test checks the param-parity property directly (gated/plain FFN param-count ratio
stays within 0.85-1.15x across several `d_model` values, not just that the formula "looks right");
end-to-end at d192 (rebalanced `D_FF=512`, vs the pre-rebalance `768`) the full engine test suite
(3.66M assertions, including the finite-difference gradient check) and a real training run both passed
clean. Should this become the DEFAULT (`--gated-ffn` defaulting to 1) for FRESH from-scratch training,
vs. staying strictly opt-in -- the user's framing ("core benefit to our sub0llm class of models") leans
toward eventually flipping the default once validated, but that is a bigger, more disruptive decision
(changes what a plain `sub0llm-configure` with no flags produces) than the rebalance math, and should
be its own deliberate follow-up after a real quality comparison, not assumed from this change alone.

**3. Weight tying (tied input embedding / output head) -- DONE (2026-07-04), CPU-only.** See
[[weight-tying-tied-embeddings]] for the full writeup: `--tie-embeddings`, `Op::TiedHead`, `layout.hpp`
drops `LmHead`/`LmBias` slots when tied (checkpoint-format-safe via existing `param_floats`
discrimination, same reasoning as gated-FFN), verified via the finite-difference gradient check at real
dims plus a real end-to-end train/resume/gen smoke test. CUDA support NOT started -- see
[[cuda-parity-backlog-deferred]].

**4. QK-norm -- DONE (2026-07-04), CPU-only.** See [[qk-norm]] for the full writeup: `--qk-norm`,
`Op::QKNorm` (a separate op from `op_rmsnorm`, not a generalization -- per-head grouping needs a
group-aware reduction), applied right after the Q/K projection and before RoPE (Gemma2 convention).
Verified via the finite-difference gradient check at 2 scales (isolated, and combined with gated-FFN
as an interop check) plus a real end-to-end train/resume/gen smoke test. A real `__restrict`-aliasing
bug was caught and fixed before shipping (see the memory note). CUDA support NOT started -- flagged as
the SIMPLEST of the 4 CPU-only features to port (plain elementwise kernel, no GEMM complications) --
see [[cuda-parity-backlog-deferred]].

**5. Attention sinks (StreamingLLM-style) -- DONE (2026-07-04), scoped down from the original idea.**
See [[attention-sinks]] for the full writeup: `--attn-sinks N` as a `gen`-time runtime flag (not a
configure-time constexpr -- this is a decoding policy, not a model-shape decision). Rather than
retrofitting the incremental KV-cache (`forward_one`/`KVCache`) with a ring buffer + RoPE position
rebasing -- real surgery on a correctness-critical shared hot path -- found that `gen_stage.cpp`'s
EXISTING "past SEQ_LEN" full-recompute fallback already had no incremental cache to preserve, so
adding sink-token retention there was a small, contained, zero-risk change (just which tokens go into
the window). No RoPE changes needed: `forward()` assigns position by row index already, so sinks land
at rows `[0,n_sink)` and the recent tail at `[n_sink,SEQ_LEN)` naturally, in-distribution, contiguous.
The fast incremental KV-cache path is untouched.

**6. Corpus-driven sizing -- DONE (2026-07-07).** See [[autosize-research-grounded-formula]] for the
full writeup: `sub0::config::autosize()` replaced the old 5-bucket hand-tuned ladder with a smooth,
formula-based shape (Chinchilla-informed target-parameter budget, a scale-dependent width/depth aspect
ratio, Heaps'-law vocab growth), then a SECOND pass grounded its constants against real prior art
(TinyStories paper, SmolLM2, FineWeb-Edu's own ablation model) after the user asked "how do we know
d192 isn't more correct" -- plus a hardware-aware VRAM clamp and a `--size-scale` minimal/ideal lever.
