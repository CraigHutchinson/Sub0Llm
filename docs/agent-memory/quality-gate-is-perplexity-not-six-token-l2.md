---
name: quality-gate-is-perplexity-not-six-token-l2
description: Judge decode-path precision changes by paired perplexity (G-PPL, --stage ppl), not the 6-token forward/forward_one L2/argmax
metadata:
  type: feedback
---

For any change to decode-path precision (quantized kernel, activation format, native weight path), the
deciding quality gate is **G-PPL**: `python scripts/run_perf_suite.py --stage ppl --arm base:... --arm new:...`.
It teacher-forces `forward_one` over the pinned `ppl_blend_v1` fixture (~2,400 tokens, built by
`scripts/make_ppl_fixture.py`, SHA-256-pinned, not committed) and compares arms per token: PASS if the
95% CI upper bound of mean dNLL <= +0.03 nats/token. The user asked for this harness to be codified for
future testing and quality gates (2026-09-26).

**Why:** the 6-token L2/argmax (G-QUALITY) flagged the native backbone at argmax 3/6 / L2 0.2926 and
nearly kept it opt-in for "quality"; over 2,418 tokens it was quality-neutral (ppl 14.85 vs 15.03, dNLL
-0.012, CI -0.035..+0.011). Six tokens cannot decide quality.

**How to apply:**
- Run G-PPL before calling any precision change shippable; G-QUALITY is only a smoke check now.
- Paired comparison cancels what both arms omit (the n-gram table), so it works before PLE exists.
- It is deterministic: one run per arm, contention does not change the score (only the tok/s it reports).
- Long-run decode tok/s from --ppl runs (4.0 bf16 / 5.5 native) is lower than the 6-token A/B's because
  attention grows with position; use for arm comparison, not as the headline.
- Related: [[verify-correctness-against-reference-before-perf]], [[three-pillar-shootout-policy]].
