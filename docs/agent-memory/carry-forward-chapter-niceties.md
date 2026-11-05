---
name: carry-forward-chapter-niceties
description: "Each new chapter must carry forward prior chapters' instrumentation and adaptivity — perf metrics, accuracy axes, self-tuning step counts"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

When building a new chapter, inherit the "niceties" established in earlier ones rather than
regressing to bare loops. Confirmed expectations (from Ch28→Ch29 handoff, 2026-06-12):

**Why:** the user explicitly flagged Ch29's first draft for losing Ch28 features — fixed step
count instead of an adaptive self-tuning mechanism, and thinner metrics.

**How to apply — checklist for every new training chapter:**
- Performance metrics in the report line: steps/s AND masked-tok/s (or tok/s), wall-clock total.
- Accuracy axes at the end: recall-vs-noise sweep, per-position (edge) profile, held-out scalar
  (NELBO/eval loss) — the "final result" must be measured on multiple axes, not one number.
- NO fixed step counts: training self-terminates (Ch28: token-gated ceiling mastery; Ch29:
  held-out-NELBO early stopping with patience). --steps is only a safety bound.
- `sub0llm::init_cpu_compute()` first line of main ([[denormal-stalls-in-training-need-ftz]]).
- Allocation-free hot loops (corrupt_into, hoisted contexts); top-level try/catch in main so
  Windows fail-fast doesn't eat the error message.
- Gotcha: TextCorpus::next_sample REASSIGNS its output tensors — never cache their data spans.
