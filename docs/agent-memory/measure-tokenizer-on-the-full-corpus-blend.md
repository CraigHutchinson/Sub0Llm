---
name: measure-tokenizer-on-the-full-corpus-blend
description: "FEEDBACK 2026-07-30: tokenizer decisions must be measured on the WIDER corpus -- cosmopedia + minipile (code) + fineweb_edu -- never one corpus. A prose-only measurement understated a real win by 2 pp"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T20:09:48.385Z
---

User, on a tokenizer change decided from cosmopedia alone:

> "We should only make the tokenizer decisions based on the wider cosmopedia and minipile and code
> corpus being included"

**Why:** the production tokenizer is learned over a BLEND, and the sources are not interchangeable.
Prose, web text and code have different casing, symbol and identifier statistics, so a number from one
source is not a result — it is a sample from the wrong distribution. Two specific traps I walked into:

* I measured the truecasing policy on **cosmopedia prose only**, then "verified" it on **TinyStories**,
  which is synthetic and contains **no code at all**. The populations most likely to break the decision
  (`ALL_CAPS` constants, CamelCase identifiers) were absent from every number I had.
* The prose-only figure was not merely narrow, it was **wrong in magnitude**: cased vocabulary
  9.7% -> 1.6% on prose versus **12.1% -> 2.0% on the blend**. Code raises the cased share, so
  prose-only UNDERSTATED the win by ~2 pp. It could equally have overstated one.

**How to apply:**
1. Any tokenizer/vocab/casing/glue measurement runs over **cosmopedia + minipile + fineweb_edu**, with a
   **per-corpus breakdown AND a combined run**. The combined stream is what the vocabulary is fitted to;
   the per-corpus rows exist to show whether code diverges from prose.
2. Draw held-out text from **each source in proportion**, or the token-cost number is silently
   prose-only even when the learn set was blended.
3. **TinyStories does not count as verification** for anything casing-, symbol- or code-sensitive. It is
   synthetic, low-entropy, templated and code-free — fine as a fast smoke corpus, not as evidence.
4. State which corpora a number came from, in the commit and in the code comment, so a later reader can
   see the scope of the claim.

`tests/casing_policy_tests.cpp` is the worked example: `kSources` drives per-corpus + combined runs.

Related: [[truecasing-collapse-unconditional]] (the decision this corrected),
[[three-pillar-shootout-policy]] (never one number in isolation),
[[tinystories-default-analysis-corpus]] (why TinyStories is the default FAST corpus, and its limits).
