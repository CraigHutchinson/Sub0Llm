---
name: boy-scout-rule-always-resolve-inconsistencies
description: Standing instruction (2026-07-07) -- always resolve inconsistencies when found, keep a clean ship (boy scout rule), not just the ones blocking current work
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's explicit instruction: "Always resolve inconsistencies when they are found - keep a clean ship -
boy scout rules."

**Why:** broader than [[resolve-preexisting-issues-blocking-our-work]] (which is scoped to breakage
that blocks validating the CURRENT change) -- this is a general standing directive: when a stale
comment, a drifted constant, a dead code path, or a second independently-hardcoded copy of a value is
noticed DURING other work, fix it then, don't just flag it and move on.

**How to apply:** when incidentally reading code for one task and spotting an unrelated inconsistency
(a comment that no longer matches the code, two places that should share one value but don't, a
stale command/flag that no longer does anything), fix it in the same pass rather than leaving a note
for later -- but still keep it as a SEPARATE, clearly-described commit from the primary task's own
commit, so the history stays readable. Prefer fixing the ROOT CAUSE (share one definition) over
patching a second copy of the same number -- see the qk-norm/attention-tile-fix pattern of hoisting a
shared named constant to catch future drift, applied again for kTokensPerParam/aspect_for_params after
`preview()`'s temp-mismatch bug turned out to be exactly this class of issue (config_util.hpp,
2026-07-07). Still respect [[only-add-arguments-we-need]] -- "resolve inconsistencies" is not license
to add speculative new knobs, only to fix things that are actually wrong or drifted.
