---
name: no-stale-numbers-in-tree
description: "User rule (Sub0ECS, 2026-10-06): do not keep superseded measurements in docs, comments or curated results; git history is the archive. Re-measure or remove."
metadata:
  type: feedback
---

On 2026-10-06, reviewing the Sub0ECS docs, the user said: "we don't need to persist old numbers where
source control keeps a history."

**Why:** the docs had accumulated figures from an earlier host and earlier code (labelled as historical,
but still there), plus "it used to read X" anecdotes. They made the docs longer and had already led to
one wrong cross-machine claim.

**How to apply:** a figure stays in the tree only while the current code produces it. When a
measurement is superseded, replace it everywhere it appears (docs, source comments, curated result
directories) or delete it; do not keep it under a "historical" banner. Prefer re-measuring to
deleting when the claim still matters. My interpretation: this is about our own measurements, not
figures quoted from published papers, and it was said about Sub0ECS; I have recorded it in that
repo's CLAUDE.md. See [[sub0ecs-v2-landing-state]], [[thermal-confounds-ab-wallclock-testing]].
