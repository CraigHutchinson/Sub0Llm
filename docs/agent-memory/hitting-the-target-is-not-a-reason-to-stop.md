---
name: hitting-the-target-is-not-a-reason-to-stop
description: "Reaching a performance target does not end the work; keep pushing while untested optimizations remain, and iterate on them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-22T19:11:52.914Z
---

When a performance goal is met, do NOT treat it as the stopping point. If there are optimizations still
on the table — untested levers, parked mechanisms, ideas a profile suggests — keep going: test them,
refine them iteratively, and spend the extra effort. The user said this explicitly (2026-09-22) while
the Qwen4 decode work was still short of its goal, as a standing instruction for after it is reached:
"even if we reached our target ... we want the most, so accept spending a little more effort to push
performance further when there are still optimizations on the table to be tested and refined
iteratively."

**Why:** a target is a floor the user set to make progress checkable, not a ceiling on what the hardware
can do. The real objective is [[optimization-target-is-hardware-ceiling]] — the machine's actual limit.
Stopping at the number leaves measured, known headroom unclaimed.

**How to apply:**
- After a gate passes, report it, then say what is still on the table and keep working, rather than
  asking whether to continue.
- Feed the next iteration from a fresh profile, not from the plan that got you here — the constraint
  usually moved (AGENTS.md §13's three-pass rule, and re-profile after any change over ~20%).
- A lever that measured flat or negative is a candidate to REFINE, not a closed question: three passes
  before judging, park behind a toggle, never revert ([[qwen4-decode-perf-b24-b35]] has several).
- Applies to the repo's own optimization backlog: an "open follow-up" row is work to pick up once the
  current thread lands, not a wishlist.
