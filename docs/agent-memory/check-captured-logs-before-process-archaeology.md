---
name: check-captured-logs-before-process-archaeology
description: When a background build/train task looks stalled, read its own captured output log FIRST -- it usually already streams progress -- before diagnosing via process trees/CIM queries
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**What happened (2026-07-03):** rebuilding `sub0llm-train` unexpectedly triggered a full re-tokenize of the
46GB fineweb_edu corpus (an unavoidable Ninja dependency-graph consequence of rebuilding `sub0llm-configure.exe`
after a `tools/configurator.cpp` change -- see [[cuda-wave6-orchestration-memory-findings]]'s Wave 7 entry).
With no new lines appearing in my last-read snapshot of the build log, I jumped straight to process-tree
diagnosis (`tasklist`, `Get-CimInstance` for child processes, thread/handle counts) to rule out a hang, on the
user's prompt "check no stray processes - builds are normally fast."

That investigation WAS the right call once done -- it correctly found a real cause (not a stray/duplicate
process, just the corpus re-tokenize) -- but it was the wrong FIRST step. The configurator was already
streaming live, itemized progress the whole time (`tokenizing: 0.8 / 46.1 GB (2%) 90 MB/s ...`) directly into
the SAME captured output file I'd stopped re-reading. A fresh `Read` on that file would have shown the
answer in one call, before any process-tree work was needed.

**How to apply:** when a background task (build, tokenize, train, any tool with its own progress reporting)
looks stalled, re-read its OWN captured output file again FIRST -- most long-running tools in this project
(configurator, train, tune) already print incremental progress lines. Only escalate to `tasklist`/
`Get-CimInstance` process-tree investigation if the captured log is GENUINELY silent for a while at a point
where output would be expected (not just "I haven't checked recently"). Don't skip the cheap check to jump to
the expensive one.
