---
name: python-not-committed-cpp-library
description: "Always prefer C++. Python is fine for QUICK LOCAL sideline experiments but must NOT be committed to the repo — the deliverable is a fast, coherent C++23/26 chapter library. Reimplement keeper-tools in C++."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User guidance (2026-06-20). The repo is a **fast, coherent C++23/26 (heading to C++26) library of chapters** for the LLM. **Always prefer C++.**

**Python:** allowed for **quick local sideline experiments / one-off analysis** only — but **do NOT commit Python to the repo**. (CLAUDE.md already says "No Python in the core library; Python only in tools/ for data prep/plotting" — this TIGHTENS it: keep even tools/ Python *uncommitted* / local. The earlier `tools/cli`, `tools/server` are C++; analysis scripts should be local scratch.)

**How to apply:**
- Run throwaway analysis in Python locally (e.g. under `/d/tmp/`), keep the **findings** (markdown/docs) in the repo, not the `.py`.
- If an analysis tool is worth keeping/repeating, **reimplement it in C++** as a proper `diffusion/tools/<name>` binary (like `sub0diff-recall-probe`).
- Caught me once: the M3 corr-decay probe `mi_decay.py` was committed then removed; its findings live in the ch32 chapter. **Backlog:** C++ port of the corr-decay (MI-vs-distance) probe if we want it permanent.

Related: [[tinystories-default-analysis-corpus]], [[cuda-first-class-iteration-time]].
