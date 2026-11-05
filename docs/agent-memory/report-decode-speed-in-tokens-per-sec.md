---
name: report-decode-speed-in-tokens-per-sec
description: "Report Sub0Llm decode/generation speed in tokens/sec (tok/s), not s/token, from 2026-09-25 on"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-25T08:46:14.742Z
---

Report decode and generation performance in **tokens per second (tok/s)** — in chat, docs, tables,
A/B results and perf_history rows. The user asked for this on 2026-09-25 ("we are now at the
performance to shift to using tokens/sec"), once decode crossed ~5 tok/s on the real Qwen4 48-layer
model (native backbone ~5.5 tok/s, default ~4.6 tok/s).

**Why:** s/token made sense when a token took seconds (the thread started at ~9 s/token). At
sub-0.2 s/token, tok/s is the natural unit, matches how llama.cpp (the oracle) reports generation
speed, and makes speedups read the right way round (bigger is better).

**How to apply:**
- Lead with tok/s; `1 / (s/token)`. Speedups as tok/s ratios or +% tok/s.
- Per-phase costs stay in ms/token — they are costs, and they add up to the token time.
- The measurement tools (`sub0llm-qwen4-forward`, `run_perf_suite.py`, `perf_history.jsonl`) still
  emit s/token; convert when reporting. When one of them is touched anyway, add a tok/s field
  alongside rather than replacing the old one (history rows must stay comparable).
- Older docs and memory notes quote s/token; convert when citing them, don't rewrite history.
