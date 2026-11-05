---
name: verify-correctness-against-reference-before-perf
description: "For Gemma/optimization work, verify output correctness against a reference oracle (llama.cpp) before claiming any speedup"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fb632281-974b-437a-89ff-47e56b5ae70c
---

When optimizing inference (Ch27 specialized build, Gemma support), correctness is
a GATE on performance — "performant but incorrect is not good."

**Why:** a specialized/monomorphized forward is easy to make fast and subtly wrong
(weight layout, RoPE convention, QK-norm, GeGLU vs SwiGLU, (1+w) norm, embed scale,
local/global window). A speedup over llama.cpp is meaningless if the logits differ.

**How to apply:** treat llama.cpp as the correctness ORACLE, not just the perf
baseline. Before reporting any tok/s or latency win for the specialized path, prove
parity vs llama.cpp on the SAME model + prompt:
- tokenizer parity (our BPE token ids == llama's),
- next-token logit parity (max-abs-diff / cosine within tolerance) — strictest,
- greedy continuation parity (identical argmax at temp 0),
- perplexity within tolerance (llama-perplexity as reference).
Build a `verify` mode into the runner and/or tests; gate perf claims on it passing.
Related: [[ch27-specialized-build-plan]], [[resolve-preexisting-issues-blocking-our-work]].
