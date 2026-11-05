---
name: optimization-target-is-hardware-ceiling
description: "Perf goal is the hardware's theoretical ceiling, not just beating llama.cpp"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 26b08af2-2bc6-47fd-985b-b0a3d9b828eb
---

llama.cpp is the CORRECTNESS oracle (byte-exact greedy parity gate) and a comparative perf yardstick — but it is NOT the optimization target. The goal is the MAXIMUM the system/hardware is physically capable of: push past llama and toward the theoretical limits (DRAM bandwidth ceiling for decode, VNNI/FMA execution-port peak for prefill, etc.).

**Why:** beating llama is necessary but not sufficient — we own the whole stack and want to know (and reach) what the silicon can actually do.

**How to apply:** when optimizing, always state where we are vs the theoretical ceiling (e.g. "95 GFLOP/s = 30% of 1-vpdpbusd/cyc peak", "94 of ~108 GB/s exercised DRAM"), not just vs llama. Actively hunt for NOVEL ways to collapse or skip work — algorithmic shortcuts, fusing passes, eliminating redundant compute, representation tricks — that make things faster than a straight implementation would be, subject to the greedy-parity gate. See [[ch27-optimization-shortlist]], [[verify-correctness-against-reference-before-perf]].
