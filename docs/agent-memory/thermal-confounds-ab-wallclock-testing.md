---
name: thermal-confounds-ab-wallclock-testing
description: "On thermally/power-constrained hardware (this laptop GPU), a single A-then-B wall-clock comparison is confounded by thermal history -- need multiple alternating trials, not one before/after pair"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

When A/B-testing wall-clock throughput on this machine's GPU (RTX 5070 Laptop, power/thermal
constrained -- see [[lm-head-tensor-core-fix-and-graceful-stop]]), **a single "run A, then run B"
comparison is not trustworthy on its own.** The GPU's boost clocks/power draw depend on recent thermal
history: whichever variant runs SECOND inherits whatever heat the FIRST variant built up, and
whichever runs FIRST gets any benefit of starting from a cooler baseline. A single before/after pair
cannot distinguish "B is genuinely faster" from "B just got the cooler slot."

**Why this came up:** testing the lm_head tensor-core fix (2026-07-02), the with-fix run (A) opened at
35 win/s then dropped to a 17-19 win/s steady state; the without-fix run (B), run immediately after,
opened at 32 win/s and was still at 31 win/s two heartbeats in -- clocks/temp/power were still visibly
different between the two measurement windows (87°C/78W for B's early readings vs the 75-82°C/53-61W
range A had settled to), meaning B's own steady state hadn't been established yet either. The user's
correction: retest both a couple of times each before trusting the comparison, and remember this for
future wall-clock testing generally.

**How to apply:** for any wall-clock (not FLOP-count or algorithmic) comparison on this hardware,
run at least 2 trials of EACH variant, ideally interleaved (A, B, A, B rather than A, A, B, B), and
only trust a difference that shows up CONSISTENTLY across trials, not a single pair. Let each trial run
long enough to reach its own steady state (watch `nvidia-smi --query-gpu=temperature.gpu,clocks.
current.sm,power.draw` settle, not just the app's own throughput number) before recording it as that
trial's result. A cooldown gap between trials (or at least noting the temperature at the START of each
trial) helps separate "this variant is faster" from "this variant got the cooler start."

**Validated the same session:** re-running A (with the lm_head fix) a second time, after B, showed
35-36 win/s steady -- nothing like trial 1's 17-19 win/s. Trial 1's low reading was the anomaly (thermal
carryover from a long stretch of nsys/ncu/rebuild churn immediately before it), not the fix. Without
the retest, the fix would have been wrongly reverted from production. See
[[lm-head-tensor-core-fix-and-graceful-stop]] for the full trial data.

**2026-09-29, the same confound on CPU decode, and now enforced in tooling:** `run_perf_suite.py` used
to rebuild each arm immediately before timing it, so every sample ran on a CPU a compile had just
heated (one reactive arm spanned 0.111-0.170 s/token). Fixed in `a8aa3ad`: arms are built once and
staged, rounds rotate the starting arm, a 20 s cooldown and a named-process check precede every sample.
Use the suite rather than a hand-rolled loop; a one-off sweep should import its helpers
(`decode_s_per_token`, `contention_check`, `contention_count`) instead of reimplementing them.

**2026-10-05, restated by the user for CPU benchmarks (Sub0ECS), as a standing rule:** "there may be
other processes on this machine, so ensure multiple samples are always taken and work interleaved to
also consider thermal drift or workload changes." It was live: a 3-round rotated capture had 3 of 8
samples start at 80-96% CPU load from other work. Always: several samples per variant (never one),
variants interleaved/rotated, a load check before each sample, and report the spread, not just a
median. In Sub0ECS use `bench/tools/rotate.py` (waits for a quiet machine, records load before/after,
drops busy samples when enough clean ones remain) rather than a hand-rolled loop.
