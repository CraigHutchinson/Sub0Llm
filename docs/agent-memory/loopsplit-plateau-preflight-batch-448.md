---
name: loopsplit-plateau-preflight-batch-448
description: "PRE-FLIGHT before resuming any LoopSplit plateau arm — every arm must be batch 448; arm C has TWO models (_s1 is the confounded 512, _s1b is the clean 448) and arm E must be forced off its baked 512 default"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T11:07:42.979Z
---

Checked 2026-07-30 while prepping the plateau runs. The baked `DEFAULT_GPU_BATCH` **differs per arm build
dir**, so trusting the default silently reintroduces the batch confound that
[[loopsplit-3arm-batch-confound]] warns about.

| arm | build dir's DEFAULT_GPU_BATCH | trained batch (models/*/config.json) |
|---|---|---|
| A `ls_A_deep16_s1` | 448 | 448 |
| B `ls_B_loop10x6_s1` | 448 | 448 |
| C `ls_C_shallow10_s1` | **512** | **512 — CONFOUNDED, do not use** |
| C `ls_C_shallow10_s1b` | 512 | **448 — this is the clean one** |
| D `ls_D_depth4_s1` | 448 | 448 |
| E (not yet created) | **512** | must be forced to 448 |

**Two traps:**

1. **Arm C has two model dirs differing by one character.** `ls_C_shallow10_s1` (batch 512) and
   `ls_C_shallow10_s1b` (batch 448). The clean 3-arm result in
   [[loopsplit-3arm-clean-result-looping-works]] came from **`_s1b`**. Resuming `_s1` continues the
   confounded run. Resume is also self-perpetuating here: `load_checkpoint` restores batch from the
   checkpoint and **overrides `--batch`**, logging "checkpoint overrides this invocation's settings" — so
   passing `--batch 448` does NOT rescue `_s1`.

2. **Arm E must be created with an explicit `--batch 448`.** Its build dir bakes 512, and at 512 memplan
   predicts 7933 MiB → ~8213 with the measured ~280 MiB overhead, against ~7891 usable on the 8151 MiB
   card. That is ~322 MiB over: it would have spilled (or now trips the no-clamp guard). At 448 it lands
   ~7236 MiB, comfortable.

Generated footprints at each arm's own baked batch (`sub0llm memplan`, which now itemizes):

```
  B  batch 448   7177 MiB  (+280 = 7457, ~434 spare)
  C  batch 512   7021 MiB  (+280 = 7301, ~590 spare)   <-- but 512 is the WRONG batch for the comparison
  E  batch 512   7933 MiB  (+280 = 8213, ~322 OVER)    <-- forces the 448 above
```

All four arm dirs (B, C, E rebuilt 2026-07-30; A left alone because it is mid-run) carry the fixes from
this session, including the guard that now STOPS a run whose batch will not fit rather than silently
shrinking it — see [[memplan-overhead-280-not-1650-and-host-is-bigger]].

Arm D additionally needs reconfiguring with `-DSUB0_LOGITS_MAX_CHUNKS=32` before its plateau run; at the
default cap of 8 its logits term is 1806 MiB and it does not fit.

Related: [[depth-attention-arm-d-running]], [[three-pillar-shootout-policy]],
[[thermal-confounds-ab-wallclock-testing]]
