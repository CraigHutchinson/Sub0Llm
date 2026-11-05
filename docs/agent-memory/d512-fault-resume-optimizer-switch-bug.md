---
name: d512-fault-resume-optimizer-switch-bug
description: RESOLVED end to end -- GPU fault silently switched Muon->AdamW on resume (confirmed via log), manually re-resumed correctly, training ran clean to a natural plateau at step 16100 (best val_nelbo 1.1653, beats d448); the underlying code bug is ALSO now fixed (see [[central-run-config-json]]) via a general RunConfig/config.json mechanism, live-verified to catch this exact scenario
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**What happened:** `sub0llm_tinystories_d512l12h8sq256v16517rgwq_20260709-174021` hit a real GPU device
fault at step 7831 (`[error] training: GPU step failed (device fault) at step 7831`). The
failure-detection hardening from earlier this session worked exactly as designed: stopped
immediately with no corrupted save, left the last periodic checkpoint (step 7728, best val_nelbo
1.18645) untouched and resumable, and a bare `sub0llm train` auto-detected the non-terminal
`status=training` and resumed. Final state: `status=plateaued`, `steps=8372` (only 644 steps /
~1 eval-tick past the resume point), `best_val_nelbo=1.18645` (the pre-fault value — the post-resume
steps never beat it).

**Real result, not a failure**: 1.18645 beats d448's sweep val_nelbo (1.19023) despite far fewer
epochs (1.20 vs 1.90) — the size-scaling trend from [[capability-roadmap-discrete-experts]]'s
underlying sweep is holding. The "plateaued early" symptom is a *training-dynamics* anomaly on top of
an otherwise-good result, not evidence the d512 point itself is bad.

**Likely root cause, found by reading `train_stage.cpp` (not yet 100% confirmed — the exact resume
CLI invocation wasn't captured before a context compaction):** `meta.txt`'s final `optimizer=0`
(AdamW) directly contradicts the pre-fault log, which clearly shows Muon running
("optimizer: Muon (hidden 2D weight matrices) + AdamW (embeddings, head, norms, biases)"). Traced
why: `sub0::AdamW opt(lr, optimizer == 1)` (`train_stage.cpp:1164`) derives the optimizer choice
purely from *this specific invocation's* `--optimizer` CLI flag — there is no code path that reads
the resumed model's own recorded `meta.txt.optimizer` and re-applies it. `m.optimizer = (optimizer
== 1) ? 1 : 0;` (`train_stage.cpp:1118`) then unconditionally overwrites the field with whatever the
latest invocation used, so the provenance of "this model was actually trained with Muon for its
first 7728 steps" is lost, not just unused. The auto-resume feature's own log message
("a bare `sub0llm train` will detect it as unfinished and offer to resume it") explicitly invites a
no-args resume — which, if that's genuinely what happened here, would silently default `optimizer`
to AdamW for the continuation. Circumstantial but strong: val_nelbo regressed immediately at the
exact resume point (1.1864 -> 1.2537 one eval-tick later) and the plateau detector gave up shortly
after — consistent with a live optimizer-identity switch destabilizing training (Muon's
orthogonalized-momentum state and AdamW's exponential-moving-average state are not interchangeable
representations for the same matrix parameters).

**Two backlog items, not fixed yet (separate module from the concurrent tokenizer redesign work,
deliberately not touched mid-way through that):**
1. **User's own ask, from watching this happen live**: "Trains just stops there and doesn't
   idle+retry. Could be worth adding some resilience via a retry." Currently a device fault halts
   correctly and safely (per [[gpu-failure-detection-hardening]]) but never retries — worth a
   real design pass on what "retry" should mean here (a fresh CUDA re-init and continue? just the
   next periodic auto-resume, which already exists via the model-directory mechanism? some bounded
   retry count before giving up for good?), not a small tweak.
2. **Auto-resume should preserve/verify the optimizer choice**, not silently default it. Read the
   resumed model's own `meta.txt.optimizer` and either apply it automatically (no `--optimizer` flag
   needed to correctly continue) or fail loudly on an explicit `--optimizer` flag that conflicts with
   the recorded history, rather than the current silent-overwrite behavior. This is the more
   likely-load-bearing fix of the two — it's what plausibly caused the actual data-quality problem
   observed here, not just an ergonomics gap.

**User also reduced GPU max frequency** as their own independent mitigation attempt for the
underlying device-fault frequency (separate from software fixes above) — noted for context, not
something to act on in software.

**Root cause CONFIRMED on 2026-07-10, not just circumstantial anymore.** Read `train.log` directly:
the post-fault resume block prints the schedule/lr lines but is missing the
`optimizer: Muon (hidden 2D weight matrices) + AdamW (...)` banner line entirely (that line only
prints when `optimizer==1`) — direct proof the resumed invocation ran with plain AdamW, matching
`meta.txt.optimizer=0`. Training loss jumped 1.0863 -> 1.2972 in the very first post-resume tick and
val_nelbo regressed 1.1864 -> 1.2537, triggering the early plateau-stop.

**Recovery action taken (2026-07-10):** all 4 checkpoints were still on disk (prune_ckpts' always-keep-
best fix from [[checkpoint-tools-and-best-retention]] meant the good 7728 checkpoint was never rotated
out despite 8372 becoming a later, worse step). Backed up the regressed `model.bin` (renamed suffix
`.bak-before-muon-resume`) and quarantined the bad-resume checkpoint (`model.bin.step000008372.ckpt`
-> same name + `.bad-adamw-resume` suffix, so `list_step_ckpts`'s suffix match no longer sees it),
which made step 7728 the new "latest" checkpoint. Re-invoked
`sub0llm-train.exe <model-dir> --optimizer muon` directly against the already-built `out/build/ts_cuda`
binary (unchanged since the original run, Jul 9 18:39 — no rebuild needed/risked). Confirmed correct on
restart: "RESUMING from step 7728 (best val_nelbo 1.1864)" + the Muon banner both printed. Training is
running again as of this session.

**Training finished cleanly (2026-07-10).** After the manual muon-explicit resume, training ran the
rest of the way with no further faults and plateaued NATURALLY at step 16100 (2.50 epochs), best
val_nelbo **1.1653** — better than both the pre-fault checkpoint (1.1864) and obviously the
bad-resume regression (1.2537). Sample output at plateau was genuinely coherent multi-sentence
narrative. Process exited 0 (not killed/crashed). This is now a closed, good sweep data point.

**Backlog item #2 (auto-resume should read back and apply the recorded optimizer choice, not trust
the invocation's CLI flag) is ALSO now fixed in code**, same session, via a general mechanism the
user asked for explicitly ("so we don't make a similar optimizer mistake again"): a central
`sub0::registry::RunConfig` struct + `config.json` per model directory (X-macro field list, so a
NEW training-recipe option is added in ONE place and automatically gets read/write coverage — see
[[central-run-config-json]] for the full design/implementation). `train_stage.cpp`'s resume path now
reads back `config.json`'s `optimizer` and overrides whatever the CLI/default says, logging the
override, exactly mirroring the existing batch/lr/seed-from-checkpoint precedent. **Live-verified**:
trained a tiny model with `--optimizer muon`, resumed it with NO `--optimizer` flag (reproducing
this exact incident's triggering condition), and confirmed the log shows
`config.json overrides this invocation's optimizer: adamw -> muon` followed by the Muon banner —
the bug class cannot recur unattended anymore. Backlog item #1 (GPU-fault auto-retry resilience)
remains open, unrelated to this fix.

See [[gpu-failure-detection-hardening]] for the fault-handling half of this story (which worked
correctly), [[model-identity-date-tag-and-comparison-table]] for the auto-resume mechanism this bug
lived in, and [[central-run-config-json]] for the fix.
