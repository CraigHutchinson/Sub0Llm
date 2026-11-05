---
name: tinystories-d448-plateau-result
description: d448/L11/H7 TinyStories GPU training plateaued naturally 2026-07-08 at step 64525 (2.50 epochs), best val_nelbo 1.2388 -- but that best checkpoint got pruned by the pre-fix binary before the plateau detector caught it; final model.bin is a real, coherent story generator. The <|endoftext|>-chaining seen in its report.txt was a preview/report generation bug, FIXED (a016d17) -- see [[eos-token-and-fixed-marker-scheme]]
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Result:** `models/sub0llm_tinystories_d448l11h7sq256v16517r_4dba86b` (GPU, batch 96, the CPU-width
fallback -- see [[gpu-batch-undertuned-vram-headroom]]) plateaued naturally at step 64525/774240
(2.50 epochs, well under the 30-epoch backstop) via the least-squares trend-plateau detector. Final
`model.bin` reflects step 64525 (val_nelbo 1.2458). The TRUE best eval was actually earlier -- val_nelbo
**1.2388** at step 56782 (2.20 epochs) -- and three subsequent evals (1.2537, 1.2561, 1.2458) never beat
it, which is exactly what finally tripped the plateau stop.

**Real, concrete case of the exact bug [[checkpoint-tools-and-best-retention]] fixed, but too late for
THIS run:** the step-56782 checkpoint (the actual best) was pruned by `prune_ckpts`' old keep=3 rotation
before training stopped, because this run used the pre-fix binary throughout (started before the fix
landed, never rebuilt mid-run since a live GPU training process can't be touched). So the best-ever
weights (1.2388) are gone; only the final, slightly-worse ones (1.2458, model.bin) and the newest-3
checkpoints (steps 59363/61944/64525) remain. The gap is small (~0.6% relative) but this is a genuine,
observed instance of the failure mode, not a hypothetical -- worth remembering if anyone asks "why isn't
the model at its true best." A future run (rebuilt binary) won't lose this.

**Sample quality at plateau (full SEQ_LEN=256 continuation from "the "):** genuinely coherent, on-topic,
thematically consistent short story (cat and mouse sharing cheese, learning to be friends) -- a real
qualitative improvement over the much-less-trained 56-step CPU smoke model tested earlier this session.
Recurs the SAME `<|endoftext|>`-overuse pattern flagged in `data/tinystories_findings.txt`'s notes on
the OLDER d192/L6/H6 reference run -- the model chains many short `<|endoftext|>`-separated morals/
codas after the main story ("...and the mouse learned that sharing is caring. <|endoftext|> and the
mouse lived happily ever after..." etc.) rather than stopping cleanly. Same pattern at two very
different model sizes/vocabs suggests this is a TinyStories-corpus habit (the source data itself likely
contains many near-duplicate moral-of-the-story codas) rather than an architecture-specific quirk --
still an open, un-investigated question, not yet root-caused.

**User's hypothesis (2026-07-08), CONFIRMED CORRECT and FIXED (`a016d17`), after CUDA parity work
landed per explicit instruction ("save the endoftext for after cuda work"):** the pattern wasn't a
quality bug in the model's learned distribution -- it was a GENERATION-side bug. `<|endoftext|>` was
the corpus's hard document-boundary marker throughout training (every training window that crosses a
document boundary treats it as a terminal token, per [[eos-token-and-fixed-marker-scheme]]) -- the
model never saw a real, coherent continuation AFTER it in training data, so sampling past it draws from
an out-of-training-distribution context, degenerating into the short, repetitive, restarting fragments
seen in the report.txt example the user quoted.

**Root cause, once actually investigated: `gen_stage.cpp` already stopped at first EOS** (landed in the
original [[eos-token-and-fixed-marker-scheme]] work, 2026-07-02) -- `sub0llm gen` was NEVER affected.
The bug was narrower than assumed: `preview_at()` (training previews + `report.txt`'s sample battery)
and `gen_self_stats()` (autotemp's coherence tuner) both independently reimplement the same
forward-loop-plus-sample_token pattern instead of calling through `gen_stage.cpp`, and neither one had
picked up the `if (next == eos_id) break;` guard when it was added. Fixed by adding the identical guard
to both, in `src/train_stage.cpp` (commit `a016d17`).

**Live-verified, not just inspected:** rebuilt the `ts_cuda` build directory (exact config match to
this model: d448/L11/H7/seq256/vocab16517, gated=0/tied=0/qknorm=0) and regenerated this model's
`report.txt` -- all 6 sample cells (2 prompts x 3 temps) now stop cleanly at the document boundary,
zero `<|endoftext|>` text anywhere. Cross-checked against a direct `sub0llm-gen` call with the identical
prompt/temp/seed to confirm both paths now diverge from the buggy output at exactly the same point
(the real EOS draw). Also ran `autotemp` end-to-end on the same model as a `gen_self_stats` smoke
test -- completed cleanly, same qualitative "underfit, cap at 1.0" diagnosis as before, no crash/
regression. See [[fable-subagent-for-big-risky-tasks]]-adjacent lesson: a background-task redirect to
`/tmp_report_out.txt` (root-relative, no permission) silently no-op'd on the first verification attempt
-- the report.txt read afterward was stale (11h old) and momentarily looked like the fix had failed;
caught by checking the file's own mtime before trusting its content.

**Closes out the standing "wait for d448 to plateau" task** from earlier this session (user: "wait
until the 448 train reaches natural plateau... or check again in 3 hours 19 minutes"). The full
generation-battery `report` (2 prompts x 3 temps, see [[checkpoint-tools-and-best-retention]]) was run
against the final plateaued model for a fuller, saved comparison point.
