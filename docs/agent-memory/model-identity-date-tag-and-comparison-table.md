---
name: model-identity-date-tag-and-comparison-table
description: Model directory identity moved from git-sha to a date+time tag (2026-07-09, 3927bbf) with explicit auto-resume/--resume/--fresh; new evalcache.hpp + sub0llm models --metrics/--refresh gives a filterable, cached cross-model comparison table (dims/params/val_nelbo/bits-per-byte/autotemp temp/optimizer). SUB0_GIT_SHA capture-timing staleness deferred to GitHub issue #1
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Trigger:** wanted a comparison table across the multi-size Chinchilla sweep (see
[[checkpoint-tools-and-best-retention]]) instead of manually re-reading N `report.txt` files. Scoping
it surfaced a real bug: `registry::model_dir()` baked `SUB0_GIT_SHA` into the directory name, and
`SUB0_GIT_SHA` is captured once by CMake's own `execute_process` at *configure* time -- not refreshed
by `sub0llm-configure`'s normal reconfigure-tool-only workflow. Confirmed empirically: `ts_cuda` hadn't
been CMake-reconfigured since 2026-07-07, so all 5 sweep models this session trained were tagged with a
three-commits-stale SHA, identical to an unrelated older model -- sha-based "this exact matrix"
filtering silently didn't work.

**Decision: fix the directory-identity granularity now, defer the CMake-capture-timing bug itself.**
User's own framing: SHA is too fine a check anyway -- an unrelated commit (e.g. a log-message fix)
between stopping and resuming a multi-day run would silently start a fresh model instead of resuming,
when architecture/weight-format compatibility (`registry::compatible()`, already SHA-independent) is
the real gate. Filed **GitHub issue #1** for the narrower, deferred CMake-timing question (does the
fix belong in `sub0llm-configure` via a first-of-its-kind subprocess call, or stay CMake-owned with a
documented caveat) -- a real design decision worth its own session, not bundled into this pass.

**Part 1: date-tagged identity + explicit auto-resume (`3927bbf`).** `registry::model_dir()` takes a
`now_datetag()` string (`YYYYMMDD-HHMMSS`, Windows-filename-safe) instead of a sha. `sub0_train_stage`'s
no-explicit-path branch now: scans for the most recent model matching this build's corpus+architecture
(`registry::compatible()`, unchanged) -> non-terminal status (`"training"`, the only one) + matching
git_sha -> silent auto-resume (today's old behavior, now explicit) -> non-terminal + MISMATCHED sha ->
halt with a named-directory error requiring `--resume`/`--fresh` -> terminal status -> starts a fresh
dated dir by default (`--resume` forces continuing it anyway). `git_sha` stays in `meta.txt` as
provenance, no longer load-bearing for resume. **All 5 branches live-verified** (not just reasoned
about): killed a real in-progress training run mid-run, confirmed silent resume; hand-edited a test
model's `git_sha` to simulate a code-version change, confirmed the halt message fires with both SHAs
named; confirmed `--resume` overrides the halt, confirmed `--fresh` forces a new dir unconditionally;
confirmed a TERMINAL (plateaued) existing model correctly triggers "starting a new dated model" instead
of silently extending it.

**Migration (one-off script, scratchpad, not committed):** pruned 2 genuinely obsolete smoke-test
models (5-step `ckpt2model_smoketest`, 56-step/ppl~1436 d448 smoke run) after explicit user
confirmation (the destructive-action classifier correctly gated this even though the plan was
pre-approved, since the SPECIFIC targets were my own choice, not user-named -- asked, got "yes, delete
both", proceeded). Renamed the remaining 6 `sub0llm_tinystories_*` dirs to the date-tag scheme (keyed
off each one's own real `created` timestamp, not "now" -- preserves chronological order). Retroactively
corrected `optimizer` (new field, defaults wrong on old files) and `git_sha` (known with certainty --
`git log` confirmed no commits landed during the sweep's `created` window) on the 5 sweep models;
left the pre-session models' `git_sha` alone (no equivalent certainty available).

**Part 2: `evalcache.hpp` + `sub0llm models --metrics`/`--refresh` (`4bf29fc`).** New engine-free cache
module (mirrors `registry.hpp`'s pattern exactly) holds RAW measurements only (train/val NELBO,
autotemp's full temperature/repetition grid) in each model's own `metrics.txt`, sibling to `meta.txt`
-- kept separate since `meta.txt` is repeatedly read-modify-written by the *active training loop* and
folding eval output into the same struct risked it being clobbered by a later resume. DERIVED values
(bits/byte, tokens/param, autotemp's interpolated recommended temperature) are recomputed fresh every
time from the cache, by whichever build of `models --metrics` happens to be running -- so a future new
comparison axis never needs the historical training/eval binary again. `report_stage`/`autotemp_stage`
were both independently calling `evaluate()` on the identical val span through two different call
conventions (hand-traced the windowing math to confirm this, not assumed) -- extracted a shared
`compute_raw_metrics()` both now call, so that duplicate pass only happens once. `models` gained
`--corpus`/`--sha`/`--since`/`--until` filters (the date filter is today's practical "this sweep"
substitute for the still-unreliable sha filter) and `--metrics` (read-only table) / `--refresh`
(compute+cache anything filtered+loadable+missing-or-stale) / `--refresh --force`.

**Real bug caught by re-testing rather than trusting the first result:** ran `--refresh --force` twice
in a row and diffed the output -- byte-identical to itself (ruling out GPU nondeterminism as the
cause) but substantially different from the standalone `autotemp` command's own baseline on the same
model (`rep4` shifted from ~0.02-0.06 to ~0.10-0.14, recommended temp from 0.33 to 1.00). Root cause:
`gen_self_stats` resolves the EOS-stop signal via `sub0::eos_token_id()`, which returns **-1** with no
tokenizer loaded -- `sub0_autotemp_stage` already calls `load_tokenizer` before its own
`compute_raw_metrics` call; the new `--refresh` path never did, so generation never stopped early and
silently inflated the measured repetition -- precisely the class of bug `a016d17` (this session's
earlier EOS-stop fix) was about. Fixed (one line) and re-verified: `--refresh`'s grid now matches
`autotemp`'s own output exactly.

**Full 6-model tinystories comparison, populated and verified live (2026-07-09):**
```
d128   3.0M  val_nelbo 1.5841  bits/B 0.685  temp 0.96  muon
d192   5.4M  val_nelbo 1.4332  bits/B 0.620  temp 0.56  muon
d256   9.1M  val_nelbo 1.3459  bits/B 0.582  temp 0.78  muon
d320  15.5M  val_nelbo 1.2678  bits/B 0.548  temp 0.91  muon
d448  41.5M  val_nelbo 1.2458  bits/B 0.539  temp 1.00  adamw  (old plain-baseline)
d448  34.3M  val_nelbo 1.2141  bits/B 0.525  temp 0.33  muon   (full feature set)
```
Notable: the full-feature-set d448 (gated-FFN+tied-embeddings+QK-norm+Muon, 34.3M params) beats the
old plain-baseline d448 (41.5M params) on val_nelbo despite having ~7M FEWER parameters -- a genuine
research result, not just a parity confirmation.

**Deferred, not fixed this pass:** cross-build auto-refresh (this build's `--refresh` invoking a
*different, older* bundled binary to backfill a dims-incompatible model) -- wasn't needed (the sweep's
5 models are all one build; the backfill model was handled by reconfiguring a matching-dims build
directory directly, same as the sweep itself). The raw-cache schema is designed so this could be added
later without a format change.
