---
name: ch29-organic-curriculum-followup
description: Why the convergence curriculum wins (global-NELBO pollution by the irreducible high-t tail) + an organic per-level-plateau-aware follow-up that self-scales
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Hypothesis (user, 2026-06-16) for WHY the Ch29 convergence curriculum beats uniform training, and a follow-up design. Full write-up: `diffusion/TRAINING_DESIGN.md` §13.2. Gate: pursue ONLY if the founded long run's end model proves USEFUL (coherent generation / worthwhile recall), not merely lower-NELBO. See [[diffusion-nelbo-ceiling-reframe]].

**Why the curriculum works — the "polluted global NELBO" hypothesis.** The win is the SIGNAL training terminates on, not the easy-first ordering. Uniform training early-stops on the GLOBAL NELBO E_t[CE], which is dominated by the IRREDUCIBLE high-t tail (per-t curve rises monotonically to H0≈5.6 and barely moves there). Learnable progress lives at low–mid t but is a small fraction of that average → the global signal plateaus early and trips patience PREMATURELY (the high-noise levels pollute the stop metric). The curriculum gates on PER-LEVEL plateaus instead, tracking each difficulty's own progress, so it never terminates on the drowned-out average. Evidence: uniform baseline early-stopped at 33k while the curriculum kept finding real gains to 66k (+3.8pt, §13.1); per-t gain largest at low t.

**Transfer down PROVEN, transfer up OPEN.** Training higher k improved base(k=1) 3.69→1.54 (denser supervision feeds the easy regime — DOWN). Whether low-k mastery gives high-k enough prior structure (UP) is read from the CLIMB VELOCITY as k rises in the founded run: steady epochs/level ⇒ foundation feeds up; ballooning epochs/level ⇒ a WALL (high-k under-resourced — same starvation that hindered uniform training). Metric is FREE from existing advance-cadence logs; richer retrospective check = `--per-t` per-level curve on any checkpoint (no restart), comparing early vs late.

**Follow-up: the ORGANIC (self-scaling) curriculum.** Don't hand-set a k schedule — train the FULL noise spread but PER-LEVEL-PLATEAU-AWARE: sample t preferentially from levels still learning; define the global stop as "ALL levels plateaued." Unifies three threads: adaptive importance sampling ([[diffusion-nelbo-ceiling-reframe]] §11) + per-level mastery (§13 curriculum) + structurally fixes the global-NELBO pollution. Needs only a per-level plateau tracker (generalize FrontierCurriculum to all levels) + a sampling weight. CHEAP metric to add at next restart: a fixed mid-k NELBO probe in the curriculum log alongside base(k=1) (forward-transfer-up signal). Don't disrupt the running founded experiment to add it; apply at the next natural restart.
