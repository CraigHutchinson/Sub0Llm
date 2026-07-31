# TUTOR_SWEEP — can a structured training order make cross-document interference identifiable?

**Status: DESIGN NOTE. Nothing here is built, and the headline recommendation is "not the thing that was
proposed".** Written against [TUTOR.md](TUTOR.md) (the scheme), [TUTOR_SPIKE.md](TUTOR_SPIKE.md) (what was
actually measured) and `include/sub0/tutorspike.hpp` (`Surface`, `EpochPlan`, `DriftProbe`).

## 0. The question, and the answer in one paragraph

The proposal: replace `sub0::tutor::EpochPlan`'s uniformly random permutation of every window tiling the
corpus with a **structured, deterministic sweep** — "linear prime-offset waves", i.e. traversal at co-prime
strides — so that when a transfer reading shows cross-document interference, the phase/interval at which it
appears back-tracks to *which* documents caused it. The premise is that random order aliases that
information away.

**The premise is wrong, and it is wrong in a way that is worth stating precisely, because the real defect it
is pointing at is real.** Random order is not what destroys the information. What destroys it is that the
transfer reading is currently integrated over an interval **one full epoch long**, so its measurement row is
the all-ones vector: 60,948 of 60,948 windows, every document, every time. A row of density ~100% carries no
pairwise information under *any* ordering, structured or not. Meanwhile a uniformly random permutation is —
by the compressed-sensing literature's own accounting — a *near-optimal* sensing design, and every
deterministic alternative proposed here is provably worse at recovery (the square-root bottleneck) while
buying only reproducibility, which a seeded PRNG already provides for free.

So: **do not restructure the sweep.** Fix the readout instead (§4), gate everything on a superposition test
that is likely to fail (§3), and restrict the question from "which document interferes with which" (538M
unknowns, unreachable) to "which *cluster* interferes with which" (~10⁴ unknowns, reachable in one run) — §7.
If per-document attribution is genuinely wanted, TRAK-style projected gradient attribution (§8) answers it
directly at O(n) cost instead of O(n²) measurements, and should be preferred.

---

## 1. The measurement model, stated formally

Let the corpus be tiled into `W = 60,948` windows over `D = 32,810` reachable documents (the numbers are
this spike's, from TUTOR_SPIKE.md: 36,000 spliced documents, minus a 2,908-document validation tail and 282
drift probes). Training proceeds in steps `t = 1..B`; step `t` trains a batch `S_t` of `b = 448` windows and
applies an update `Δw_t`.

For document `i`, `Surface::record` computes

```
transfer_i(k) = [ nelbo_i(visit k+1) − nelbo_post_i(visit k) ] / [ G(visit k+1) − G(visit k) ]
```

where `G` is global applied learning. Write `I(k) = { t : visit k < t < visit k+1 }` for the steps in the
interval — document `i` is not trained in any of them, by construction.

To first order in the step size,

```
nelbo_i(after) − nelbo_i(before)  ≈  ∇ℓ_i(w) · Σ_{t ∈ I(k)} Δw_t                                  (1)
```

If additionally each step's update decomposes over its batch members, `Δw_t = Σ_{j ∈ S_t} u_j`, then

```
Δ_i(k)  ≈  Σ_j  a_{k,j} · x_{ij},        a_{k,j} = (applied learning j received in I(k))            (2)
                                          x_{ij}  = ∇ℓ_i · u_j / (unit applied learning)
```

Equation (2) is the linear model the identification question assumes: one reading is `⟨A_row, x_i⟩`, `A` is
the interval-composition matrix, `x_i` is the vector of per-document effects on `i`. **Everything downstream
depends on (1) and (2) both holding. §3 argues (2) is the fragile one and is probably false as written.**

### 1a. The design matrix we currently have, and why it is rank-degenerate

`EpochPlan` visits every window exactly once per epoch. A document `i` with `w_i` windows is therefore
visited `w_i` times per epoch, and the mean gap between consecutive visits is `W / w_i` windows.

| | windows/doc/epoch | mean interval length | distinct docs in `A_row` | row density |
|---|---|---|---|---|
| median doc (`w_i = 1`) | 1 | 60,948 windows = one epoch | ~32,810 | **~100%** |
| long cosmopedia doc (`w_i ≈ 30`) | 30 | ~2,032 windows ≈ 4.5 steps | ~2,000 | ~3.3% |

For the *majority* of documents `A_row` is the all-ones vector. Rows that are all-ones are mutually
identical: `rank(A) ≈ 1` over that population, and the only quantity estimable from them is the global drift
scalar — which is exactly what `DriftProbe` already reports, and exactly what the observed transfer numbers
turned out to be (population means at the 1e-5 level, sign-unstable across reads, TUTOR_SPIKE.md "the
transfer sign is not stable across training").

**This is the diagnosis. It is a readout defect, not an ordering defect.** No permutation of the corpus
changes the fact that a full-epoch interval touches every document. Structuring the sweep cannot fix it;
shortening the interval can.

### 1b. The hard floor on interval resolution

One update happens per step. Therefore the finest possible measurement row is **one batch**: 448 of 60,948
windows, density 0.735%. You cannot attribute below the granularity of a batch, ever, by any ordering.

To obtain a single-batch row for document `i` you need `nelbo_i` scored at step `t` and again at step `t+1`
with `i` in neither batch. `Surface::record_post` + a per-step re-score of a fixed panel gives exactly that.
This is §4 and it is the whole of the actionable recommendation.

---

## 2. The scale constraint, with the numbers

| quantity | value |
|---|---|
| reachable documents `D` | 32,810 |
| unordered pairs | 538,231,645 |
| transfer readings, one 10-epoch run (measured) | 22,809 |
| unknowns per measurement | **23,597×** |
| transfer readings **per document** per run | 22,809 / 32,810 = **0.70** |

The 23,600× headline understates the problem. The operative number is the last row: **each document
currently gets less than one transfer reading per entire 10-epoch run.** Estimating anything per-document
from 0.7 samples is not an underdetermination problem, it is an absence-of-data problem.

There is a second, harder floor that no scheme can move. In the linear model (2), the standard error of the
estimate of `x_{ij}` scales as `σ / √c_j`, where `c_j` is the **column weight** — the number of measurement
rows that contain document `j`. Under constraint 4 (every token trained exactly once per epoch), `c_j` is
*pinned*:

```
c_j = visits of j over the run = w_j × epochs ≈ 1.86 × 8.16 ≈ 15.2   (median document, 10 counted epochs)
```

So even with a perfect design and single-batch rows, the per-partner resolution is **σ / √15.2 = 0.26σ**, and
**no reordering can improve it** — reordering permutes which rows contain `j`, it cannot change how many.
The only three levers are:

1. **more epochs** — `c_j ∝ epochs`, so resolution improves as `√epochs`. 4× the compute buys 2×.
2. **sparsity** — assume `x_i` has few large entries, then use `m = O(s log(n/s))` rows instead of `n`.
3. **aggregation** — group documents; a group of size `g` has column weight `g × 15.2`. This is the only
   lever that gains orders of magnitude, and it is the recommendation (§7).

---

## 3. THE GATING EXPERIMENT — test superposition first, and expect it to fail

Everything above assumes (2): the effect on `i` of training `{j,k}` equals the effect of `j` plus the effect
of `k`. **This is not merely "neural training is nonlinear, be careful". There is a specific, named mechanism
in this codebase that breaks it exactly, and this spike has already recorded it.**

### 3a. The optimizer breaks additivity before the network gets a chance to

Equation (1) is fine: `Δℓ_i ≈ ∇ℓ_i · Δw` is a first-order Taylor step and is the same approximation TracIn is
built on — *"ℓ(wₜ,z′) − ℓ(wₜ₊₁,z′) ≈ ηₜ ∇ℓ(wₜ,z′) · ∇ℓ(wₜ,zₜ)"*
([Pruthi et al., NeurIPS 2020](https://arxiv.org/abs/2002.08484)). Linearity in `Δw` is exact to first order.

Equation (2) requires something stronger and different: that `Δw_t` itself decomposes additively over the
batch. Under plain SGD it does — `Δw = −η Σ_j g_j`. **Under Adam it does not.** The update is
`−lr · m̂ / (√v̂ + ε)`, which is a sign-like, scale-normalising function of the aggregate gradient. Doubling
one document's gradient contribution does not double its share of the update; it changes the denominator
that every *other* document's share is divided by. The batch members are coupled through `v̂`.

TUTOR_SPIKE.md already found the same fact from the other end and recorded it as a finding:

> repeat count is *not* interchangeable with loss weight … N sequential steps at `lr` is not one step at
> `N·lr` under Adam: the optimizer's moment normalisation makes the two differ.

That is the identical non-additivity, observed in the time axis instead of the batch axis. It is not a
speculative risk; it is a measured property of this trainer. The hybrid Muon(matrices) + AdamW configuration
is worse still — Newton–Schulz orthogonalisation of the momentum matrix is strongly non-additive by design.

### 3b. The experiment

Cheap, decisive, read-only, no GPU-hours to speak of, and it uses only seams that already exist
(`save_checkpoint`/`load_checkpoint`, `sub0_dev_forward_loss`, `DriftProbe`'s panel).

```
for each checkpoint w0 in {early (~1 ep), mid (~5 ep), late (~10 ep)}:      # the regime may be maturity-dependent
  for each of R = 30 disjoint batch pairs (S_A, S_B), each of size b/2:
      score  the probe panel P at w0                       ->  L0[P]
      load w0; reset optimizer state; train ONE step on S_A ; score P  ->  LA[P]
      load w0; reset optimizer state; train ONE step on S_B ; score P  ->  LB[P]
      load w0; reset optimizer state; train ONE step on S_A ∪ S_B; score P -> LAB[P]

  residual_p = (LAB[p] − L0[p]) − ((LA[p] − L0[p]) + (LB[p] − L0[p]))
  report  |residual| / |LAB − L0|   distribution over p, and its ratio to the frozen-model null
```

**Controls that are not optional:**

* The **frozen-model null** — TUTOR_SPIKE.md control arm 5. Score `P` twice at `w0` with no training. The
  delta should be exactly 0; whatever it is instead (bf16 accumulation order, atomics) is the floor beneath
  every number in this experiment, including the residual. Run it first; it can invalidate the instrument.
* **Optimizer state must be reset identically in all three arms**, or the `v̂` coupling contaminates the
  comparison with a state-carryover artefact rather than the effect being tested.
* Run the whole thing **twice**: once under Adam/Muon as configured, once under plain SGD at matched applied
  learning. If superposition holds under SGD and fails under Adam, the finding is "the optimizer is the
  obstruction", which is separable and actionable. If it fails under both, the linear model is void and
  §5–§7 are all moot.

**Pass criterion, stated before the data:** superposition is usable if the median `|residual| / |ΔLAB|` is
**below 0.2** at the mid and late checkpoints. Above 0.5 the linear model is not worth building on. Between
the two, it is usable only for aggregate/group questions (§7) and never for per-document claims.

**Prediction, registered:** it fails under Adam at realistic LR (residual ≳ 0.5) and passes under SGD. If
that is the outcome, the correct response is *not* to switch the trainer to SGD — it is to abandon
loss-delta attribution and move to gradient-alignment attribution (§8), where linearity is exact by
construction because the quantity measured is `∇ℓ_i · Δw` itself rather than its consequence.

**Cost:** 3 checkpoints × 30 pairs × 3 single-step trainings = 270 steps plus ~1,100 panel scorings. Minutes.
It is the cheapest thing in this document and it gates every other thing in it.

---

## 4. The readout fix — the only unconditional recommendation

Independent of superposition, independent of any identification scheme, and independent of the sweep:

**Re-score a fixed panel of held-out probe documents after EVERY step, and record each reading against the
single batch that separates it from the previous one.**

This converts the measurement rows from density ~100% (§1a) to density 0.735%, which is the hard floor from
§1b. It is the difference between a rank-1 design and a full-rank one. It requires:

* No change to `EpochPlan`, no change to training order, no change to what training does. Constraint 4 is
  untouched because the panel is `DriftProbe`'s never-trained set, already excluded by construction.
* A fix to the memory hazard TUTOR_SPIKE.md already found: `sub0_dev_forward_loss` goes through
  `fwd_alloc(full=true)` and materialises an unchunked `[n·T, VOCAB]` logits buffer, which is what capped the
  re-score at ~22 windows and OOM'd at full batch. A panel of ~64–128 probe windows is inside
  `eval::device_batch`'s existing `DEVICE_LOGITS_BUDGET_BYTES` derivation, so this needs no new kernel — it
  needs the panel to be small and fixed, which it already is.
* Cost: one forward over ~128 windows per step against a 448-window forward+backward ≈ **128/448 × 1/3 ≈ 9.5%
  throughput**. Not free. Bounded, and it is the entire information budget of the exercise.

Yield: `B = 1,110` single-batch readings **per panel document per 10-epoch run**, against the current 0.70
full-epoch readings per document. That is a ~1,600× increase in usable measurements, obtained without
touching the sweep.

**Note the asymmetry this creates and accept it.** Panel documents are never trained, so they yield
`x_{ij}` for `i ∈ panel` only — "what does the corpus do to these 128 documents". It does not yield a full
`D × D` matrix. That is the restricted question of §7 and it is the right one to be asking.

---

## 5. Prior art, evaluated — which formalisations actually apply

### 5a. Co-prime strides and CRT — **category error, reject**

Co-prime sampling's win is the **difference co-array**: two uniform arrays with co-prime spacings `M`, `N`
give `O(MN)` degrees of freedom from `M + N` sensors, because the cross-difference set has `O(MN)` distinct
lags ([Vaidyanathan & Pal, IEEE T-SP 59(2):573–586, 2011](https://arxiv.org/pdf/1808.07505)). That works
because the estimated object is a **second-order statistic of a stationary process** — the autocorrelation
`R(lag)` — recovered by averaging many independent snapshots.

Neither premise holds here. There is no stationarity: the model changes irreversibly at every step, so the
"same" pair of documents at epoch 1 and epoch 8 are not two snapshots of one quantity. And there is no
lag-indexed autocorrelation to estimate: our measurement is a *sum over a set*, not a lagged product.

CRT is a further step removed. CRT resolves ambiguity when a scalar is observed **modulo** several co-prime
moduli — range/frequency disambiguation, where wraparound is the ambiguity. Our observation has no modular
wraparound. CRT would give each document a unique residue-tuple *index*, which is unique labelling, not
identification: we already have unique labels (the ordinal). **Unique indexing is not the missing ingredient;
rank is.**

Third, as a *training order* co-prime strides are actively hazardous here. A stride-`s` traversal with
`gcd(s, W) = 1` is a valid permutation, but batch `t` becomes an arithmetic progression `{ t·b·s + i·s }`,
which is maximally coherent — consecutive rows are shifts of each other, exactly the structure the square-root
bottleneck penalises (§5d). Worse, the tutorspike manifest **interleaves populations in runs of 64**, so the
per-batch population composition becomes a deterministic function of `s mod 64`; a stride congruent to 0 mod
64 puts one population in a whole batch. That is a curriculum confound introduced by the measurement design,
which is the failure `blend_schedule.hpp` exists to prevent.

**Verdict: reject. Wrong formalism, no rank gain, and a training-order hazard.**

### 5b. Sidon sets / B₂ sets / Singer difference sets / Golomb rulers — **arithmetically decisive, reject**

This is the closest honest reading of "prime offset waves". A Sidon set has all pairwise differences distinct
(Singer proved that for every prime power `q`, a perfect difference set of size `q+1` exists in
`ℤ/(q²+q+1)ℤ`, cited in [Bhowmick et al., arXiv:2003.04929](https://arxiv.org/pdf/2003.04929)), so an
observed lag identifies a pair **uniquely**. That is precisely the property the proposal wants.

It does not survive contact with the numbers. The Erdős–Turán bound is that a Sidon set in `[1, N]` has at
most `N^{1/2} + N^{1/4} + 1` elements ([Erdős & Turán 1941; see
arXiv:2103.15850](https://arxiv.org/pdf/2103.15850)). Inverting it:

| | value |
|---|---|
| documents to place at distinct Sidon offsets | 32,810 |
| schedule length required (`≈ k²`; Singer: `q²+q+1`, `q ≥ 32,809`) | **1.076 × 10⁹ slots** |
| slots actually available in one epoch | 60,948 |
| **dilation required** | **17,663×** |

Each document would appear **once** in the whole Sidon layout, so this buys one epoch at 17,663× the schedule
length — i.e. 17,662 idle slots between every trained window. That is not a training order.

And even ignoring cost, the mechanism does not do what is wanted. Sidon's "one pair per lag" property assumes
**one event per position**. Here every position is a batch of 448 windows, so a given lag is shared by
448 × 448 ≈ 200,000 candidate pairs. The distinctness is destroyed by the batch before the lag is ever read.

**Verdict: reject. Off by four orders of magnitude, and the batch destroys the defining property anyway.**

### 5c. Hadamard / frequency-multiplexed measurement — **rejected by the token budget, and the arithmetic is exact**

The multiplex (Fellgett) advantage is real and quantified: for `N` resolution elements, Hadamard/S-matrix
encoding improves SNR by a factor of order `(N/2)^{1/2}` over one-at-a-time measurement
([Nelson & Fredman, Appl. Opt. 13(11):2662, 1974](https://opg.optica.org/ao/abstract.cfm?uri=ao-13-11-2662)).
That would be a `√(32810/2) ≈ 128×` gain — the single largest number in this document, if it were reachable.

It is not. A Hadamard/S-matrix design requires each **column** (each document) to be present in ~half of all
rows. With `B = 1,110` steps that is a column weight of ~555, i.e. **555 visits per document per run**.
Constraint 4 pins the column weight at 15.2 (§2). The design is over-subscribed by **36×** in tokens, which
is 36× the compute, on a corpus where the run already takes 8.16 real epochs.

The deeper point: Hadamard designs are *dense* (`ρ = 1/2`); our achievable design is *extremely sparse*
(`ρ = 15.2/1110 = 1.4%`). Sparse designs are the group-testing / compressed-sensing regime, not the
multiplexing regime. That is what §5e is for.

**Verdict: reject at a fixed token budget. Record the reason — it is the clean quantitative statement of why
"give every document an orthogonal code" cannot work here.**

### 5d. Compressed sensing — **random is already near-optimal; this is the argument against structuring**

Random matrices satisfy the restricted isometry property at `m = O(k log(n/k))` rows
([Baraniuk, Davenport, DeVore & Wakin, *Constr. Approx.* 28:253–263,
2008](https://link.springer.com/article/10.1007/s00365-007-9003-x)). Explicit deterministic constructions do
not reach this. They are *"notorious for performing at the 'square-root bottleneck,' i.e., they only accept
sparsity levels on the order of the square root of the number of measurements"*
([Bandeira, Mixon & Moreira, arXiv:1403.3427](https://arxiv.org/abs/1403.3427)), because they are proved RIP
via coherence (Gershgorin), which inevitably costs `m ~ k²`. The only known explicit matrix beating it is
Bourgain–Dilworth–Ford–Konyagin–Kutzarova's, and it beats it by an epsilon in the exponent.

Concretely for us, at `n = 32,810` unknowns and `s = 30` assumed interferers:

| design | rows needed |
|---|---|
| random (RIP, `≈ 4 s log(n/s)`) | ~840 |
| coherence-based deterministic (`~s²`) | ~900 … but the guarantee degrades as `s²` and at `s = 100` needs ~10⁴ |

**So a structured sweep must justify itself on grounds other than recovery performance, because on recovery
performance it loses.** The three grounds usually offered:

* *Reproducibility* — already free. `EpochPlan::next` shuffles with the caller's RNG, seeded from the run
  seed; the permutation is exactly reconstructible.
* *No restricted-isometry constant to estimate* — a real advantage in general, but moot here: for random
  designs RIP holds with overwhelming probability and needs no per-instance estimation; and we would in any
  case be solving a least-squares problem, not certifying exact recovery.
* *Deterministic back-tracking without storing `A`* — the stated motivation. Storing `A` for this corpus is
  0.6 MB (`EpochPlan`'s own slot array), regenerable from one integer. **The benefit being purchased is
  0.6 MB and one `std::shuffle`.**

**Verdict: the structured sweep is strictly dominated. Keep the seeded random permutation.**

### 5e. Combinatorial group testing / disjunct matrices / Kautz–Singleton — **the one that actually fits, and it fits surprisingly well**

If interference is *sparse* — few strong interferers per document — this is the right frame. The
Kautz–Singleton construction takes a Reed–Solomon code over `GF(q)` and replaces each `q`-ary symbol with a
weight-1 binary vector of length `q`, yielding a `d`-disjunct matrix; it is order-optimal at `Θ(d log N)`
tests in the probabilistic setting for `d = Ω(log² N)`
([Inan, Kairouz, Wootters & Ozgur, arXiv:1808.01457](https://arxiv.org/abs/1808.01457)).

Its structure maps onto our constraint **exactly**, which is the interesting part. Take the RS code
`[N, k]_q` with `N` = number of epochs and `q` = steps per epoch:

* each column (window) has exactly **one 1 per symbol block** — i.e. **every window is trained exactly once
  per epoch**. That is constraint 4, satisfied by construction, not by a side condition.
* codewords agree in at most `k − 1` positions, so covering a column needs `d ≥ N/(k−1)` others:
  `d`-disjunct with `d = ⌈N/(k−1)⌉ − 1`.

Instantiated at this spike's actual configuration:

| parameter | value |
|---|---|
| columns `n` (windows) | 60,948 |
| `q` (steps per epoch, prime) | **137** → batch = 60,948/137 = **445** (vs the run's 448) |
| `N` (blocks = epochs) | **10** |
| rows `m = N·q` | **1,370** (vs the run's 1,110 steps) |
| RS message length `k` | 3 (needs `q^k = 2.57M ≥ 60,948`) ✓ |
| disjunctness `d = ⌈N/(k−1)⌉ − 1` | **4** |

So a Kautz–Singleton schedule at *this corpus's own batch size and epoch count* is 4-disjunct: it exactly
identifies up to 4 interfering windows per target, using a design that is a legal epoch permutation. At 20
epochs `d = 9`; disjunctness grows as `epochs/2 − 1`. At **group** granularity (§7, `G = 181` clusters) `k = 2`
suffices and `d = 9` at 10 epochs.

This is a genuine positive result and it should be recorded as such. **It is nevertheless not the
recommendation**, for three reasons, in order of weight:

1. **It is gated on §3 and on sparsity, neither established.** Disjunctness is a guarantee about a sparse
   boolean model. We have no evidence that `x_i` is 4-sparse, or 30-sparse, or sparse at all.
2. **Our measurements are real-valued and additive, not boolean OR.** Additive measurements are strictly more
   informative, so disjunctness is a *sufficient but pessimistic* criterion; the correct frame is §5d's
   `m = O(s log(n/s))`, and there random already wins.
3. **It buys `d = 4` where random buys `s ≈ 30` at the same `m`.** The square-root bottleneck, restated.

**Verdict: the best of the structured candidates by a wide margin, mathematically clean, constraint-4-native
— and still dominated by the random permutation we already have. Adopt only if a deterministic
exactly-certifiable design is later wanted for its own sake.**

---

## 6. SNR — what is actually detectable

Three floors, in ascending order of severity.

**Arithmetic floor — not binding.** The per-window readout agrees with the batch scalar to ~6e-9 relative
(TUTOR_SPIKE.md stage 0), usable to ~1e-7. At `nelbo ≈ 2` that is ~2e-7 absolute. One step applies ≈ 900 units
of global applied learning (from the recorded `applied ≈ 1e6` over 1,110 steps), and the measured drift floor
at 10 epochs is 8.50e-6 per unit applied — so a **single-batch** reading has expected magnitude
`8.5e-6 × 900 ≈ 7.7e-3` nelbo. The per-partner share, if 448 windows contribute equally, is `1.7e-5` —
**85× above the arithmetic floor.** Float precision is not the constraint. `Entry`'s `float` loss fields are
correctly sized.

**Nondeterminism floor — unmeasured, and it must be measured first.** bf16 accumulation order and atomics
give a nonzero delta between two scorings of the same weights. TUTOR_SPIKE.md control arm 5 measures it in
minutes. Until it is measured, the 85× above is an upper bound on the true headroom and could be any amount
smaller. **Run arm 5 before anything else in this document except §3.**

**Stochastic floor — the binding one, and unmeasured.** Let `σ` be the per-reading standard deviation of a
single-batch transfer measurement. From §1b, the standard error on a per-document coefficient is `σ/√c_j` with
`c_j = 15.2` pinned by the token budget:

```
detectable per-document effect  ≈  2 × σ / √15.2  ≈  0.51 σ            (2-sigma, 10 counted epochs)
                                ≈  0.36 σ                              (20 epochs)
                                ≈  0.11 σ                              (aggregated into groups of 10)
```

`σ` is not known and **cannot be derived** — it is exactly what TUTOR_SPIKE.md control arm 2 (the
duplicate-content control: splice ~500 documents in twice at different ordinals, and read the divergence
between the identical copies) exists to supply. Two identical documents' readings differ only by noise, so
their spread *is* `σ`.

**Therefore: arms 5 and 2 of TUTOR_SPIKE.md are not refinements of the result, they are prerequisites of this
entire design note.** Neither is expensive. Both should run before any measurement-design work begins, and
the note's numbers should be revisited against the measured `σ`.

---

## 7. Where the extra information actually comes from — restrict the question

Four tiers of question, with an honest verdict on each.

| tier | question | unknowns per target | rows available (§4) | verdict |
|---|---|---|---|---|
| 0 | per-**population** (3 groups) | 3 | 1,110 | **already answered** — this is the existing table, and it works |
| 1 | per-**cluster**, `G ≈ 100–200` | ≤ 200 | 1,110 | **reachable in one run, no sparsity assumption** |
| 2 | per-**document**, top-`s` interferers, `s ≲ 30` | 32,810 sparse | 1,110 | conditional: needs §3 to pass **and** sparsity to hold |
| 3 | full pairwise `D × D` | 538,231,645 | 22,809 | **unreachable; do not attempt** |

**Tier 1 is the recommendation.** Cluster the 32,810 documents into `G ≈ 181 ≈ √D` groups; a group's column
weight becomes `15.2 × 32810/181 ≈ 2,750`, the per-group standard error drops to `σ/52`, and with 1,110
single-batch rows against ≤ 200 unknowns the system is **6× overdetermined with no sparsity assumption at
all**. This is the only lever in §2 that gains orders of magnitude, and it gains four.

Clustering must be defined from something that is not the measurement, or the finding is circular.
Defensible choices, cheapest first: the manifest population; the corpus source; tokenizer-level features
(document length, type/token ratio, caps fraction — this repo already computes all three); k-means over
mean-pooled embeddings. **The result to look for is a `G × G` transfer matrix with visible off-diagonal
structure** — "shuffled conflicts with everything, cosmopedia reinforces cosmopedia" — which is a diagnostic
nothing in this project can currently produce, and which is TUTOR_SPIKE.md's "transfer is a product, not an
error term" delivered at a granularity the data can actually support.

Tier 3 deserves one explicit statement so it is never revisited: closing a 23,600× gap by brute force means
~23,600 runs to reach `m = n` with *zero* noise margin, at ~16 min/run on this toy corpus ≈ **262 GPU-days**,
for a corpus 1/3000th the size of fineweb, to obtain a determined-but-unregularised system. It is not a
question of finding a cleverer order.

---

## 8. The alternative that probably dominates all of it — gradient attribution

The ML literature attacks precisely this question and does not go through aggregate loss deltas at all.

* **Influence functions** ([Koh & Liang, ICML 2017](https://proceedings.mlr.press/v70/koh17a.html)) —
  `dℓ(z_test)/dε` for up-weighting a training point, via `∇ℓ_test ᵀ H⁻¹ ∇ℓ_train`. Requires Hessian-vector
  products; known to degrade on non-convex deep models. Expensive here, and the `H⁻¹` is the hard part.
* **TracIn** ([Pruthi et al., NeurIPS 2020](https://arxiv.org/abs/2002.08484)) — the same first-order step
  used in eq. (1), but read *forward*: `TracInCP(z,z′) = Σᵢ ηᵢ ∇ℓ(w_{tᵢ}, z) · ∇ℓ(w_{tᵢ}, z′)` over saved
  checkpoints. **No identification problem exists**: the pairwise quantity is computed directly, not inferred
  from aggregates. Cost is `O(D)` gradient evaluations, not `O(D²)` measurements.
* **Datamodels** ([Ilyas et al., ICML 2022](https://proceedings.mlr.press/v162/ilyas22a.html)) — the honest
  reference point for what aggregate-measurement identification actually costs. To fit a *linear*
  (per-example, **not** pairwise) datamodel on CIFAR-10's 50,000 examples they trained
  [**300,000 to 1,500,000 models**](https://github.com/MadryLab/datamodels-data) depending on the subsampling
  fraction α. That is the empirical price of the exact problem shape being proposed here, for a problem one
  order easier (linear, not quadratic) on a training set of comparable size. It is the strongest single
  argument against tier 3.
* **TRAK** ([Park et al., ICML 2023](https://arxiv.org/abs/2303.14186)) — datamodel-quality attribution from
  *a handful* of trained models, by random (Johnson–Lindenstrauss) projection of per-example gradients.
* **Example forgetting** ([Toneva et al., ICLR 2019](https://arxiv.org/abs/1812.05159)) — worth noting because
  it is the *cheap* version of what `Entry::velocity`'s sign already measures: a forgetting event is a
  correct→incorrect transition, and unforgettable examples are prunable. This is complementary, not
  competing, and it is nearly free from data the surface already carries.

**Costed for this model** (d256 L8, vocab 16,517, ≈ 12.6M parameters):

| approach | storage | compute | yields |
|---|---|---|---|
| full-gradient TracIn | 12.6M × 32,810 × 4B = **1.65 TB** | 32,810 per-example backward passes | exact `D × D` Gram |
| JL-projected (TRAK-style), `k = 4096` | 4096 × 32,810 × 4B = **537 MB** | same, plus a projection | approximate `D × D` Gram |
| **last-layer only** (TracIn's own "cherry-picking layers") | activation ⊗ softmax-error factorises; no `D²` materialisation | one forward per document | top-`s` neighbours per document |

The last row is the point. Per-example gradients of the LM head factorise as an outer product, so the dot
product `∇ℓ_i · ∇ℓ_j` reduces to a product of two low-rank terms and never needs the full gradient
materialised. This project's chunked `lm_head` + CE path already computes both factors.

**Honest comparison.** Gradient attribution gets the full pairwise matrix for ~537 MB and one epoch's worth of
gradient work, versus 262 GPU-days for the measurement route. Its weakness is that it is a *first-order
proxy*: it answers "do these gradients align at this checkpoint", not "what did training on `j` actually do
to `i`" — and it inherits the same eq. (1) approximation, so §3's failure would compromise its
*interpretation* too (though not its computation). Its cost here is real: per-example gradients are a genuine
engine change, since gradients are currently accumulated over the batch and never exposed per row.

**But the sweep's own measurement is a strictly worse instrument for the same question,** and the correct
comparison is not "sweep vs nothing", it is "sweep vs TracIn-on-the-last-layer". On that comparison the sweep
loses on every axis except one: the sweep measures the *actual* trajectory of the *actual* run, which is
what TUTOR's controller would eventually act on. That is not nothing. It is not worth 262 GPU-days.

---

## 9. Recommendation

**Ranked, with the gate first.**

1. **Run the frozen-model null (TUTOR_SPIKE arm 5) and the superposition experiment (§3).** Hours, not days.
   If superposition fails under the production optimizer — which is the registered prediction — §5 and §7
   tier 2 are void and the note ends at §8. *This is the first thing to do and nothing else should start
   before it.*
2. **Run the duplicate-content control (TUTOR_SPIKE arm 2) to obtain `σ`.** Every detectability claim in §6
   is parameterised by it and none can be settled without it. A splice-tool flag plus a rerun.
3. **Fix the readout (§4): per-step re-score of a fixed, small, held-out panel.** Unconditional — it is
   correct regardless of how 1 and 2 land, because the current transfer rows are ~100% dense and therefore
   rank-degenerate. ~9.5% throughput, ~1,600× more usable measurements, zero change to training order.
4. **Restrict the question to tier 1 (§7): a `G × G` cluster transfer matrix, `G ≈ 100–200`.** Fully
   determined by one run given step 3, with no sparsity assumption. This is the deliverable that pays, and
   it is the natural continuation of TUTOR_SPIKE.md's "transfer is a product, not an error term".
5. **Do NOT restructure `EpochPlan`.** Co-prime/prime-offset waves are a category error (§5a) and a
   training-order hazard given the run-64 population interleave. Sidon layouts are 17,663× too long (§5b).
   Hadamard needs 36× the token budget (§5c). Kautz–Singleton is mathematically clean and constraint-4-native
   (§5e) but buys `d = 4` where the *existing* random permutation buys `s ≈ 30` at the same row count (§5d).
   The only benefit a deterministic order confers over a seeded shuffle is not having to store 0.6 MB.
6. **If per-document attribution is genuinely wanted, use last-layer TracIn/TRAK (§8), not the sweep.**
   537 MB and one epoch of gradient work versus 262 GPU-days.

## 10. Where I am uncertain

Stated plainly, because several of these could flip the conclusion.

* **`σ` is unknown.** Every SNR number in §6 is a shape, not a magnitude. If `σ` turns out to be large
  relative to `7.7e-3`, tier 1 shrinks too and the answer becomes "population-level only, which we already
  have".
* **Whether `x_i` is sparse at all.** §5e and §7 tier 2 both assume it. It is plausible (shared vocabulary
  and topic should localise interference) and equally plausible that interference in a 12.6M-parameter model
  at 32,810 documents is *dense and diffuse* — every document mildly affects every other through shared
  capacity — in which case sparse recovery is the wrong tool entirely and only aggregation works.
* **The superposition prediction.** I expect Adam to break it, and I am confident about the mechanism
  (`v̂` coupling) but not about the magnitude. It may be that at small per-step LR the coupling is a second-order
  correction and the residual comes in under 0.2. The experiment is cheap precisely because I cannot argue
  this one from first principles.
* **The 0.735% row density claim assumes the panel is re-scored between *every* pair of steps.** If the
  re-score has to run at a cadence for throughput reasons, row weight scales with the cadence and §4's 1,600×
  degrades proportionally. At cadence 4 it is 400×, still decisive; at cadence 20 (the old value) it is 80×
  and the rows are 20 batches wide.
* **Kautz–Singleton's fit is almost suspiciously neat** — `q = 137` giving batch 445 against the run's 448 is
  a coincidence of this corpus's size, not a property of the construction. At any other corpus scale the
  parameters would need re-derivation and the batch size would not land where the trainer wants it. I have
  not checked whether the RS-derived batch composition interacts badly with the population interleave the way
  co-prime strides do; it should be checked before any implementation, not assumed.
* **Everything measured so far is single-seed.** This repo has been burned twice
  (`gqa-ab-three-pillar-result`, `loopsplit-3arm-clean-result`). Nothing in this note is safe to act on past
  step 1 without TUTOR_SPIKE.md control arm 1.

---

## Sources

* R. Baraniuk, M. Davenport, R. DeVore, M. Wakin, *A Simple Proof of the Restricted Isometry Property for
  Random Matrices*, Constr. Approx. 28:253–263, 2008 —
  https://link.springer.com/article/10.1007/s00365-007-9003-x
* A. Bandeira, D. Mixon, J. Moreira, *Explicit Matrices with the Restricted Isometry Property: Breaking the
  Square-Root Bottleneck*, arXiv:1403.3427 — https://arxiv.org/abs/1403.3427
* H. Inan, P. Kairouz, M. Wootters, A. Ozgur, *On the Optimality of the Kautz-Singleton Construction in
  Probabilistic Group Testing*, arXiv:1808.01457 — https://arxiv.org/abs/1808.01457
* P. P. Vaidyanathan, P. Pal, *Sparse Sensing with Co-prime Samplers and Arrays*, IEEE T-SP 59(2):573–586,
  2011; and *Coprime Sensing via Chinese Remaindering over Quadratic Fields, Part I*, arXiv:1808.07505 —
  https://arxiv.org/pdf/1808.07505
* P. Erdős, P. Turán (1941) Sidon-set bound `≤ N^{1/2} + N^{1/4} + 1`, as stated in *An upper bound on the
  size of Sidon sets*, arXiv:2103.15850 — https://arxiv.org/pdf/2103.15850
* J. Singer's prime-power perfect-difference-set construction, as stated in *An asymptotic version of the
  prime power conjecture for perfect difference sets*, arXiv:2003.04929 — https://arxiv.org/pdf/2003.04929
* M. Nelson, M. Fredman, *Hadamard Spectroscopy* / theoretical comparison of singly multiplexed Hadamard
  transform spectrometers, Appl. Opt. 13(11):2662, 1974 —
  https://opg.optica.org/ao/abstract.cfm?uri=ao-13-11-2662
* G. Pruthi, F. Liu, S. Kale, M. Sundararajan, *Estimating Training Data Influence by Tracing Gradient
  Descent*, NeurIPS 2020, arXiv:2002.08484 — https://arxiv.org/abs/2002.08484
* P. W. Koh, P. Liang, *Understanding Black-box Predictions via Influence Functions*, ICML 2017 —
  https://proceedings.mlr.press/v70/koh17a.html
* A. Ilyas, S. M. Park, L. Engstrom, G. Leclerc, A. Madry, *Datamodels: Predicting Predictions from Training
  Data*, ICML 2022 — https://proceedings.mlr.press/v162/ilyas22a.html ; model counts from
  https://github.com/MadryLab/datamodels-data
* S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, A. Madry, *TRAK: Attributing Model Behavior at Scale*, ICML
  2023, arXiv:2303.14186 — https://arxiv.org/abs/2303.14186
* M. Toneva, A. Sordoni, R. Tachet des Combes, A. Trischler, Y. Bengio, G. Gordon, *An Empirical Study of
  Example Forgetting during Deep Neural Network Learning*, ICLR 2019, arXiv:1812.05159 —
  https://arxiv.org/abs/1812.05159
