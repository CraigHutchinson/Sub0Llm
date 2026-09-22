# O4 — thread GDN across heads and channels; dynamic scheduling for the routed experts

**Status:** merged 2026-09-22. Decode **0.236 → 0.211 s/token** at the recommended flags, driven entirely
by the GDN mixer (**90.0 → 73.3 ms/token, −18.5%**). Bit-exact: L2 0.23252 and argmax 4/6 in every run of
the A/B, all three neutral fingerprints unchanged, both suites exact.

Built by a delegated Sonnet agent; re-verified independently before merge (§4).

## 1. Context

After O3, decode's profile at the recommended flags was mixer ~124 ms (GDN ~90, QSA ~33), routed experts
~44, Gated Residual ~32, lm_head ~21. Every GEMV in it was already threaded through `gemv::axpy` — but
GDN's own body was not. The conv1d, the delta-rule recurrence and RMSNormGated ran on one core, which is
why the mixer stayed the largest phase.

## 2. Lever 1 — GDN's per-head and per-channel work, threaded (bit-exact)

`gdn::forward` gained `detail::parallel_for<Threads>`, shaped exactly like `gemv::axpy`'s own dispatch:
`if constexpr (Threads <= 1)` elides the pragma entirely, and a call from inside an existing team runs
serially instead of nesting. Four changes use it:

- **`in_proj_b` / `in_proj_a`** were a hand-written output-major dot loop. They now go through
  `gemv::axpy<Threads>` into `beta`/`gg`'s own rows, transformed in place afterwards. The accumulation
  order per output is the same i-ascending sum, so `gemv.hpp`'s bit-exactness argument carries over.
- **conv1d + SiLU and the history-roll were fused into one per-channel body** and threaded over `c`
  (`conv_dim` = 10240 at real dims). One pass instead of two. Per channel the read-before-write order is
  unchanged; only the order between independent channels differs.
- **The delta-rule recurrence** is threaded over physical k-heads (Hk = 16). Each owns a disjoint set of
  `rep` = 3 v-heads, so each state and output slot is written by exactly one thread.
- **RMSNormGated** is threaded over v-heads (Hv = 48), each writing its own `gated` slice.

**Why this is bit-exact rather than merely close:** no threaded loop reduces across the axis it splits.
Every iteration reads only its own inputs and writes only its own output slice, so thread count changes
which core runs an iteration, never the arithmetic. Confirmed by measurement, not just argument: L2
stayed exactly 0.23252 in all 8 runs of the A/B below, at both 8 and 10 expert threads.

## 3. Lever 2 — `ParallelExperts`: `schedule(static)` → `schedule(dynamic)`

A token's `EXPERTS_PER_TOK` experts are not uniform work: each selected plane may be a different sidecar
format, and O1's microbenchmark measured a real per-format spread. With `static`, a thread that draws the
expensive format cannot be helped. `dynamic` lets any thread claim the next pending expert.

Bit-exactness is structural and predates this change: `moe_math.hpp`'s phase 1 computes each selected
expert into its own buffer, and phase 2 sums them in fixed k order. Which thread computed k, and when,
was never part of the answer.

**The authoring agent measured this lever alone at `--moe-decode-threads 8`: 38.3 → 35.2 ms/token
(−8%), 4 rounds.** See §4 for what re-verification found at the recommended 10.

## 4. Independent re-verification (primary agent, before merge)

Both arms built from the same build dir, binaries copied out, then run alternately with the arm order
reversed on even rounds. Real 48-layer artifact, `--tokens 6`, flags
`--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 10 --decode-omp-spin 1 --profile-phases 1`.
The two arms' `sub0_core.dll` differ in size, which is the check that they really are two builds (§10.2).

| round | order | base s/token | O4 s/token |
|---|---|---:|---:|
| 1 | base first | 0.334 (cold) | 0.240 |
| 2 | O4 first | 0.236 | 0.230 |
| 3 | base first | 0.236 | 0.211 |
| 4 | O4 first | 0.228 | 0.208 |

Round 1's base run was the session's first touch of the sidecar (routed experts 106 ms against ~44 warm),
so the warm rounds are the comparison: **base median 0.236, O4 median 0.211, −10.6%.** O4 is faster in
every round, whether it ran first or second.

Phase means over the warm rounds (ms/token):

| phase | base | O4 | delta |
|---|---:|---:|---:|
| **mixer: GDN** | **90.0** | **73.3** | **−18.5%** |
| mixer: QSA | 32.6 | 31.9 | −2.2% |
| MoE: routed experts | 44.3 | 45.7 | +3.3% |
| Gated Residual | 31.6 | 30.9 | −2.2% |

**Lever 1 reproduces, slightly better than reported (−18.5% against the agent's ~15%). Lever 2 does not
reproduce at `--moe-decode-threads 10`** — the routed-expert phase is if anything marginally worse, well
inside that phase's own run-to-run spread (42.8–47.7 ms across both arms). That is not a contradiction of
the agent's number: it measured at 8 threads, where two of ten experts must queue behind another and a
bad draw cannot be rebalanced. At 10 threads each expert already has its own thread, so there is nothing
for work-stealing to fix. See §5 for the direct 8-thread check.

**Neutral gates, rebuilt from this merge:** `sub0_tests` 28,969,623 / 147 and `sub0_frontend_tests`
144,457 / 263, both exact; fingerprints `45ab9af849227297` / `9b83bc6a4d6b8574` / `816c4a54ad49b8cf`
unchanged. Note what these do and do not prove: the neutral config has `DECODE_GEMV_THREADS = 1` and only
`decode.cpp` passes a thread count, so the suites verify the `Threads <= 1` path is byte-identical. The
threaded path is exercised only by the real-artifact run, whose L2 is the gate for it.

## 5. Direct check of lever 2 at 8 expert threads

Both arms rebuilt with `--moe-decode-threads 8`, everything else identical, 3 interleaved rounds with the
order reversed on the even round (ms/token for the phases):

| round | order | base total | O4 total | base MoE routed | O4 MoE routed | base GDN | O4 GDN |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | base first | 0.233 | 0.226 | 41.0 | 40.4 | 90.0 | 80.0 |
| 2 | O4 first | 0.251 | 0.214 | 42.6 | 40.9 | 97.7 | 75.9 |
| 3 | base first | 0.218 | 0.201 | 41.5 | 40.9 | 84.8 | 68.8 |

**The routed-expert phase moves 41.7 → 40.7 ms, −2.4%, inside its own spread.** The agent's −8% (38.3 →
35.2) did not reproduce here at the thread count it was measured at. GDN reproduces again (90.8 → 74.9,
−17.5%), and total decode goes 0.233 → 0.214 (−8%), so the end-to-end win at 8 threads is lever 1's as
well. L2 stayed 0.23252 in all six runs.

**Disposition: lever 2 stays, with weak evidence stated rather than dressed up.** It is bit-exact, costs
one scheduling word, and is the principled choice for non-uniform work items. It is not what makes this
merge worth taking, and nothing downstream should cite an 8% MoE win from it. Two plausible reasons the
original number did not survive: the phase's own run-to-run spread (42.8–47.7 ms at 10 threads across
both arms) is wider than the effect, and both measurements were taken on a host with other agents active.

Also worth recording, since it was free: **8 and 10 expert threads now perform the same** (base 0.233 vs
0.236, O4 0.214 vs 0.211, medians). The recommended flags keep 10.

## 6. Lever 3 — GDN out-projection prefetch: attempted, and NOT banked

`gdn out` (6144 → 2560) runs at ~44–51 GB/s at 8 threads against `in_qkv`'s ~58–71 GB/s, both through the
same primitive. The hypothesis was that the narrower per-thread output slice means more DRAM row-stride
jumps per byte, which a prefetch of the next 4-row group could hide. Implemented as `_mm_prefetch` hints
in `accumulate_avx2`, measured over 3 interleaved rounds: no signal either way, and `in_qkv` moved as much
run to run as `gdn out` did, i.e. host noise dominated.

**Process gap, recorded rather than smoothed over:** the agent then REVERTED it. AGENTS.md §13 says a
mechanism gets three passes before judgement, and that a parked finding stays in the tree behind its own
toggle — "a reverted branch is a finding thrown away". This had one inconclusive pass and no toggle, so
the work is not banked and whoever resumes starts from zero. Two untried candidates it named:

1. widen the row unroll from 4 to 8 rows per group (independent of prefetching);
2. force the private-tile path for narrow per-thread ranges below the current 128-output boundary.

## 6a. `cpp-review` pass over the diff (primary agent, pre-merge)

No MUST findings. What it changed and what it recorded:

- **SHOULD, applied:** `decode.cpp`'s scheduling comment ran 12 lines and asserted a performance
  rationale that §5 could not reproduce. Trimmed to the bit-exactness argument plus an explicit note
  that the measured effect is inside noise, with a pointer here.
- **NICE, not applied:** `gdn::detail::parallel_for` restates `gemv::axpy`'s dispatch shape (serial at
  `Threads <= 1`, no nesting inside a team). One consumer today, so extracting a shared primitive would
  add surface nothing else reads (AGENTS.md §8). Revisit if QSA or MoE want the same pattern.
- **PRE-EXISTING, not fixed:** `gdn::forward` writes the conv history through `float tmp[MAX_K - 1]`
  with `MAX_K = 32`, while `K = d.conv_kernel` is a RUNTIME field of `Dims`. Nothing checks
  `K <= MAX_K`. Unreachable from the engine (`GDN_CONV_KERNEL` is 4, and `layout.hpp` asserts only the
  lower bound) but reachable from a test or tool that builds its own `Dims`. Predates this diff, which
  only moved the buffer inside the per-channel body. Worth a bound assertion or a `static_assert` on the
  engine constant, as its own change rather than inside a merge.

The long rationale comments in `gdn_math.hpp` were left alone: long-form WHY comments are this
codebase's house style, and each one states a bit-exactness argument a reader needs.

## 7. Next

At ~0.211 s/token against the ~117 ms bandwidth floor for a bf16-resident backbone, roughly 1.8x remains.
The profile is now GDN 73, routed experts 46, QSA 32, GR 31, lm_head 21. GDN is still the largest single
phase, and its remaining cost is the serial-in-time recurrence rather than the GEMVs. The larger lever is
O5: keeping the backbone in its native GGUF quantization cuts the floor itself by about 2.4x, which no
amount of threading on the bf16 path can do.
