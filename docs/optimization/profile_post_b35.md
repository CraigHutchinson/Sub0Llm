# O0/O1 profile at the fused operating point (post-B35)

**Measured 2026-09-21**, real 48-layer BF16 artifact, `--moe-quant-dot 1`, 6 tokens, zero contention.
Temporary phase scaffold in `decode.cpp` (atomics + destructor printer, B27's pattern), reverted
immediately after — `git status` confirmed clean.

This exists because `docs/OPTIMIZATION_PROCESS.md` §5 requires re-deriving the profile after any change
over ~20%, and B35 was 2.4x. Every lever choice made between B35 landing and this measurement was aimed
at a pre-B35 map.

## The measurement

| Phase | ms/token | Share |
|---|---:|---:|
| **MoE** (router + fused quantized expert FFN + shared expert) | **975** | **68.7%** |
| Mixer (GDN / QSA / attention) | 301 | 21.2% |
| `lm_head` (248,320 × 2,560) | 77 | 5.4% |
| Gated Residual (read + write rows) | 65 | 4.6% |
| unattributed | 0.7 | 0.0% |
| **total** | **1419** | |

Attribution is essentially complete (0.0% unattributed), so these shares can be trusted as a basis for
lever selection.

## What this corrects

**I claimed, when dispatching the B34-on-B35 combination test, that B35 had shifted the binding
constraint off the MoE path and that "the remaining float reductions are now a far larger share of what
is left." Half right, and the wrong half was load-bearing.**

- Correct: the mixer's share *did* grow — it was a minor cost pre-B35 and is now 21.2%.
- Wrong: MoE is **still 68.7% dominant**, not displaced. B35 made the dominant phase 2.4x cheaper; it
  did not stop it being dominant.

That explains the −4% result directly. `--simd-reduce` targets the float reductions inside GDN/QSA and
the shared-expert gate logit — i.e. a slice that caps at ~21% of total. Even a strong win there is
bounded by that ceiling, and the multi-accumulator scaffolding's fixed per-call cost is paid across a
large number of individually small reductions. The combination was never going to pay at that ratio,
and the profile would have said so before the experiment was run. This is precisely the §5a failure:
selecting an O3/O4 lever without first re-establishing O0/O1.

## What it says to do next

1. **MoE remains the target, by a wide margin** (68.7%). B35's own named follow-up is well-aimed:
   IQ1_S is 47% of planes and is the one format whose dot lands on the wider `vpmulld` shape
   (8 lanes/multiply) rather than `vpmaddwd` (16 lanes). That is an O2/O4 change *inside the dominant
   phase*, which is the right shape per §5a.
2. **The mixer at 21.2% is genuinely untouched** and is the only other phase with real headroom. Note
   this is per-token cost for 48 layers of GDN plus 12 of QSA — worth its own O1 split (GDN vs QSA)
   before any lever is chosen, since they are different code with different characteristics.
3. **`lm_head` at 5.4% and Gated Residual at 4.6% are not worth optimizing** until the two above are
   exhausted. Note `lm_head` is only 77 ms/token despite being a 248,320 × 2,560 projection — it is
   already efficient relative to its size.
4. Re-run this profile after the next change over ~20%. The map goes stale by construction.

## Reproducing

The scaffold is deliberately not kept in the tree (it is dead weight in every build that is not
profiling). To re-derive: insert `prof::Scope` RAII timers at `gr_read_row`/`gr_write_row`, the
`do_full_attn_mixer` lambda, the GDN branch, the `USE_MOE` block, and the `lm_head` call in
`Model::forward_one`, accumulating into relaxed atomics printed by a static destructor. Roughly 40
lines; B27 and B39 used the same approach. Revert before committing and verify with `git status`.
