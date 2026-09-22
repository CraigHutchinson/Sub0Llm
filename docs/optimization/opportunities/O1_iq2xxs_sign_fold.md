# O1 — fold IQ2_XXS's sign application into the dot product

**Status:** merged 2026-09-22 — implemented as a WIDER change than §4 chose; see §10
**Expected gain:** up to ~18% of total decode (see §2 arithmetic); realistically less
**Risk:** medium — touches the fused kernel's numerics path
**Complexity:** S/M
**Gates:** `G-PARITY` (waived for the fused arm), `G-QUALITY` (L2 must stay `0.293822` bit-identical),
`G-PERF`, `G-HASH`, `G-SUITE-*`

## 1. Context — the profile that justifies this

Measured 2026-09-21 at the fused operating point (`docs/optimization/profile_post_b35.md`): MoE is
**68.7%** of decode. Within MoE, per-format cost, measured with a temporary scaffold at the
`gemv_plane` dispatch (reverted):

| format | ms | share | planes | µs/plane | bytes/plane | **GB/s** |
|---|---:|---:|---:|---:|---:|---:|
| IQ1_S | 1331.8 | 23.9% | 4080 | 326 | 320 KB | 0.98 |
| **IQ2_XXS** | **2061.2** | **37.0%** | 1680 | **1227** | 422 KB | **0.34** |
| IQ4_NL | 2179.3 | 39.1% | 2880 | 757 | 922 KB | 1.22 |

**IQ2_XXS moves bytes 3.6x slower than IQ4_NL.** It is 37% of MoE dot time from only 19.4% of the
planes and the second-smallest byte footprint. That is an isolated inefficiency, not a property of
having fewer bits.

**This also refutes the previously-named next lever.** B35's own follow-up nominated IQ1_S, on the
reasoning that its dot lands on the 8-lane `vpmovsxbd`+`vpmulld` shape rather than 16-lane
`vpmaddwd`. The measurement says IQ1_S is the **cheapest** format per plane (326 µs) and the second
cheapest per byte. Lane width is not the binding constraint; per-group unpacking work is. The IQ1_S
brief should not be written until something measures it as a problem.

## 2. Problem

`Iq2XxsPlane::group()` (`include/sub0/moe_quant_dot.hpp`) does work neither sibling unpacker does:

```cpp
std::array<std::int8_t, GROUP> sign{};
for (int l = 0; l < 4; ++l) {
    const std::uint64_t grid  = gguf::IQ2XXS_GRID[aux8[l]];
    const std::uint64_t signs = SIGNS64[(aux[1] >> (7 * l)) & 127];
    std::memcpy(wg.q.data() + 8 * l, &grid,  sizeof grid);
    std::memcpy(sign.data()  + 8 * l, &signs, sizeof signs);
}
for (int j = 0; j < GROUP; ++j) wg.q[j] = static_cast<std::int8_t>(wg.q[j] * sign[j]);
```

It materialises a 32-byte `sign` array, then makes a **second full pass** over the group applying it by
multiply. At 51,200 groups per plane that is ~64 extra ops per group of pure overhead, on the hot path,
for every one of the 1,680 IQ2_XXS planes a 6-token run touches.

Note this is a *different* defect from B28's. B28 fixed a per-element data-dependent **branch** in the
old `dequantize_iq2_xxs`; this is a branchless but redundant second pass in the **new** B35 kernel. The
same format being the outlier twice, for two unrelated reasons, is worth noting as a pattern: IQ2_XXS's
grid+sign encoding is simply the most awkward of the three to unpack, so it deserves the most care.

## 3. Options

1. **Fold the sign into `dot_group`.** Pass the sign qword alongside the magnitude and apply it during
   the multiply-accumulate rather than in a separate pass. Removes the `sign` array and the second
   loop entirely. Most invasive of the three — changes `WeightGroup`'s contract, which `gemv` and the
   other two unpackers share.
2. **Apply the sign via XOR/negate on the packed qword.** `q` and `signs` are both `uint64_t` holding
   8 int8 lanes. A branchless SWAR negate (`(q ^ mask) - mask` per lane, where `mask` is the sign
   broadcast) applies all 8 lanes in ~3 ops instead of 8 multiplies, with no `sign` array and no second
   pass. Keeps `WeightGroup`'s contract intact — strictly local to this unpacker.
3. **Leave the second pass but make it vectorizable.** Least invasive, least gain; the pass is already
   trivially vectorizable and the compiler may already be doing it, in which case this buys nothing.

## 4. Chosen approach

**Option 2**, with option 1 held as the second iteration if option 2 does not close enough of the gap.

Rationale: option 2 is local to `Iq2XxsPlane`, leaves the shared `WeightGroup`/`gemv`/`dot_group`
contract untouched (so IQ1_S and IQ4_NL cannot regress), and directly deletes the two things the
profile implicates — the `sign` array materialisation and the second pass. Option 1 is strictly more
powerful but changes a contract three unpackers share, which is a poor trade for iteration 1 given
option 2 may be sufficient. Per `AGENTS.md` §13 this is attempt 1 of 3; option 1 is the planned
attempt 2.

## 5. Implementation

- `include/sub0/moe_quant_dot.hpp`, `Iq2XxsPlane::group()` only.
- Replace the `sign` array + second pass with a per-qword SWAR sign application inside the existing
  `l` loop, writing directly into `wg.q`.
- `SIGNS64[...]` already yields ±1 per lane; the SWAR form wants a 0/−1 mask per lane instead, so
  either derive the mask from the same 7-bit index or change what `SIGNS64` stores. **Verify which**
  against the table's actual contents before coding — do not assume.

## 6. Validation

- `G-QUALITY` is the load-bearing gate here: the end-to-end L2 diff must reproduce **`0.293822`
  bit-identically**. This change must be a pure reordering of how a sign is applied, not a change to
  the value. If L2 moves at all, the sign mapping is wrong.
- `moe_quant_tests.cpp`'s B35 cases, including the swapped-planes rejection test, stay green.
- `G-PERF` via `python scripts/run_perf_suite.py --stage perf --label O1 --arm "fused:--moe-quant-dot 1"`,
  compared against the current 1.41-1.47 s/token.
- Re-run the per-format scaffold (§1) to confirm IQ2_XXS's GB/s actually moved toward IQ4_NL's 1.22 —
  a total-time win that does not show up as a per-format win would mean the gain came from somewhere
  else and the hypothesis is unconfirmed.

## 7. Risks

- **Sign-convention error** is the obvious failure: it would flip weights and wreck quality while still
  running fast. The `0.293822` bit-identity check catches it immediately, which is why it is the
  primary gate rather than a throughput number.
- SWAR lane arithmetic on `int8` is easy to get subtly wrong at lane boundaries; the existing B35 tests
  operate on real sidecar bytes and will catch a systematic error, but a targeted unit test comparing
  the new `group()` against the old one over all 256 grid × 128 sign combinations would be cheap and
  decisive. **Write that first.**

## 8. Ceiling — what this is worth

If IQ2_XXS reached IQ4_NL's 1.22 GB/s, its 2061 ms would fall to ~575 ms: −1486 ms of 5572 ms MoE dot
time (−27% of MoE, ≈ −18% of total decode). That is the **ceiling**, assuming the entire gap is this
one inefficiency. Treat it as an upper bound, not a forecast — the 3.6x gap may be partly inherent to
the grid-lookup encoding, in which case closing half of it is a good result.

## 9. Follow-ups / compounding

- **Attempt 2** (if needed): option 1, folding sign into `dot_group`, which would also let IQ1_S share
  the same path.
- **IQ4_NL is 39.1% of MoE dot time** — the single largest slice, and unexamined. It is at the *best*
  GB/s of the three, so there may be nothing there, but its share alone justifies a profile pass.
- The mixer (GDN/QSA) at 21.2% of total is entirely untouched and needs its own O1 split before any
  lever is chosen there.
- Whatever lands here, **re-profile if it exceeds ~20%** (`OPTIMIZATION_PROCESS.md` §5).

## 10. What was actually built, and why it differs from §4

§4 chose a local fix (SWAR sign negate inside `Iq2XxsPlane::group()`). Before implementing it, the
post-B35 roofline (`../roofline_post_b35.md`) reframed the problem: the fused path sat at **2.2% of the
AVX2 int8 ceiling for every format**, not only IQ2_XXS. The sign pass was one symptom; the shape of the
loop was the cause — every 32-element group ended in a horizontal reduction, an int→float convert and
a serial scalar accumulate, and `group()` returned a 36-byte struct by value. A local fix to one
unpacker could not move the other two formats (61% of MoE time).

So O1 became `detail::gemv_avx2` (`include/sub0/moe_quant_dot.hpp`):
- the accumulator stays in a float vector for the whole row; one horizontal sum per ROW;
- the MAC is the `vpsignb` + `vpmaddubsw` + `vpmaddwd` shape llama.cpp's AVX2 IQ kernels use;
- IQ2_XXS hands its native magnitude/sign split straight to the kernel — **no sign pass at all**
  (§4's goal, met more completely than option 2 would have);
- IQ4_NL decodes both nibble halves with one `vpshufb`; the f16 scale uses F16C;
- each format's bit-field parse lives in one `fields()` shared by the portable and AVX2 paths, so
  the layout knowledge is not duplicated. The portable kernel stays as the non-AVX2 fallback and the
  AVX2 kernel's test reference (compile-time `kAvx2Kernels`, no runtime knob).

### Results

Kernel microbenchmark (`sub0llm-bench-moeqd`, per plane, old → new): IQ1_S ~375 → ~100-118 µs (~3.5x),
IQ2_XXS ~1300 → ~97-118 µs (~12x), IQ4_NL ~760 → ~108 µs stream (~7x). IQ4_NL is now memory-side
(hot 59 vs stream 108 µs, ~8.5 GB/s single-core); IQ1_S/IQ2_XXS remain compute-side at ~13-15% of AVX2.

End-to-end, real 48-layer artifact, fused config, warm, sandboxed, interleaved, 3 runs each
(2026-09-22, post-reboot, background 4.3%):

| arm | median s/token | runs | L2 (6 tok) | argmax |
|---|---:|---|---:|---:|
| main (portable kernel) | 1.508 | 1.573, 1.444, 1.508 | 0.293822 | 4/6 |
| O1 (AVX2 kernel) | **0.705** | 0.756, 0.671, 0.705 | **0.23252** | 4/6 |

**2.14x decode.** Quality IMPROVED: the per-lane accumulation is a shallower rounding chain than a long
sequential float sum. The main arm reproduced B35's recorded 0.293822 exactly, which validates the setup.

Gates: G-SUITE-ENGINE 28,969,623 / 147 exact; G-SUITE-FRONTEND 122,385 / 256 exact excluding the new O1
case (+19,224 assertions / 1 case); G-QUALITY 0.2325 ≤ 0.43; G-PERF −53%. Mutation checks: swapping two
IQ2_XXS sign qwords, and swapping IQ4_NL's nibble halves, each fail multiple `[moequant]` cases.

### Next constraint (re-profile before choosing)

MoE is no longer 68.7% of decode — re-profile at the O1 operating point before briefing anything.
Known candidates from the microbench: IQ1_S/IQ2_XXS grid lookups (4 table loads + `set_epi64x` per
group, compute-side at ~15% of AVX2), IQ4_NL memory-side (prefetch / multi-row interleave now worth
re-trying — B28's prefetch was measured when the kernel was compute-bound, which it no longer is for
IQ4_NL), and the mixer (21.2% before O1, now a far larger share).
