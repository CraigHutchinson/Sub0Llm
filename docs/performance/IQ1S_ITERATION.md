# IQ1_S packed float-dot iteration

Tracking #60, parent #18/#30. The two-run profile in #58 attributes ~71% of dense 27B evaluation wall
time to IQ1_S projections. Hypothesis: fusing packed decode with the float dot, then accumulating
vectors, can remove scratch traffic and the serial accumulation dependency without quantizing activations.

The new IQ1_S module owns geometry, packed fields and decode. The legacy GGUF decoder and MoE Plane
forward to that shared core. Dense projection retains its original baseline and adds fused scalar and
AVX2 float candidates. On a target without AVX2, native falls back to fused scalar. Whole-block row,
span and scratch validation remain common to all paths. No allocation is added inside the row core.

`--iq1s-kernel baseline|scalar|native` is a temporary proving-ground evaluation knob, default baseline.
TODO(#60): after the independent quality/memory and controlled speed gates justify promotion, select
the supported proven kernel through the generated configuration; preserve an explicit baseline for
future experiments. Production chat artifacts are preserved during evaluation.

Reference layout was checked against the clean pinned llama.cpp `cf659f240b232e15700b1721bbb43c1d973c52e9`,
`ggml/src/ggml-quants.c::dequantize_row_iq1_s` (line 2666). Each 256-element block has an F16 scale,
32 low grid-index bytes and eight 16-bit group descriptors. Each group describes 32 contiguous input
columns; its four grid entries decode eight signed bytes each. Scale and delta equations, row-contiguous
GGUF output layout, and the 0..2047 grid indices are retained. SIMD changes the reduction order, not
activation precision or weight values.

## Verification before timing

- Generic and native benchmark fixtures compare all three kernels with an independently written
  packed-byte decoder and double-precision dot at 256, 5120 and 17408 input widths, 32 output rows,
  unaligned packed/input offsets, full codebook coverage across the fixtures, group signs/scales,
  zero/subnormal/normal/negative F16 scales, partial dequantization and truncated/partial-block refusal.
  Fixture vector relative L2 limit is 0.00002. Near-zero components can exhibit cancellation error
  even in the original scalar baseline; the vector metric avoids dividing by those individual values.
  The production logit gate remains 0.001.
- Independent unmodified ggml matches 1,494 sampled real-model blocks with zero differing values.
- `check_iq1s_dots.cpp` checks 792 real IQ1_S rows with ggml decoding plus double dots; all kernels pass.
- Fresh frontend suite: 220,980 assertions / 327 test cases pass. The complete engine suite passes
  44,292,784 assertions / 147 cases with a generated tokenizer and realistic 16,794-token vocabulary.
  An initial external-vocabulary engine build failed two tokenizer-dependent cases because that mode
  emits no tokenizer; the corpus-backed configuration resolves those failures without test changes.
  Seven performance CTest checks pass.
  The earlier local frontend executable was stale (203,457 / 317); it was rebuilt before this result.
- ASan/UBSan native fixture checks pass on the host. LeakSanitizer requires running outside the sandbox’s
  ptrace boundary; the successful host run retained leak checking.
- Migrated baseline is bit-identical to the prior native binary at all sixteen full-model positions.
  SIMD passes the unchanged independent gate (worst relative L2 0.0000794629).
  Fused scalar is bit-identical to the migrated baseline at all sixteen positions.

Timing results will be appended after staged, interleaved runs. This note does not yet claim a speed win
or establish full-window quality beyond the tested positions.
