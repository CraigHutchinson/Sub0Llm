# IQ2_XXS fused float projection — preparation

Tracking #66 / #18 / #30. Status on CrogBuntu1, 2026-11-08: opt-in candidate with an 18% complete-model generation gain;
full-corpus qualification remains pending. AUTO and the live chat default remain preserved.
The existing #63 native-all qualification retains preserved executables, frozen inputs and its verified
first-window cache; capture stopped at that boundary to publish/test this increment and will resume.

## Hypothesis and implementation

The post-#62 diagnostic profile attributed 21.60% of complete decode wall time to IQ2_XXS.
Hypothesis: avoid decoded scratch stores/loads and scalar magnitude/sign expansion by converting packed
codebook groups directly into AVX2 float lanes before the existing eight-lane reduction. Activations
remain float. Finite weight values and lane accumulation order match prior native-all.

`quant/iq2_xxs.hpp` owns the format's geometry, explicit little-endian unaligned packed fields,
scalar decoder and fused float row cores. The registry and retained compatibility function forward
to that module. `Iq2XxsPlane` shares its packed fields/grid/sign/scale helpers; its integer-activation
algorithms remain the existing implementations. Scalar/generic fallback is retained.

The separate `fused-iq2` experiment enables that core for IQ2_XXS and retains native-all behavior for
other formats. The generated configurator accepts it explicitly; AUTO remains baseline pending #63.
This is a scheduling/implementation selection, not a weight-layout or model-semantic fingerprint axis.
Runtime strategy dispatch remains a proving-ground experiment override. TODO #29: after quality and
performance qualification, instantiate the generated selection with `if constexpr` in the production
engine adapter; preserve baseline experiments in the proving ground.

## Current evidence

[Raw row evidence](qwen27b_iq2xxs_row_evidence.json) pins sources, oracle executable and model/reference
identities. Against independent pinned ggml decoding, 1,494 original real rows across 11 types and
12 re-encoded real rows for four absent types pass. Decoder floats are bit-identical; fused-iq2 equals
prior native-all bit for bit for all 1,506 rows. IQ2_XXS has 177 real sampled rows, relative L2
0.000000615582 against independent double dots, within the unchanged per-format 0.00002 limit.

Independent synthetic wire/double-dot fixtures cover widths 256, 5,120 and 17,408, signed/zero/subnormal
and large finite half scales, every magnitude/sign codebook index, all scale nibbles, unaligned data,
tails and truncated/overflowing lengths. Generic, AVX2 and native builds pass; ASan/UBSan pass.
A separate Plane reconstruction check covers 22,784 groups in scalar/cached/SIMD/cached-SIMD forms.
These are short correctness checks; their elapsed times are not performance comparisons.

The cpp-review pass checked that mapped rows and input spans outlive all field/dot calls, that
Plane fields remain values, that no pointer/reference survives a resized output, and that the fused
core uses only caller-owned input/output plus register/stack temporaries. No actionable lifetime or
ownership finding remains. Compiler warnings are errors; executed row/tail paths also pass sanitizers.

The shared benchmark executable registers six new prior/fused row cases (79 total). Generic and
AVX2 list-only runs pass every correctness fixture. Its normal fixture checks
execute before listing or measuring. The extended real-row oracle additionally requires bit equality
with the prior vector reduction. All eight tooling CTest groups pass. The 18 Python script tests pass with the new chat override. The freshly rebuilt complete frontend
suite passes 220,980 assertions / 327 cases, identical to the pre-migration #60 baseline counts. The freshly rebuilt complete neutral engine suite
also passes the unchanged 44,292,784 assertions / 147 cases.

## First complete-model window and actual chat

The separate fused candidate passes every one of 128 positions in `literature-0` against the pinned
float reference (worst relative L2 0.000008736925, unchanged 0.001 limit). All captured logits, NLLs and
token traces are bit-identical to the preserved native-all candidate for that window. Peak RSS is
5,706.16 MiB, within 8 GiB. This is one window, not the full 4,096-target quality verdict.

Actual non-thinking chat returns “Hi there!” with EOS, then “Hi” when asked to repeat the first
message; replayed prompt grows from 13 to 31 tokens. A 242-token overflow is refused without changing
accepted messages/context/completion state. Reset clears state and reproduces the initial reply and
13-token context. A one-token cap produces “Hi” with `finish=length`. Both CI ISA variants pass.

## Controlled performance

[Raw evidence](qwen27b_iq2xxs_evidence.json) records commands, hashes, every warmup and measured
sample, calibration, load, power, phase arrays, generated IDs and peak RSS. Same downloaded model,
eight physical cores CPU 0–7 on one CCD, 13 pretokenized prompt IDs and eight additional greedy
evaluations (EOS ignored for fixed count). Four staged arms, one excluded warmup each and three
rotated measured rounds. All 12 measured samples are untainted; measured output IDs agree across
all arms. These are warm-cache short-workload backend rates, not long-context or chat-server rates.

| Arm | Prompt tokens/s | Generated tokens/s | Max measured RSS MiB |
|---|---:|---:|---:|
| Previous published native-all | 1.279814 | 1.274102 | 5703.14 |
| Migrated decoder, native-all | 1.372124 | 1.370793 | 5703.00 |
| Fused IQ2_XXS | 1.514203 | 1.502826 | 5703.06 |
| Clean native llama.cpp | 17.034562 | 4.270162 | 6249.79 |

Medians: 1.183× prompt / 1.180× generation over the published candidate. The decoder/module
migration contributes 1.076× generation; fusion adds 1.096× over that migrated path. Fused generation
samples span 1.489366–1.509335 tokens/s (robust CV 0.64%); prior samples span 1.250647–1.279678
(0.65%). The reference still generates 2.841× faster and processes this prompt 11.250× faster.
Memory stays inside the unchanged 8 GiB budget. No reference-parity or hardware-limit claim.

Five interleaved single-core native kernel rounds show prior/fused medians (ns per 32-row projection):

| Width | Prior vector | Fused | Ratio |
|---|---:|---:|---:|
| 256 | 2374.44 | 844.94 | 2.810× |
| 5120 | 42355.59 | 14214.00 | 2.980× |
| 17408 | 143649.55 | 48096.33 | 2.987× |

All five rounds have <=1% measured load and stable calibration. These warm-cache slices isolate
kernel cost; their ratio is not substituted for the measured complete-model gain.

Separate post-change diagnostic attribution (no foreign-load rejection): IQ1_S is 28.59% of complete
decode wall time, IQ1_M 18.50%, Q3_K 16.42%, IQ2_XS 13.87%, IQ2_S 8.58% and IQ2_XXS 5.38%.
Projection calls sum to 95.29%; the remainder stays visible. Use this ranking for the next hypothesis.

## Remaining acceptance

Draft PR #67 contains the reviewed implementation and evidence; full frozen corpus qualification
remains mandatory before closing #66/#63 or promoting defaults. Continue the unchanged fixture,
protocol, scalar baseline and reference cache with the faster candidate manifest. This selects the
candidate using measured speed without changing any quality threshold or window. Retain all prior
artifacts/cache records. Review subsequent code edits, complete generated AUTO/ISA selection and
final promoted-artifact checks, and merge accepted increments through linked PRs. Full context here
is bounded to 128; general engine integration, batching, reference parity and hardware limits remain.
