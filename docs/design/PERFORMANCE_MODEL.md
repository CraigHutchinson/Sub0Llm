# Performance model: what limits the 27B on this machine, and what is attainable

Status: DESIGN, grounded in recorded measurements. Tracking: #77 (goal), #74, #63, #66, #70, #78.
Legend: **[V]** verified from a recorded measurement or the code; **[D]** derived arithmetic from [V] inputs;
**[U]** unknown or unverified, to be settled by a measurement named here.

Machine provenance matters (#80): every number here was measured on the AMD Ryzen 9 8945HX box (16C/32T,
Zen 4, AVX-512), Linux, eight physical cores pinned on one CCD, unless stated. Numbers from the earlier Intel
Core Ultra 9 box in older docs are not used.

## 1. Inputs

| Quantity | Value | Source |
|---|---|---|
| Encoded weight bytes read per evaluation | 5,764,050,576 B (5.37 GiB) | [V] `docs/performance/QWEN27B_SPEED_COMPARISON.md` |
| Logical projection FLOPs per evaluation | 51,247,841,280 | [V] same |
| 8-core FP32 FMA roof | 1,386.89 GFLOP/s | [V] `tools/reference/bench_host_roofs.cpp` |
| 8-core INT8 dot roof | 5,546.30 GOP/s | [V] same |
| 8-core sequential read roof (1 GiB buffer) | 32.894 GB/s | [V] same |
| 16-core / 32-thread versions of the three roofs | unknown | [U] #78 |
| Theoretical DRAM bandwidth (channels x MT/s x 8) | unknown | [U] needs DIMM speed; `dmidecode` refused without root (#78) |
| Clean llama.cpp (pinned cf659f24) | 17.03 prompt / 4.27-4.28 gen tok/s | [V] speed comparison and PR #67 table |
| Sub0Llm `baseline` kernel, 8 threads | 0.25 prompt / 0.25 gen tok/s | [V] measured 2026-11-09 (3 runs agree) |
| Sub0Llm `native-all` | 1.28 / 1.27 | [V] PR #67 table |
| + migrated IQ2_XXS module | 1.37 / 1.37 | [V] PR #67 |
| + fused IQ2_XXS (PR #67, draft) | 1.514 / 1.503 | [V] PR #67 |

The probes are descriptive ceilings of one eight-core slice, not whole-machine limits. The workload is a 13-token
prompt plus eight greedy evaluations, which is short.

## 2. Decode (generation): memory roof versus where we are

- Roof [D]: 5.764e9 B / 32.894e9 B/s = 0.1752 s per token, so **5.71 tok/s** at the 8-core probe bandwidth.
- llama.cpp at 4.27 tok/s is **74.8%** of that roof [D] (about 24.6 GB/s of weight payload).
- Sub0Llm fused-IQ2 at 1.503 tok/s is **26.3%** [D] (8.7 GB/s). The same run does 77 GFLOP/s of logical work, 5.6%
  of the FP32 roof [D].

Conclusion [D]: today's decode is **not memory-bound**. It is bound by the work per weight: dequantise each
block to float and do a float dot (`denseq::project`), which uses 5.6% of the compute roof and 26% of the
bandwidth roof. The first 3-4x is therefore kernel work, not hardware. Past llama.cpp the headroom is thin: if
the 8-core probe is the practical bandwidth roof, the most that is left above 4.27 tok/s is 5.71 (+34%). **Earlier
text in #77 that put llama.cpp at about 30% of an assumed 89.6 GB/s DRAM peak was wrong**: that peak was an
unverified guess; the measured probe is 32.9 GB/s.

What decides whether "exceed" is possible at all [U]: the bandwidth a 16-core / 32-thread read achieves (more
cores and both CCDs may raise it), and whether quantised-activation dots beat float dots per byte. Both are
measurements (#78), not design choices.

## 3. Prefill: compute roof, and why batching is the whole story

- Serial prefill streams all weights once per prompt token: 13 tokens move 13 x 5.37 GiB. With a batch of B tokens
  the weight stream is paid once per B tokens, so the bandwidth term falls by B and the work becomes compute-bound [D].
- Compute roof [D]: FP32 51.25 GFLOP / 1,386.89 GFLOP/s = **27.1 tok/s**; INT8 dot **108 tok/s** (ideal: ignores the
  dequantise cost, so a ceiling, not a target).
- llama.cpp at 17.03 tok/s is **63%** of the FP32 roof and 16% of the INT8 roof [D]. Sub0Llm at 1.51 is 5.6% of FP32.

Consequences [D]:
1. With float-activation math the ceiling is about 27 tok/s on these eight cores: reaching llama.cpp (17) needs 63%
   of peak FP32, which is hard but not impossible with a good blocked kernel; **exceeding it with float math is
   capped at about +60%**.
2. Clearly exceeding llama.cpp on prefill needs **INT8 activation dots** (AVX512-VNNI / AVX-VNNI) and/or all 16
   cores. The int8 roof is 4x the FP32 roof. Whether Zen 4's VNNI throughput is reachable after IQ-format decode is [U].
3. Batched prefill (#74) is necessary and not sufficient: it removes the bandwidth term; the kernel quality sets
   the rest. See `PREFILL_BATCHING.md`.

## 4. What this implies for the plan

| Lever | Affects | Expected size | Evidence / note |
|---|---|---|---|
| Faster per-format projection kernels (#63, #66, #70, #31-#41) | decode and prefill | decode up to about 4x before the memory roof | proven so far: 0.25 -> 1.51 tok/s |
| Batched prefill (#74) | prompt only | removes the bandwidth term; then kernel-bound | llama.cpp batches; 13.35x gap quoted in #74 |
| INT8-activation kernels for IQ formats | prefill compute ceiling, some decode | up to 4x the FP32 roof on paper | needs a design; see `QUANT_LIBRARY.md` |
| Use all 16 cores / both CCDs | both | [U] | measure bandwidth/compute scaling first (#78) |
| iGPU (#75) | prefill slice, maybe | small: 2-CU part about 0.56 TFLOPS vs the 8-core FP32 roof 1.39 TFLOPS, shared DRAM | decide by the experiment in `MEASUREMENT_PLATFORM.md` |
| Fewer bytes per token (different quant, speculation) | decode | outside the current scope | not planned |

Order of attack: kernels first (largest, proven), batching next (prompt), then the 16-core/INT8 question once
the machine test has measured it. Every result is reported **as a fraction of the measured roof** and with a
machine id, not only as tok/s.

## 5. Gates that keep speed honest (unchanged)

16-position independent F32 logit gate at relative L2 <= 0.001; identical greedy continuation; the 4,096-target
corpus qualification (#63) before any default is promoted; fresh-process medians with the shared `sub0perf`
controls and an untainted-sample rule; peak RSS gate. A speed number without its correctness record is not accepted.
