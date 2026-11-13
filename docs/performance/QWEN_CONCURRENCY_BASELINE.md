# Concurrent reference baseline — CrogBuntu1, 2026-11-13

Tracking #90/#77; harness `scripts/compare_qwen_concurrency.py`. Native shared-weight kernel follow-up:
[#105](https://github.com/CraigHutchinson/Sub0Llm/issues/105). This establishes a reference service profile,
not a native Sub0Llm concurrency result or a tuned maximum.

The downloaded Qwen3.8-27B UD-IQ1_S (SHA256 `3895b6ea...`) runs on pinned llama.cpp `cf659f24`,
server build `b1-cf659f2`, CPU-only. Eight physical CPUs 0-7, 8 generation/batch workers, four server
slots, total context 2048 (512 per slot), continuous batching enabled. Same 13 raw ChatML prompt IDs
and eight generated tokens per session, greedy sampling, EOS ignored for the fixed budget, prompt
cache disabled. Seeds are explicit; this is not an MTP or speculative-decoding experiment.

One warmup per occupancy and three rotating measured rounds at 1/2/4 concurrent sessions. All 12
samples are untainted under the calibrated load/power checks. Executable and model hashes, process
start identity, worker CPU-budget checks, runner/calibrator hashes, request token counts and timing
records are in [the evidence](qwen_concurrency_2026_11_13.json). The owned local server was stopped
after capture. UI/chat-template metadata is omitted from the published copy with hashes retained.

| Active sessions | Aggregate generated tokens/s, median | Three measured rates | TTFT p50 / p95, s | Request latency p50 / p95, s |
|---|---:|---|---|---|
| 1 | 2.482 | 2.498, 2.475, 2.482 | 1.543 / 1.555 | 3.223 / 3.232 |
| 2 | 3.105 | 3.152, 3.105, 2.541 | 2.227 / 3.539 | 5.153 / 6.298 |
| 4 | 3.193 | 3.210, 2.763, 3.193 | 4.440 / 5.983 | 10.022 / 11.580 |

**Denominator:** total generated tokens divided by complete round wall time, including prompt
processing, queueing and request overhead. This is not decode-only tokens/s and must not be compared
directly with the historical 4.27 decode-only result. TTFT means first nonempty streamed content.
Three rounds describe this short profile; the spread at occupancy 2/4 is substantial and does not
establish a hardware ceiling or a broad confidence interval.

Peak server RSS observed across the process lifetime: **7.890 GiB**. RSS and PSS snapshots are
recorded per sample; the peak is not reset between occupancies, and all four slots remain configured.
The context, batch/ubatch choices, slot capacity and sampling are part of the result, not universal
recommendations. Further tuning needs its own matched, interleaved profile.

The measured aggregate gain at four sessions is about 1.29x, while p95 request latency rises sharply.
Concurrency must therefore be evaluated with latency and memory, not only request count. The next
Sub0Llm mechanism targets reuse of each packed weight/decode across session rows, with one compiled
batched path supporting occupancy one, per-session state, runtime admission and an active mask.
Native pooled inference and verified MTP remain future work; they are not claimed by this benchmark.
