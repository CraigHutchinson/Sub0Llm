# Concurrent inference: measurement and design

`scripts/compare_qwen_concurrency.py` measures an already running, local llama.cpp-compatible server. It does not start, rebuild, or configure an engine. It validates `/props` and `/slots`, and refuses to claim native pooled Sub0Llm support: the current proving ground is not a resident multi-session inference engine. Benchmark that engine only after the pooled backend exists.

The pinned llama.cpp reference is [`cf659f240b232e15700b1721bbb43c1d973c52e9`](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/tools/server/README.md#post-completion). Its raw `/completion` endpoint accepts an array of token IDs, streaming, `n_predict`, `ignore_eos`, and `cache_prompt`; final streaming results include the slot, token counts, truncation, stop reason, and timings ([result implementation](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/tools/server/server-task.cpp#L335)). Prompt token IDs avoid tokenizer or implicit BOS differences. The runner requires every requested token, uncached prompt, a limit stop, and first nonempty streamed content; refusals and incomplete streams are errors. It requests progress events and checks `prompt_progress.cache` for reused prompt tokens. The final `tokens_cached` is informational: the pinned server sets it to the slot's current prompt token count, not the reused prompt count ([source](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/tools/server/server-context.cpp#L2434)).

Create a JSON manifest (fill in actual values):

```json
{
  "engine_kind": "llama-server", "url": "http://127.0.0.1:8080",
  "engine_id": "llama.cpp pinned build", "engine_sha256": "<64 lowercase hex>",
  "model_id": "model and quantization", "model_sha256": "<64 lowercase hex>",
  "server_build": "exact /props build_info", "server_model_path": "exact /props model_path",
  "server_pid": 12345, "cpus": [0, 1, 2, 3],
  "context_per_slot": 4096, "server_cap": 4,
  "prompt_ids": [1, 2, 3], "tokens_per_request": 128,
  "seed": 42, "concurrency": [1, 2, 4], "rounds": 3,
  "timeout_seconds": 600, "calibrator": "/path/to/sub0perf-calibrator"
}
```

Use the **same actual raw prompt IDs**, token budget, per-slot context, sampling settings and hardware for every engine arm. Set each server's capacity and thread/affinity settings before the run; no automatic server manipulation occurs. The calibrator must be a compatible binary implementing `--calibration`. Run:

```sh
python scripts/compare_qwen_concurrency.py --manifest manifest.json --out out/concurrency/llama.json
python -m unittest discover -s scripts/tests -p test_qwen_concurrency.py
```

One warmup per level precedes at least three measured rounds with rotated level order. The runner uses the shared performance lock, checks idle load and trusted real power, and brackets each sample with calibration. It checks server PID start time and main-thread CPU affinity, hashes the running executable through `/proc/PID/exe` and the declared model path before and after the run, and verifies build, model path, slot count and per-slot context through the server API. Each sample records global and selected-CPU activity less the server process CPU time; above 5% foreign activity or a calibration drift above 5% marks it tainted. It retries at most three times, preserving all completed attempts in a partial JSON file and refusing final results if the gate cannot pass. It never changes host or server settings.

The file records round and per-request wall, first-content TTFT, token count, text and token hashes, server timings and aggregate generated tokens divided by round wall including prompt processing. p95 is nearest-rank. This is throughput for a fixed prompt and output budget, not decode-only tokens/s, coding quality, or proof of hardware saturation. These checks establish PID/executable and file identities. The HTTP listening socket and model mappings are not independently bound to that PID/path; worker affinity is sampled at round boundaries. The lock is cooperative; noncooperating activity may escape coarse CPU accounting or perturb memory bandwidth. Account for those limits when comparing engines.

For a native backend, bake a `MAX_SEQUENCES` capacity at configure time and use a runtime active mask within that fixed capacity. Keep per-session KV cache, GDN/recurrent state, RNG, activation and sampling scratch distinct and owned for the session lifetime. Group active decode rows so compatible sequences reuse a weight read; preserve independent state and output ordering. Quantized and float execution choices must retain their existing correctness gates and avoid per-token heap allocation. Report aggregate multi-session throughput alongside per-session TTFT and p50/p95 latency; increased aggregate tokens/s does not establish improved single-session latency.

One batched decode path should also cover occupancy one; the active mask must skip inactive slots, with no dummy computation or hot-path allocation. A memory planner should admit only the sessions that fit the actual model, context and headroom budget; compile-time capacity must not imply committing maximum context for every slot. For the current dense model, the planning estimate is about 144 MiB of GDN state per session plus 128 KiB of KV per token: at 4,096 tokens that is about 656 MiB per session. Eight such sessions plus roughly 5.4 GiB of weights exceed the 8 GiB process gate. These figures depend on the model, precision and allocation strategy and must be measured against RSS and PSS. Record admission decisions, queue delay, TTFT and p95 latency by session, and check that fairness does not degrade as occupancy rises.

MTP or different-seed candidate trees require target-model verification of proposed tokens and state before acceptance. The GDN state estimate also makes naive full-state cloning for every speculative branch potentially expensive: measure snapshot and rollback costs, use preallocated branch state or a validated structured reuse scheme, and include every copy and rejected branch in the timing and memory budget. Count accepted output tokens over the full proposal **and** verification wall time, with acceptance rate, memory and quality gates; raw branch tokens/s is not a comparable speedup. Do not call a speculative path implemented or qualified until those pieces are present and tested.

Live validation found that progress-only frames contain a placeholder token ID zero. They are excluded
from generated-token counts; a real generated token zero is counted normally. The runner also records
RSS/PSS snapshots and process-lifetime peak RSS (not a per-round reset), checks worker CPU masks before
and after each round, and hashes the runner and calibrator before and after the run.

The [first measured reference profile](../performance/QWEN_CONCURRENCY_BASELINE.md) covers 1/2/4 sessions.
The next native kernel increment is [#105](https://github.com/CraigHutchinson/Sub0Llm/issues/105):
a compiled 2-output-row x 2-session IQ1_S Q16 tile, as recommended by Astra's expert review.
