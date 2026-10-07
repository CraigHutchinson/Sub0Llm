# O14 -- the parameter arena as a read-only file view (O13 step C1)

Status: **implemented, default OFF (`--param-arena auto` resolves to heap), 2026-10-07.** The primary agent
re-measures and decides the auto default.

## What it does

`sub0llm-configure --param-arena {auto,heap,mapped}` bakes `constexpr bool PARAM_ARENA_MAPPED`. With
`mapped`, `load_model` validates the header, dtype and file size exactly as before, then makes the parameter
arena a read-only `FileMap` view of the model file at `sizeof(Header)` instead of reading the blob into a
heap arena. Cold pages are clean file-backed pages, so the OS drops them for free and re-faults the ~2% decode
reads. No file format, sidecar, cache or kernel changed.

Classification (AGENTS.md S10.3): storage placement only. `PARAM_FLOATS`, the `Header` and every computed
value are unchanged, so it joins neither `ARCH_FINGERPRINT` nor `ARCH_FINGERPRINT2`.

Legal only for a forward-only build (`--hc-count >= 2`, `--num-experts >= 2` or QSA on) on `--compute 0`;
the configurator refuses anything else and names the reason. The backend refuses again at its own seam
(`adopt_param_file_view` returns false in a build that can train), and `load_model` then falls back to the read.

## Design

- One owner type, `ParamArena { unique_ptr heap; FileMap map; param_t* base; }` in `backend.cpp`. Every reader
  (`mk_param`, `param_store_ptr`, `param_store_view`, the residency report) goes through `base`.
- **Pointer capture.** `mk_param` bakes `base + off` into the calling thread's parameter Nodes, and
  `build_layout` is idempotent and rewrites them in place. So moving the arena = set `base`, run
  `build_layout()` again on the calling thread, then free the old owner. If any OTHER thread has built a graph
  its Nodes cannot be reached from here: adoption is refused (heap read with a note), and a later writer
  aborts with a message rather than leave a dangling pointer. The order "load before the first graph" (the
  qwen4 tools) never allocates a heap arena at all; "build_model then load_model" (gen/eval stages) frees the
  heap arena after the rebind.
- **Writers.** A read-only view faults on write, so every writer converts the arena to a heap copy first
  (`make_param_arena_writable`, copies the bytes, rebinds, unmaps): `param_store_ptr()` (the only writable raw
  accessor, used by `load_model`'s fallback), `params_ptr()` (f32 view; tests, train, checkpoints), and
  `build_model()` (random init). `save_model` uses the new read-only `param_store_view()`, so saving never
  un-maps. `param_write_ptr` (init) aborts if it ever sees a mapped arena. Not reachable with a mapped arena:
  AdamW/`param_master_f32`, `grad_ptr`/`adam_*_ptr`, `train_stage` checkpoint code, device upload/download
  hooks (all need a trainable build or a device backend, both refused at configure time).
  One consequence: `save_model` to the very path that is mapped fails (Windows holds the file).
- **Alignment.** `sizeof(Header)` is 48, the map base is page-aligned, so every element type is aligned
  (static_assert in `load_model`). No kernel uses an aligned load on `pdata` (`gemv.hpp`'s only aligned object
  is its own stack tile).
- **Report.** The end-of-run residency line (cache builds) now says `parameters (heap|mapped)`.

## Measurements (real 48-layer model, 2000 G-PPL tokens, ppl_blend_v2)

- Bit-exact: perplexity 11.9443 in both arms, paired dNLL +0.0000, per-token dumps byte-identical (SHA-256
  equal). Same under the regime-2 cache arms.
- Idle host, `load_model`: heap 27.8 / 30.3 s, mapped 18.7 / 20.8 s (warm file). Cold-evicted blob: heap 29.2 /
  35.4 s, mapped 29.9 / 21.4 s (the pairing-identity hash reads the file either way; inconclusive).
- Peak working set, reactive arms: 44.3 GiB heap, 43.0 GiB mapped (2000-token run); 60-token run, after load
  12.47 vs 3.32 GiB, final 29.00 vs 20.03 GiB.
- End-of-run residency (cache build): mapped parameters 2.0% of 9.16 GiB resident, matching the 2% decode reads.
- First touch: no prefault added. A 60-token run is 7.4-8.3 tok/s mapped against 7.6-8.0 heap (noise-sized);
  the cost is ~47k soft faults, ~0.2 s once, against a 9 GiB read at load. Prefaulting the hot set needs a
  tensor-level hot list, which is new surface with no measured need.
- Regime 2 (`--ballast-room 10`), scoring run / one speed round:

| Arm | tok/s (scoring, speed) | hit rate | miss waits | peak WS |
|---|---|---:|---:|---:|
| heap17 | 7.82, 7.87 | 97.2% | 8417 ms | 32.1 / 35.0 GiB |
| mapped17 | 7.86, 7.54 | 97.2% | 8315 ms | 27.7 GiB |
| mapped26 | 7.95, 6.59 | 98.6% | 5256 ms | 36.3 / 36.7 GiB |

  The speed rounds ran while sibling builds were active (the suite logged competing processes), one round
  each, so single numbers are not comparable; mapped26's waits fell by 37% as the trace replay predicts, but
  that is ~3 s of a ~260 s run and no tok/s gain is demonstrated here.

## Tests

`tests/param_arena_tests.cpp` (own target `sub0_param_arena_tests`, so `sub0_tests` counts are untouched):
refusal in a trainable build; in a forward-only build, adoption after nodes exist gives byte-identical arenas
and bitwise-identical logits, writers convert back, `build_model` after a view re-randomizes on the heap,
`save_model` from a view does not un-map; `load_model` follows `PARAM_ARENA_MAPPED`.

## Unverified

POSIX `mmap` path (compiles through `FileMap`, not run here). Load-before-first-graph is covered by the real
tool only, not by a unit test (the test process always has nodes by then).
