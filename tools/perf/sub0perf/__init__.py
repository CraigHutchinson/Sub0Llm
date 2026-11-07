"""Sub0Math performance tooling. `tools/perf/bench_check.py` is the command-line entry point (a thin shim over `cli`).

Modules, lowest layer first:

  common         constants, benchmark grouping, atomic file writes
  platform_proc  OS specifics: CPU pinning, priority, process launch, pid checks, CPU time counters
  load           system load sampling and waiting for a quiet machine
  stats          robust CV, calibration-scaled medians, pair ratios
  baseline       sample-record and baseline formats: build, read, write
  gitstate       git repository state: refs, HEAD, clean tree, local records, documentation-only pushes
  runner         measuring: interleaved calibrated rounds, extending rounds
  judge          verdicts: benchmarks, pairs, counters, environment controls, method mismatch
  confirm        confirmation of regressions and retry of inconclusive results
  report         the tables
  accept         recording a baseline and accepting changes (the logged `accepted` list)
  lock           one measuring run per repository
  hooks          installing the git hooks and chaining existing ones
  gate           the pre-push gate
  cli            argument parsing and the commands
"""
