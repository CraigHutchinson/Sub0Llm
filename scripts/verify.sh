#!/usr/bin/env bash
# scripts/verify.sh [build-dir] -- build everything, run ctest, print ONLY the verdict.
#
# Full logs go to <build-dir>/verify_build.log and verify_ctest.log, so an agent (or a human) reads two
# summary lines instead of hundreds of build/test lines. The build dir must already be configured and
# `sub0llm-configure` already run in it (see docs/HANDOVER.md step 8; the build fails otherwise).
# JOBS (default 8) caps parallel tests: -j$(nproc) was OOM-killed on a 29 GiB host.
# Exit status: 0 = build OK and every test passed, 1 = build failed, 2 = tests failed.
set -u
B="${1:-out/build/native}"
[ -f "$B/build.ninja" ] || { echo "verify: '$B' is not a configured build directory" >&2; exit 1; }

if ! cmake --build "$B" > "$B/verify_build.log" 2>&1; then
    echo "BUILD FAILED -- first errors (full log: $B/verify_build.log):"
    grep -m5 -E "error:|FAILED" "$B/verify_build.log"
    exit 1
fi
echo "build OK ($(grep -c '^\[' "$B/verify_build.log") steps)"

# The full per-test listing stays in the log; only the summary lines below are printed.
( cd "$B" && ctest --output-on-failure -j"${JOBS:-8}" > verify_ctest.log 2>&1 )
rc=$?
grep -E "tests passed|tests failed|did not run|Skipped|\*\*\*Failed" "$B/verify_ctest.log" | cut -c1-110 | head -12
[ $rc -eq 0 ] && echo "tests OK" && exit 0
echo "TESTS FAILED (full log: $B/verify_ctest.log)"; exit 2
