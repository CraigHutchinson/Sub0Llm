#!/usr/bin/env bash
# WP4b neutral-identity harness: reconfigure + rebuild + run sub0_tests at the three standard
# shapes and print the assertion count + arch-identity fingerprints for each.
# Usage: bash scripts/wp4b_check.sh [label]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS=D:/Craig/GitHub/Sub0Llm/data/tinystories.txt
LABEL="${1:-run}"

run_shape() {
  local dir="$1"; shift
  local desc="$1"; shift
  cd "$ROOT"
  cmake --build "out/build/$dir" --target sub0llm-configure > /dev/null
  "./out/build/$dir/sub0llm-configure.exe" --corpus "$CORPUS" \
      -o "out/build/$dir/generated/sub0_config.hpp" "$@" --vocab 26260 --corpus-pretok 0 > /dev/null 2>&1
  cmake --build "out/build/$dir" --target sub0_tests > /dev/null
  cd "$ROOT/out/build/$dir/tests"
  PATH="$PATH:../" ./sub0_tests.exe > "out_$LABEL.txt" 2>&1 || true
  echo "--- $desc ($dir) ---"
  grep -E "assertions in|forward: |grad:    |decode:  |FAILED|failed" "out_$LABEL.txt" | head -20
}

run_shape s1 "d96 L8 H2 seq128"      --dmodel 96  --layers 8  --heads 2 --seq 128
run_shape s2 "d132 L11 H4 kv2 seq96" --dmodel 132 --layers 11 --heads 4 --kv-heads 2 --seq 96
run_shape s3 "d196 L11 H7 seq256"    --dmodel 196 --layers 11 --heads 7 --seq 256
