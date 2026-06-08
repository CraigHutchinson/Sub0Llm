#!/usr/bin/env bash
# Interleaved A/B/A/B decode-throughput bench: sub0llm-gemma vs llama.cpp.
#
# Both engines are separate processes that reload the model, so we compare each engine's
# OWN decode-loop timer (load- and prompt-excluded): ours prints "(X tok/s)"; llama-bench
# reports tg. We ALTERNATE the two per round so thermal drift (Arrow Lake-HX throttles
# under sustained load) averages out instead of biasing whichever ran first/last.
#
# usage: bench_vs_llama.sh [model.gguf] [n_tokens] [threads] [rounds]
set -u
MODEL="${1:-models/gemma-4-12b-it-Q8_0.gguf}"
N="${2:-64}"
T="${3:-20}"
ROUNDS="${4:-5}"
LLAMA_DIR="D:/tools/llamacpp/vendor/llama.cpp-prebuilt/b9334/cpu"
export PATH="$LLAMA_DIR:$PATH"
PROMPT="The capital of France is"

ours_toks() {   # ours decode tok/s (its own timer, excludes load + prompt forward)
  ./build-native/bin/sub0llm-gemma --model "$MODEL" --mode greedy --text "$PROMPT" \
      -n "$N" -t "$T" 2>&1 | grep -oE '\(([0-9.]+) tok/s\)' | grep -oE '[0-9.]+'
}
llama_toks() {  # llama tg tok/s (its own eval timer, excludes load + prompt).
  # The tg row is "...| tg64 | 6.05 ± 0.01 |" — the t/s is the float right before "±"
  # (NOT the first float, which is the 11.78 GiB model size).
  "$LLAMA_DIR/llama-bench.exe" -m "$MODEL" -p 0 -n "$N" -t "$T" -r 1 2>/dev/null \
      | grep -iE 'tg[0-9]' | awk -F'±' '{print $1}' | grep -oE '[0-9]+\.[0-9]+' | tail -1
}

echo "interleaved bench: ours vs llama.cpp | n=$N t=$T rounds=$ROUNDS"
ours=(); llama=()
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 1 )); then a=$(ours_toks);  b=$(llama_toks)
  else                      b=$(llama_toks); a=$(ours_toks); fi   # swap order each round
  ours+=("$a"); llama+=("$b")
  printf "  round %d: ours %5s | llama %5s tok/s\n" "$r" "$a" "$b"
done

med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }
max() { printf '%s\n' "$@" | sort -n | tail -1; }
om=$(med "${ours[@]}"); lm=$(med "${llama[@]}")
ob=$(max "${ours[@]}"); lb=$(max "${llama[@]}")
echo "----"
printf "ours  : median %s  best %s tok/s\n" "$om" "$ob"
printf "llama : median %s  best %s tok/s\n" "$lm" "$lb"
awk -v o="$om" -v l="$lm" 'BEGIN{ printf "ours/llama (median): %.1f%%\n", 100*o/l }'
