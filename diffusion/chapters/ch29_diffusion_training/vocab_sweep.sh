#!/usr/bin/env bash
# Ch29 vocab-size sweep (sub-axis of tokenization, axis A).
# V=512 over-fragments Shakespeare (--inspect shows single-char word-starts ·f ·st ·com),
# so common words are never single tokens. Larger vocab → less fragmentation → cleaner word
# prediction + more semantic content per 64-token window, BUT each token rarer on our 2.19M-
# token corpus → embedding data-sparsity. Sweet spot unknown. Compare with WORD-LEVEL recall
# (the tokenizer-comparable metric; token recall is NOT comparable across vocab sizes).
#
# Reuses the per-token proxy as the V=512 datapoint; this runs V=1024 and V=2048 only.
# Run from the scratch binary so the canonical exe stays free for rebuilds.
set -u
cd /d/Craig/GitHub/Sub0Llm
EXE=./build-native/bin/ch29_train_long.exe
CORPUS=data/complete_shakespeare.txt
for V in 1024 2048; do
  DIR=/d/tmp/ch29_vocab_$V
  LOG=/d/tmp/ch29_vocab_$V.log
  echo "=== vocab $V → $DIR ==="
  "$EXE" --ckpt-dir "$DIR" --paragraphs 10000 --corpus "$CORPUS" \
    --vocab-size "$V" --threads 6 --eval-train > "$LOG" 2>&1
  echo "--- vocab $V done ---"
done
echo "########## VOCAB SWEEP SUMMARY (word-level recall is the comparable metric) ##########"
for V in 512 1024 2048; do
  if [ "$V" = "512" ]; then LOG=/d/tmp/ch29_ww_pertoken.log; else LOG=/d/tmp/ch29_vocab_$V.log; fi
  echo "### V=$V ($LOG) ###"
  grep -E "Denoiser: V=|overall:|breakdown:|TRAIN overall" "$LOG" 2>/dev/null
done
