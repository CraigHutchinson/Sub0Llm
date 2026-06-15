#!/usr/bin/env bash
# Ch29 vocab sweep — BUDGET-MATCHED (resolves the confound: the naive sweep let bigger
# vocab early-stop sooner — V=512/1024/2048 trained 24030/15515/6222 steps, so "bigger
# vocab worse" tracked the step counts, not the vocab). Here ALL vocab sizes train for the
# SAME fixed step budget with early-stop OFF, so the comparison is fair. Compare on the
# tokenizer-comparable WORD-LEVEL recall. (Proxy is ~136K tokens → still over-penalizes
# large vocab via data-sparsity; if a vocab catches up here it was pure budget, if it still
# loses it's sparsity → needs a full-corpus run. Decomposes the two confounds.)
set -u
cd /d/Craig/GitHub/Sub0Llm
EXE=./build-native/bin/ch29_exp.exe   # has --pin + token cache; canonical stays free
CORPUS=data/complete_shakespeare.txt
STEPS=40000      # > V=512's natural 24030; generous equal budget for all
# Pin to the 4 P-cores the undertrain does NOT use ([0,1,10,11]) → no oversubscription.
PIN="12,13,22,23"
for V in 512 1024 2048; do
  DIR=/d/tmp/ch29_vocabbm_$V
  LOG=/d/tmp/ch29_vocabbm_$V.log
  rm -rf "$DIR"
  echo "=== budget-matched vocab $V ($STEPS steps, no early-stop, pin $PIN) → $DIR ==="
  "$EXE" --ckpt-dir "$DIR" --paragraphs 10000 --corpus "$CORPUS" \
    --vocab-size "$V" --threads 4 --pin "$PIN" --eval-train \
    --steps "$STEPS" --patience 1000000 --min-epochs 1000000 > "$LOG" 2>&1
  echo "--- vocab $V done ---"
done
echo "########## BUDGET-MATCHED VOCAB SUMMARY (all $STEPS steps; word-level = comparable) ##########"
for V in 512 1024 2048; do
  echo "### V=$V ###"
  grep -E "Denoiser: V=|Training done|overall:|breakdown:|TRAIN overall" /d/tmp/ch29_vocabbm_$V.log 2>/dev/null
done
