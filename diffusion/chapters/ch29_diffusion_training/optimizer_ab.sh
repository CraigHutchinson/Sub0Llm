#!/usr/bin/env bash
# Ch29 optimizer A/B: Adam vs AdamW vs Muon, budget-matched (fixed 20k steps, early-stop off),
# same arch/corpus. Each optimizer at its sensible default LR (Adam/AdamW 1e-3, Muon auto 0.02).
# Run CONCURRENTLY on DISJOINT cores (the --pin feature) so the A/B finishes in ~one run's wall.
# Compare on word-level + word-START recall (the honest metrics). Tests axis F: is the model
# optimization-limited (Muon, which targets faster/lower convergence, should win if so)?
set -u
cd /d/Craig/GitHub/Sub0Llm
EXE=./build-native/bin/ch29_opt.exe
COMMON="--paragraphs 10000 --vocab-size 512 --corpus data/complete_shakespeare.txt \
  --steps 20000 --patience 1000000 --min-epochs 1000000 --threads 4 --eval-train"

$EXE $COMMON --optimizer adam  --ckpt-dir /d/tmp/ch29_opt_adam  --pin 0,1,10,11   > /d/tmp/ch29_opt_adam.log  2>&1 &
$EXE $COMMON --optimizer adamw --ckpt-dir /d/tmp/ch29_opt_adamw --pin 12,13,22,23 > /d/tmp/ch29_opt_adamw.log 2>&1 &
$EXE $COMMON --optimizer muon  --ckpt-dir /d/tmp/ch29_opt_muon  --pin E           > /d/tmp/ch29_opt_muon.log  2>&1 &
wait

echo "########## OPTIMIZER A/B SUMMARY (20k steps; word-level/START = honest metrics) ##########"
for OPT in adam adamw muon; do
  echo "### $OPT ###"
  grep -E "optimizer:|Training done|held-out NELBO \(|  overall:|breakdown:|TRAIN overall" \
    /d/tmp/ch29_opt_$OPT.log 2>/dev/null
done
