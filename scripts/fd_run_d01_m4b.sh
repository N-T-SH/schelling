#!/usr/bin/env bash
# Fork D, campaign D01: M4b reference re-run on heldout_item at the fork SHA (seeds 0 1 2), plus B1.
# One job at a time, two threads, lowest priority: the mainline has the machine (fork README).
# PROCESSED points at the main checkout's data (read-only); the worktree has no data/.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
renice -n 19 -p $$ > /dev/null 2>&1
PROCESSED=${PROCESSED:?set PROCESSED to the main checkout data/processed}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p runs
for s in $SEEDS; do
  name=fd-m4b-item-s$s
  if [ ! -f "runs/$name/final/config.json" ]; then
    uv run schelling train --config configs/twotower-a.yaml --set "name=$name" --set "out_dir=runs/$name" \
      --set "seed=$s" --set split=heldout_item --set "processed_dir=$PROCESSED" > "runs/$name.log" 2>&1 \
      || echo "FAILED $name"
  fi
  if [ ! -f "reports/FD-m4b-item-s$s-val/metrics.json" ]; then
    uv run schelling eval --model "runs/$name/final" --split heldout_item --partition val \
      --processed "$PROCESSED" --out "reports/FD-m4b-item-s$s-val" --seed 0 > "runs/$name-eval.log" 2>&1 \
      || echo "FAILED eval $name"
  fi
done
[ -f reports/FD-b1-item-val/metrics.json ] || uv run schelling eval --model b1 --split heldout_item \
  --partition val --processed "$PROCESSED" --out reports/FD-b1-item-val --seed 0 > runs/fd-b1-item-eval.log 2>&1 \
  || echo "FAILED eval b1"
echo ALLDONE
