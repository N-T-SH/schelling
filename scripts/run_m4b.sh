#!/usr/bin/env bash
# Every training and evaluation run behind research/reports/M4b.md. Run from a clean checkout.
# Training uses configs/twotower-a.yaml; only seed, ablation flags and the split change.
set -u
cd "$(dirname "$0")/.."
CFG=configs/twotower-a.yaml
train() { # name, extra --set args...
  local name=$1; shift
  uv run schelling train --config "$CFG" --set "name=$name" --set "out_dir=runs/$name" "$@" > "runs/$name.log" 2>&1 || echo "FAILED $name"
}
export -f train; export CFG
mkdir -p runs
# Stage 1: training (3 at a time)
{
  for s in 1 2 3 4; do echo "tt-s$s --set seed=$s"; done
  for s in 0 1 2; do echo "tt-np-s$s --set seed=$s --set model.use_person=false"; done
  for s in 0 1 2; do echo "tt-nopr-s$s --set seed=$s --set model.prior_weight=0"; done
  for k in 0 1 2 3 4 5 6 7; do echo "tt-f$k --set split=heldout_experiment_f$k"; done
  echo "tt-item --set split=heldout_item"
} | xargs -P 3 -L 1 bash -c 'train $0 "$@"'
# Stage 2: evaluation through the unchanged harness
{
  for k in 0 1 2 3 4 5 6 7; do echo "uv run schelling eval --model runs/tt-f$k/final --split heldout_experiment_f$k --partition val --out reports/M4b-tt-f$k-val --seed 0"; done
  echo "uv run schelling eval --model runs/tt-item/final --split heldout_item --partition val --out reports/M4b-tt-item-val --seed 0"
  for m in b0 b1 b2 b3; do echo "uv run schelling eval --model $m --split heldout_item --partition val --out reports/M4b-$m-heldout_item-val --seed 0"; done
} | xargs -P 3 -I{} bash -c '{} > /dev/null 2>&1 || echo "FAILED: {}"'
# Stage 3: pooled treatment effects over the eight leave-one-experiment-out folds
runs=""; for k in 0 1 2 3 4 5 6 7; do runs="$runs --run reports/M4b-tt-f$k-val"; done
uv run schelling effects-pool $runs --out reports/M4b-pooled-tt/effects.json
echo ALLDONE
