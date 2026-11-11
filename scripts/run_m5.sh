#!/usr/bin/env bash
# M5 paraphrase-augmentation ablation: m5-aug vs m5-base (configs/twotower-a.yaml), same seeds.
# Needs data/paraphrases/twin2k-gpt-oss-120b-v1.jsonl (scripts/make_paraphrases.py).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
renice -n 19 -p $$ > /dev/null 2>&1
SEEDS=${SEEDS:-"0 1 2"}
STAGES=${STAGES:-"1 2 3"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
train() { # variant config seed
  local v=$1 cfg=$2 s=$3 name=m5-$1-s$3
  [ -f "runs/$name/final/config.json" ] && { echo "skip $name (finished)"; return; }
  uv run schelling train --config "$cfg" --set "name=$name" --set "out_dir=runs/$name" --set "seed=$s" \
    > "runs/$name.log" 2>&1 || echo "FAILED $name"
}
export -f train
mkdir -p runs
if has 1; then
  for s in $SEEDS; do echo "aug configs/m5-aug.yaml $s"; echo "base configs/twotower-a.yaml $s"; done \
    | xargs -P 2 -L 1 bash -c 'train "$@"' _
fi
if has 2; then
  for s in $SEEDS; do for v in aug base; do
    [ -f "reports/M5-$v-s$s-val/metrics.json" ] && continue
    uv run schelling eval --model "runs/m5-$v-s$s/final" --split heldout_person --partition val \
      --out "reports/M5-$v-s$s-val" --seed 0 > "runs/m5-$v-s$s-eval.log" 2>&1 || echo "FAILED eval $v $s"
  done; done
fi
if has 3; then
  mkdir -p reports/M5-guards
  for s in $SEEDS; do for v in aug base; do
    uv run schelling guards --run "reports/M5-$v-s$s-val" --reference reports/M2-b1-heldout_person-val \
      --out "reports/M5-guards/$v-s$s.json" || echo "FAILED guards $v $s"
  done; done
fi
echo ALLDONE
