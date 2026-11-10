#!/usr/bin/env bash
# Every run behind campaign C01 idea I01 (research/campaigns/C01-person-conditioning/ideas).
# Training: I01 and an M4b re-run at the same commit, seeds 0 to 2, two jobs at a time.
# Then the unchanged harness on heldout_person val, one evaluation at a time, then guards.
# STAGES="2 3" re-runs only evaluation and guards.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
renice -n 19 -p $$ > /dev/null 2>&1
STAGES=${STAGES:-"1 2 3"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
train() { # config, name, extra --set args...
  local cfg=$1 name=$2; shift 2
  uv run schelling train --config "$cfg" --set "name=$name" --set "out_dir=runs/$name" "$@" > "runs/$name.log" 2>&1 || echo "FAILED $name"
}
export -f train
mkdir -p runs
if has 1; then {
  for s in 0 1 2; do
    echo "configs/c01-i01.yaml c01-i01-s$s --set seed=$s"
    echo "configs/twotower-a.yaml c01-m4b-s$s --set seed=$s"
  done
} | xargs -P 2 -L 1 bash -c 'train "$@"' _; fi
if has 2; then
  for s in 0 1 2; do
    for v in i01 m4b; do
      uv run schelling eval --model "runs/c01-$v-s$s/final" --split heldout_person --partition val \
        --out "reports/C01-$v-s$s-val" --seed 0 > /dev/null 2>&1 || echo "FAILED eval $v $s"
    done
  done
fi
if has 3; then
  for s in 0 1 2; do
    for v in i01 m4b; do
      uv run schelling guards --run "reports/C01-$v-s$s-val" --reference reports/M2-b1-heldout_person-val \
        --out "reports/C01-guards/$v-s$s.json" || echo "FAILED guards $v $s"
    done
  done
fi
echo ALLDONE
