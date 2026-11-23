#!/usr/bin/env bash
# Fork D, D02 A1 probe (campaigns/D02-distilled-prior/A1-probe.md): A0 (M4b) and A1 (M4b + teacher
# prior) on heldout_item_f0..f4, seeds 0 1 2, val only; plus B1 and the teacher alone per fold.
# One job at a time (16 GB machine). Run from a clean worktree so provenance is clean.
#   PROCESSED=/abs/data/processed REPORTS=/abs/reports TEACHER=/abs/teacher_prior.json \
#   SPEC=/abs/a1-deepseek-high32k.json bash scripts/fd_run_a1_folds.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${THREADS:-6} MKL_NUM_THREADS=${THREADS:-6}
PROCESSED=${PROCESSED:?} REPORTS=${REPORTS:?} TEACHER=${TEACHER:?} SPEC=${SPEC:?}
SEEDS=${SEEDS:-"0 1 2"} FOLDS=${FOLDS:-"0 1 2 3 4"}
mkdir -p runs
ev() {  # model out
  [ -f "$REPORTS/$2/metrics.json" ] && return
  uv run schelling eval --model "$1" --split "$split" --partition val --processed "$PROCESSED" \
    --out "$REPORTS/$2" --seed 0 > "runs/$2-eval.log" 2>&1 || echo "FAILED eval $2"
}
for f in $FOLDS; do
  split=heldout_item_f$f
  ev b1 "FD-a1-b1-f$f-val"
  ev "$SPEC" "FD-a1-teacher-f$f-val"
done
for s in $SEEDS; do
  for f in $FOLDS; do
    split=heldout_item_f$f
    for arm in a0 a1; do
      name=fd-$arm-f$f-s$s
      extra=()
      [ "$arm" = a1 ] && extra=(--set "model.teacher_prior_path=$TEACHER")
      if [ ! -f "runs/$name/final/config.json" ]; then
        uv run schelling train --config configs/twotower-a.yaml --set "name=$name" \
          --set "out_dir=runs/$name" --set "seed=$s" --set "split=$split" \
          --set "processed_dir=$PROCESSED" "${extra[@]}" > "runs/$name.log" 2>&1 \
          || { echo "FAILED $name"; continue; }
      fi
      ev "runs/$name/final" "FD-$arm-f$f-s$s-val"
      echo "done $name $(date +%H:%M)"
    done
  done
done
echo ALLDONE
