# M4b Two-tower model: report

**Result: the M4 done-criterion is met on Twin-2K-500 `heldout_person` validation, by a small margin.** The two-tower model (decision 0004) has a validation NLL of 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean --> averaged over five seeds (SD 0.0030 <!-- src: reports/M4b-train/summary.json#seeds.sd -->), against 0.9976 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> for the item-mean baseline B1. All five seeds beat B1. The margin is small, it is not a clear win over the LightGBM baseline B3 (0.9921 <!-- src: reports/M2-b3-heldout_person-val/metrics.json#individual.by_source.twin2k.nll -->), and all of it comes from the person tower.

Everything here is the validation partition. The test partition was never read. The acceptance criterion is unchanged from `docs/SPEC.md`, with Psych-101 excluded from the bar (decision 0001).

## What was run

- Model: frozen ModernBERT-base, vectors cached, two-tower head (`configs/twotower-a.yaml`, Twin-2K-500 only, 4000 steps of batch 1024). Psych-101 is not in this training stage.
- Tuning disclosure: two configurations were tried on validation. The first (`tt-a`, no input standardisation) reached 1.0052 <!-- src: reports/M4b-train/tt-a/metrics.json#final_val_nll -->, worse than B1. Standardising the frozen vectors (`tt-a2`) fixed it. Everything below uses the second configuration.
- Seed 0 (`tt-a2`) was trained at 50eab8a and evaluated through the unchanged harness. Seeds 1 to 4, the ablations and the folds were trained at c45e7aa (the item run at 6818abd). Seeds 1 to 4 are scored on the training-time cached path. A parity test shows the cached path equals live inference, and for seed 0 the two agree.
- Harness evaluations of the folds and held-out items ran at f8fa545 on a clean tree. Every `metrics.json` records its own git SHA and dirty flag.
- Reproduce with `scripts/run_m4b.sh` and `scripts/summarise_m4b.py`.

## Criterion: Twin-2K-500, `heldout_person` validation

| model | NLL | note |
| --- | --- | --- |
| B1 item mean | 0.9976 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | baseline to beat |
| B3 LightGBM | 0.9921 <!-- src: reports/M2-b3-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | strongest baseline |
| M4 cross-encoder | 1.0434 <!-- src: reports/M4-base-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | previous attempt, failed |
| two-tower, seed 0 (harness) | 0.9864 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | full harness evaluation |
| two-tower, seed 1 | 0.9861 <!-- src: reports/M4b-train/tt-s1/metrics.json#final_val_nll --> | cached path |
| two-tower, seed 2 | 0.9897 <!-- src: reports/M4b-train/tt-s2/metrics.json#final_val_nll --> | cached path |
| two-tower, seed 3 | 0.9910 <!-- src: reports/M4b-train/tt-s3/metrics.json#final_val_nll --> | cached path |
| two-tower, seed 4 | 0.9931 <!-- src: reports/M4b-train/tt-s4/metrics.json#final_val_nll --> | cached path |

Seed mean 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean -->. The gap to B1 is about three seed SDs, so it is real at this scale, but it is small in absolute terms. The mean is within about one seed SD of B3, so I make no claim over B3.

## Ablations (3 seeds each)

| variant | mean NLL | SD |
| --- | --- | --- |
| full model (5 seeds) | 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean --> | 0.0030 <!-- src: reports/M4b-train/summary.json#seeds.sd --> |
| no person tower | 0.9991 <!-- src: reports/M4b-train/summary.json#no_person.mean --> | 0.0001 <!-- src: reports/M4b-train/summary.json#no_person.sd --> |
| no prior-only loss | 0.9921 <!-- src: reports/M4b-train/summary.json#no_prior_loss.mean --> | 0.0010 <!-- src: reports/M4b-train/summary.json#no_prior_loss.sd --> |

Without the person tower the model is level with B1 (0.9976 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.twin2k.nll -->), so the item prior is doing what B1 does and the person tower supplies the whole improvement. The prior-only loss helps a little. The ablation seeds are 0 to 2; the full-model mean uses seeds 0 to 4. No paired bootstrap was run.

## Guards (seed 0, harness)

| check | two-tower | B1 |
| --- | --- | --- |
| ECE | 0.0040 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#individual.by_source.twin2k.ece --> | 0.0084 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.twin2k.ece --> |
| SD ratio | 0.9946 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> | 0.9955 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> |
| population TVD (lower is better) | 0.0573 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.tvd.mean --> | 0.0537 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#population.twin2k.summary.tvd.mean --> |
| population W1 (lower is better) | 0.1119 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.w1.mean --> | 0.1074 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#population.twin2k.summary.w1.mean --> |
| option-permutation max change | 0.0000002 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#robustness.permutation_max_abs_dp --> | 0.0000000 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#robustness.permutation_max_abs_dp --> |

Calibration, SD ratio and permutation invariance pass. **Population TVD and W1 are slightly worse than B1.** The model predicts individuals a little better but reproduces the overall answer distribution a little worse. For a project aimed at realistic population outcomes this matters more than the NLL gain, and the spec does not guard it.

## Held-out items (`heldout_item`, validation)

| model | NLL |
| --- | --- |
| B0, B1, B2 (no item information, identical) | 1.3649 <!-- src: reports/M4b-b1-heldout_item-val/metrics.json#individual.overall.nll --> |
| B3 LightGBM | 1.4457 <!-- src: reports/M4b-b3-heldout_item-val/metrics.json#individual.overall.nll --> |
| two-tower (one seed) | 1.3078 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.overall.nll --> |

The model beats every baseline on unseen items, so it uses question text rather than item identity. One seed only.

## Leave-one-experiment-out folds (8 folds, one seed each)

The two-tower model beats B1 on 6 <!-- src: reports/M4b-train/summary.json#folds_better_than_b1 --> of 8 folds and loses on two. Per-fold NLL difference against B1 (negative favours the model):

| fold | difference |
| --- | --- |
| f0 | 0.0058 <!-- src: reports/M4b-train/summary.json#folds.f0.delta --> |
| f1 | -0.1035 <!-- src: reports/M4b-train/summary.json#folds.f1.delta --> |
| f2 | -0.0032 <!-- src: reports/M4b-train/summary.json#folds.f2.delta --> |
| f3 | -0.1204 <!-- src: reports/M4b-train/summary.json#folds.f3.delta --> |
| f4 | -0.1530 <!-- src: reports/M4b-train/summary.json#folds.f4.delta --> |
| f5 | 0.0344 <!-- src: reports/M4b-train/summary.json#folds.f5.delta --> |
| f6 | -0.1610 <!-- src: reports/M4b-train/summary.json#folds.f6.delta --> |
| f7 | -0.1454 <!-- src: reports/M4b-train/summary.json#folds.f7.delta --> |

Each fold has 412 validation rows, and B1 degenerates to uniform on an unseen experiment, so this tests whether question text carries over to new experiments. The two losing folds are small and noisy; I draw no conclusion about them.

## Treatment effects (pooled over the 8 folds)

| metric | two-tower | B1 |
| --- | --- | --- |
| correlation with human effect sizes | 0.0248 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.correlation.mean --> | -0.1290 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.correlation.mean --> |
| sign agreement | 0.4250 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.sign_agreement.mean --> | 0.4350 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.sign_agreement.mean --> |
| false-positive rate on human-null pairs | 0.0917 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.false_positive_rate_pairs.mean --> | 0.0250 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.false_positive_rate_pairs.mean --> |

The model does not yet reproduce human treatment effects: correlation is near zero and sign agreement is at chance. This is the clearest gap against the project goal, and it is the case for the next campaign.

## Psych-101 (uninformative under decision 0001)

Seed 0 scores 1.9175 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#individual.by_source.psych101.nll --> against 2.9989 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.psych101.nll --> for B1. Psych-101 was not in this training stage, and its options are participant-specific key letters, so I do not read anything into this.

## Speed

Seed 0 serves 653 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#efficiency.decisions_per_s --> decisions per second on CPU in the harness, against 8481 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#efficiency.decisions_per_s --> for B1.

## Limitations

- The margin over B1 is small, and over B3 it is not established.
- Seeds 1 to 4 and the ablations are scored on the cached path, not the full harness. Folds and held-out items are one seed each.
- The model was tuned on this validation partition (two configurations), and the test partition has not been touched.
- Population-level TVD and W1 are slightly worse than B1, and treatment-effect fidelity is at chance.
- Stage A only: backbone frozen, Twin-2K-500 only, no history. Psych-101 and fine-tuning are untried.

## Decisions for the human

1. Whether M4 counts as met on a margin this small. My recommendation: yes for the letter of the spec, with the caveats above recorded.
2. Whether population TVD and held-out-person NLL should become guard metrics (an eval-change branch).
3. Next step: C01 (person conditioning) targets the treatment-effect gap. M5 is not started.
