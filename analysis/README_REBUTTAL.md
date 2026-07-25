# Rebuttal analysis scripts (NeurIPS #27899)

Standalone, **artifact-driven** analyses that produce the rebuttal evidence
*without* re-running the training pipeline in `src/` (repo cleanup is deferred
until acceptance). Each script loads precomputed artifacts — embeddings
(`.pth`/`.npy`), score dicts (`{idx: float}` JSON: full `S`, subset `S_s`,
extrapolated), and (Item 2) downstream per-sample predictions — and computes
tables/metrics. Every hyperparameter is selected on an `S_s` validation split
(the non-oracle rule that fixes App B.4), so nothing here is an oracle.

## Self-test (no GPU, no torch, no data)
```bash
python analysis/run_all_smoke.py     # runs every script's --smoke synthetic test
```
Each script also has its own `--smoke` flag.

## Scripts → rebuttal items
| Script | Rebuttal item | Reviewers |
|---|---|---|
| `item1_knn_k_selection.py` | Non-oracle KNN `k`-selection (mandatory) | Dcja, bWqr |
| `item4_regression_baselines.py` | Stronger regression/interpolation baselines | RpJS W4 |
| `item8_composition.py` | Qualitative pruning composition | Dcja, RpJS W3 |
| `item3_timing_report.py` | Extrapolation-vs-DUAL wall-clock | bWqr W3/Q2 |
| `item2_behavior_preservation.py` | Error-set overlap + subgroup/tail accuracy | myyu W2 |
| `b1_calibrate_scores.py` | Ranking-preserving score calibration | bWqr/myyu/RpJS |
| `b3p1_uncertainty_error.py` | Uncertainty as error predictor | Dcja W1, RpJS W3 |
| `longtail_fidelity.py`, `lt_utils.py` | Long-tail score fidelity (replaces dist-shift) | myyu, RpJS W3 |
| `make_embeddings.py` | Regenerate proxy embeddings (not persisted on disk) | helper |
| `rebuttal_common.py` | shared loaders / metrics / fixtures | — |

## Real-run examples
```bash
# Item 1 — deployable k vs oracle-k gap
python analysis/item1_knn_k_selection.py \
  --embeddings   .../embeddings/CIFAR10/embeddings_dict.pth \
  --subset_scores .../extrapolation/subset/CIFAR10_dynamic_uncertainty_*.json \
  --full_scores  .../prune/CIFAR10_dynamic_uncertainty_0.json \
  --k_values 10 20 50 100 --out results/item1_cifar10.json

# Item 4 — baselines (+ drop-in score dicts for src/prune/prune_with_scores.py)
python analysis/item4_regression_baselines.py \
  --embeddings ... --subset_scores ... --full_scores ... \
  --out_metrics results/item4_cifar10.json --out_scores_dir results/item4_dicts/

# Item 2 — needs two per-sample prediction sets (GT-pruned vs extrapolation-pruned),
# saved as npz {sample_idx,label,pred} from loaded *_model.pth checkpoints
python analysis/item2_behavior_preservation.py \
  --pred_gt preds_gt.npz --pred_ext preds_ext.npz --out results/item2.json

# Embeddings are NOT persisted (save_embeddings: false everywhere) — regenerate
# from a proxy checkpoint. IMAGENET has both full + subset proxies:
python analysis/make_embeddings.py \
  --dataset IMAGENET \
  --ckpt .../models/imagenet/subset_data/tdds_256234.pth \
  --out  .../savedir/embeddings/imagenet/embeddings_dict.pth
```

## Deferred (Tier C — need retraining/compute)
Item 5 seed error-bars, B2 residual/GCNII GNN, B3 Part 2 uncertainty-aware
selection, Item 7 foundation embeddings, and the long-tail retraining follow-up
are specified as protocols in the rebuttal's `experiments_details.md`.
