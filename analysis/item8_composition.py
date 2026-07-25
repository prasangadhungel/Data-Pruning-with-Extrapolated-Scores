"""Item 8 - Qualitative pruning-composition analysis.

Rebuttal target: reviewers Dcja (qualitative pruning behavior) + RpJS W3
(oversmoothing of atypical samples). Purely from score dicts + labels (no
retraining), we compare WHAT gets pruned under the ground-truth scores ``S`` vs
the extrapolated scores, at each pruning rate:

  * retained-set overlap (Jaccard, agreement) GT vs extrapolation;
  * per-class retention counts (does extrapolation distort class balance?);
  * score-distribution shift (mean/std/quantiles of retained scores);
  * "oversmoothing" probe: how many GT tail (low-score, atypical) samples the
    extrapolation *rescues* or *drops* relative to GT.

Self-test:
    python analysis/item8_composition.py --smoke
"""

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). Root on the shared store:
#   ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
#   (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
# This script needs full scores S + an extrapolated score dict + labels:
#   --full_scores         ROOT/scores/prune/CIFAR10_dynamic_uncertainty_0.json          (= S)
#   --extrapolated_scores ROOT/scores/extrapolation/extrapolated/
#                         gnn_DU_CIFAR10_resnet50-self-trained_k_10_seed_20000_euclidean.json
#     Extrapolated dicts on disk (gnn / knn):
#       IMAGENET  : gnn__du_IMAGENET_..._k_50_seed_256234_euclidean.json,
#                   gnn__tdds_IMAGENET_..._k_10_seed_256234_euclidean_4_27.json
#       PLACES_365: gnn_extrapolation_DU_PLACES_365_..._k_10_seed_450000_euclidean.json,
#                   gnn_tdds_PLACES_365_..._k_10_seed_450000_euclidean.json,
#                   knn_extrapolation_PLACES_365_..._k_50_seed_450000_euclidean.json
#       SYNTH_100M: gnn_extrapolation_SYNTHETIC_CIFAR100_1M_..._k_10_seed_200000_euclidean.json,
#                   knn_extrapolation_SYNTHETIC_CIFAR100_1M_..._k_50_seed_200000_euclidean.json
#   --labels  <DS>_labels.npy  -- NOT stored with the scores. Dump once from
#             utils.dataset.prepare_data(cfg.dataset) (it returns per-sample labels).
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List

import numpy as np

import rebuttal_common as rc


def composition(
    full_scores: Dict[int, float],
    extrapolated_scores: Dict[int, float],
    labels: np.ndarray,
    keep_fracs: List[float],
) -> Dict:
    n = len(labels)
    full_arr = rc.scores_to_array(full_scores, n)
    ext_arr = rc.scores_to_array(extrapolated_scores, n)

    rows = []
    for keep in keep_fracs:
        gt_keep = set(rc.topk_retained(full_scores, keep))
        ex_keep = set(rc.topk_retained(extrapolated_scores, keep))
        agreement = len(gt_keep & ex_keep) / max(1, len(gt_keep))

        gt_classes = Counter(int(labels[i]) for i in gt_keep)
        ex_classes = Counter(int(labels[i]) for i in ex_keep)
        classes = sorted(set(labels.tolist()))
        # L1 distance between normalized class-retention distributions
        gt_vec = np.array([gt_classes.get(c, 0) for c in classes], dtype=float)
        ex_vec = np.array([ex_classes.get(c, 0) for c in classes], dtype=float)
        gt_vec /= gt_vec.sum() + 1e-9
        ex_vec /= ex_vec.sum() + 1e-9
        class_l1 = float(np.abs(gt_vec - ex_vec).sum())

        # oversmoothing probe: GT-dropped tail that extrapolation rescues, etc.
        gt_drop = set(range(n)) - gt_keep
        rescued = len(gt_drop & ex_keep)  # GT would drop, extrapolation keeps
        newly_dropped = len(gt_keep - ex_keep)  # GT keeps, extrapolation drops

        rows.append({
            "keep_frac": keep,
            "prune_rate": round(1 - keep, 3),
            "jaccard": rc.jaccard(gt_keep, ex_keep),
            "agreement": agreement,
            "class_balance_l1": class_l1,
            "retained_score_mean_gt": float(full_arr[list(gt_keep)].mean()),
            "retained_score_mean_ext_on_gtscale": float(full_arr[list(ex_keep)].mean()),
            "rescued_gt_dropped": rescued,
            "newly_dropped_gt_kept": newly_dropped,
        })
    return {"n": n, "num_classes": len(set(labels.tolist())), "rows": rows}


def _print(res: Dict) -> None:
    print(f"  n={res['n']} classes={res['num_classes']}")
    hdr = ("prune", "jaccard", "agree", "cls_L1", "mean_gt", "mean_ext",
           "rescued", "dropped")
    print("  " + " ".join(f"{h:>8}" for h in hdr))
    for r in res["rows"]:
        print("  " + " ".join(f"{v:>8}" for v in (
            r["prune_rate"], f"{r['jaccard']:.3f}", f"{r['agreement']:.3f}",
            f"{r['class_balance_l1']:.3f}", f"{r['retained_score_mean_gt']:.3f}",
            f"{r['retained_score_mean_ext_on_gtscale']:.3f}",
            r["rescued_gt_dropped"], r["newly_dropped_gt_kept"])))


def run_smoke() -> Dict:
    print("[item8] SMOKE: pruning-composition analysis on synthetic data")
    emb, subset, full, labels = rc.make_fixture(n=800, d=16, subset_frac=0.3, seed=5)
    # cheap 'extrapolated' = KNN on the fixture
    from item1_knn_k_selection import knn_extrapolate
    n = emb.shape[0]
    seed_idx, res_idx = rc.seed_and_residual(subset, n)
    s_all = rc.scores_to_array(subset, n)
    pred = knn_extrapolate(emb, seed_idx, s_all[seed_idx], res_idx, 10)
    ext = {int(i): float(v) for i, v in zip(res_idx, pred)}
    for i in seed_idx:
        ext[int(i)] = float(s_all[i])
    res = composition(full, ext, labels, [0.5, 0.2, 0.1])
    _print(res)
    assert all(0 <= r["jaccard"] <= 1 for r in res["rows"])
    print("[item8] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 8: pruning-composition analysis")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full_scores")
    ap.add_argument("--extrapolated_scores")
    ap.add_argument("--labels", help="npy of int labels indexed by sample id")
    ap.add_argument("--keep_fracs", type=float, nargs="+",
                    default=[0.5, 0.2, 0.1, 0.05])
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.full_scores and args.extrapolated_scores and args.labels):
        ap.error("need --full_scores --extrapolated_scores --labels (or --smoke)")

    full = rc.load_scores(args.full_scores)
    ext = rc.load_scores(args.extrapolated_scores)
    labels = np.load(args.labels)
    res = composition(full, ext, labels, args.keep_fracs)
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[item8] wrote {args.out}")


if __name__ == "__main__":
    main()
