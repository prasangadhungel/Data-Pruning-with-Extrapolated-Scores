"""Long-tail score-fidelity analysis (replaces the distribution-shift item).

Rebuttal target: reviewer myyu (subgroup / long-tail robustness) + RpJS W3
(oversmoothing of atypical samples). Per the user, long-tail matters more here
than corruption shift. This is the POST-HOC (no-retraining) part: on a
CIFAR-100-LT-style class-imbalanced version of the (synthetic) CIFAR-100, we ask
whether score extrapolation preserves TAIL-class importance as well as it does
head-class importance.

Metrics (computed only from score dicts + labels + the LT index set):
  * per-class Pearson/Spearman of extrapolated vs ground-truth ``S``;
  * head vs tail fidelity gap;
  * tail-vs-head RETENTION at each pruning rate (does pruning by extrapolated
    scores drop tail samples at a different rate than pruning by GT scores?).

The retrain follow-up (train on LT-pruned data, measure tail accuracy) is the
deferred Tier-C item; this script produces the evidence available without GPUs.

Self-test:
    python analysis/longtail_fidelity.py --smoke
"""

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). Root on the shared store:
#   ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
#   (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
# This script needs full scores S + an extrapolated score dict + labels.
# Prefer SYNTHETIC_CIFAR100_1M (100 classes) so head/tail is meaningful:
#   --full_scores         ROOT/scores/prune/SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_0.json  (= S)
#   --extrapolated_scores ROOT/scores/extrapolation/extrapolated/
#                         gnn_extrapolation_SYNTHETIC_CIFAR100_1M_resnet50-self-trained_k_10_seed_200000_euclidean.json
#                         (or knn_extrapolation_SYNTHETIC_CIFAR100_1M_..._k_50_seed_200000_euclidean.json)
#   --labels  SYNTHETIC_CIFAR100_1M_labels.npy  -- NOT stored with the scores. Dump
#             once from utils.dataset.prepare_data(cfg.dataset) (returns per-sample labels).
# The LT index set is built in-memory by lt_utils.make_longtail_indices(labels, imb_factor);
# no separate long-tail data file is needed. Other datasets: swap the DS name in
# scores/prune/<DS>_*.json and scores/extrapolation/extrapolated/{gnn,knn}_*<DS>_*.json.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

import rebuttal_common as rc
import lt_utils


def longtail_fidelity(
    full_scores: Dict[int, float],
    extrapolated_scores: Dict[int, float],
    labels: np.ndarray,
    lt_indices: np.ndarray,
    class_sizes: Dict[int, int],
    keep_fracs: List[float],
    tail_frac: float = 0.5,
) -> Dict:
    n = len(labels)
    full_arr = rc.scores_to_array(full_scores, n)
    ext_arr = rc.scores_to_array(extrapolated_scores, n)

    lt_set = set(lt_indices.tolist())
    head, tail = lt_utils.head_tail_split(class_sizes, tail_frac)
    head_set, tail_set = set(head), set(tail)

    # restrict fidelity to LT samples that were extrapolated (finite ext)
    def class_fidelity(cls_set):
        idx = [i for i in lt_indices
               if int(labels[i]) in cls_set and np.isfinite(ext_arr[i])]
        if len(idx) < 2:
            return {"pearson": float("nan"), "spearman": float("nan"), "n": len(idx)}
        a = ext_arr[idx]
        b = full_arr[idx]
        return {"pearson": rc.pearson(a, b), "spearman": rc.spearman(a, b),
                "n": len(idx)}

    head_fid = class_fidelity(head_set)
    tail_fid = class_fidelity(tail_set)

    # per-class fidelity table
    per_class = {}
    for c in sorted(class_sizes.keys()):
        per_class[int(c)] = {**class_fidelity({c}), "size": int(class_sizes[c])}

    # retention: among LT samples, top-k by GT vs by extrapolated, split head/tail
    lt_labels = labels[lt_indices]
    full_lt = full_arr[lt_indices]
    ext_lt = ext_arr[lt_indices]
    rows = []
    for keep in keep_fracs:
        n_keep = int(keep * len(lt_indices))
        gt_keep = set(lt_indices[np.argsort(-full_lt)[:n_keep]].tolist())
        ex_keep = set(lt_indices[np.argsort(-ext_lt)[:n_keep]].tolist())

        def tail_retention(keep_set):
            tail_in_lt = [i for i in lt_indices if int(labels[i]) in tail_set]
            if not tail_in_lt:
                return float("nan")
            return len(keep_set & set(tail_in_lt)) / len(tail_in_lt)

        rows.append({
            "keep_frac": keep,
            "prune_rate": round(1 - keep, 3),
            "tail_retention_gt": tail_retention(gt_keep),
            "tail_retention_ext": tail_retention(ex_keep),
            "tail_retention_gap": abs(tail_retention(gt_keep) - tail_retention(ex_keep)),
            "retained_overlap": rc.jaccard(gt_keep, ex_keep),
        })

    return {
        "n_lt": int(len(lt_indices)),
        "num_classes": len(class_sizes),
        "head_fidelity": head_fid,
        "tail_fidelity": tail_fid,
        "head_tail_pearson_gap": (head_fid["pearson"] - tail_fid["pearson"])
        if np.isfinite(head_fid["pearson"]) and np.isfinite(tail_fid["pearson"])
        else float("nan"),
        "retention_rows": rows,
        "per_class": per_class,
    }


def _print(res: Dict) -> None:
    print(f"  LT samples={res['n_lt']} classes={res['num_classes']}")
    print(f"  head fidelity: P={res['head_fidelity']['pearson']:.4f} "
          f"Sp={res['head_fidelity']['spearman']:.4f} n={res['head_fidelity']['n']}")
    print(f"  tail fidelity: P={res['tail_fidelity']['pearson']:.4f} "
          f"Sp={res['tail_fidelity']['spearman']:.4f} n={res['tail_fidelity']['n']}")
    print(f"  head-tail Pearson gap = {res['head_tail_pearson_gap']:.4f}")
    print(f"  {'prune':>7} {'tail_ret_gt':>12} {'tail_ret_ext':>13} {'gap':>7} {'overlap':>8}")
    for r in res["retention_rows"]:
        print(f"  {r['prune_rate']:>7} {r['tail_retention_gt']:>12.4f} "
              f"{r['tail_retention_ext']:>13.4f} {r['tail_retention_gap']:>7.4f} "
              f"{r['retained_overlap']:>8.4f}")


def run_smoke() -> Dict:
    print("[longtail] SMOKE: long-tail score fidelity on synthetic data")
    emb, subset, full, labels = rc.make_fixture(n=1500, d=16, subset_frac=0.3, seed=8)
    lt_indices, sizes = lt_utils.make_longtail_indices(labels, imb_factor=0.1, seed=8)
    # cheap 'extrapolated' scores = KNN
    from item1_knn_k_selection import knn_extrapolate
    n = emb.shape[0]
    seed_idx, res_idx = rc.seed_and_residual(subset, n)
    s_all = rc.scores_to_array(subset, n)
    pred = knn_extrapolate(emb, seed_idx, s_all[seed_idx], res_idx, 10)
    ext = {int(i): float(v) for i, v in zip(res_idx, pred)}
    for i in seed_idx:
        ext[int(i)] = float(s_all[i])
    res = longtail_fidelity(full, ext, labels, lt_indices, sizes, [0.5, 0.2, 0.1])
    _print(res)
    assert res["n_lt"] < n and res["n_lt"] > 0
    assert all(0 <= r["retained_overlap"] <= 1 for r in res["retention_rows"])
    print("[longtail] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-tail score-fidelity analysis")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full_scores")
    ap.add_argument("--extrapolated_scores")
    ap.add_argument("--labels", help="npy int labels indexed by sample id")
    ap.add_argument("--imb_factor", type=float, default=0.01)
    ap.add_argument("--tail_frac", type=float, default=0.5)
    ap.add_argument("--keep_fracs", type=float, nargs="+", default=[0.5, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
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
    lt_indices, sizes = lt_utils.make_longtail_indices(labels, args.imb_factor, args.seed)
    res = longtail_fidelity(full, ext, labels, lt_indices, sizes,
                            args.keep_fracs, args.tail_frac)
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[longtail] wrote {args.out}")


if __name__ == "__main__":
    main()
