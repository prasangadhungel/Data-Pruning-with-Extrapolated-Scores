"""Item 1 (mandatory) - Non-oracle KNN k-selection for score extrapolation.

Rebuttal target: reviewers Dcja (Q on non-oracle k) + bWqr; App B.4
(`\\label{Sec:Extrapolation}`) currently picks the KNN neighbourhood size ``k``
by the highest Pearson correlation of the extrapolated scores against the
GROUND-TRUTH scores ``S`` on the residual set ``D_r`` -- an ORACLE that is
unavailable at deployment. This script shows the deployable alternative:
select ``k`` using only a held-out validation split of the subset scores
``S_s`` (the ``S_s``-val rule), and reports the (small) gap to the oracle ``k``.

It is standalone -- it re-implements the distance-weighted KNN extrapolation
with ``sklearn.NearestNeighbors`` (so no ``torch_cluster`` needed) and does NOT
modify ``src/extrapolate/knn_extrapolate.py``.

Usage (real artifacts):
    python analysis/item1_knn_k_selection.py \
        --embeddings   .../embeddings/CIFAR10/embeddings_dict.pth \
        --subset_scores .../extrapolation/subset/CIFAR10_*.json \
        --full_scores  .../prune/CIFAR10_dynamic_uncertainty_0.json \
        --k_values 10 20 50 100 --val_frac 0.1 --distance euclidean \
        --out results/item1_cifar10.json

Self-test (no data / no torch):
    python analysis/item1_knn_k_selection.py --smoke
"""

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). Root on the shared store:
#   ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
#   (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
# This script needs embeddings + subset scores S_s + full scores S:
#   --embeddings    ROOT/savedir/embeddings/CIFAR10/embeddings_dict.pth
#                   dict{idx->vec}; SAVED for CIFAR10. For IMAGENET / PLACES_365 /
#                   SYNTHETIC_CIFAR100_1M the config path exists but save=false, so
#                   recompute from a proxy checkpoint (e.g.
#                   ROOT/models/CIFAR10/full_data/dual_model.pth).
#   --subset_scores ROOT/scores/extrapolation/subset/CIFAR10_dynamic_uncertainty_0.json  (= S_s)
#   --full_scores   ROOT/scores/prune/CIFAR10_dynamic_uncertainty_0.json                 (= S, eval only)
# Other datasets (swap CIFAR10 -> DS; metric = dynamic_uncertainty | tdds):
#   S    : scores/prune/{IMAGENET_dynamic_uncertainty_0_4_20, IMAGENET_tdds_0_4_26,
#          PLACES_365_dynamic_uncertainty_0, PLACES_365_tdds_0,
#          SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_0}.json
#   S_s  : scores/extrapolation/subset/{CIFAR10_tdds_0, PLACES_365_dynamic_uncertainty_0,
#          SYNTHETIC_CIFAR100_1M_dynamic_uncertainty_0}.json  (IMAGENET subset not located)
#   embeds: savedir/embeddings/{imagenet,places365,SYNTHETIC_CIFAR100_1M}/embeddings_dict.pth
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

import rebuttal_common as rc


def knn_extrapolate(
    emb: np.ndarray,
    src_idx: np.ndarray,
    src_scores: np.ndarray,
    tgt_idx: np.ndarray,
    k: int,
    distance: str = "euclidean",
    weighted: bool = True,
) -> np.ndarray:
    """Extrapolate scores for ``tgt_idx`` from ``src_idx`` via distance-weighted KNN."""
    from sklearn.neighbors import NearestNeighbors

    metric = "cosine" if distance == "cosine" else "minkowski"
    k = min(k, len(src_idx))
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(emb[src_idx])
    dist, nbr = nn.kneighbors(emb[tgt_idx])  # [T, k]
    neigh_scores = src_scores[nbr]  # [T, k]
    if weighted:
        w = np.exp(-dist)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        return (neigh_scores * w).sum(axis=1) / w_sum.ravel()
    return neigh_scores.mean(axis=1)


def select_k(
    emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    k_values: List[int],
    val_frac: float,
    distance: str,
    weighted: bool,
    seed: int,
) -> Dict:
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    fit_idx, val_idx = rc.fit_val_split(seed_idx, val_frac, rng)

    s_all = rc.scores_to_array(subset_scores, n)
    full_all = rc.scores_to_array(full_scores, n)
    fit_scores = s_all[fit_idx]

    per_k = []
    for k in k_values:
        # (a) VAL selection signal: extrapolate to the val split from fit only,
        #     compare against the KNOWN subset scores S_s on val.
        val_pred = knn_extrapolate(
            emb, fit_idx, fit_scores, val_idx, k, distance, weighted
        )
        val_corr = rc.pearson(val_pred, s_all[val_idx])

        # (b) ORACLE signal (for reporting the gap only): extrapolate to the
        #     residual D_r from the full seed set, compare against S on D_r.
        res_pred = knn_extrapolate(
            emb, seed_idx, s_all[seed_idx], residual_idx, k, distance, weighted
        )
        res_corr = rc.pearson(res_pred, full_all[residual_idx])
        res_spear = rc.spearman(res_pred, full_all[residual_idx])
        per_k.append(
            {
                "k": int(k),
                "val_pearson_Ss": val_corr,
                "residual_pearson_S": res_corr,
                "residual_spearman_S": res_spear,
            }
        )

    valid = [r for r in per_k if not np.isnan(r["val_pearson_Ss"])]
    k_val = max(valid, key=lambda r: r["val_pearson_Ss"])  # deployable choice
    k_oracle = max(per_k, key=lambda r: r["residual_pearson_S"])  # unavailable
    gap = k_oracle["residual_pearson_S"] - k_val["residual_pearson_S"]
    return {
        "n_samples": int(n),
        "n_seed": int(len(seed_idx)),
        "n_val": int(len(val_idx)),
        "distance": distance,
        "weighted": weighted,
        "per_k": per_k,
        "selected_k_val": k_val["k"],
        "selected_k_oracle": k_oracle["k"],
        "test_corr_val_selected": k_val["residual_pearson_S"],
        "test_corr_oracle_selected": k_oracle["residual_pearson_S"],
        "oracle_gap_pearson": float(gap),
    }


def _print(res: Dict) -> None:
    print(f"  samples={res['n_samples']} seed={res['n_seed']} val={res['n_val']}")
    print(f"  {'k':>6} {'val P(S_s)':>12} {'test P(S)':>12} {'test Sp(S)':>12}")
    for r in res["per_k"]:
        print(
            f"  {r['k']:>6} {r['val_pearson_Ss']:>12.4f} "
            f"{r['residual_pearson_S']:>12.4f} {r['residual_spearman_S']:>12.4f}"
        )
    print(
        f"  -> k*(val, deployable)  = {res['selected_k_val']}  "
        f"test corr = {res['test_corr_val_selected']:.4f}"
    )
    print(
        f"  -> k*(oracle, upper bd) = {res['selected_k_oracle']}  "
        f"test corr = {res['test_corr_oracle_selected']:.4f}"
    )
    print(f"  -> oracle gap (Pearson) = {res['oracle_gap_pearson']:.4f}")


def run_smoke() -> Dict:
    print("[item1] SMOKE: non-oracle KNN k-selection on synthetic data")
    emb, subset, full, _ = rc.make_fixture(n=500, d=16, subset_frac=0.3, seed=1)
    res = select_k(
        emb, subset, full, [3, 5, 10, 20, 50], 0.2, "euclidean", True, seed=1
    )
    _print(res)
    assert 0.0 <= res["oracle_gap_pearson"] < 0.5, res["oracle_gap_pearson"]
    assert res["test_corr_val_selected"] > 0.3
    print("[item1] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 1: non-oracle KNN k-selection")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--embeddings")
    ap.add_argument("--subset_scores")
    ap.add_argument("--full_scores")
    ap.add_argument("--k_values", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    if not (args.embeddings and args.subset_scores and args.full_scores):
        ap.error("--embeddings, --subset_scores, --full_scores required (or --smoke)")

    emb = rc.load_embeddings(args.embeddings)
    subset = rc.load_scores(args.subset_scores)
    full = rc.load_scores(args.full_scores)
    res = select_k(
        emb, subset, full, args.k_values, args.val_frac, args.distance,
        not args.unweighted, args.seed,
    )
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[item1] wrote {args.out}")


if __name__ == "__main__":
    main()
