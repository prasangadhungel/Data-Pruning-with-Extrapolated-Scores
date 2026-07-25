"""B3 Part 1 - Uncertainty as an error predictor (post-hoc, no retraining).

Rebuttal target: reviewer Dcja W1 ("uncertainty") + RpJS W3 ("oversmoothing of
atypical/ambiguous samples"). If a cheap per-sample uncertainty of the
extrapolated score correlates with the actual extrapolation error
``|extrapolated - S|``, then the method KNOWS where it is unreliable -- which is
exactly the hook for the (deferred) uncertainty-aware selection (B3 Part 2).

Uncertainty estimators (KNN-based, no torch needed):
  * neighbor_std   - std of the k neighbour scores (disagreement)
  * neighbor_dist  - mean distance to the k neighbours (isolation / low density)

We report, on the residual set D_r:
  * Spearman(uncertainty, |extrapolated - S|)
  * AUROC of uncertainty for detecting the top-``err_quantile`` largest errors.

Self-test:
    python analysis/b3p1_uncertainty_error.py --smoke
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

import rebuttal_common as rc


def knn_predict_with_uncertainty(
    emb: np.ndarray,
    src_idx: np.ndarray,
    src_scores: np.ndarray,
    tgt_idx: np.ndarray,
    k: int,
    distance: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (weighted prediction, neighbor_std, neighbor_mean_dist)."""
    from sklearn.neighbors import NearestNeighbors

    metric = "cosine" if distance == "cosine" else "minkowski"
    k = min(k, len(src_idx))
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(emb[src_idx])
    dist, nbr = nn.kneighbors(emb[tgt_idx])
    ns = src_scores[nbr]
    w = np.exp(-dist)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    pred = (ns * w).sum(axis=1) / w_sum.ravel()
    return pred, ns.std(axis=1), dist.mean(axis=1)


def uncertainty_error_analysis(
    emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    k: int,
    distance: str,
    err_quantile: float,
) -> Dict:
    n = emb.shape[0]
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    s_all = rc.scores_to_array(subset_scores, n)
    full = rc.scores_to_array(full_scores, n)

    pred, u_std, u_dist = knn_predict_with_uncertainty(
        emb, seed_idx, s_all[seed_idx], residual_idx, k, distance)
    err = np.abs(pred - full[residual_idx])

    thr = np.quantile(err, 1.0 - err_quantile)
    large = err >= thr

    out = {"k": int(k), "n_residual": int(len(residual_idx)),
           "err_quantile": err_quantile, "estimators": {}}
    for name, u in (("neighbor_std", u_std), ("neighbor_dist", u_dist)):
        out["estimators"][name] = {
            "spearman_unc_vs_err": rc.spearman(u, err),
            "pearson_unc_vs_err": rc.pearson(u, err),
            "auroc_large_error": rc.roc_auc(u, large),
        }
    return out


def _print(res: Dict) -> None:
    print(f"  k={res['k']} residual={res['n_residual']} "
          f"top-{res['err_quantile']:.0%} errors flagged")
    print(f"  {'estimator':>14} {'spearman':>9} {'pearson':>9} {'auroc':>9}")
    for name, m in res["estimators"].items():
        print(f"  {name:>14} {m['spearman_unc_vs_err']:>9.4f} "
              f"{m['pearson_unc_vs_err']:>9.4f} {m['auroc_large_error']:>9.4f}")


def run_smoke() -> Dict:
    print("[b3p1] SMOKE: uncertainty as error predictor on synthetic data")
    emb, subset, full, _ = rc.make_fixture(n=600, d=16, subset_frac=0.3,
                                           noise=0.5, seed=4)
    res = uncertainty_error_analysis(emb, subset, full, k=10,
                                     distance="euclidean", err_quantile=0.1)
    _print(res)
    # at least one estimator should be a usable error detector
    aurocs = [m["auroc_large_error"] for m in res["estimators"].values()]
    assert max(aurocs) > 0.55, aurocs
    print("[b3p1] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="B3 Part 1: uncertainty -> error")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--embeddings")
    ap.add_argument("--subset_scores")
    ap.add_argument("--full_scores")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--err_quantile", type=float, default=0.1)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.embeddings and args.subset_scores and args.full_scores):
        ap.error("need --embeddings --subset_scores --full_scores (or --smoke)")

    emb = rc.load_embeddings(args.embeddings)
    subset = rc.load_scores(args.subset_scores)
    full = rc.load_scores(args.full_scores)
    res = uncertainty_error_analysis(emb, subset, full, args.k, args.distance,
                                     args.err_quantile)
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[b3p1] wrote {args.out}")


if __name__ == "__main__":
    main()
