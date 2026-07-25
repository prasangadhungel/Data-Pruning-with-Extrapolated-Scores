"""Item 4 - Stronger regression / interpolation baselines for extrapolation.

Rebuttal target: reviewer RpJS W4 ("baselines too weak; wants stronger
regression/interpolation baselines"). We show that the distance-weighted KNN /
GNN used in the paper are competitive with (or better than) standard regressors
for extrapolating importance scores from ``S_s`` to ``D_r``, and that none is an
oracle: every hyperparameter is chosen on an ``S_s`` validation split (Item-1
rule), then scored on the residual set against the ground-truth ``S``.

Baselines:
  * knn_weighted   - the paper's distance-weighted KNN (reference)
  * kernel_ridge   - RBF Kernel Ridge Regression
  * svr_rbf        - RBF Support Vector Regression
  * random_forest  - Random Forest regressor
  * label_prop     - graph label propagation over a kNN graph (transductive)

Each produces an extrapolated score dict in the SAME JSON format as the repo,
so it is a drop-in for ``src/prune/prune_with_scores.py`` on the real machine.

Cheap stage only (no downstream training): report Pearson/Spearman vs S on D_r.

Self-test:
    python analysis/item4_regression_baselines.py --smoke
"""

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). Root on the shared store:
#   ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
#   (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
# This script needs embeddings + subset scores S_s + full scores S:
#   --embeddings    ROOT/savedir/embeddings/CIFAR10/embeddings_dict.pth
#                   dict{idx->vec}; SAVED for CIFAR10. For IMAGENET / PLACES_365 /
#                   SYNTHETIC_CIFAR100_1M recompute from a proxy checkpoint (e.g.
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
from typing import Callable, Dict, List, Tuple

import numpy as np

import rebuttal_common as rc
from item1_knn_k_selection import knn_extrapolate


def _fit_predict_sklearn(
    make_model: Callable,
    grid: List[dict],
    emb: np.ndarray,
    fit_idx: np.ndarray,
    fit_y: np.ndarray,
    val_idx: np.ndarray,
    val_y: np.ndarray,
    seed_idx: np.ndarray,
    seed_y: np.ndarray,
    residual_idx: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Select hyperparameters on the S_s-val split, refit on the full seed set."""
    best, best_corr = None, -np.inf
    for params in grid:
        model = make_model(**params)
        model.fit(emb[fit_idx], fit_y)
        corr = rc.pearson(model.predict(emb[val_idx]), val_y)
        if not np.isnan(corr) and corr > best_corr:
            best_corr, best = corr, params
    model = make_model(**(best or {}))
    model.fit(emb[seed_idx], seed_y)
    return model.predict(emb[residual_idx]), (best or {})


def label_propagation(
    emb: np.ndarray,
    seed_idx: np.ndarray,
    seed_y: np.ndarray,
    residual_idx: np.ndarray,
    k: int = 10,
    n_iter: int = 50,
) -> np.ndarray:
    """Transductive regression label propagation over a symmetric kNN graph."""
    from sklearn.neighbors import NearestNeighbors

    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(emb)
    dist, nbr = nn.kneighbors(emb)
    y = np.zeros(n)
    clamp = np.zeros(n, dtype=bool)
    y[seed_idx] = seed_y
    clamp[seed_idx] = True
    sigma = np.median(dist[:, 1:]) + 1e-9
    for _ in range(n_iter):
        w = np.exp(-(dist[:, 1:] ** 2) / (2 * sigma ** 2))
        w_sum = w.sum(axis=1) + 1e-12
        new_y = (y[nbr[:, 1:]] * w).sum(axis=1) / w_sum
        y = np.where(clamp, y, new_y)
    return y[residual_idx]


def evaluate_baselines(
    emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    val_frac: float,
    seed: int,
    methods: List[str],
) -> Dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR

    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    fit_idx, val_idx = rc.fit_val_split(seed_idx, val_frac, rng)
    s_all = rc.scores_to_array(subset_scores, n)
    full_all = rc.scores_to_array(full_scores, n)
    y_true = full_all[residual_idx]

    def score(pred):
        return {
            "pearson": rc.pearson(pred, y_true),
            "spearman": rc.spearman(pred, y_true),
            "mse": float(np.mean((pred - y_true) ** 2)),
        }

    results, dicts = {}, {}

    def register(name, pred):
        results[name] = score(pred)
        d = {str(int(i)): float(p) for i, p in zip(residual_idx, pred)}
        for i in seed_idx:  # keep known subset scores
            d[str(int(i))] = float(s_all[i])
        dicts[name] = d

    if "knn_weighted" in methods:
        # select k on val (deployable), then extrapolate residual
        best_k, best_c = 10, -np.inf
        for k in (5, 10, 20, 50):
            p = knn_extrapolate(emb, fit_idx, s_all[fit_idx], val_idx, k)
            c = rc.pearson(p, s_all[val_idx])
            if c > best_c:
                best_c, best_k = c, k
        register(
            "knn_weighted",
            knn_extrapolate(emb, seed_idx, s_all[seed_idx], residual_idx, best_k),
        )

    if "kernel_ridge" in methods:
        grid = [{"alpha": a, "gamma": g, "kernel": "rbf"}
                for a in (1e-2, 1e-1, 1.0) for g in (None, 0.1, 1.0)]
        pred, _ = _fit_predict_sklearn(
            KernelRidge, grid, emb, fit_idx, s_all[fit_idx], val_idx,
            s_all[val_idx], seed_idx, s_all[seed_idx], residual_idx)
        register("kernel_ridge", pred)

    if "svr_rbf" in methods:
        grid = [{"C": c, "gamma": "scale", "kernel": "rbf"} for c in (1.0, 10.0)]
        pred, _ = _fit_predict_sklearn(
            SVR, grid, emb, fit_idx, s_all[fit_idx], val_idx,
            s_all[val_idx], seed_idx, s_all[seed_idx], residual_idx)
        register("svr_rbf", pred)

    if "random_forest" in methods:
        grid = [{"n_estimators": 200, "max_depth": md, "random_state": seed}
                for md in (None, 8, 16)]
        pred, _ = _fit_predict_sklearn(
            RandomForestRegressor, grid, emb, fit_idx, s_all[fit_idx], val_idx,
            s_all[val_idx], seed_idx, s_all[seed_idx], residual_idx)
        register("random_forest", pred)

    if "label_prop" in methods:
        register(
            "label_prop",
            label_propagation(emb, seed_idx, s_all[seed_idx], residual_idx),
        )

    return {"n": int(n), "n_seed": int(len(seed_idx)),
            "metrics": results, "score_dicts": dicts}


ALL_METHODS = ["knn_weighted", "kernel_ridge", "svr_rbf", "random_forest", "label_prop"]


def _print(res: Dict) -> None:
    print(f"  samples={res['n']} seed={res['n_seed']}")
    print(f"  {'method':>15} {'pearson':>9} {'spearman':>9} {'mse':>9}")
    for name, m in sorted(res["metrics"].items(), key=lambda kv: -kv[1]["pearson"]):
        print(f"  {name:>15} {m['pearson']:>9.4f} {m['spearman']:>9.4f} {m['mse']:>9.4f}")


def run_smoke() -> Dict:
    print("[item4] SMOKE: regression/interpolation baselines on synthetic data")
    emb, subset, full, _ = rc.make_fixture(n=500, d=16, subset_frac=0.3, seed=2)
    res = evaluate_baselines(emb, subset, full, 0.2, seed=2, methods=ALL_METHODS)
    _print(res)
    assert res["metrics"]["knn_weighted"]["pearson"] > 0.3
    assert len(res["score_dicts"]["kernel_ridge"]) == res["n"]
    print("[item4] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 4: stronger regression baselines")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--embeddings")
    ap.add_argument("--subset_scores")
    ap.add_argument("--full_scores")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_metrics")
    ap.add_argument("--out_scores_dir",
                    help="dir to dump one JSON score dict per method (drop-in for prune)")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.embeddings and args.subset_scores and args.full_scores):
        ap.error("--embeddings, --subset_scores, --full_scores required (or --smoke)")

    emb = rc.load_embeddings(args.embeddings)
    subset = rc.load_scores(args.subset_scores)
    full = rc.load_scores(args.full_scores)
    res = evaluate_baselines(emb, subset, full, args.val_frac, args.seed, args.methods)
    _print(res)
    if args.out_metrics:
        rc.save_json({"n": res["n"], "n_seed": res["n_seed"],
                      "metrics": res["metrics"]}, args.out_metrics)
        print(f"[item4] wrote {args.out_metrics}")
    if args.out_scores_dir:
        import os
        for name, d in res["score_dicts"].items():
            p = os.path.join(args.out_scores_dir, f"baseline_{name}.json")
            rc.save_json(d, p)
        print(f"[item4] wrote score dicts to {args.out_scores_dir}")


if __name__ == "__main__":
    main()
