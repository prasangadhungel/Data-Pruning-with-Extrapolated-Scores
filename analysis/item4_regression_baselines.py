"""Item 4 - Stronger regression / interpolation baselines for extrapolation.

Rebuttal target: reviewer RpJS W4 ("baselines too weak; wants stronger
regression/interpolation baselines"). We show that the distance-weighted KNN /
GNN used in the paper are competitive with (or better than) standard regressors
for extrapolating importance scores from ``S_s`` to ``D_r``, and that none is an
oracle: every hyperparameter is chosen on an ``S_s`` validation split (Item-1
rule), then scored on the residual set against the ground-truth ``S``.

Baselines:
  * knn_weighted   - the paper's distance-weighted KNN (reference)
  * knn_mean       - unweighted KNN (ablation: does the distance weighting matter?)
  * ridge          - plain linear Ridge on the embeddings (cheapest floor, O(N d))
  * kernel_ridge   - RBF Kernel Ridge Regression (Nystrom-approx when large)
  * svr_rbf        - RBF Support Vector Regression (RBFSampler+LinearSVR when large)
  * hist_gbr       - Histogram Gradient Boosting regressor (lean, near-linear)
  * random_forest  - Random Forest regressor
  * label_prop     - graph label propagation over a kNN graph (transductive)

Each produces an extrapolated score dict in the SAME JSON format as the repo,
so it is a drop-in for ``src/prune/prune_with_scores.py`` on the real machine.

Cheap stage only (no downstream training): report Pearson/Spearman vs S on D_r.

Memory-lean for ImageNet-1M:
  * embeddings loaded as float32 (halves the [N, d] footprint);
  * kernel_ridge / svr_rbf auto-switch from exact RBF (small data) to a
    linear-memory approximation (Nystrom+Ridge / RBFSampler+LinearSVR) once the
    training set exceeds ``--exact_max`` -> no dense N_train x N_train kernel;
  * heavy regressors train on a random subsample capped at ``--max_train``;
  * all predictions run in ``--pred_chunk`` batches (no huge residual x train
    intermediate);
  * label_prop builds its kNN graph in memory-bounded chunks and is skipped
    above ``--label_prop_max`` unless ``--force_label_prop`` (brute O(N^2)).

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



ROOT = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning"
SUBSET_SCORES_PATH = f"{ROOT}/scores/extrapolation/subset/IMAGENET_dynamic_uncertainty_0_256234_4_19.json"
SUBSET_FULL_SCORES_PATH = f"{ROOT}/scores/extrapolation/extrapolated/gnn__du_IMAGENET_resnet18-self-trained_k_20_seed_256234_euclidean.json"
SUBSET_KNN_SCORES_PATH = f"{ROOT}/scores/extrapolation/extrapolated/knn__du_IMAGENET_weighted_resnet18-self-trained_k_20_seed_256234_euclidean__4_20.json"
FULL_SCORES_PATH = f"{ROOT}/scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json"
embeddings_path = f"{ROOT}/savedir/embeddings/imagenet/submodel_embedding.pth" 
Outfolder = f"{ROOT}/analysis_reports/neurips26"


def _f32(emb: np.ndarray) -> np.ndarray:
    """Contiguous float32 view (halves memory, speeds up BLAS)."""
    return np.ascontiguousarray(emb, dtype=np.float32)


def _cap(idx: np.ndarray, y: np.ndarray, max_n: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Random subsample (idx, y) to at most ``max_n`` rows (0/None = no cap)."""
    if max_n and len(idx) > max_n:
        sel = rng.choice(len(idx), size=max_n, replace=False)
        return idx[sel], y[sel]
    return idx, y


def _chunk_predict(model, emb: np.ndarray, idx: np.ndarray, chunk: int) -> np.ndarray:
    """Predict over ``idx`` in ``chunk``-sized batches -> bounded memory."""
    out = np.empty(len(idx), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        out[s:s + chunk] = model.predict(emb[j])
    return out


def _default_gamma(emb: np.ndarray) -> float:
    """sklearn 'scale'-style RBF gamma = 1 / (n_features * Var(X))."""
    v = float(emb.var())
    d = emb.shape[1]
    return 1.0 / (d * v) if v > 0 else 1.0 / d


def _krr_builder(approx: bool, gamma0: float, n_components: int):
    """Return (make_model, grid) for RBF Kernel Ridge (exact or Nystrom-approx)."""
    if not approx:
        from sklearn.kernel_ridge import KernelRidge
        grid = [{"alpha": a, "gamma": g, "kernel": "rbf"}
                for a in (1e-2, 1e-1, 1.0) for g in (None, 0.1, 1.0)]
        return KernelRidge, grid
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    def make(alpha=1.0, gamma=gamma0, n_components=n_components):
        return make_pipeline(
            Nystroem(kernel="rbf", gamma=gamma, n_components=n_components,
                     random_state=0),
            Ridge(alpha=alpha))

    grid = [{"alpha": a, "gamma": g}
            for a in (1e-2, 1e-1, 1.0) for g in (gamma0 * 0.1, gamma0, gamma0 * 10)]
    return make, grid


def _svr_builder(approx: bool, gamma0: float, n_components: int):
    """Return (make_model, grid) for RBF SVR (exact or RBFSampler+LinearSVR)."""
    if not approx:
        from sklearn.svm import SVR
        return SVR, [{"C": c, "gamma": "scale", "kernel": "rbf"} for c in (1.0, 10.0)]
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVR

    def make(C=1.0, gamma=gamma0, n_components=n_components):
        return make_pipeline(
            RBFSampler(gamma=gamma, n_components=n_components, random_state=0),
            LinearSVR(C=C, max_iter=5000))

    grid = [{"C": c, "gamma": g}
            for c in (1.0, 10.0) for g in (gamma0, gamma0 * 10)]
    return make, grid


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
    pred_chunk: int = 100_000,
) -> Tuple[np.ndarray, dict]:
    """Select hyperparameters on the S_s-val split, refit on the (capped) seed set.

    ``fit_idx`` / ``seed_idx`` are expected to be already subsampled by the
    caller (``--max_train``); prediction on ``residual_idx`` runs in
    ``pred_chunk`` batches so the residual x train kernel is never materialised.
    """
    best, best_corr = None, -np.inf
    for params in grid:
        model = make_model(**params)
        model.fit(emb[fit_idx], fit_y)
        vp = _chunk_predict(model, emb, val_idx, pred_chunk)
        corr = rc.pearson(vp, val_y)
        if not np.isnan(corr) and corr > best_corr:
            best_corr, best = corr, params
    model = make_model(**(best or {}))
    model.fit(emb[seed_idx], seed_y)
    return _chunk_predict(model, emb, residual_idx, pred_chunk), (best or {})


def _knn_graph(emb: np.ndarray, k: int, chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact kNN graph over all N rows, built in memory-bounded row chunks.

    Uses the ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` trick with a float32
    ``chunk x N`` distance block (peak ~= chunk * N * 4 bytes), so memory stays
    bounded regardless of N. Time is O(N^2 d); fine for medium N, gated for 1M.
    Column 0 of the result is the self-match (distance 0), matching the
    ``kneighbors`` convention the caller relies on.
    """
    n = emb.shape[0]
    k = min(k, n)
    sq = np.einsum("ij,ij->i", emb, emb)  # ||row||^2, float32
    nbr = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        q = emb[s:s + chunk]
        d2 = sq[None, :] + np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ emb.T)
        np.maximum(d2, 0, out=d2)
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        pd = d2[rows, part]
        order = np.argsort(pd, axis=1)
        nbr[s:s + chunk] = part[rows, order]
        dist[s:s + chunk] = np.sqrt(pd[rows, order])
    return dist, nbr


def label_propagation(
    emb: np.ndarray,
    seed_idx: np.ndarray,
    seed_y: np.ndarray,
    residual_idx: np.ndarray,
    k: int = 10,
    n_iter: int = 50,
    knn_chunk: int = 512,
) -> np.ndarray:
    """Transductive regression label propagation over a symmetric kNN graph.

    The kNN graph is built with :func:`_knn_graph` in ``knn_chunk`` row batches
    (bounded memory) instead of ``NearestNeighbors.fit`` over the whole matrix.
    """
    n = emb.shape[0]
    dist, nbr = _knn_graph(emb, min(k + 1, n), knn_chunk)
    y = np.zeros(n)
    clamp = np.zeros(n, dtype=bool)
    y[seed_idx] = seed_y
    clamp[seed_idx] = True
    sigma = float(np.median(dist[:, 1:])) + 1e-9
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
    max_train: int = 30_000,
    pred_chunk: int = 100_000,
    exact_max: int = 8_000,
    n_components: int = 1_000,
    label_prop_max: int = 200_000,
    force_label_prop: bool = False,
    knn_chunk: int = 512,
) -> Dict:
    from sklearn.ensemble import RandomForestRegressor

    emb = _f32(emb)
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    fit_idx, val_idx = rc.fit_val_split(seed_idx, val_frac, rng)
    s_all = rc.scores_to_array(subset_scores, n)
    full_all = rc.scores_to_array(full_scores, n)
    y_true = full_all[residual_idx]

    # Capped training sets for the heavy regressors (random subsample of S_s).
    fit_c, fit_yc = _cap(fit_idx, s_all[fit_idx], max_train, rng)
    val_c, val_yc = _cap(val_idx, s_all[val_idx], max_train, rng)
    seed_c, seed_yc = _cap(seed_idx, s_all[seed_idx], max_train, rng)
    approx = len(seed_c) > exact_max
    gamma0 = _default_gamma(emb)

    print(f"[item4] n={n} seed={len(seed_idx)} residual={len(residual_idx)} "
          f"train_cap={len(seed_c)} approx_kernels={approx} dtype={emb.dtype}")
    assert len(residual_idx) != 0

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
        # select k on the (capped) val split (deployable, non-oracle), then
        # extrapolate the residual from the full seed set (the paper's reference).
        best_k, best_c = 20, -np.inf
        for k in (5, 10, 20, 50):
            p = knn_extrapolate(emb, fit_c, fit_yc, val_c, k)
            c = rc.pearson(p, val_yc)
            if not np.isnan(c) and c > best_c:
                best_c, best_k = c, k
        register(
            "knn_weighted",
            knn_extrapolate(emb, seed_idx, s_all[seed_idx], residual_idx, best_k),
        )

    if "knn_mean" in methods:
        # unweighted KNN ablation: same k-selection, uniform averaging.
        best_k, best_c = 20, -np.inf
        for k in (5, 10, 20, 50):
            p = knn_extrapolate(emb, fit_c, fit_yc, val_c, k, weighted=False)
            c = rc.pearson(p, val_yc)
            if not np.isnan(c) and c > best_c:
                best_c, best_k = c, k
        register(
            "knn_mean",
            knn_extrapolate(emb, seed_idx, s_all[seed_idx], residual_idx,
                            best_k, weighted=False),
        )

    if "ridge" in methods:
        from sklearn.linear_model import Ridge
        grid = [{"alpha": a} for a in (1e-2, 1e-1, 1.0, 10.0)]
        pred, _ = _fit_predict_sklearn(
            Ridge, grid, emb, fit_c, fit_yc, val_c, val_yc,
            seed_c, seed_yc, residual_idx, pred_chunk)
        register("ridge", pred)

    if "hist_gbr" in methods:
        from sklearn.ensemble import HistGradientBoostingRegressor
        grid = [{"learning_rate": lr, "max_iter": 300, "max_depth": md,
                 "random_state": seed}
                for lr in (0.05, 0.1) for md in (None, 8)]
        pred, _ = _fit_predict_sklearn(
            HistGradientBoostingRegressor, grid, emb, fit_c, fit_yc,
            val_c, val_yc, seed_c, seed_yc, residual_idx, pred_chunk)
        register("hist_gbr", pred)

    if "kernel_ridge" in methods:
        make, grid = _krr_builder(approx, gamma0, n_components)
        pred, _ = _fit_predict_sklearn(
            make, grid, emb, fit_c, fit_yc, val_c, val_yc,
            seed_c, seed_yc, residual_idx, pred_chunk)
        register("kernel_ridge", pred)

    if "svr_rbf" in methods:
        make, grid = _svr_builder(approx, gamma0, n_components)
        pred, _ = _fit_predict_sklearn(
            make, grid, emb, fit_c, fit_yc, val_c, val_yc,
            seed_c, seed_yc, residual_idx, pred_chunk)
        register("svr_rbf", pred)

    if "random_forest" in methods:
        grid = [{"n_estimators": 200, "max_depth": md, "random_state": seed,
                 "n_jobs": -1} for md in (None, 8, 16)]
        pred, _ = _fit_predict_sklearn(
            RandomForestRegressor, grid, emb, fit_c, fit_yc, val_c, val_yc,
            seed_c, seed_yc, residual_idx, pred_chunk)
        register("random_forest", pred)

    if "label_prop" in methods:
        if n > label_prop_max and not force_label_prop:
            print(f"[item4] label_prop SKIPPED: n={n} > label_prop_max="
                  f"{label_prop_max} (transductive O(N^2)); pass "
                  f"--force_label_prop to run the chunked brute-force graph.")
        else:
            register(
                "label_prop",
                label_propagation(emb, seed_idx, s_all[seed_idx], residual_idx,
                                  knn_chunk=knn_chunk),
            )

    return {"n": int(n), "n_seed": int(len(seed_idx)),
            "metrics": results, "score_dicts": dicts}


ALL_METHODS = ["knn_weighted", "knn_mean", "ridge", "kernel_ridge", "svr_rbf",
               "hist_gbr", "random_forest", "label_prop"]


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
    ap.add_argument("--embeddings", default=embeddings_path)
    ap.add_argument("--subset_scores",default=SUBSET_SCORES_PATH)
    ap.add_argument("--full_scores", default=FULL_SCORES_PATH)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=30_000,
                    help="cap the random S_s subsample used to TRAIN the heavy "
                         "regressors (kernel_ridge/svr/rf); 0 disables the cap")
    ap.add_argument("--pred_chunk", type=int, default=100_000,
                    help="batch size for chunked prediction over the residual set")
    ap.add_argument("--exact_max", type=int, default=8_000,
                    help="train size above which kernel_ridge/svr switch from "
                         "exact RBF to the linear-memory approximation")
    ap.add_argument("--n_components", type=int, default=1_000,
                    help="landmarks / random Fourier features for the approx kernels")
    ap.add_argument("--label_prop_max", type=int, default=200_000,
                    help="skip label_prop above this N (transductive O(N^2))")
    ap.add_argument("--force_label_prop", action="store_true",
                    help="run label_prop even above --label_prop_max")
    ap.add_argument("--knn_chunk", type=int, default=512,
                    help="row-chunk for the memory-bounded label_prop kNN graph")
    ap.add_argument("--out_metrics",default=f"{Outfolder}/item4_baseline_metrics.json",
                    help="path to dump JSON metrics dict")
    ap.add_argument("--out_scores_dir", default=f"{Outfolder}/item4_baseline_scores",
                    help="dir to dump one JSON score dict per method (drop-in for prune)")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.embeddings and args.subset_scores and args.full_scores):
        ap.error("--embeddings, --subset_scores, --full_scores required (or --smoke)")

    emb = rc.load_embeddings(args.embeddings, dtype=np.float32)
    subset = rc.load_scores(args.subset_scores)
    full = rc.load_scores(args.full_scores)
    print(f"[item4] running baselines on {len(list(subset.keys()))} subset scores, {len(list(full.keys()))} full scores")
    res = evaluate_baselines(
        emb, subset, full, args.val_frac, args.seed, args.methods,
        max_train=args.max_train, pred_chunk=args.pred_chunk,
        exact_max=args.exact_max, n_components=args.n_components,
        label_prop_max=args.label_prop_max,
        force_label_prop=args.force_label_prop, knn_chunk=args.knn_chunk)
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
