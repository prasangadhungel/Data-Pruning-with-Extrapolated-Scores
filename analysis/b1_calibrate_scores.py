"""B1 - Ranking-preserving score calibration.

Rebuttal target: the "moderate correlation vs 'accurate'" critique (bWqr W1,
myyu W1, RpJS). A monotone (rank-preserving) calibration map improves the
*value fidelity* of the extrapolated scores WITHOUT changing their ranking -- so
top-k pruning selection is provably unchanged, yet MSE / calibration error
against ``S`` drop. Deployable, no retraining.

Honesty: the calibration map is fit ONLY on a held-out validation split of the
subset scores ``S_s`` (the Item-1 rule). Concretely we split the seed set into
fit/val, extrapolate to the *val* samples from the *fit* samples (so the val
extrapolated values are genuinely out-of-sample), and fit the monotone map
    extrapolated_val -> S_s(val).
The map is then applied to the residual-set extrapolation (fit on the full seed
set). No ground-truth ``S`` on ``D_r`` is ever used for fitting.

Maps:
  * isotonic - monotone non-parametric (sklearn IsotonicRegression)
  * platt    - monotone affine fit (slope clamped >= 0)

Self-test:
    python analysis/b1_calibrate_scores.py --smoke
"""


from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

import rebuttal_common as rc
from item1_knn_k_selection import knn_extrapolate

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


ROOT = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning"
SUBSET_SCORES_PATH = f"{ROOT}/scores/extrapolation/subset/CIFAR10_dynamic_uncertainty_0.json"
SUBSET_SCORES_PATH = f"{ROOT}/scores/extrapolation/extrapolated/gnn__du_IMAGENET_resnet18-self-trained_k_20_seed_128117_euclidean.json"
FULL_SCORES_PATH = f"{ROOT}/scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json"
embeddings_path = f"{ROOT}/savedir/embeddings/imagenet/submodel_embedding.pth" 
Outfolder = f"{ROOT}/analysis_reports/neurips26"




def fit_isotonic(x: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(x, y)
    return lambda v: ir.predict(v)


def fit_platt(x: np.ndarray, y: np.ndarray):
    a, b = np.polyfit(x, y, 1)
    a = max(a, 0.0)
    return lambda v: a * np.asarray(v) + b


def calibrate(
    emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    method: str,
    k: int,
    val_frac: float,
    distance: str,
    seed: int,
) -> Dict:
    n = emb.shape[0]
    if not np.isfinite(emb).all():
        bad = int((~np.isfinite(emb)).any(axis=1).sum())
        raise ValueError(
            f"[b1] embeddings contain non-finite values in {bad}/{n} rows -> "
            "distances become NaN and every metric is None. Regenerate the "
            "embeddings (analysis/make_embeddings.py) or drop the bad rows.")
    max_key = max(max(subset_scores, default=-1), max(full_scores, default=-1))
    if max_key >= n:
        raise ValueError(
            f"[b1] score dicts reference sample index {int(max_key)} >= "
            f"n_embeddings={n}: the embeddings and score dicts live in DIFFERENT "
            "index spaces. A dict{idx->vec} embedding file is re-stacked by sorted "
            "key, so row i != sample_idx i unless the keys are exactly 0..N-1. "
            "Use dense Tensor[N,d] embeddings whose row i == sample_idx i "
            "(analysis/make_embeddings.py --format tensor).")
    rng = np.random.default_rng(seed)
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    fit_idx, val_idx = rc.fit_val_split(seed_idx, val_frac, rng)
    s_all = rc.scores_to_array(subset_scores, n)
    full = rc.scores_to_array(full_scores, n)

    print(f"[b1] seed/residual sizes: {len(seed_idx)}/{len(residual_idx)}, {s_all.shape} , {full.shape} ")
    # Coverage guard: the ground-truth full scores S must cover the residual
    # targets, otherwise y is NaN and pearson/spearman/mse are all None.
    y = full[residual_idx]
    covered = np.isfinite(y)
    if int(covered.sum()) < 2:
        raise ValueError(
            f"[b1] full_scores covers only {int(covered.sum())}/{len(residual_idx)} "
            "residual samples -> targets are (almost) all NaN, so every metric is "
            "None. Pass the FULL score dict for this dataset, keyed by the SAME "
            f"sample indices as the embeddings (full: n={len(full_scores)}, "
            f"max_key={max(full_scores) if full_scores else 'NA'}; n_emb={n}).")
    if int(covered.sum()) < len(residual_idx):
        print(f"[b1] WARNING: full_scores misses "
              f"{len(residual_idx) - int(covered.sum())}/{len(residual_idx)} "
              "residual samples; metrics computed on the covered subset.")

    # Held-out extrapolation on the val split -> honest (ext, target) pairs
    ext_val = knn_extrapolate(emb, fit_idx, s_all[fit_idx], val_idx, k, distance)
    fitter = {"isotonic": fit_isotonic, "platt": fit_platt}[method]
    cal_map = fitter(ext_val, s_all[val_idx])

    # Residual extrapolation from the full seed set, then calibrate
    ext_res = knn_extrapolate(emb, seed_idx, s_all[seed_idx], residual_idx, k, distance)
    cal_res = np.asarray(cal_map(ext_res), dtype=np.float64)

    # Degenerate-map fallback: a constant map (isotonic collapse / platt slope
    # clamped to 0 on a weak val fit) destroys all signal and makes the
    # correlations NaN. Fall back to identity, which is still rank-preserving.
    if float(np.std(cal_res[covered])) < 1e-12:
        print("[b1] WARNING: calibration map collapsed to a constant "
              "(weak/non-monotone validation fit); using identity map instead.")
        cal_res = ext_res.copy()

    def stats(pred):
        m = covered & np.isfinite(pred)
        return {"pearson": rc.pearson(pred[m], y[m]),
                "spearman": rc.spearman(pred[m], y[m]),
                "mse": float(np.mean((pred[m] - y[m]) ** 2)),
                "n_eval": int(m.sum())}

    before, after = stats(ext_res), stats(cal_res)
    print(before,after)
    # Rank preservation <=> the fitted map is monotone non-decreasing over the
    # observed range (plateaus/ties are allowed; order is never reversed).
    order = np.argsort(ext_res)
    rank_preserved = bool(np.all(np.diff(cal_res[order]) >= -1e-9))

    calibrated_dict = {str(int(i)): float(v) for i, v in zip(residual_idx, cal_res)}
    for i in seed_idx:
        calibrated_dict[str(int(i))] = float(s_all[i])

    return {"method": method, "k": int(k), "n_val": int(len(val_idx)),
            "before": before, "after": after,
            "mse_reduction": before["mse"] - after["mse"],
            "ranking_preserved": rank_preserved,
            "calibrated_scores": calibrated_dict}


def _print(res: Dict) -> None:
    print(f"  method={res['method']} k={res['k']} val={res['n_val']} "
          f"ranking_preserved={res['ranking_preserved']}")
    print(f"  {'':>10} {'pearson':>9} {'spearman':>9} {'mse':>9}")
    for tag in ("before", "after"):
        m = res[tag]
        print(f"  {tag:>10} {m['pearson']:>9.4f} {m['spearman']:>9.4f} {m['mse']:>9.4f}")
    print(f"  -> MSE reduction = {res['mse_reduction']:.4f}")


def run_smoke() -> Dict:
    print("[b1] SMOKE: ranking-preserving calibration on synthetic data")
    # Guaranteed invariant: a monotone map never reverses the ranking, so top-k
    # selection is unchanged. The MSE effect is dataset-dependent (it helps when
    # the raw extrapolation is mis-scaled, e.g. KNN shrinkage on real data); the
    # script reports it either way. Here we only assert the guarantees.
    emb, subset, full, _ = rc.make_fixture(n=1500, d=16, subset_frac=0.5,
                                           noise=0.4, seed=3)
    res = calibrate(emb, subset, full, "platt", k=15, val_frac=0.3,
                    distance="euclidean", seed=3)
    _print(res)
    assert res["ranking_preserved"], "platt map must preserve ranking"
    assert res["mse_reduction"] > -0.05, res["mse_reduction"]
    res_iso = calibrate(emb, subset, full, "isotonic", k=15, val_frac=0.3,
                        distance="euclidean", seed=3)
    assert res_iso["ranking_preserved"], "isotonic map must preserve ranking"
    print("[b1] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="B1: ranking-preserving calibration")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--embeddings",default=embeddings_path)
    ap.add_argument("--subset_scores", default=SUBSET_SCORES_PATH)
    ap.add_argument("--full_scores", default=FULL_SCORES_PATH)
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "platt"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_scores", default=f"{Outfolder}/calibrated_scores.json")
    ap.add_argument("--out_metrics", default=f"{Outfolder}/calibration_metrics.json")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.embeddings and args.subset_scores and args.full_scores):
        ap.error("need --embeddings --subset_scores --full_scores (or --smoke)")

    emb = rc.load_embeddings(args.embeddings)
    subset = rc.load_scores(args.subset_scores)
    full = rc.load_scores(args.full_scores)
    print(emb.shape)

    res = calibrate(emb, subset, full, args.method, args.k, args.val_frac,
                    args.distance, args.seed)
    _print(res)
    if args.out_scores:
        rc.save_json(res["calibrated_scores"], args.out_scores)
        print(f"[b1] wrote calibrated score dict {args.out_scores}")
    if args.out_metrics:
        rc.save_json({k: v for k, v in res.items() if k != "calibrated_scores"},
                     args.out_metrics)
        print(f"[b1] wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
