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
import os
from collections import Counter
from typing import Dict, List

import numpy as np

import rebuttal_common as rc


ROOT = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning"
SUBSET_SCORES_PATH = f"{ROOT}/scores/extrapolation/subset/IMAGENET_dynamic_uncertainty_0_11_20.json"
EXTRAPOLATED_SCORES_PATH = f"{ROOT}/scores/extrapolation/extrapolated/gnn__du_IMAGENET_resnet18-self-trained_k_20_seed_128117_euclidean.json"
FULL_SCORES_PATH = f"{ROOT}/scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json"
embeddings_path = f"{ROOT}/savedir/embeddings/imagenet/submodel_embedding.pth" 
Outfolder = f"{ROOT}/analysis_reports/neurips26"


def load_labels(dataset_name: str, cache: str | None = None) -> np.ndarray:
    """Return a per-sample-index int label array for ``dataset_name`` using the
    project's own data pipeline (utils.dataset.get_dataset), so ordering matches
    the sample_idx used everywhere else.

    Labels are not stored next to the scores, so we materialise them once from
    the trainset (which yields ``(image, label, index)``) and optionally cache
    them to ``cache`` (.npy) for reuse. Run from the repo root; imports torch
    lazily so ``--smoke`` works without it.
    """
    if cache and os.path.exists(cache):
        return np.load(cache)

    import sys
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from torch.utils.data import DataLoader
    from utils.dataset import get_dataset

    trainset, _ = get_dataset(dataset_name)
    n = len(trainset)
    labels = np.full(n, -1, dtype=np.int64)
    loader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=4)
    for _imgs, lbls, idxs in loader:
        idxs = idxs.to("cpu").numpy().astype(np.int64)
        labels[idxs] = lbls.to("cpu").numpy().astype(np.int64)
    if (labels < 0).any():
        raise RuntimeError(
            f"{int((labels < 0).sum())} sample indices never appeared in the "
            f"{dataset_name} loader; index space mismatch.")
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        np.save(cache, labels)
        print(f"[item8] cached labels -> {cache}")
    return labels


def embed_projection_plot(
    embeddings: np.ndarray,
    full_scores: Dict[int, float],
    extrapolated_scores: Dict[int, float],
    labels: np.ndarray,
    keep_frac: float,
    out_png: str,
    method: str = "umap",
    color_by: str = "status",
    max_points: int = 20000,
    seed: int = 0,
) -> str:
    """2-D UMAP/TSNE projection of the embeddings, highlighting where the
    extrapolation's pruning decision agrees/disagrees with the ground truth at
    ``keep_frac``. Nice qualitative exhibit for the rebuttal (Dcja / RpJS W3).

    color_by:
      * ``status`` -- 4 way agree-keep / agree-drop / GT-drop-EXT-keep (rescued)
        / GT-keep-EXT-drop (newly dropped). Shows *where* disagreements live.
      * ``class``  -- colour by class label.
      * ``score``  -- colour by ground-truth score.

    Heavy deps (torch not needed here; needs matplotlib and umap-learn or
    sklearn) are imported lazily so ``--smoke`` stays dependency-free.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = np.asarray(embeddings, dtype=np.float32)
    n = emb.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"labels ({labels.shape[0]}) vs embeddings ({n}) mismatch")

    gt_keep = set(rc.topk_retained(full_scores, keep_frac))
    ex_keep = set(rc.topk_retained(extrapolated_scores, keep_frac))

    rng = np.random.default_rng(seed)
    sel = np.arange(n)
    if n > max_points:
        sel = rng.choice(n, size=max_points, replace=False)

    # 2-D projection
    method = method.lower()
    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=seed)
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[item8] umap unavailable ({exc}); falling back to TSNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, init="pca")
    xy = reducer.fit_transform(emb[sel])

    fig, ax = plt.subplots(figsize=(7, 6))
    if color_by == "status":
        status = np.empty(sel.shape[0], dtype=object)
        for j, i in enumerate(sel):
            i = int(i)
            in_gt, in_ex = i in gt_keep, i in ex_keep
            status[j] = ("agree_keep" if in_gt and in_ex else
                         "agree_drop" if not in_gt and not in_ex else
                         "rescued" if not in_gt and in_ex else
                         "newly_dropped")
        palette = {"agree_keep": "#4c9f70", "agree_drop": "#cccccc",
                   "rescued": "#d1495b", "newly_dropped": "#3d5a80"}
        z = {"agree_drop": 0, "agree_keep": 1, "rescued": 2, "newly_dropped": 3}
        for name in sorted(set(status), key=lambda s: z[s]):
            m = status == name
            ax.scatter(xy[m, 0], xy[m, 1], s=4, c=palette[name], label=name,
                       alpha=0.6 if name == "agree_drop" else 0.85,
                       linewidths=0)
        ax.legend(markerscale=3, fontsize=8, loc="best")
    elif color_by == "class":
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=4, c=labels[sel],
                        cmap="tab20", alpha=0.7, linewidths=0)
        fig.colorbar(sc, ax=ax, label="class")
    elif color_by == "score":
        gt_arr = rc.scores_to_array(full_scores, n)
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=4, c=gt_arr[sel],
                        cmap="viridis", alpha=0.7, linewidths=0)
        fig.colorbar(sc, ax=ax, label="ground-truth score")
    else:
        raise ValueError(f"unknown color_by={color_by!r}")

    ax.set_title(f"{method.upper()} of embeddings @ keep={keep_frac:g} "
                 f"(prune={1 - keep_frac:g}), colour={color_by}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[item8] wrote projection -> {out_png}")
    return out_png


def _score_mass_recovery(full_arr: np.ndarray, gt_keep: set, ex_keep: set) -> float:
    """Retained GT-score-mass by extrapolation / oracle max mass (same budget).

    Forgiving: two different retained sets with similar GT score mass prune
    equally well. 1.0 = ext keeps as much GT-important mass as the oracle.
    Shifted so a min-mass (worst) set maps toward 0 not an arbitrary floor.
    """
    order = np.argsort(full_arr)  # ascending
    k = len(gt_keep)
    if k == 0:
        return float("nan")
    total = float(full_arr.sum())
    max_mass = float(full_arr[order[-k:]].sum())      # oracle keeps top-k mass
    min_mass = float(full_arr[order[:k]].sum())        # worst keeps bottom-k mass
    ex_mass = float(full_arr[list(ex_keep)].sum())
    denom = max_mass - min_mass
    if denom <= 1e-12:
        return 1.0
    return (ex_mass - min_mass) / denom


def _rbo(gt_rank: List[int], ex_rank: List[int], p: float = 0.98) -> float:
    """Rank-biased overlap of two ranked lists (top-weighted, [0,1]).

    p near 1 = deep lists matter; smaller p = weight only the very top.
    Truncated RBO over the given depth (extrapolated set size).
    """
    d = min(len(gt_rank), len(ex_rank))
    if d == 0:
        return float("nan")
    seen_gt, seen_ex = set(), set()
    overlap = 0
    s = 0.0
    for i in range(d):
        seen_gt.add(gt_rank[i])
        seen_ex.add(ex_rank[i])
        overlap = len(seen_gt & seen_ex)
        s += (overlap / (i + 1)) * (p ** i)
    return float((1 - p) * s / (1 - p ** d)) if p ** d < 1 else float(overlap / d)


def _soft_jaccard(full_scores, ext_scores, keep: float, tol_frac: float = 0.02) -> float:
    """Tolerance Jaccard: near-tie boundary confusions count as matches.

    A sample in one keep-set but not the other still counts as agreement if
    its rank in the *other* ranking is within ``tol_frac`` of the cutoff.
    Softens the hard boundary flips that tank plain Jaccard at high prune.
    """
    gt_ord = [k for k, _ in sorted(full_scores.items(), key=lambda kv: kv[1], reverse=True)]
    ex_ord = [k for k, _ in sorted(ext_scores.items(), key=lambda kv: kv[1], reverse=True)]
    n = len(gt_ord)
    n_keep = int(keep * n)
    band = max(1, int(tol_frac * n))
    gt_rank = {k: i for i, k in enumerate(gt_ord)}
    ex_rank = {k: i for i, k in enumerate(ex_ord)}
    gt_keep = set(gt_ord[:n_keep])
    ex_keep = set(ex_ord[:n_keep])
    cutoff = n_keep + band
    soft_inter = 0
    for k in gt_keep | ex_keep:
        in_gt = k in gt_keep or gt_rank[k] < cutoff
        in_ex = k in ex_keep or ex_rank[k] < cutoff
        if in_gt and in_ex:
            soft_inter += 1
    union = len(gt_keep | ex_keep)
    return float(soft_inter / union) if union else 1.0


def composition(
    full_scores: Dict[int, float],
    extrapolated_scores: Dict[int, float],
    labels: np.ndarray,
    keep_fracs: List[float],
) -> Dict:
    n = len(labels)
    full_arr = rc.scores_to_array(full_scores, n)
    ext_arr = rc.scores_to_array(extrapolated_scores, n)
    gt_order = [k for k, _ in sorted(full_scores.items(), key=lambda kv: kv[1], reverse=True)]
    ex_order = [k for k, _ in sorted(extrapolated_scores.items(), key=lambda kv: kv[1], reverse=True)]

    rows = []
    for keep in keep_fracs:
        gt_keep = set(rc.topk_retained(full_scores, keep))
        ex_keep = set(rc.topk_retained(extrapolated_scores, keep))
        agreement = len(gt_keep & ex_keep) / max(1, len(gt_keep))
        k_keep = len(gt_keep)
        mass_rec = _score_mass_recovery(full_arr, gt_keep, ex_keep)
        rbo = _rbo(gt_order[:k_keep], ex_order[:k_keep], p=0.98)
        soft_jac = _soft_jaccard(full_scores, extrapolated_scores, keep, tol_frac=0.02)

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
            "soft_jaccard": soft_jac,
            "rbo": rbo,
            "score_mass_recovery": mass_rec,
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
    hdr = ("prune", "jaccard", "softjac", "rbo", "massrec", "agree", "cls_L1",
           "mean_gt", "mean_ext", "rescued", "dropped")
    print("  " + " ".join(f"{h:>8}" for h in hdr))
    for r in res["rows"]:
        print("  " + " ".join(f"{v:>8}" for v in (
            r["prune_rate"], f"{r['jaccard']:.3f}", f"{r['soft_jaccard']:.3f}",
            f"{r['rbo']:.3f}", f"{r['score_mass_recovery']:.3f}",
            f"{r['agreement']:.3f}",
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
    # exercise the projection exhibit (TSNE keeps deps light; umap optional)
    import tempfile
    out_png = os.path.join(tempfile.gettempdir(), "item8_projection_smoke.png")
    try:
        embed_projection_plot(emb, full, ext, labels, 0.2, out_png,
                              method="tsne", color_by="status", max_points=400)
        assert os.path.exists(out_png)
        print(f"[item8] projection smoke OK -> {out_png}")
    except Exception as exc:  # pragma: no cover - sklearn/matplotlib optional
        print(f"[item8] projection smoke skipped ({type(exc).__name__}: {exc})")
    print("[item8] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 8: pruning-composition analysis")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full_scores", default=FULL_SCORES_PATH)
    ap.add_argument("--extrapolated_scores", default=EXTRAPOLATED_SCORES_PATH)
    ap.add_argument("--labels", help="npy of int labels indexed by sample id")
    ap.add_argument("--dataset", default="IMAGENET", help="dataset name; if given (and --labels not), "
                    "labels are materialised via utils.dataset.get_dataset")
    ap.add_argument("--labels_cache", help="npy path to cache/reuse dataset labels")
    ap.add_argument("--keep_fracs", type=float, nargs="+",
                    default=[0.5, 0.2, 0.1, 0.05])
    ap.add_argument("--out")
    # optional UMAP/TSNE projection exhibit
    ap.add_argument("--embeddings", help="embeddings .pth/.npy/.npz for projection plot")
    ap.add_argument("--plot_out", help="path to write the projection PNG")
    ap.add_argument("--plot_keep_frac", type=float, default=0.1)
    ap.add_argument("--plot_method", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--plot_color_by", choices=["status", "class", "score"],
                    default="status")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.full_scores and args.extrapolated_scores):
        ap.error("need --full_scores --extrapolated_scores (or --smoke)")
    if not (args.labels or args.dataset):
        ap.error("need --labels or --dataset (to materialise labels)")

    full = rc.load_scores(args.full_scores)
    ext = rc.load_scores(args.extrapolated_scores)
    if args.labels:
        labels = np.load(args.labels)
    else:
        labels = load_labels(args.dataset, cache=args.labels_cache)
    res = composition(full, ext, labels, args.keep_fracs)
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[item8] wrote {args.out}")

    if args.embeddings and args.plot_out:
        emb = rc.load_embeddings(args.embeddings)
        embed_projection_plot(
            emb, full, ext, labels, args.plot_keep_frac, args.plot_out,
            method=args.plot_method, color_by=args.plot_color_by,
        )


if __name__ == "__main__":
    main()
