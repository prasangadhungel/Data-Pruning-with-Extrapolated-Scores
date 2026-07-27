"""Dcja Q2 - KNN score extrapolation on a PRETRAINED (foundation) backbone.

Reviewer Dcja, Q2 ("Sensitivity to proxy embedding quality"):
    "did the authors compare against pretrained or foundation-model embeddings
     for supervised pruning?"

The paper extrapolates scores in the embedding space of the PROXY model ``F_s``
that is trained only on the subset ``D_s``.  This raises the concern that a
poorly-generalising ``F_s`` degrades extrapolation.  This script answers the
question directly for IMAGENET + Dynamic-Uncertainty (DU) by running the *same*
distance-weighted KNN extrapolation on the embeddings of an OFF-THE-SHELF,
ImageNet-pretrained backbone (torchvision, e.g. resnet50 IMAGENET1K weights) and
comparing its extrapolation quality against the paper's proxy embeddings.

Two modes:
  (A) ``embed``       : extract penultimate features from a torchvision pretrained
                        backbone over the trainset (sample_idx-aligned with the
                        score dicts). Requires torch + torchvision (cluster).
  (B) ``extrapolate`` : run non-oracle KNN extrapolation on one or two embedding
                        files (pretrained vs, optionally, proxy) and report
                        residual Pearson/Spearman side-by-side.
  ``--smoke``         : numpy-only self-test (no torch, no data).

--------------------------------------------------------------------------------
DATA TO LOAD (real run). Root on the shared store:
  ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
  IMAGENET DU scores:
    S_s (subset) : ROOT/scores/extrapolation/subset/IMAGENET_dynamic_uncertainty_*.json
    S   (full)   : ROOT/scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json  (eval only)
  imagenet64 data (read by utils.dataset.get_dataset("IMAGENET")):
    /ceph/ssd/shared/datasets/imagenet/imagenet64/{train_data_batch_1..10.npz, val_data.npz}
  Proxy embeddings (for the comparison column) come from make_embeddings.py:
    ROOT/savedir/embeddings/imagenet/embeddings_dict.pth

TYPICAL WORKFLOW (cluster):
  # 1a. supervised pretrained backbone (foundation-model features)
  python analysis/q2_pretrained_backbone_knn.py embed \
    --dataset IMAGENET --source torchvision --backbone resnet50 \
    --weights IMAGENET1K_V2 --resize 224 --imagenet_norm \
    --out /ceph/.../savedir/embeddings/imagenet/pretrained_resnet50.pth
  # 1b. DINOv2 self-supervised foundation backbone (TURTLE-style; strongest test)
  python analysis/q2_pretrained_backbone_knn.py embed \
    --dataset IMAGENET --source dinov2 --backbone dinov2_vitb14 \
    --resize 224 --imagenet_norm \
    --out /ceph/.../savedir/embeddings/imagenet/dinov2_vitb14.pth
  # 2. proxy backbone embeddings (paper F_s) -- see make_embeddings.py
  python analysis/make_embeddings.py --dataset IMAGENET \
    --ckpt .../models/imagenet/subset_data/dynamic_uncertainty_256234.pth \
    --features --out /ceph/.../savedir/embeddings/imagenet/proxy_du_feat.pth
  # 3. compare extrapolation quality of the two embedding spaces
  python analysis/q2_pretrained_backbone_knn.py extrapolate \
    --pretrained_embeddings /ceph/.../dinov2_vitb14.pth \
    --proxy_embeddings      /ceph/.../proxy_du_feat.pth \
    --subset_scores .../scores/extrapolation/subset/IMAGENET_dynamic_uncertainty_0.json \
    --full_scores   .../scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json \
    --k_values 10 20 50 100 --out results/q2_imagenet_du.json

Self-test (no data / no torch):
  python analysis/q2_pretrained_backbone_knn.py --smoke
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import rebuttal_common as rc
from item1_knn_k_selection import knn_extrapolate

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# --------------------------------------------------------------------------- #
# Default artifact paths (mirrors item4_regression_baselines.py so the script
# runs directly with no flags on the cluster). IMAGENET + Dynamic-Uncertainty.
# --------------------------------------------------------------------------- #
ROOT = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning"
SUBSET_SCORES_PATH = f"{ROOT}/scores/extrapolation/subset/IMAGENET_dynamic_uncertainty_0_256234_4_19.json"
FULL_SCORES_PATH = f"{ROOT}/scores/prune/IMAGENET_dynamic_uncertainty_0_4_20.json"
PROXY_EMB_PATH = f"{ROOT}/savedir/embeddings/imagenet/submodel_embedding.pth"
EMB_DIR = f"{ROOT}/savedir/embeddings/imagenet"
DINOV2_EMB_PATH = f"{EMB_DIR}/dinov2_vitb14.pth"
PRETRAINED_EMB_PATH = f"{EMB_DIR}/pretrained_resnet50.pth"
OUTFOLDER = f"{ROOT}/analysis_reports/neurips26"


# --------------------------------------------------------------------------- #
# (B) extrapolate + compare  (numpy-only; the science of the answer)
# --------------------------------------------------------------------------- #
def extrapolate_eval(
    emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    k_values: List[int],
    val_frac: float,
    distance: str,
    weighted: bool,
    seed: int,
) -> Dict:
    """Non-oracle KNN extrapolation quality of ONE embedding space.

    ``k`` is chosen by the deployable ``S_s``-val rule (item1); we then report
    the residual-set (``D_r``) Pearson/Spearman against the full scores ``S`` at
    that ``k`` (and the oracle-``k`` upper bound for context).
    """
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    seed_idx, residual_idx = rc.seed_and_residual(subset_scores, n)
    fit_idx, val_idx = rc.fit_val_split(seed_idx, val_frac, rng)

    s_all = rc.scores_to_array(subset_scores, n)
    full_all = rc.scores_to_array(full_scores, n)
    fit_scores = s_all[fit_idx]

    per_k = []
    for k in k_values:
        val_pred = knn_extrapolate(emb, fit_idx, fit_scores, val_idx, k, distance, weighted)
        val_corr = rc.pearson(val_pred, s_all[val_idx])
        res_pred = knn_extrapolate(
            emb, seed_idx, s_all[seed_idx], residual_idx, k, distance, weighted
        )
        # evaluate only where the full score S is defined
        m = ~np.isnan(full_all[residual_idx])
        res_corr = rc.pearson(res_pred[m], full_all[residual_idx][m])
        res_spear = rc.spearman(res_pred[m], full_all[residual_idx][m])
        per_k.append({
            "k": int(k),
            "val_pearson_Ss": val_corr,
            "residual_pearson_S": res_corr,
            "residual_spearman_S": res_spear,
        })

    valid = [r for r in per_k if not np.isnan(r["val_pearson_Ss"])]
    k_val = max(valid, key=lambda r: r["val_pearson_Ss"])
    k_oracle = max(per_k, key=lambda r: r["residual_pearson_S"])
    return {
        "embed_dim": int(emb.shape[1]),
        "n_samples": int(n),
        "n_seed": int(len(seed_idx)),
        "per_k": per_k,
        "selected_k_val": k_val["k"],
        "test_pearson_val_selected": k_val["residual_pearson_S"],
        "test_spearman_val_selected": k_val["residual_spearman_S"],
        "selected_k_oracle": k_oracle["k"],
        "test_pearson_oracle_selected": k_oracle["residual_pearson_S"],
    }


def compare(
    pretrained_emb: np.ndarray,
    subset_scores: Dict[int, float],
    full_scores: Dict[int, float],
    k_values: List[int],
    val_frac: float,
    distance: str,
    weighted: bool,
    seed: int,
    proxy_emb: Optional[np.ndarray] = None,
) -> Dict:
    out = {
        "distance": distance,
        "weighted": weighted,
        "pretrained": extrapolate_eval(
            pretrained_emb, subset_scores, full_scores, k_values,
            val_frac, distance, weighted, seed,
        ),
    }
    if proxy_emb is not None:
        out["proxy"] = extrapolate_eval(
            proxy_emb, subset_scores, full_scores, k_values,
            val_frac, distance, weighted, seed,
        )
        dp = out["pretrained"]["test_pearson_val_selected"]
        pp = out["proxy"]["test_pearson_val_selected"]
        out["delta_pearson_pretrained_minus_proxy"] = float(dp - pp)
    return out


def _print(res: Dict) -> None:
    def _blk(name, r):
        print(f"  [{name}] dim={r['embed_dim']} n={r['n_samples']} seed={r['n_seed']}")
        print(f"    {'k':>6} {'val P(S_s)':>12} {'test P(S)':>12} {'test Sp(S)':>12}")
        for x in r["per_k"]:
            print(f"    {x['k']:>6} {x['val_pearson_Ss']:>12.4f} "
                  f"{x['residual_pearson_S']:>12.4f} {x['residual_spearman_S']:>12.4f}")
        print(f"    -> k*(val)={r['selected_k_val']}  test P={r['test_pearson_val_selected']:.4f} "
              f"Sp={r['test_spearman_val_selected']:.4f}  "
              f"(oracle k={r['selected_k_oracle']} P={r['test_pearson_oracle_selected']:.4f})")

    _blk("pretrained backbone", res["pretrained"])
    if "proxy" in res:
        _blk("proxy F_s", res["proxy"])
        d = res["delta_pearson_pretrained_minus_proxy"]
        sign = "+" if d >= 0 else ""
        print(f"  => delta Pearson (pretrained - proxy) = {sign}{d:.4f}")


# --------------------------------------------------------------------------- #
# (A) embed  (torch/torchvision; cluster only)
# --------------------------------------------------------------------------- #
def run_embed(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from utils.dataset import get_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[q2.embed] device={device} source={args.source} backbone={args.backbone} "
          f"weights={args.weights} resize={args.resize} imagenet_norm={args.imagenet_norm}")

    captured = {}
    if args.source == "dinov2":
        # self-supervised foundation backbone (as used by TURTLE). torch.hub returns
        # the feature extractor directly: model(x) -> CLS embedding [B, D]. No classifier
        # head, so no hook. vitb14->768, vits14->384, vitl14->1024, vitg14->1536.
        model = torch.hub.load("facebookresearch/dinov2", args.backbone).to(device).eval()

        def _forward(imgs):
            captured["feat"] = model(imgs).detach()
    else:  # torchvision supervised backbone: hook the penultimate features
        import torchvision
        import torch.nn as nn

        weights = None if args.weights.lower() in ("none", "random") else args.weights
        ctor = getattr(torchvision.models, args.backbone)
        model = ctor(weights=weights).to(device).eval()
        last = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is None:
            raise RuntimeError("No nn.Linear layer to hook for penultimate features.")
        last.register_forward_hook(
            lambda _m, inp, _o: captured.__setitem__("feat", inp[0].detach())
        )

        def _forward(imgs):
            model(imgs)

    trainset, _ = get_dataset(args.dataset)
    n_total = len(trainset)
    print(f"[q2.embed] trainset size={n_total}")

    # DINOv2 (and torchvision pretrained) expect ImageNet-normalised RGB. The repo
    # pipeline applies its own normalisation, so optionally re-normalise here.
    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _prep(imgs):
        if args.resize and args.resize > 0:
            imgs = torch.nn.functional.interpolate(
                imgs, size=(args.resize, args.resize),
                mode="bilinear", align_corners=False,
            )
        if args.imagenet_norm:
            imgs = (imgs - _mean) / _std
        return imgs

    sample0 = _prep(trainset[0][0].unsqueeze(0).to(device))
    with torch.no_grad():
        _forward(sample0)
    dim = int(captured["feat"].shape[1])
    print(f"[q2.embed] embedding_dim={dim}")

    store = torch.zeros(n_total, dim)
    loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    seen = 0
    with torch.no_grad():
        for imgs, _labels, idxs in loader:
            imgs = _prep(imgs.to(device, non_blocking=True))
            _forward(imgs)
            store[idxs.to(torch.long)] = captured["feat"].cpu()
            seen += imgs.shape[0]
            if seen % (args.batch_size * 20) == 0:
                print(f"[q2.embed]   {seen}/{n_total} done")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(store, args.out)
    print(f"[q2.embed] saved Tensor[{n_total}, {dim}] -> {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def run_smoke() -> Dict:
    print("[q2] SMOKE: pretrained-vs-proxy KNN extrapolation on synthetic data")
    # "pretrained" = clean informative embedding; "proxy" = noisier version
    emb, subset, full, _ = rc.make_fixture(n=600, d=16, subset_frac=0.3, seed=7)
    rng = np.random.default_rng(0)
    proxy = emb + rng.normal(0, 0.6, size=emb.shape)  # degrade to mimic weak F_s
    res = compare(
        emb, subset, full, [5, 10, 20, 50], 0.2, "euclidean", True, seed=7,
        proxy_emb=proxy,
    )
    _print(res)
    assert res["pretrained"]["test_pearson_val_selected"] > 0.3
    # cleaner embedding should not be worse than the degraded proxy
    assert res["delta_pearson_pretrained_minus_proxy"] > -0.05
    print("[q2] SMOKE PASSED")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Dcja Q2: pretrained-backbone KNN extrapolation")
    sub = ap.add_subparsers(dest="mode")
    ap.add_argument("--smoke", action="store_true")

    pe = sub.add_parser("embed", help="extract pretrained/foundation backbone features")
    pe.add_argument("--dataset", default="IMAGENET")
    pe.add_argument("--source", default="dinov2", choices=["torchvision", "dinov2"],
                    help="torchvision supervised backbone (penultimate features) or "
                         "dinov2 self-supervised foundation backbone (CLS features, TURTLE-style)")
    pe.add_argument("--backbone", default="dinov2_vitb14",
                    help="torchvision ctor (resnet50, vit_b_16, ...) OR dinov2 hub id "
                         "(dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)")
    pe.add_argument("--weights", default="IMAGENET1K_V2",
                    help="torchvision weights enum string, or 'none' (ignored for dinov2)")
    pe.add_argument("--resize", type=int, default=224,
                    help="bilinear-resize inputs to RxR (0=off). Use a multiple of 14 "
                         "for dinov2 (e.g. 224, 518)")
    pe.add_argument("--imagenet_norm", action=argparse.BooleanOptionalAction, default=True,
                    help="re-normalise inputs with ImageNet mean/std before the backbone "
                         "(default on; --no-imagenet_norm to disable)")
    pe.add_argument("--batch-size", type=int, default=256)
    pe.add_argument("--num-workers", type=int, default=4)
    pe.add_argument("--out", default=DINOV2_EMB_PATH,
                    help=f"output .pth (default: {DINOV2_EMB_PATH})")

    px = sub.add_parser("extrapolate", help="KNN extrapolation quality: pretrained vs proxy")
    px.add_argument("--pretrained_embeddings", default=DINOV2_EMB_PATH,
                    help=f"foundation-backbone embeddings (default: {DINOV2_EMB_PATH})")
    px.add_argument("--proxy_embeddings", default=PROXY_EMB_PATH,
                    help=f"paper proxy F_s embeddings (default: {PROXY_EMB_PATH}); "
                         "pass 'none' to skip the comparison column")
    px.add_argument("--subset_scores", default=SUBSET_SCORES_PATH)
    px.add_argument("--full_scores", default=FULL_SCORES_PATH)
    px.add_argument("--k_values", type=int, nargs="+", default=[10, 20, 50, 100])
    px.add_argument("--val_frac", type=float, default=0.1)
    px.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    px.add_argument("--unweighted", action="store_true")
    px.add_argument("--seed", type=int, default=42)
    px.add_argument("--out", default=f"{OUTFOLDER}/q2_imagenet_du.json")

    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return 0
    if args.mode == "embed":
        return run_embed(args)
    if args.mode == "extrapolate":
        pre = rc.load_embeddings(args.pretrained_embeddings, dtype=np.float32)
        proxy = None
        if args.proxy_embeddings and args.proxy_embeddings.lower() != "none":
            proxy = rc.load_embeddings(args.proxy_embeddings, dtype=np.float32)
        subset = rc.load_scores(args.subset_scores)
        full = rc.load_scores(args.full_scores)
        res = compare(
            pre, subset, full, args.k_values, args.val_frac, args.distance,
            not args.unweighted, args.seed, proxy_emb=proxy,
        )
        _print(res)
        if args.out:
            rc.save_json(res, args.out)
            print(f"[q2] wrote {args.out}")
        return 0
    ap.error("choose a mode: embed | extrapolate  (or --smoke)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
