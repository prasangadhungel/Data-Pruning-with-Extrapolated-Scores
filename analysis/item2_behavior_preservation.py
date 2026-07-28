"""Item 2 - Behavior preservation: error-set overlap + subgroup accuracy.

Rebuttal target: reviewer myyu W2 ("does the pruned model fail on the SAME
examples?") + subgroup / tail robustness. This is a Tier-B, no-retraining
analysis: it consumes the per-sample TEST predictions of two already-trained
downstream models -- one pruned with the ground-truth scores ``S``, one pruned
with the extrapolated scores -- and quantifies whether they behave the same:

  * error-set overlap (Jaccard) and prediction agreement;
  * per-class / worst-group / tail accuracy gaps.

On the real machine the predictions are computed DIRECTLY from the pruned
downstream checkpoints (default mode): the script builds the downstream model +
test loader, loads each checkpoint, and runs test predictions via
``predict_from_checkpoint``. Defaults compare the GT-score-pruned PLACES_365 TDDS
model against the KNN-extrapolation-pruned model (swap ``--ckpt_ext`` to the GNN
checkpoint). For standalone/CI use, pass ``--pred_gt``/``--pred_ext`` ``.npz``
arrays (fields ``sample_idx, label, pred``) to bypass checkpoint loading.

OOD robustness: pass ``--corruption`` (e.g. ``gaussian_blur``) with ``--severity``
1..5 to apply an ImageNet-C-style perturbation to the test set ON THE FLY (see
``perturbations.py``). Both models are evaluated on the SAME corrupted inputs, so
the behaviour-preservation metrics then report whether score extrapolation still
matches the GT-pruned model under a blurry/noisy distribution shift.

Run directly (real machine, all defaults):
    python analysis/item2_behavior_preservation.py

Run with a blurry OOD test set:
    python analysis/item2_behavior_preservation.py --corruption gaussian_blur --severity 3

Self-test (no torch, no data):
    python analysis/item2_behavior_preservation.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, format="{time:MM-DD HH:mm} - {message}")
except ModuleNotFoundError:  # loguru optional: fall back to stdlib logging
    import logging
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s - %(message)s", datefmt="%m-%d %H:%M",
    )
    logger = logging.getLogger("item2")
import rebuttal_common as rc
import perturbations

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). Root on the shared store:
#   ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
#   (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
# This script compares the DOWNSTREAM behaviour of two pruned models:
#   --pred_gt     predictions of a model trained on the GT-score-pruned set
#   --pred_ext    predictions of a model trained on the extrapolation-pruned set
# Each is an .npz with arrays {sample_idx, label, pred} over the SAME test set.
#
# NOTE: these two checkpoints DO NOT EXIST yet. Only proxy checkpoints are on disk
#   (ROOT/models/<DS>/full_data/dual_model.pth and
#    ROOT/models/<DS>/subset_data/<metric>_<seed>.pth, e.g. CIFAR10
#    dynamic_uncertainty_10000.pth, imagenet dynamic_uncertainty_256234.pth /
#    tdds_256234.pth) -- NOT the pruned-downstream models this item needs.
# To produce them (Tier-C, needs GPU): run src/prune/prune_with_scores.py at a fixed
# prune rate twice -- once with the GT score dict (scores/prune/<DS>_*.json), once with
# the extrapolated dict (scores/extrapolation/extrapolated/{gnn,knn}_*<DS>_*.json) --
# then evaluate each on the test set and dump {sample_idx,label,pred} to the npz above.
# ---------------------------------------------------------------------------

ROOT = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning"
gnn_extra_checkpoint = f"{ROOT}/models/pruned_models/PLACES_365/gnn__TDDS_PLACES_365_resnet50-self-trained_k_10_seed_360692_euclidean_5_31/model_pruned_80.pth"
knn_extra_checkpoint = f"{ROOT}/models/pruned_models/PLACES_365/knn_TDDS_PLACES_365_weighted_resnet50-self-trained_k_20_seed_360692_euclidean__5_31/model_pruned_80.pth"
original_score_checkpoint = f"{ROOT}/models/pruned_models/PLACES_365/tdds/PLACES_365_last_tdds_0/model_pruned_80.pth"


def build_model_and_testloader(
    dataset: str,
    ref_ckpt: str,
    model_name: str,
    num_classes: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device,
    corruption: str = "none",
    severity: int = 3,
    corruption_seed: int = 0,
):
    """Build the downstream architecture + the shared test loader (real machine).

    ``ref_ckpt`` is only used so ``load_model_by_name`` can instantiate the right
    architecture; ``predict_from_checkpoint`` reloads the per-model state anyway,
    so the same model object is reused for GT and extrapolation checkpoints.

    When ``corruption`` is not ``"none"`` the test set is wrapped in
    :class:`perturbations.CorruptedDataset`, so an ImageNet-C-style perturbation
    (e.g. a blur) is applied to every test image ON THE FLY. Both the GT-pruned
    and extrapolation-pruned models are then evaluated on the SAME OOD inputs.
    """
    import torch
    from torch.utils.data import DataLoader

    from utils.dataset import get_dataset
    from utils.models import load_model_by_name

    _train, testset = get_dataset(dataset)
    logger.info(f"[item2] {dataset} testset size={len(testset)}")
    if corruption and corruption != "none":
        testset = perturbations.CorruptedDataset(
            testset, corruption, severity=severity, seed=corruption_seed
        )
        logger.info(
            f"[item2] OOD test corruption='{corruption}' severity={severity} "
            f"(applied on the fly)"
        )
    model = load_model_by_name(model_name, num_classes, image_size, ref_ckpt, device)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(getattr(device, "type", "cpu") == "cuda"),
    )
    return model, test_loader

def predict_from_checkpoint(checkpoint_path, model, test_loader, device="cuda"):
    """Per-sample test predictions from a loaded model (real-machine path).

    ``test_loader`` must yield ``(images, labels, sample_idx)``. Returns arrays
    ``(sample_idx, label, pred)``. Kept dependency-light: torch is imported here.
    """
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    model.to(device).eval()
    idxs, labs, preds = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images, labels, sample_idx = batch
            out = model(images.to(device))
            p = out.argmax(1).cpu().numpy()
            idxs.append(np.asarray(sample_idx))
            labs.append(np.asarray(labels))
            preds.append(p)
    return (np.concatenate(idxs), np.concatenate(labs), np.concatenate(preds))


def behavior_preservation(
    label: np.ndarray,
    pred_gt: np.ndarray,
    pred_ext: np.ndarray,
    tail_frac: float = 0.2,
    class_freq: Optional[np.ndarray] = None,
) -> Dict:
    correct_gt = pred_gt == label
    correct_ext = pred_ext == label
    err_gt = set(np.nonzero(~correct_gt)[0].tolist())
    err_ext = set(np.nonzero(~correct_ext)[0].tolist())

    classes = sorted(set(label.tolist()))
    per_class = {}
    for c in classes:
        m = label == c
        per_class[int(c)] = {
            "acc_gt": float(correct_gt[m].mean()),
            "acc_ext": float(correct_ext[m].mean()),
            "n": int(m.sum()),
        }
    worst_gap = max(abs(v["acc_gt"] - v["acc_ext"]) for v in per_class.values())
    mean_gap = float(np.mean([abs(v["acc_gt"] - v["acc_ext"])
                              for v in per_class.values()]))

    # tail = rarest classes (by class_freq if given, else fewest test samples)
    if class_freq is not None:
        order = np.argsort(class_freq)
    else:
        order = np.argsort([per_class[c]["n"] for c in classes])
    n_tail = max(1, int(tail_frac * len(classes)))
    tail_classes = [classes[i] for i in order[:n_tail]]
    tail_mask = np.isin(label, tail_classes)
    tail = {
        "classes": [int(c) for c in tail_classes],
        "acc_gt": float(correct_gt[tail_mask].mean()),
        "acc_ext": float(correct_ext[tail_mask].mean()),
    }

    return {
        "n_test": int(len(label)),
        "overall_acc_gt": float(correct_gt.mean()),
        "overall_acc_ext": float(correct_ext.mean()),
        "error_set_jaccard": rc.jaccard(err_gt, err_ext),
        "prediction_agreement": float(np.mean(pred_gt == pred_ext)),
        "error_agreement": float(
            np.mean((~correct_gt) == (~correct_ext))
        ),
        "worst_group_gap": worst_gap,
        "mean_class_gap": mean_gap,
        "tail": tail,
        "tail_gap": abs(tail["acc_gt"] - tail["acc_ext"]),
        "per_class": per_class,
    }


def _print(res: Dict) -> None:
    logger.info(f"  n_test={res['n_test']} acc_gt={res['overall_acc_gt']:.4f} "
          f"acc_ext={res['overall_acc_ext']:.4f}")
    logger.info(f"  error-set Jaccard   = {res['error_set_jaccard']:.4f}")
    logger.info(f"  prediction agreement= {res['prediction_agreement']:.4f}")
    logger.info(f"  error agreement     = {res['error_agreement']:.4f}")
    logger.info(f"  worst-group gap     = {res['worst_group_gap']:.4f}")
    logger.info(f"  mean class gap      = {res['mean_class_gap']:.4f}")
    logger.info(f"  tail classes {res['tail']['classes']}: "
          f"acc_gt={res['tail']['acc_gt']:.4f} acc_ext={res['tail']['acc_ext']:.4f} "
          f"gap={res['tail_gap']:.4f}")


def _load_pred_npz(path):
    z = np.load(path)
    order = np.argsort(z["sample_idx"])
    return z["label"][order], z["pred"][order]


def run_smoke() -> Dict:
    logger.info("[item2] SMOKE: behavior preservation on synthetic predictions")
    rng = np.random.default_rng(6)
    n, k = 2000, 10
    label = rng.integers(0, k, size=n)
    # GT-pruned model: 85% acc. Extrapolation-pruned: highly correlated errors.
    base_correct = rng.random(n) < 0.85
    pred_gt = np.where(base_correct, label, (label + rng.integers(1, k, n)) % k)
    flip = rng.random(n) < 0.05  # small independent disagreement
    ext_correct = np.where(flip, ~base_correct, base_correct)
    pred_ext = np.where(ext_correct, label, (label + rng.integers(1, k, n)) % k)
    res = behavior_preservation(label, pred_gt, pred_ext, tail_frac=0.2)
    _print(res)
    assert res["error_agreement"] > 0.8, res["error_agreement"]
    assert res["error_set_jaccard"] > 0.5

    # OOD corruption path: ImageNet-C-style blur applied on the fly (torch-free)
    class _NumpyImgDS:
        def __init__(self, n, rng):
            self.imgs = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return self.imgs[i], int(label[i % len(label)]), i

    ds = perturbations.CorruptedDataset(_NumpyImgDS(8, rng), "gaussian_blur", 3, seed=1)
    corr_img, corr_lab, corr_idx = ds[2]
    assert corr_img.shape == (3, 16, 16) and corr_idx == 2
    assert not np.allclose(corr_img, ds.base[2][0]), "corruption left image unchanged"
    logger.info(f"  OOD corruption smoke ok (gaussian_blur, {len(ds)} imgs)")

    logger.info("[item2] SMOKE PASSED")
    return res


def _align_preds(gt_triple, ext_triple):
    """Align two (idx,label,pred) triples on their common sample_idx order."""
    gi, gl, gp = gt_triple
    ei, el, ep = ext_triple
    common = np.intersect1d(gi, ei)
    if len(common) == 0:
        raise ValueError("GT and extrapolation predictions share no sample_idx.")
    g_map = {int(i): j for j, i in enumerate(gi)}
    e_map = {int(i): j for j, i in enumerate(ei)}
    gsel = np.array([g_map[int(i)] for i in common])
    esel = np.array([e_map[int(i)] for i in common])
    label = gl[gsel]
    assert np.array_equal(label, el[esel]), "label mismatch on shared sample_idx"
    return label, gp[gsel], ep[esel]


def run_from_checkpoints(args) -> Dict:
    """Load pruned downstream checkpoints, run test predictions, compare behaviour."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[item2] device={device} dataset={args.dataset}")
    model, test_loader = build_model_and_testloader(
        args.dataset, args.ckpt_gt, args.model_name, args.num_classes,
        args.image_size, args.batch_size, args.num_workers, device,
        corruption=args.corruption, severity=args.severity,
        corruption_seed=args.corruption_seed,
    )
    logger.info(f"[item2] GT-pruned ckpt : {args.ckpt_gt}")
    gt_triple = predict_from_checkpoint(args.ckpt_gt, model, test_loader, device)
    logger.info(f"[item2] extrap  ckpt : {args.ckpt_ext}")
    ext_triple = predict_from_checkpoint(args.ckpt_ext, model, test_loader, device)

    label, pred_gt, pred_ext = _align_preds(gt_triple, ext_triple)
    cf = np.load(args.class_freq) if args.class_freq else None
    res = behavior_preservation(label, pred_gt, pred_ext, args.tail_frac, cf)
    res["ckpt_gt"] = args.ckpt_gt
    res["ckpt_ext"] = args.ckpt_ext
    res["corruption"] = args.corruption
    res["severity"] = args.severity if args.corruption != "none" else None
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        logger.info(f"[item2] wrote {args.out}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 2: behavior preservation")
    ap.add_argument("--smoke", action="store_true")
    # --- checkpoint mode (real machine; default) --------------------------------
    ap.add_argument("--ckpt_gt", default=original_score_checkpoint,
                    help="downstream model pruned with GT scores (default: PLACES_365 TDDS)")
    ap.add_argument("--ckpt_ext", default=knn_extra_checkpoint,
                    help="downstream model pruned with extrapolated scores "
                         "(default: KNN; pass the gnn_extra_checkpoint for the GNN variant)")
    ap.add_argument("--dataset", default="PLACES_365")
    ap.add_argument("--model-name", default="resnet50-self-trained")
    ap.add_argument("--num-classes", type=int, default=365)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    # --- ImageNet-C-style on-the-fly OOD test corruption ------------------------
    ap.add_argument(
        "--corruption", default="none", choices=list(perturbations.CORRUPTION_CHOICES),
        help="apply an ImageNet-C-like perturbation (e.g. a blur) to the test set "
             "on the fly; both models are evaluated on the SAME corrupted inputs",
    )
    ap.add_argument("--severity", type=int, default=3, choices=list(perturbations.SEVERITIES),
                    help="corruption severity 1..5 (ignored when --corruption none)")
    ap.add_argument("--corruption-seed", type=int, default=0,
                    help="seed for the on-the-fly corruption noise")
    # --- npz fallback (standalone / CI) -----------------------------------------
    ap.add_argument("--pred_gt", help="npz sample_idx,label,pred (skips checkpoint mode)")
    ap.add_argument("--pred_ext", help="npz sample_idx,label,pred (skips checkpoint mode)")
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--class_freq", help="optional npy of per-class train frequency")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    if args.pred_gt and args.pred_ext:  # explicit npz fallback
        label_gt, pred_gt = _load_pred_npz(args.pred_gt)
        label_ext, pred_ext = _load_pred_npz(args.pred_ext)
        assert np.array_equal(label_gt, label_ext), "test label order mismatch"
        cf = np.load(args.class_freq) if args.class_freq else None
        res = behavior_preservation(label_gt, pred_gt, pred_ext, args.tail_frac, cf)
        _print(res)
        if args.out:
            rc.save_json(res, args.out)
            logger.info(f"[item2] wrote {args.out}")
        return

    run_from_checkpoints(args)


if __name__ == "__main__":
    main()
