"""Item 2 - Behavior preservation: error-set overlap + subgroup accuracy.

Rebuttal target: reviewer myyu W2 ("does the pruned model fail on the SAME
examples?") + subgroup / tail robustness. This is a Tier-B, no-retraining
analysis: it consumes the per-sample TEST predictions of two already-trained
downstream models -- one pruned with the ground-truth scores ``S``, one pruned
with the extrapolated scores -- and quantifies whether they behave the same:

  * error-set overlap (Jaccard) and prediction agreement;
  * per-class / worst-group / tail accuracy gaps.

On the real machine the predictions come from loaded checkpoints (the repo
already loads ``*_model.pth`` in ``analysis/analyse_scores.py``); use
``predict_from_checkpoint`` for that. For standalone/CI use, the script also
accepts prediction arrays saved as ``.npz`` with fields
``sample_idx, label, pred``.

Self-test:
    python analysis/item2_behavior_preservation.py --smoke
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np

import rebuttal_common as rc

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
    print(f"  n_test={res['n_test']} acc_gt={res['overall_acc_gt']:.4f} "
          f"acc_ext={res['overall_acc_ext']:.4f}")
    print(f"  error-set Jaccard   = {res['error_set_jaccard']:.4f}")
    print(f"  prediction agreement= {res['prediction_agreement']:.4f}")
    print(f"  error agreement     = {res['error_agreement']:.4f}")
    print(f"  worst-group gap     = {res['worst_group_gap']:.4f}")
    print(f"  mean class gap      = {res['mean_class_gap']:.4f}")
    print(f"  tail classes {res['tail']['classes']}: "
          f"acc_gt={res['tail']['acc_gt']:.4f} acc_ext={res['tail']['acc_ext']:.4f} "
          f"gap={res['tail_gap']:.4f}")


def _load_pred_npz(path):
    z = np.load(path)
    order = np.argsort(z["sample_idx"])
    return z["label"][order], z["pred"][order]


def run_smoke() -> Dict:
    print("[item2] SMOKE: behavior preservation on synthetic predictions")
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
    print("[item2] SMOKE PASSED")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 2: behavior preservation")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pred_gt", help="npz with sample_idx,label,pred (GT-pruned model)")
    ap.add_argument("--pred_ext", help="npz with sample_idx,label,pred (extrap-pruned)")
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--class_freq", help="optional npy of per-class train frequency")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.pred_gt and args.pred_ext):
        ap.error("need --pred_gt and --pred_ext npz files (or --smoke)")

    label_gt, pred_gt = _load_pred_npz(args.pred_gt)
    label_ext, pred_ext = _load_pred_npz(args.pred_ext)
    assert np.array_equal(label_gt, label_ext), "test label order mismatch"
    cf = np.load(args.class_freq) if args.class_freq else None
    res = behavior_preservation(label_gt, pred_gt, pred_ext, args.tail_frac, cf)
    _print(res)
    if args.out:
        rc.save_json(res, args.out)
        print(f"[item2] wrote {args.out}")


if __name__ == "__main__":
    main()
