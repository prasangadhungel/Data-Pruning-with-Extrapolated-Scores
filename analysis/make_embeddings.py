"""Standalone embedding generator (rebuttal helper).

Reproduces the exact embedding computation used by the extrapolation pipeline
(gnn_extrapolate.py / pprgo_extrapolate.py) as a standalone script, because no
``embeddings_dict.pth`` is persisted on disk (``save_embeddings: false`` in every
config, so the pipeline recomputes embeddings in-memory each run).

Defaults target IMAGENET (imagenet64), for which both a full-data and a
subset-data proxy checkpoint exist.

--------------------------------------------------------------------------------
WHAT IT DOES (mirrors src/extrapolate/*_extrapolate.py):
  1. build the proxy backbone with utils.models.load_model_by_name(model_name, ...)
  2. load the proxy state_dict from --ckpt
  3. run it (eval, no_grad) over the trainset returned by utils.dataset.get_dataset
  4. store one vector per sample, keyed by the dataset's sample_idx
  5. save as a dense Tensor[N, D]  (--format tensor, default; what gnn/knn expect)
     or as dict{sample_idx -> Tensor[1, D]} (--format dict; what pprgo expects)

IMPORTANT — logits vs features:
  load_model_by_name returns the FULL model (ResNetEmbedding is commented out in
  models.py), so the default embedding is the model's fc/linear LOGITS
  => D == num_classes (1000 for IMAGENET). This is faithful to the current paper
  pipeline. Pass --features to instead capture the PENULTIMATE features (input to
  the final linear layer, e.g. 512-d for ResNet18) via a forward hook. Using
  --features is a deliberate deviation from the paper's runs.

--------------------------------------------------------------------------------
DATA TO LOAD (real run). Root on the shared store:
  ROOT = /ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning
  (older mirror: /nfs/homedirs/dhp/unsupervised-data-pruning)
  IMAGENET proxy checkpoints (state_dict of a resnet18-self-trained backbone;
  the config's ``models.resnet50.path`` key is just a path holder — the real
  architecture is ``models.names[0] = "resnet18-self-trained"``):
    subset : ROOT/models/imagenet/subset_data/tdds_256234.pth               (config default)
             ROOT/models/imagenet/subset_data/dynamic_uncertainty_256234.pth
    full   : ROOT/models/imagenet/full_data/dual_model.pth
  imagenet64 data (read by utils.dataset.get_dataset("IMAGENET")):
    /ceph/ssd/shared/datasets/imagenet/imagenet64/{train_data_batch_1..10.npz, val_data.npz}
  Output (choose any local/CEPH path):
    ROOT/savedir/embeddings/imagenet/embeddings_dict.pth

EXAMPLE (subset proxy, faithful logits):
  python analysis/make_embeddings.py \
    --dataset IMAGENET \
    --ckpt /ceph/.../models/imagenet/subset_data/tdds_256234.pth \
    --out  /ceph/.../savedir/embeddings/imagenet/embeddings_dict.pth

EXAMPLE (full proxy, penultimate features, pprgo dict format):
  python analysis/make_embeddings.py \
    --dataset IMAGENET \
    --ckpt /ceph/.../models/imagenet/full_data/dual_model.pth \
    --features --format dict \
    --out  /ceph/.../savedir/embeddings/imagenet/embeddings_full_feat.pth

Other datasets: --dataset CIFAR10 --num-classes 10 --image-size 32 --model-name
resnet50-self-trained (etc.); CIFAR10/CIFAR100 auto-download to ./data.

Run from the REPO ROOT so ``from utils...`` resolves (the script also injects
``<repo>/src`` on sys.path). Requires torch + torchvision; a GPU is optional.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _last_linear(model):
    """Return the last nn.Linear submodule (``fc`` for ResNet, ``linear`` for the
    ResNet_32/ResNet_64 variants)."""
    import torch.nn as nn

    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise RuntimeError("No nn.Linear layer found to hook for --features.")
    return last


def main() -> int:
    p = argparse.ArgumentParser(description="Generate proxy embeddings for extrapolation.")
    p.add_argument("--dataset", default="IMAGENET",
                   help="Dataset name understood by utils.dataset.get_dataset (default: IMAGENET).")
    p.add_argument("--ckpt", required=True,
                   help="Path to the proxy backbone state_dict (.pth).")
    p.add_argument("--out", required=True,
                   help="Output path for the embeddings (.pth).")
    p.add_argument("--model-name", default="resnet18-self-trained",
                   help="Backbone id for load_model_by_name (IMAGENET config uses "
                        "resnet18-self-trained; CIFAR uses resnet50-self-trained).")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--features", action="store_true",
                   help="Capture penultimate features (input to the final linear) "
                        "instead of logits. Deviates from the paper pipeline.")
    p.add_argument("--format", choices=["tensor", "dict"], default="tensor",
                   help="tensor: Tensor[N, D] indexed by sample_idx (gnn/knn). "
                        "dict: {sample_idx -> Tensor[1, D]} (pprgo).")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on #samples processed (debug; 0 = all).")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from utils.dataset import get_dataset
    from utils.models import load_model_by_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[make_embeddings] device={device} dataset={args.dataset} "
          f"model={args.model_name} features={args.features} format={args.format}")

    trainset, _ = get_dataset(args.dataset)
    n_total = len(trainset)
    n = n_total if args.limit <= 0 else min(args.limit, n_total)
    print(f"[make_embeddings] trainset size={n_total} (processing {n})")

    model = load_model_by_name(
        args.model_name, args.num_classes, args.image_size, args.ckpt, device
    ).eval()

    # Optional penultimate-feature capture via a forward hook on the final linear.
    captured = {}
    hook_handle = None
    if args.features:
        def _hook(_module, inputs, _output):
            captured["feat"] = inputs[0].detach()
        hook_handle = _last_linear(model).register_forward_hook(_hook)

    def embed(batch_imgs):
        out = model(batch_imgs)
        return captured["feat"] if args.features else out

    # Infer embedding dim from a single sample.
    sample0 = trainset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        dim = embed(sample0).shape[1]
    print(f"[make_embeddings] embedding_dim={dim} "
          f"({'penultimate features' if args.features else 'logits'})")

    loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    if args.format == "tensor":
        store = torch.zeros(n_total, dim)
    else:
        store = {}

    seen = 0
    with torch.no_grad():
        for imgs, _labels, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            vecs = embed(imgs).cpu()
            idxs = idxs.to(torch.long)
            if args.format == "tensor":
                store[idxs] = vecs
            else:
                for j, sample_idx in enumerate(idxs.tolist()):
                    store[int(sample_idx)] = vecs[j : j + 1]
            seen += imgs.shape[0]
            if args.limit > 0 and seen >= args.limit:
                break
            if seen % (args.batch_size * 20) == 0:
                print(f"[make_embeddings]   {seen}/{n} done")

    if hook_handle is not None:
        hook_handle.remove()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(store, args.out)
    kind = "Tensor[N, D]" if args.format == "tensor" else "dict{idx -> Tensor[1, D]}"
    print(f"[make_embeddings] saved {kind} for {seen} samples (dim={dim}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
