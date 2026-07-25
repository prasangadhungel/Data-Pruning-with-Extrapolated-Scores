"""Shared helpers for the rebuttal (#27899) post-hoc analysis scripts.

These scripts are *standalone* and artifact-driven: they LOAD precomputed
artifacts (embeddings, score dicts, optional checkpoints / prediction arrays)
and compute rebuttal evidence. They intentionally do NOT touch or re-run the
training pipeline in ``src/`` (that clean-up is deferred until acceptance).

Everything here works on plain ``numpy`` arrays so the scripts (and their
``--smoke`` self-tests) run without a GPU, without ``torch`` installed, and
without the real datasets. On the machine that has the artifacts, ``torch`` is
available and ``load_embeddings`` transparently reads the ``.pth`` tensors.

Artifact formats (mirrors the existing repo/notebooks):
  * embeddings : ``savedir/embeddings/<DS>/embeddings_dict.pth`` -> Tensor[N, d]
                 (also accepts ``.npy`` / ``.npz`` for local testing)
  * scores     : JSON ``{ "<idx>": <float>, ... }``  (full S, subset S_s, or
                 extrapolated), keyed by string sample index.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_scores(path: str) -> Dict[int, float]:
    """Load a ``{str(idx): float}`` score dict as ``{int: float}``."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings as a ``float`` ``np.ndarray`` of shape ``[N, d]``.

    Accepts ``.pth`` (torch), ``.npy`` or ``.npz`` (numpy). ``torch`` is only
    imported when a ``.pth`` file is actually loaded, so local/smoke runs never
    require it.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pth", ".pt"):
        import torch  # local import on purpose

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            # dict{idx -> vector}: order by index
            idxs = sorted(int(k) for k in obj.keys())
            arr = np.stack([np.asarray(obj[k]).reshape(-1) for k in idxs])
        else:
            arr = obj.detach().cpu().numpy()
        return np.asarray(arr, dtype=np.float64)
    if ext == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if ext == ".npz":
        z = np.load(path)
        key = "embeddings" if "embeddings" in z else list(z.keys())[0]
        return np.asarray(z[key], dtype=np.float64)
    raise ValueError(f"Unsupported embeddings extension: {ext}")


def scores_to_array(scores: Dict[int, float], n: int, default: float = np.nan) -> np.ndarray:
    """Densify a score dict to an array of length ``n`` (missing -> ``default``)."""
    out = np.full(n, default, dtype=np.float64)
    for k, v in scores.items():
        if 0 <= k < n:
            out[k] = v
    return out


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Index bookkeeping
# --------------------------------------------------------------------------- #
def seed_and_residual(
    subset_scores: Dict[int, float], n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (seed indices = D_s, residual indices = D_r) for ``n`` samples."""
    seed = np.array(sorted(subset_scores.keys()), dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[seed] = False
    residual = np.nonzero(mask)[0]
    return seed, residual


def fit_val_split(
    seed_idx: np.ndarray, val_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the seed set (D_s / S_s) into fit and validation indices.

    The validation split is the ONLY signal allowed for hyperparameter
    selection (the ``S_s``-val rule that fixes the oracle-``k`` critique).
    """
    perm = rng.permutation(len(seed_idx))
    n_val = max(1, int(round(val_frac * len(seed_idx))))
    val = seed_idx[perm[:n_val]]
    fit = seed_idx[perm[n_val:]]
    return fit, val


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of ``scores`` for binary ``labels`` (rank-based, no sklearn dep)."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def topk_retained(scores: Dict[int, float], keep_frac: float) -> List[int]:
    """Indices retained when keeping the top ``keep_frac`` by score (repo rule)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n_keep = int(keep_frac * len(ordered))
    return [k for k, _ in ordered[:n_keep]]


# --------------------------------------------------------------------------- #
# Smoke fixtures (no torch, no data)
# --------------------------------------------------------------------------- #
def make_fixture(
    n: int = 400,
    d: int = 16,
    subset_frac: float = 0.25,
    noise: float = 0.35,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[int, float], Dict[int, float], np.ndarray]:
    """Synthetic (embeddings, subset_scores S_s, full_scores S, labels).

    Scores are a smooth function of the embeddings plus noise, so that
    neighbourhood-based extrapolation is meaningful but imperfect (mirrors the
    "moderate correlation" regime the reviewers ask about).
    """
    rng = np.random.default_rng(seed)
    num_classes = 5
    labels = rng.integers(0, num_classes, size=n)
    centers = rng.normal(size=(num_classes, d))
    emb = centers[labels] + rng.normal(scale=0.6, size=(n, d))
    w = rng.normal(size=d)
    latent = emb @ w
    full = latent + rng.normal(scale=noise * np.std(latent), size=n)
    full = (full - full.min()) / (np.ptp(full) + 1e-9)
    full_scores = {int(i): float(full[i]) for i in range(n)}
    n_sub = int(subset_frac * n)
    seed_idx = rng.choice(n, size=n_sub, replace=False)
    subset_scores = {int(i): float(full[i]) for i in seed_idx}
    return emb, subset_scores, full_scores, labels
