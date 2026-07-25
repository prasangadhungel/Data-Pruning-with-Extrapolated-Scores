"""Long-tail helpers - build a CIFAR-100-LT style class-imbalanced index set.

Used by ``longtail_fidelity.py``. Exponential-profile imbalance (the standard
CIFAR-100-LT construction): class ``c`` keeps ``n_c = n_max * imb_factor**(c/(C-1))``
samples, where ``imb_factor`` (e.g. 0.01) is the ratio of the rarest to the most
frequent class. Operates on index arrays only -- no image data needed, so it
composes with the artifact-driven score analyses.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def longtail_class_sizes(
    class_counts: Dict[int, int], imb_factor: float = 0.01
) -> Dict[int, int]:
    """Target per-class sizes for an exponential long-tail profile."""
    classes = sorted(class_counts.keys())
    C = len(classes)
    sizes = {}
    for rank, c in enumerate(classes):
        frac = imb_factor ** (rank / max(1, C - 1))
        sizes[c] = max(1, int(round(class_counts[c] * frac)))
    return sizes


def make_longtail_indices(
    labels: np.ndarray, imb_factor: float = 0.01, seed: int = 0
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Subsample indices to an exponential long-tail; return (indices, sizes)."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels.tolist()))
    counts = {c: int((labels == c).sum()) for c in classes}
    # most-frequent class first so rank 0 is the head
    ordered = sorted(classes, key=lambda c: -counts[c])
    ranked_counts = {c: counts[c] for c in ordered}
    sizes = longtail_class_sizes(ranked_counts, imb_factor)
    keep = []
    for c in ordered:
        idx_c = np.nonzero(labels == c)[0]
        take = min(sizes[c], len(idx_c))
        keep.append(rng.choice(idx_c, size=take, replace=False))
    indices = np.sort(np.concatenate(keep))
    return indices, sizes


def head_tail_split(
    sizes: Dict[int, int], tail_frac: float = 0.5
) -> Tuple[List[int], List[int]]:
    """Split classes into head vs tail by frequency (rarest -> tail)."""
    ordered = sorted(sizes.keys(), key=lambda c: sizes[c])  # rarest first
    n_tail = max(1, int(tail_frac * len(ordered)))
    tail = ordered[:n_tail]
    head = ordered[n_tail:]
    return head, tail
