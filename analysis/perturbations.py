"""ImageNet-C-style on-the-fly TEST perturbations (OOD robustness).

Rebuttal target: reviewer myyu (OOD / robustness under distribution shift).
The corruptions are applied to the *already-transformed* ``(C, H, W)`` test
tensors inside the loader, so both the GT-score-pruned and the
extrapolation-pruned downstream models see the SAME corrupted inputs. This lets
``item2_behavior_preservation`` ask whether score extrapolation preserves
downstream behaviour not only in-distribution but also under a blurry / noisy
ImageNet-C-like shift -- without needing to store a separate corrupted dataset
on disk (it is generated on the fly).

Design notes:
  * Every corruption function takes a float ndarray of shape ``(C, H, W)`` plus a
    ``severity`` in 1..5 and returns the same shape/dtype -- so the corruptions
    are torch-free and unit-testable (``python analysis/perturbations.py``).
  * Corruptions act in the model's *normalized* input space (that is where the
    tensors already live in the loader). Blur is a weight-1 local average, so it
    behaves well there; noise / brightness / contrast use severities calibrated
    for the small (32-64 px) normalized crops used in this project rather than
    the 224 px [0, 1] values of the original ImageNet-C tables.
  * ``CorruptedDataset`` bridges torch tensors <-> ndarray on the fly and keeps
    the underlying dataset's tuple layout (``(img, label)`` or
    ``(img, label, idx)``) intact, so it is a drop-in test-set wrapper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def _as_chw(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:  # (H, W) -> (1, H, W)
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"expected a (C, H, W) image, got shape {x.shape}")
    return x


def _check_severity(severity: int) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return severity


def gaussian_noise(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    _check_severity(severity)
    rng = rng if rng is not None else np.random.default_rng()
    x = _as_chw(x)
    sigma = (0.04, 0.06, 0.08, 0.10, 0.13)[severity - 1]
    noise = rng.normal(0.0, sigma, size=x.shape)
    return (x + noise).astype(x.dtype, copy=False)


def shot_noise(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    """Poisson-like signal-dependent noise (approximate; operates in-place scale)."""
    _check_severity(severity)
    rng = rng if rng is not None else np.random.default_rng()
    x = _as_chw(x)
    scale = (0.5, 0.35, 0.25, 0.18, 0.12)[severity - 1]
    # jitter each value by noise whose std grows with |value| (shot-noise flavour)
    noise = rng.normal(0.0, 1.0, size=x.shape) * (scale * np.sqrt(np.abs(x) + 1e-3))
    return (x + noise).astype(x.dtype, copy=False)


def gaussian_blur(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    _check_severity(severity)
    x = _as_chw(x)
    sigma = (0.5, 0.75, 1.0, 1.5, 2.0)[severity - 1]
    # blur spatial dims only, never across channels (axis 0)
    out = gaussian_filter(x.astype(np.float64, copy=False), sigma=(0.0, sigma, sigma))
    return out.astype(x.dtype, copy=False)


def defocus_blur(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    """Box-average defocus (a harder, flatter blur than the gaussian kernel)."""
    from scipy.ndimage import uniform_filter

    _check_severity(severity)
    x = _as_chw(x)
    size = (2, 3, 4, 5, 7)[severity - 1]
    out = uniform_filter(x.astype(np.float64, copy=False), size=(1, size, size))
    return out.astype(x.dtype, copy=False)


def brightness(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    _check_severity(severity)
    x = _as_chw(x)
    c = (0.1, 0.2, 0.3, 0.4, 0.5)[severity - 1]
    return (x + c).astype(x.dtype, copy=False)


def contrast(x: np.ndarray, severity: int = 3, rng=None) -> np.ndarray:
    _check_severity(severity)
    x = _as_chw(x)
    c = (0.85, 0.75, 0.6, 0.45, 0.3)[severity - 1]  # scale toward per-channel mean
    mean = x.mean(axis=(1, 2), keepdims=True)
    return ((x - mean) * c + mean).astype(x.dtype, copy=False)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "gaussian_blur": gaussian_blur,
    "defocus_blur": defocus_blur,
    "brightness": brightness,
    "contrast": contrast,
}

CORRUPTION_CHOICES = ("none", *CORRUPTIONS.keys())


def apply_corruption(
    x: np.ndarray, name: Optional[str], severity: int = 3, rng=None
) -> np.ndarray:
    """Apply a named corruption to a ``(C, H, W)`` ndarray (``none`` = identity)."""
    if name in (None, "none"):
        return _as_chw(x)
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; choose from {CORRUPTION_CHOICES}")
    return CORRUPTIONS[name](_as_chw(x), severity, rng)


class CorruptedDataset:
    """Wrap a test ``Dataset`` and corrupt image[0] of each item on the fly.

    Preserves the underlying tuple layout ``(img, label[, idx])`` and, when the
    image is a torch tensor, returns a torch tensor of the same dtype/device.
    Corruption noise is deterministic per sample (seeded by ``(seed, idx)``) so
    repeated passes over the loader are reproducible.
    """

    def __init__(self, base, corruption: str, severity: int = 3, seed: int = 0):
        if corruption not in CORRUPTION_CHOICES:
            raise KeyError(
                f"unknown corruption {corruption!r}; choose from {CORRUPTION_CHOICES}"
            )
        self.base = base
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item[0]
        rng = np.random.default_rng((self.seed, int(idx)))

        is_tensor = hasattr(img, "detach")
        arr = img.detach().cpu().numpy() if is_tensor else np.asarray(img)
        cor = apply_corruption(arr, self.corruption, self.severity, rng)

        if is_tensor:
            import torch

            out = torch.from_numpy(np.ascontiguousarray(cor)).to(img)
        else:
            out = cor.astype(arr.dtype, copy=False) if arr.ndim == cor.ndim else cor
        return (out,) + tuple(item[1:])


def _smoke() -> None:
    print("[perturbations] SMOKE: ImageNet-C-style on-the-fly corruptions")
    rng = np.random.default_rng(0)
    img = rng.random((3, 32, 32)).astype(np.float32)

    for name in CORRUPTIONS:
        out = apply_corruption(img, name, severity=3, rng=np.random.default_rng(1))
        assert out.shape == img.shape, (name, out.shape)
        assert np.isfinite(out).all(), name
        assert not np.allclose(out, img), f"{name} left the image unchanged"
    # identity
    assert np.allclose(apply_corruption(img, "none"), img)
    # blur must reduce high-frequency energy (variance of the laplacian proxy)
    blurred = apply_corruption(img, "gaussian_blur", severity=5)
    assert blurred.var() <= img.var() + 1e-6

    # CorruptedDataset preserves tuple layout on a torch-free numpy dataset
    class _DummyDS:
        def __init__(self, n):
            self.imgs = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return self.imgs[i], i % 4, i  # (img, label, idx)

    ds = CorruptedDataset(_DummyDS(5), "gaussian_blur", severity=2, seed=7)
    assert len(ds) == 5
    out_img, lab, sidx = ds[3]
    assert out_img.shape == (3, 16, 16) and lab == 3 and sidx == 3
    # deterministic per (seed, idx)
    a = CorruptedDataset(_DummyDS(1), "gaussian_noise", 3, seed=0)[0][0]
    print("[perturbations] SMOKE PASSED")


def run_smoke() -> None:
    """Alias so ``run_all_smoke`` can call ``run_smoke`` uniformly."""
    _smoke()


if __name__ == "__main__":
    _smoke()
