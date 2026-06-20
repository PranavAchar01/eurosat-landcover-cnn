"""Data loading for EuroSAT-style land-cover patches.

Supports two sources:
  1. A real on-disk ImageFolder layout (root/<class_name>/*.jpg|png|tif).
  2. A synthetic generator that produces class-correlated RGB patches so the
     training loop, tests, and CI run end-to-end without a multi-hundred-MB
     download. Each EuroSAT class is given a distinct colour/texture prior that
     a CNN can learn, which keeps the demo honest about what the model does.

The 10 EuroSAT classes follow Helber et al. (2019), "EuroSAT: A Novel Dataset
and Deep Learning Benchmark for Land Use and Land Cover Classification".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

try:  # torch is optional at import time so docs/tests can introspect classes
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

CLASSES: List[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Per-class colour prior (mean RGB, 0-1) loosely echoing real spectra:
# vegetation greens, water blues, built-up greys, bare/crop browns.
_CLASS_PRIOR = {
    "AnnualCrop": (0.55, 0.52, 0.30),
    "Forest": (0.13, 0.38, 0.16),
    "HerbaceousVegetation": (0.40, 0.55, 0.28),
    "Highway": (0.45, 0.45, 0.47),
    "Industrial": (0.60, 0.58, 0.56),
    "Pasture": (0.38, 0.60, 0.33),
    "PermanentCrop": (0.50, 0.48, 0.25),
    "Residential": (0.58, 0.50, 0.48),
    "River": (0.20, 0.35, 0.55),
    "SeaLake": (0.10, 0.25, 0.55),
}

IMG_SIZE = 64


def make_synthetic_patch(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Return a (H, W, 3) uint8 patch with a class-specific colour + texture."""
    base = np.array(_CLASS_PRIOR[class_name], dtype=np.float32)
    img = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img[:] = base
    # Spatial texture: low-frequency gradient + speckle, scaled per class.
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] / IMG_SIZE
    grad = (0.15 * (np.sin(6 * xx + rng.uniform(0, 6)) +
                    np.cos(6 * yy + rng.uniform(0, 6))))[..., None]
    speckle = rng.normal(0, 0.05, size=img.shape)
    img = img + grad + speckle
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def make_synthetic_dataset(out_dir: str, per_class: int = 60, seed: int = 0) -> str:
    """Write a synthetic ImageFolder dataset to ``out_dir`` and return the path."""
    rng = np.random.default_rng(seed)
    for cls in CLASSES:
        d = os.path.join(out_dir, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(per_class):
            patch = make_synthetic_patch(cls, rng)
            Image.fromarray(patch).save(os.path.join(d, f"{cls}_{i:04d}.png"))
    return out_dir


@dataclass
class _Sample:
    path: str
    label: int


class EuroSATDataset(Dataset):
    """ImageFolder-style dataset returning (CHW float tensor, label)."""

    def __init__(self, root: str, classes: List[str] | None = None,
                 normalize: bool = True):
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to use EuroSATDataset.")
        self.root = root
        self.classes = classes or CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.normalize = normalize
        self.samples: List[_Sample] = []
        for cls in self.classes:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    self.samples.append(_Sample(os.path.join(d, fn),
                                                self.class_to_idx[cls]))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {root!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        arr = np.asarray(Image.open(s.path).convert("RGB"), dtype=np.float32) / 255.0
        if self.normalize:
            arr = (arr - 0.5) / 0.5
        chw = np.transpose(arr, (2, 0, 1)).copy()
        return torch.from_numpy(chw), s.label


def train_val_split(n: int, val_frac: float = 0.2, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]
