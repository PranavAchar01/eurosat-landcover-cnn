import os
import numpy as np
from eurosat_cnn.data import (CLASSES, make_synthetic_dataset, make_synthetic_patch,
                              EuroSATDataset, train_val_split)


def test_classes_count():
    assert len(CLASSES) == 10


def test_synthetic_patch_shape():
    rng = np.random.default_rng(0)
    p = make_synthetic_patch("Forest", rng)
    assert p.shape == (64, 64, 3) and p.dtype == np.uint8


def test_dataset_roundtrip(tmp_path):
    root = make_synthetic_dataset(str(tmp_path / "ds"), per_class=4, seed=1)
    ds = EuroSATDataset(root)
    assert len(ds) == 40
    x, y = ds[0]
    assert tuple(x.shape) == (3, 64, 64)
    assert 0 <= y < 10


def test_split_disjoint():
    tr, va = train_val_split(100, 0.2, seed=0)
    assert len(tr) == 80 and len(va) == 20
    assert set(tr).isdisjoint(set(va))
