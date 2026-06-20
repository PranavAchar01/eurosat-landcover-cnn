# EuroSAT Land-Cover CNN

A compact, CPU-trainable convolutional neural network for **land use / land cover (LULC) classification** of Sentinel-2 satellite image patches, following the 10-class [EuroSAT benchmark](https://github.com/phelber/EuroSAT) (Helber et al., 2019).

This project demonstrates an end-to-end **image classification** pipeline for remote-sensing imagery: data ingestion, normalization, a convolutional classifier, training/validation, and single-patch inference. The full loop runs in seconds on a laptop using a bundled synthetic data generator, while also accepting the real EuroSAT dataset unchanged.

## Why this matters for Earth observation

Land cover maps derived from satellite imagery underpin NASA Earth Action themes such as agriculture, ecological conservation, and water resources. Classifying surface type from spectral/spatial patterns is the core building block — this repo implements that block from scratch with PyTorch.

## Quickstart

```bash
pip install -e ".[dev]"          # installs torch, numpy, pillow, pytest

# Train on an auto-generated synthetic dataset (no download needed)
eurosat-train --epochs 5

# Or point at a real EuroSAT ImageFolder (root/<ClassName>/*.jpg)
eurosat-train --data-dir /path/to/EuroSAT --epochs 20 --out model.pt

# Predict on a single patch
eurosat-predict path/to/patch.png --ckpt model.pt
```

Run the bundled demo (synthetic data -> train -> predict):

```bash
python examples/quickstart.py
```

## Real EuroSAT data

Download the RGB version from the [official repository](https://github.com/phelber/EuroSAT) and unzip into an ImageFolder layout:

```
EuroSAT/
  AnnualCrop/  Forest/  HerbaceousVegetation/  Highway/  Industrial/
  Pasture/  PermanentCrop/  Residential/  River/  SeaLake/
```

Then pass `--data-dir EuroSAT`. No code changes required.

## Architecture

`LandCoverCNN` is a VGG-style network: three convolutional blocks (Conv-BN-ReLU x2 + MaxPool) widening 32->64->128 channels, global average pooling, and a dropout + linear head. ~0.5M parameters — small enough to train on CPU, deep enough to learn spatial texture rather than just colour.

| Component | Choice |
|-----------|--------|
| Input | 64x64x3, normalized to [-1, 1] |
| Backbone | 3 conv blocks, BatchNorm, MaxPool |
| Head | GAP -> Dropout(0.3) -> Linear(10) |
| Loss / Opt | CrossEntropy / Adam |

## The synthetic generator

To keep CI and demos fast and reproducible, `make_synthetic_dataset` writes class-correlated RGB patches: each EuroSAT class gets a distinct colour prior (vegetation greens, water blues, built-up greys) plus low-frequency texture and speckle. This is a stand-in for spectra, not a claim of real-world accuracy — swap in `--data-dir` for genuine results.

## Tests

```bash
pytest -q
```

Tests cover dataset round-tripping, model output shapes, parameter budget, and a learning check that the network drives loss down on a fixed batch.

## License

MIT — see [LICENSE](LICENSE).
