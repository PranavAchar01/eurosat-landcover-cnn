"""EuroSAT land-cover classification with a compact CNN (PyTorch)."""
from .model import LandCoverCNN
from .data import EuroSATDataset, CLASSES, make_synthetic_dataset

__all__ = ["LandCoverCNN", "EuroSATDataset", "CLASSES", "make_synthetic_dataset"]
__version__ = "0.1.0"
