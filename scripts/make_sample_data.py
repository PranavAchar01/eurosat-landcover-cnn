"""CLI: write a synthetic EuroSAT-style dataset for quick experiments."""
import argparse
from eurosat_cnn.data import make_synthetic_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--per-class", type=int, default=80)
    a = p.parse_args()
    path = make_synthetic_dataset(a.out, per_class=a.per_class)
    print(f"Wrote synthetic dataset -> {path}")
