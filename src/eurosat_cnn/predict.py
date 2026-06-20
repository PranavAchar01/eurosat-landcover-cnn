"""Run inference on a single image patch with a trained checkpoint."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from .model import LandCoverCNN


def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = LandCoverCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def predict_image(model, classes, image_path: str):
    arr = np.asarray(Image.open(image_path).convert("RGB").resize((64, 64)),
                     dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None].copy())
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = int(probs.argmax())
    return classes[top], float(probs[top]), {c: float(p) for c, p in zip(classes, probs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--ckpt", default="landcover_cnn.pt")
    args = p.parse_args()
    model, classes = load_model(args.ckpt)
    label, conf, _ = predict_image(model, classes, args.image)
    print(f"{label}  ({conf:.1%})")


if __name__ == "__main__":
    main()
