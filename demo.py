"""Entirely synthetic integration test, NOT a medical accuracy benchmark."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class ToyEncoder(nn.Module):
    def encode_image(self, x):
        """O(BHW) toy color feature; no clinical meaning."""
        return x.mean(dim=(-2, -1))


def toy_transform(image):
    """O(HW) time/memory."""
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255


def make_demo(root):
    """O(NHW): reproducible artificial images, no personal/medical data."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng, rows = np.random.default_rng(2026), []
    for split, n in (("train", 40), ("val", 20), ("test", 20)):
        for i in range(n):
            label = int(i % 4 == 0)
            image = rng.integers(0, 80, (32, 32, 3), dtype=np.uint8)
            image[:, :, label*2] += 140
            name = f"{split}_{i}.png"
            if not cv2.imwrite(str(root / name), image):
                raise OSError("Cannot write synthetic image")
            rows.append({"path": name, "patient_id": f"synthetic_{split}_{i}",
                         "label": label, "site": "synthetic_B" if split == "test" else "synthetic_A",
                         "date": "2025-01-01" if split == "test" else "2024-01-01", "split": split})
    path = root / "manifest.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    from train import run
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/demo")
    args = parser.parse_args()
    base = Path(args.output)
    manifest = make_demo(base / "data")
    result = run(manifest, None, base / "result", epochs=8, demo=True)
    print(json.dumps({"status": "synthetic demo completed", "demo_only": result["demo_only"]}))
