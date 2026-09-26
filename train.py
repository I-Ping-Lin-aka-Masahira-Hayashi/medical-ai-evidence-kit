"""Frozen official MobileCLIP-S0 + patient-weighted binary linear probe.

This is an original experiment harness, not a reproduction of a medical paper.
No checkpoint, patient information, or third-party implementation is distributed.
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
# Set before importing torch / creating a CUDA context; see official randomness notes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader

from core import (load_manifest, audit_pixels, aggregate_patients, report_by_site,
                  sha256_file)


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """O(HW + preprocessing) time; reject lossy implicit 16-bit conversion."""
        if Path(self.rows[index]["path"]).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Only PNG/JPEG input is supported")
        raw = cv2.imread(self.rows[index]["path"], cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError("Only curated 8-bit 3-channel PNG/JPEG images are supported")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb))


def seed_worker(worker_id):
    """O(1); avoid identical worker RNG states and OpenCV thread oversubscription."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def extract_features(model, transform, rows, target, device, batch=16, workers=0):
    """O(N F_enc) time; batch-sized tensors plus OS-managed O(ND) mmap pages."""
    loader = DataLoader(Images(rows, transform), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, persistent_workers=False)
    model.eval().requires_grad_(False)
    features, offset = None, 0
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            # FP32 default: no assumed speedup or diagnostic equivalence from AMP.
            z = torch.nn.functional.normalize(model.encode_image(images).float(), dim=-1)
            array = z.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite encoder output")
            if features is None:
                features = np.lib.format.open_memmap(target, mode="w+", dtype="float32",
                                                    shape=(len(rows), array.shape[1]))
            features[offset:offset+len(array)] = array
            offset += len(array)
    if offset != len(rows):
        raise ValueError("Feature count mismatch")
    features.flush()
    return features


def patient_weights(rows):
    """O(N); each patient contributes equal total training weight."""
    counts = Counter(r["patient_id"] for r in rows)
    w = np.array([1 / counts[r["patient_id"]] for r in rows], dtype=np.float32)
    return w / w.mean()


def predict(head, features, indices, device, batch=256):
    """O(MD) time; O(batch*D+M) auxiliary memory, GPU graph disabled."""
    output = np.empty(len(indices), dtype=np.float64)
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            ix = indices[start:start+batch]
            x = torch.from_numpy(np.array(features[ix], copy=True)).to(device)
            output[start:start+len(ix)] = head(x).squeeze(-1).sigmoid().cpu().numpy()
    return output


def fit_probe(features, rows, device, epochs=50, batch=256, seed=42, patience=5):
    """O(E N_train D + E N_val D), O(batch*D + D + N) extra memory."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    weights = patient_weights([rows[i] for i in train_ix])
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    head = nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, stalled, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(train_ix))
        for start in range(0, len(order), batch):
            local_ix = order[start:start+batch]
            ix = train_ix[local_ix]
            x = torch.from_numpy(np.array(features[ix], copy=True)).to(device)
            y = torch.from_numpy(labels[ix]).to(device)
            w = torch.from_numpy(weights[local_ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(-1), y, reduction="none")
            loss = (losses * w).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        val_p = predict(head, features, val_ix, device, batch)
        yv, pv, _ = aggregate_patients([rows[i] for i in val_ix], val_p)
        pv = np.clip(pv, 1e-7, 1-1e-7)
        val_loss = float(-np.mean(yv*np.log(pv)+(1-yv)*np.log1p(-pv)))
        history.append({"epoch": epoch+1, "validation_patient_bce": val_loss})
        if val_loss < best:
            best, stalled = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stalled += 1
        if stalled >= patience:
            break
    head.load_state_dict(best_state)
    return head, history


def run(manifest, checkpoint, output, batch=16, workers=0, epochs=50, seed=42, demo=False):
    """Orchestrator: see docs/complexity.md for composed costs."""
    if min(batch, epochs) < 1 or workers < 0:
        raise ValueError("Invalid run parameters")
    rows = load_manifest(manifest)
    audit = audit_pixels(rows)
    output = Path(output)
    # No accidental reuse of a test set/result directory.
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if demo:
        from demo import ToyEncoder, toy_transform
        model, transform = ToyEncoder().to(device), toy_transform
        checkpoint_hash = "SYNTHETIC_DEMO_NO_PRETRAINED_MODEL"
    else:
        import mobileclip
        checkpoint_hash = sha256_file(checkpoint)
        model, _, transform = mobileclip.create_model_and_transforms(
            "mobileclip_s0", pretrained=str(checkpoint))
        model = model.to(device)
    features = extract_features(model, transform, rows, output / "features.npy", device, batch, workers)
    del model  # drop encoder before learning a small head; no per-batch empty_cache.
    head, history = fit_probe(features, rows, device, epochs=epochs, seed=seed)
    test_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    probabilities = predict(head, features, test_ix, device)
    y, p, sites = aggregate_patients([rows[i] for i in test_ix], probabilities)
    report = report_by_site(y, p, sites)
    report.update({"demo_only": demo, "audit": audit, "history": history,
                   "seed": seed, "checkpoint_sha256": checkpoint_hash,
                   "manifest_sha256": sha256_file(manifest), "device": str(device),
                   "versions": {"torch": str(torch.__version__), "opencv": cv2.__version__, "numpy": np.__version__}})
    torch.save({"state_dict": head.cpu().state_dict(), "feature_dim": features.shape[1],
                "encoder_sha256": checkpoint_hash, "threshold": 0.5}, output / "probe.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))
