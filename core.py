"""Original audit/evaluation code. Binary labels; one prediction per patient."""
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


def sha256_file(path):
    """O(file bytes) time, O(1 MiB) auxiliary memory."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path):
    """O(N) metadata; reject patient overlap and non-external test sites."""
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array")
    patient_info, paths = {}, set()
    for r in rows:
        for key in ("patient_id", "site", "split", "path", "label", "date"):
            if key not in r or r[key] is None or r[key] == "":
                raise ValueError(f"Missing {key}")
        if type(r["label"]) is not int or r["label"] not in (0, 1):
            raise ValueError("Labels must be integer 0 or 1")
        if r["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if not isinstance(r["patient_id"], str) or not isinstance(r["site"], str):
            raise ValueError("Patient and site IDs must be strings")
        if date.fromisoformat(r["date"]).isoformat() != r["date"]:
            raise ValueError("Dates must use YYYY-MM-DD")
        info = (r["split"], r["label"], r["site"])
        old = patient_info.setdefault(r["patient_id"], info)
        if old != info:
            raise ValueError("Patient overlap or inconsistent label/site")
        image_path = (path.parent / r["path"]).resolve()
        if not image_path.is_file() or image_path in paths:
            raise ValueError("Missing or repeated image path")
        paths.add(image_path)
        r["path"] = str(image_path)
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    dev_sites = {r["site"] for r in rows if r["split"] != "test"}
    if dev_sites & {r["site"] for r in rows if r["split"] == "test"}:
        raise ValueError("Test sites must be external to train/validation")
    if max(r["date"] for r in rows if r["split"] != "test") >= min(
        r["date"] for r in rows if r["split"] == "test"
    ):
        raise ValueError("External test must be later than development data")
    return rows


def audit_pixels(rows):
    """O(total decoded pixels) time; reject exact decoded duplicates across splits."""
    import cv2
    seen = {}
    for r in rows:
        pixels = cv2.imread(r["path"], cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError("Unreadable image")
        h = hashlib.sha256()
        h.update(str((pixels.shape, str(pixels.dtype))).encode())
        h.update(pixels.tobytes())
        digest = h.hexdigest()
        if digest in seen and seen[digest] != r["split"]:
            raise ValueError("Decoded pixel duplicate across splits")
        seen[digest] = r["split"]
    return {"images": len(rows), "unique_pixel_hashes": len(seen)}


def aggregate_patients(rows, probabilities):
    """O(N) time/memory: fixed arithmetic mean per patient, decided before testing."""
    if len(rows) != len(probabilities):
        raise ValueError("Prediction count mismatch")
    groups = {}
    for r, p in zip(rows, probabilities):
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid probability")
        key = r["patient_id"]
        if key not in groups:
            groups[key] = [r["label"], r["site"], 0.0, 0]
        g = groups[key]
        if (g[0], g[1]) != (r["label"], r["site"]):
            raise ValueError("Inconsistent patient metadata")
        g[2] += float(p)
        g[3] += 1
    values = list(groups.values())
    return (np.array([v[0] for v in values]),
            np.array([v[2] / v[3] for v in values]),
            np.array([v[1] for v in values]))


def binary_metrics(y, p, threshold=0.5):
    """O(M log M) time, O(M) memory. Exact tie-aware AUROC and step-wise AP."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected equal nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels/probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    y = y.astype(np.int64)
    pred = p >= threshold
    tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    fp, fn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    positives, negatives = int(y.sum()), int((1-y).sum())
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    auroc = ap = None
    if positives and negatives:
        order = np.argsort(-p, kind="stable")
        sy, sp = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sp)), len(sp)-1]
        tps = np.cumsum(sy)[ends]
        fps = ends + 1 - tps
        recall = np.r_[0., tps / positives]
        fpr = np.r_[0., fps / negatives]
        auroc = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
        ap = float(np.sum(np.diff(recall) * tps / (tps + fps)))
    return {"patients": len(y), "positive": positives, "negative": negatives,
            "prevalence": float(y.mean()), "auroc": auroc, "average_precision": ap,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity)/2 if positives and negatives else None,
            "precision": tp/(tp+fp) if tp+fp else None,
            "brier": float(np.mean((p-y)**2)), "threshold": threshold,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def bootstrap_ci(y, p, repeats=500, seed=42):
    """O(R M log M), O(M+R) memory, stratified patient bootstrap."""
    y, p = np.asarray(y), np.asarray(p)
    binary_metrics(y, p)
    if repeats < 20:
        raise ValueError("Use at least 20 bootstrap replicates")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return {"status": "undefined_single_class"}
    rng, samples = np.random.default_rng(seed), defaultdict(list)
    for _ in range(repeats):
        ix = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        result = binary_metrics(y[ix], p[ix])
        for name in ("auroc", "average_precision", "balanced_accuracy", "brier"):
            samples[name].append(result[name])
    return {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in samples.items()}


def report_by_site(y, p, sites, repeats=500):
    """O(R M log M + M K + sum M_k log M_k); K sites."""
    result = {"overall": binary_metrics(y, p), "patient_bootstrap_95_ci": bootstrap_ci(y, p, repeats),
              "sites": {}, "ci_scope": "conditional on observed sites and class counts"}
    for site in np.unique(sites):
        mask = sites == site
        result["sites"][str(site)] = binary_metrics(y[mask], p[mask])
    return result
