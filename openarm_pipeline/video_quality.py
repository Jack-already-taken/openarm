\
from __future__ import annotations
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from .utils import to_numpy

def normalize_frame(frame: Any) -> np.ndarray:
    img = to_numpy(frame)
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def blur_score(frame: Any) -> float:
    img = normalize_frame(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def exposure_score(frame: Any) -> float:
    img = normalize_frame(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray < 10) + np.mean(gray > 245))

def frame_difference(frame_a: Any, frame_b: Any) -> float:
    a = normalize_frame(frame_a).astype(np.float32)
    b = normalize_frame(frame_b).astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(np.abs(a - b)))

def classify_frame_quality(frame: Any, prev_frame: Optional[Any] = None, blur_threshold=30.0, exposure_threshold=0.50, frozen_frame_threshold=0.5) -> Dict[str, Any]:
    reasons: List[str] = []
    b = blur_score(frame)
    e = exposure_score(frame)
    if b < blur_threshold:
        reasons.append("blur")
    if e > exposure_threshold:
        reasons.append("bad_exposure")
    diff = None
    if prev_frame is not None:
        diff = frame_difference(prev_frame, frame)
        if diff < frozen_frame_threshold:
            reasons.append("frozen_or_duplicate")
    return {"valid": len(reasons) == 0, "reasons": reasons, "blur_score": b, "exposure_score": e, "frame_difference": diff}

def audit_video_stream(dataset: Any, indices: List[int], image_key: Optional[str], stride=5, blur_threshold=30.0, exposure_threshold=0.50, frozen_frame_threshold=0.5) -> Dict[str, Any]:
    if image_key is None:
        return {"available": False, "image_key": None, "num_sampled_frames": 0, "num_bad_frames": 0, "bad_frame_ratio": None, "frame_reports": []}
    frame_reports, prev_frame = [], None
    for idx in indices[::max(1, stride)]:
        sample = dataset[idx]
        q = classify_frame_quality(sample[image_key], prev_frame, blur_threshold, exposure_threshold, frozen_frame_threshold)
        q["dataset_index"] = int(idx)
        frame_reports.append(q)
        prev_frame = sample[image_key]
    num_bad = sum(1 for r in frame_reports if not r["valid"])
    return {"available": True, "image_key": image_key, "num_sampled_frames": len(frame_reports), "num_bad_frames": int(num_bad), "bad_frame_ratio": float(num_bad / max(1, len(frame_reports))), "frame_reports": frame_reports}
