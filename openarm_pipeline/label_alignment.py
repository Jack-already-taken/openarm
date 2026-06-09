\
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

def nearest_index(timestamps: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(timestamps) - t)))

def align_temporal_labels_to_states(label_json_path: str | Path, timestamps: np.ndarray) -> List[Dict[str, Any]]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    aligned = []
    for label in labels:
        start_idx = nearest_index(timestamps, float(label["start_time"]))
        end_idx = nearest_index(timestamps, float(label["end_time"]))
        aligned.append({"label": label["label"], "start_time": float(label["start_time"]), "end_time": float(label["end_time"]), "start_joint_index": start_idx, "end_joint_index": end_idx})
    return aligned

def frame_level_label_table(timestamps: np.ndarray, aligned_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table = []
    for i, t in enumerate(timestamps):
        active = [seg["label"] for seg in aligned_segments if seg["start_joint_index"] <= i <= seg["end_joint_index"]]
        table.append({"index": i, "timestamp": float(t), "labels": active})
    return table
