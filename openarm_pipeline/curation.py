\
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from scipy.signal import savgol_filter
from .loaders import DatasetKeys, get_episode_arrays

def smooth_states(states: np.ndarray, window=5) -> np.ndarray:
    if window <= 1 or len(states) < window:
        return states
    if window % 2 == 0:
        window += 1
    return savgol_filter(states, window_length=window, polyorder=min(2, window - 1), axis=0)

def build_alignment_mask(report: Dict[str, Any], num_steps: int, source_indices: List[int]) -> np.ndarray:
    valid = np.ones(num_steps, dtype=bool)
    index_to_local = {idx: i for i, idx in enumerate(source_indices)}
    for fr in report["egocentric_video_report"].get("frame_reports", []):
        local_i = index_to_local.get(int(fr["dataset_index"]))
        if local_i is not None and not fr["valid"]:
            valid[local_i] = False
    return valid

def curate_dataset(dataset: Any, episode_to_indices: Dict[int, List[int]], keys: DatasetKeys, audit_reports: List[Dict[str, Any]], output_dir: str | Path, smooth_window=5, drop_bad_frames_with_mask=True) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    arrays_dir = output_dir / "curated_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    report_by_ep = {int(r["episode_id"]): r for r in audit_reports}
    manifest = []
    for ep_id in sorted(report_by_ep.keys()):
        report = report_by_ep[ep_id]
        indices = episode_to_indices[ep_id]
        keep = bool(report["keep_candidate"])
        states, actions, timestamps = get_episode_arrays(dataset, indices, keys)
        mask = build_alignment_mask(report, len(indices), indices)
        if not drop_bad_frames_with_mask:
            mask[:] = True
        item = {"episode_id": ep_id, "kept": keep, "drop_reasons": list(report["filter_reasons"]), "num_original_steps": len(indices), "num_valid_aligned_steps": int(mask.sum()), "original_indices": [int(i) for i in indices], "valid_alignment_mask": mask.astype(int).tolist()}
        if keep:
            save_path = arrays_dir / f"episode_{ep_id:06d}.npz"
            payload = {"states": states[mask], "states_smooth": smooth_states(states[mask], smooth_window), "timestamps": timestamps[mask], "original_indices": np.asarray(indices)[mask]}
            if actions is not None:
                payload["actions"] = actions[mask]
            np.savez_compressed(save_path, **payload)
            item["array_path"] = str(save_path)
        manifest.append(item)
    return manifest
