\
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import torch
from tqdm.auto import tqdm

def to_numpy(x: Any) -> Any:
    """Convert torch tensor / scalar / list / image-like object to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return x

def scalar(x: Any) -> Any:
    arr = to_numpy(x)
    if isinstance(arr, np.ndarray):
        if arr.shape == ():
            return arr.item()
        return arr.reshape(-1)[0].item()
    return x

def find_first_key(sample: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in sample:
            return key
    return None

def find_state_key(sample: Dict[str, Any]) -> Optional[str]:
    key = find_first_key(sample, ["observation.state", "observation.joint_positions", "observation.joints", "state"])
    if key is not None:
        return key
    for k in sample.keys():
        kl = k.lower()
        if "state" in kl or "joint" in kl:
            arr = to_numpy(sample[k])
            if isinstance(arr, np.ndarray) and arr.ndim <= 2:
                return k
    return None

def find_action_key(sample: Dict[str, Any]) -> Optional[str]:
    key = find_first_key(sample, ["action", "actions"])
    if key is not None:
        return key
    for k in sample.keys():
        if "action" in k.lower():
            return k
    return None

def find_timestamp_key(sample: Dict[str, Any]) -> Optional[str]:
    key = find_first_key(sample, ["timestamp", "timestamps", "time"])
    if key is not None:
        return key
    for k in sample.keys():
        if "time" in k.lower():
            return k
    return None

def find_image_keys(sample: Dict[str, Any]) -> List[str]:
    image_keys: List[str] = []
    for k, v in sample.items():
        kl = k.lower()
        if "image" in kl or "camera" in kl or "frame" in kl:
            arr = to_numpy(v)
            if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                image_keys.append(k)
    return image_keys

def choose_wrist_or_ego_camera(image_keys: List[str]) -> Optional[str]:
    for term in ["wrist", "ego", "egocentric", "hand"]:
        for k in image_keys:
            if term in k.lower():
                return k
    return image_keys[0] if image_keys else None

def get_episode_indices(dataset: Any) -> Dict[int, List[int]]:
    episode_to_indices: Dict[int, List[int]] = defaultdict(list)
    if hasattr(dataset, "hf_dataset"):
        hfds = dataset.hf_dataset
        if hasattr(hfds, "column_names") and "episode_index" in hfds.column_names:
            for idx, ep in enumerate(hfds["episode_index"]):
                episode_to_indices[int(ep)].append(idx)
            return dict(episode_to_indices)
    for idx in tqdm(range(len(dataset)), desc="Scanning episode_index"):
        sample = dataset[idx]
        if "episode_index" in sample:
            ep = int(scalar(sample["episode_index"]))
        elif "episode_id" in sample:
            ep = int(scalar(sample["episode_id"]))
        else:
            ep = 0
        episode_to_indices[ep].append(idx)
    return dict(episode_to_indices)

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return json_safe(obj.detach().cpu().numpy())
    return obj
