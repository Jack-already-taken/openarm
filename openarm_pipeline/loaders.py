\
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np
from .utils import choose_wrist_or_ego_camera, find_action_key, find_image_keys, find_state_key, find_timestamp_key, scalar, to_numpy

def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset

@dataclass
class DatasetKeys:
    state_key: str
    action_key: Optional[str]
    timestamp_key: Optional[str]
    image_keys: List[str]
    wrist_image_key: Optional[str]

def load_lerobot_dataset(repo_id: str):
    return import_lerobot_dataset()(repo_id)

def detect_dataset_keys(dataset: Any, state_key=None, action_key=None, timestamp_key=None, wrist_image_key=None) -> DatasetKeys:
    sample = dataset[0]
    detected_state = state_key or find_state_key(sample)
    detected_action = action_key or find_action_key(sample)
    detected_timestamp = timestamp_key or find_timestamp_key(sample)
    image_keys = find_image_keys(sample)
    if wrist_image_key is not None and wrist_image_key not in sample:
        print(f"Warning: configured wrist_image_key={wrist_image_key!r} not found. Falling back to auto-detected camera.")
        wrist_image_key = None
    detected_wrist = wrist_image_key or choose_wrist_or_ego_camera(image_keys)
    if detected_state is None:
        raise KeyError("Could not detect robot state key. Set state_key manually in config.")
    return DatasetKeys(detected_state, detected_action, detected_timestamp, image_keys, detected_wrist)

def get_episode_arrays(dataset: Any, indices: List[int], keys: DatasetKeys) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    states, actions, timestamps = [], [], []
    for local_i, idx in enumerate(indices):
        sample = dataset[idx]
        states.append(to_numpy(sample[keys.state_key]).reshape(-1))
        if keys.action_key is not None:
            actions.append(to_numpy(sample[keys.action_key]).reshape(-1))
        if keys.timestamp_key is not None:
            timestamps.append(float(scalar(sample[keys.timestamp_key])))
        else:
            timestamps.append(float(local_i))
    return np.stack(states, axis=0), (np.stack(actions, axis=0) if actions else None), np.asarray(timestamps, dtype=np.float64)
