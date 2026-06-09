\
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .loaders import DatasetKeys, get_episode_arrays
from .video_quality import audit_video_stream

def audit_joint_states(states: np.ndarray, timestamps: np.ndarray, max_joint_velocity=20.0, max_timestamp_gap_ratio=3.0) -> Dict[str, Any]:
    report: Dict[str, Any] = {"num_steps": int(len(states)), "state_dim": int(states.shape[1]), "has_nan_or_inf": bool(not np.isfinite(states).all())}
    report.update({"state_min": np.nanmin(states, axis=0).tolist(), "state_max": np.nanmax(states, axis=0).tolist(), "state_mean": np.nanmean(states, axis=0).tolist(), "state_std": np.nanstd(states, axis=0).tolist()})
    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        dt_safe = np.maximum(dt, 1e-6)
        median_dt = float(np.median(dt_safe))
        gaps = dt_safe > max_timestamp_gap_ratio * median_dt
        velocity = np.diff(states, axis=0) / dt_safe[:, None]
        max_abs_vel = np.nanmax(np.abs(velocity), axis=0)
        report.update({"median_dt": median_dt, "max_dt": float(np.max(dt_safe)), "num_timestamp_gaps": int(gaps.sum()), "max_abs_joint_velocity_per_dim": max_abs_vel.tolist(), "max_abs_joint_velocity": float(np.nanmax(max_abs_vel)), "has_velocity_spike": bool(np.nanmax(max_abs_vel) > max_joint_velocity)})
    else:
        report.update({"median_dt": None, "max_dt": None, "num_timestamp_gaps": 0, "max_abs_joint_velocity_per_dim": [], "max_abs_joint_velocity": 0.0, "has_velocity_spike": False})
    return report

def teleop_filter_reasons(ep_len: int, joint_report: Dict[str, Any], min_episode_len=20) -> List[str]:
    reasons = []
    if ep_len < min_episode_len: reasons.append("too_short")
    if joint_report["has_nan_or_inf"]: reasons.append("nan_or_inf_joint_state")
    if joint_report["has_velocity_spike"]: reasons.append("joint_velocity_spike")
    if joint_report["num_timestamp_gaps"] > 0: reasons.append("timestamp_gap")
    return reasons

def audit_dataset(dataset: Any, episode_to_indices: Dict[int, List[int]], keys: DatasetKeys, max_episodes: Optional[int] = None, min_episode_len=20, max_joint_velocity=20.0, max_timestamp_gap_ratio=3.0, blur_threshold=30.0, exposure_threshold=0.50, frozen_frame_threshold=0.5, max_bad_frame_ratio=0.25, video_frame_stride=5) -> List[Dict[str, Any]]:
    all_episode_ids = sorted(episode_to_indices.keys())
    if max_episodes is not None:
        all_episode_ids = all_episode_ids[:max_episodes]
    reports = []
    for ep_id in tqdm(all_episode_ids, desc="Auditing episodes"):
        indices = episode_to_indices[ep_id]
        states, actions, timestamps = get_episode_arrays(dataset, indices, keys)
        joint_report = audit_joint_states(states, timestamps, max_joint_velocity, max_timestamp_gap_ratio)
        video_report = audit_video_stream(dataset, indices, keys.wrist_image_key, video_frame_stride, blur_threshold, exposure_threshold, frozen_frame_threshold)
        reasons = teleop_filter_reasons(len(indices), joint_report, min_episode_len)
        if video_report["available"] and video_report["bad_frame_ratio"] is not None and video_report["bad_frame_ratio"] > max_bad_frame_ratio:
            reasons.append("high_bad_egocentric_frame_ratio")
        reports.append({"episode_id": int(ep_id), "num_steps": int(len(indices)), "state_key": keys.state_key, "action_key": keys.action_key, "timestamp_key": keys.timestamp_key, "image_keys": keys.image_keys, "wrist_image_key": keys.wrist_image_key, "joint_report": joint_report, "egocentric_video_report": video_report, "keep_candidate": len(reasons) == 0, "filter_reasons": reasons})
    return reports

def summarize_audit_reports(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in reports:
        jr, vr = r["joint_report"], r["egocentric_video_report"]
        rows.append({"episode_id": r["episode_id"], "num_steps": r["num_steps"], "keep_candidate": r["keep_candidate"], "filter_reasons": ", ".join(r["filter_reasons"]), "has_nan_or_inf": jr["has_nan_or_inf"], "max_abs_joint_velocity": jr["max_abs_joint_velocity"], "num_timestamp_gaps": jr["num_timestamp_gaps"], "bad_frame_ratio": vr["bad_frame_ratio"], "num_bad_frames": vr["num_bad_frames"], "num_sampled_video_frames": vr["num_sampled_frames"]})
    return pd.DataFrame(rows)

def video_reason_counts(reports: List[Dict[str, Any]]) -> Counter:
    counts = Counter()
    for r in reports:
        for fr in r["egocentric_video_report"]["frame_reports"]:
            for reason in fr["reasons"]:
                counts[reason] += 1
    return counts
