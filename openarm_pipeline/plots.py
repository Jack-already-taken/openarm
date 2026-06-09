\
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd

def _save_hist(values, xlabel, ylabel, title, path, bins=30, vline=None):
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins)
    if vline is not None:
        plt.axvline(vline, linestyle="--", label=f"threshold={vline}")
        plt.legend()
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return path

def plot_trajectory_lengths(summary_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    return _save_hist(summary_df["num_steps"], "Episode length, frames", "Count", "Trajectory Length Distribution", output_dir / "trajectory_length_distribution.png")

def plot_joint_velocity(summary_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    return _save_hist(summary_df["max_abs_joint_velocity"].dropna(), "Max absolute joint velocity", "Episode count", "Joint Velocity Spike Audit", output_dir / "joint_velocity_audit.png")

def plot_bad_frame_ratio(summary_df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if "bad_frame_ratio" not in summary_df or not summary_df["bad_frame_ratio"].notna().any():
        return None
    return _save_hist(summary_df["bad_frame_ratio"].dropna(), "Bad wrist/egocentric frame ratio", "Episode count", "Wrist / Egocentric Video Quality Audit", output_dir / "egocentric_bad_frame_ratio.png", bins=20)

def collect_video_scores(reports: List[Dict[str, Any]]) -> dict:
    scores = {"blur_scores": [], "exposure_scores": [], "frame_diffs": []}
    for r in reports:
        for fr in r["egocentric_video_report"]["frame_reports"]:
            scores["blur_scores"].append(fr["blur_score"])
            scores["exposure_scores"].append(fr["exposure_score"])
            if fr["frame_difference"] is not None:
                scores["frame_diffs"].append(fr["frame_difference"])
    return scores

def plot_video_score_distributions(reports: List[Dict[str, Any]], output_dir: str | Path, blur_threshold: float, exposure_threshold: float) -> list[Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    scores, paths = collect_video_scores(reports), []
    if scores["blur_scores"]:
        paths.append(_save_hist(scores["blur_scores"], "Blur score, variance of Laplacian", "Sampled frame count", "Blur Score Distribution", output_dir / "blur_score_distribution.png", vline=blur_threshold))
    if scores["exposure_scores"]:
        paths.append(_save_hist(scores["exposure_scores"], "Exposure score, saturated pixel ratio", "Sampled frame count", "Exposure Score Distribution", output_dir / "exposure_score_distribution.png", vline=exposure_threshold))
    return paths
