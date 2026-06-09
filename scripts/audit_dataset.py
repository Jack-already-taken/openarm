\
from __future__ import annotations
import argparse, json
from pathlib import Path
from openarm_pipeline.audit import audit_dataset, summarize_audit_reports, video_reason_counts
from openarm_pipeline.loaders import detect_dataset_keys, load_lerobot_dataset
from openarm_pipeline.plots import plot_bad_frame_ratio, plot_joint_velocity, plot_trajectory_lengths, plot_video_score_distributions
from openarm_pipeline.utils import get_episode_indices, json_safe
from common import load_config, none_if_missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"]); figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_lerobot_dataset(cfg["repo_id"])
    keys = detect_dataset_keys(dataset, none_if_missing(cfg,"state_key"), none_if_missing(cfg,"action_key"), none_if_missing(cfg,"timestamp_key"), none_if_missing(cfg,"wrist_image_key"))
    episodes = get_episode_indices(dataset)
    reports = audit_dataset(dataset, episodes, keys, cfg.get("max_episodes"), cfg["min_episode_len"], cfg["max_joint_velocity"], cfg["max_timestamp_gap_ratio"], cfg["blur_threshold"], cfg["exposure_threshold"], cfg["frozen_frame_threshold"], cfg["max_bad_frame_ratio"], cfg["video_frame_stride"])
    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f: json.dump(json_safe(reports), f, indent=2)
    summary_df = summarize_audit_reports(reports); summary_df.to_csv(output_dir / "audit_summary.csv", index=False)
    metadata = {"repo_id": cfg["repo_id"], "num_frames": len(dataset), "num_episodes": len(episodes), "audited_episodes": len(reports), "keys": keys.__dict__, "video_reason_counts": dict(video_reason_counts(reports)), "thresholds": {k: cfg[k] for k in ["min_episode_len","max_joint_velocity","max_timestamp_gap_ratio","blur_threshold","exposure_threshold","frozen_frame_threshold","max_bad_frame_ratio","video_frame_stride"]}}
    with open(output_dir / "audit_metadata.json", "w", encoding="utf-8") as f: json.dump(json_safe(metadata), f, indent=2)
    plot_trajectory_lengths(summary_df, figures_dir); plot_joint_velocity(summary_df, figures_dir); plot_bad_frame_ratio(summary_df, figures_dir); plot_video_score_distributions(reports, figures_dir, cfg["blur_threshold"], cfg["exposure_threshold"])
    print("Saved audit outputs under", output_dir)

if __name__ == "__main__":
    main()
