\
from __future__ import annotations
import argparse, json
from pathlib import Path
from openarm_pipeline.audit import audit_dataset, summarize_audit_reports
from openarm_pipeline.curation import curate_dataset
from openarm_pipeline.loaders import detect_dataset_keys, load_lerobot_dataset
from openarm_pipeline.utils import get_episode_indices, json_safe
from common import load_config, none_if_missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_lerobot_dataset(cfg["repo_id"])
    keys = detect_dataset_keys(dataset, none_if_missing(cfg,"state_key"), none_if_missing(cfg,"action_key"), none_if_missing(cfg,"timestamp_key"), none_if_missing(cfg,"wrist_image_key"))
    episodes = get_episode_indices(dataset)
    reports = audit_dataset(dataset, episodes, keys, cfg.get("max_episodes"), cfg["min_episode_len"], cfg["max_joint_velocity"], cfg["max_timestamp_gap_ratio"], cfg["blur_threshold"], cfg["exposure_threshold"], cfg["frozen_frame_threshold"], cfg["max_bad_frame_ratio"], cfg["video_frame_stride"])
    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f: json.dump(json_safe(reports), f, indent=2)
    summarize_audit_reports(reports).to_csv(output_dir / "audit_summary.csv", index=False)
    manifest = curate_dataset(dataset, episodes, keys, reports, output_dir, cfg["smooth_window"], cfg["drop_bad_frames_with_mask"])
    with open(output_dir / "curated_manifest.json", "w", encoding="utf-8") as f: json.dump(json_safe(manifest), f, indent=2)
    kept = sum(1 for item in manifest if item["kept"])
    print(f"Saved curated manifest under {output_dir}; kept {kept}/{len(manifest)} audited episodes")

if __name__ == "__main__":
    main()
