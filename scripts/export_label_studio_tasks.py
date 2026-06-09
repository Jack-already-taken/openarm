\
from __future__ import annotations
import argparse, json
from pathlib import Path
from common import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curation.yaml")
    parser.add_argument("--manifest", default="outputs/curated_manifest.json")
    parser.add_argument("--out", default="outputs/label_studio_tasks.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.manifest, "r", encoding="utf-8") as f: manifest = json.load(f)
    tasks = []
    for item in manifest:
        if item["kept"]:
            tasks.append({"data": {"episode_id": item["episode_id"], "repo_id": cfg["repo_id"], "wrist_image_key": cfg.get("wrist_image_key"), "array_path": item.get("array_path")}, "meta": {"num_original_steps": item["num_original_steps"], "num_valid_aligned_steps": item["num_valid_aligned_steps"]}})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} task stubs to {out}")

if __name__ == "__main__":
    main()
