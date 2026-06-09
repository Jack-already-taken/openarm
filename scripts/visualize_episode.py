\
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from openarm_pipeline.loaders import detect_dataset_keys, load_lerobot_dataset
from openarm_pipeline.utils import get_episode_indices
from openarm_pipeline.video_quality import normalize_frame
from common import load_config, none_if_missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curation.yaml")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = load_lerobot_dataset(cfg["repo_id"])
    keys = detect_dataset_keys(dataset, none_if_missing(cfg,"state_key"), none_if_missing(cfg,"action_key"), none_if_missing(cfg,"timestamp_key"), none_if_missing(cfg,"wrist_image_key"))
    episodes = get_episode_indices(dataset)
    if args.episode_id not in episodes: raise KeyError(f"episode_id={args.episode_id} not found")
    if keys.wrist_image_key is None: raise KeyError("No wrist image key detected")
    indices = episodes[args.episode_id]
    chosen = np.linspace(0, len(indices)-1, args.num_frames).astype(int)
    plt.figure(figsize=(3*args.num_frames,3))
    for plot_i, local_i in enumerate(chosen):
        idx = indices[local_i]
        plt.subplot(1, args.num_frames, plot_i+1)
        plt.imshow(normalize_frame(dataset[idx][keys.wrist_image_key]))
        plt.axis("off"); plt.title(f"idx={idx}")
    plt.suptitle(f"Episode {args.episode_id}, camera={keys.wrist_image_key}"); plt.tight_layout()
    out = Path(args.out) if args.out else Path(cfg["output_dir"]) / "figures" / f"episode_{args.episode_id:06d}_wrist_frames.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print("Saved:", out)

if __name__ == "__main__":
    main()
