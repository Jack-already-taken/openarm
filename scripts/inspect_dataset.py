\
from __future__ import annotations
import argparse
from openarm_pipeline.loaders import detect_dataset_keys, load_lerobot_dataset
from openarm_pipeline.utils import get_episode_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/libero_object_image")
    args = parser.parse_args()
    dataset = load_lerobot_dataset(args.repo_id)
    print("Loaded:", args.repo_id)
    print("Number of frames:", len(dataset))
    sample = dataset[0]
    print("\nSample keys:")
    for k, v in sample.items():
        print(f"  {k}: type={type(v).__name__}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")
    keys = detect_dataset_keys(dataset)
    print("\nDetected keys:", keys)
    episodes = get_episode_indices(dataset)
    lengths = [len(v) for v in episodes.values()]
    print("\nEpisodes:", len(episodes), "min:", min(lengths), "mean:", sum(lengths)/len(lengths), "max:", max(lengths))

if __name__ == "__main__":
    main()
