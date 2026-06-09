# OpenArm 2.0 — Teleoperation & Egocentric Data Pipeline

Phase 2 repository for the OpenArm 2.0 take-home. It turns the Colab prototype into a scriptable, reproducible pipeline for LeRobot datasets with teleoperation state/action data and wrist-camera video.

Default dataset:

```text
lerobot/libero_object_image
```

## Completed tasks

- Task 1: Dataset exploration and quality audit
- Task 2: Labeling schema design
- Task 3: Data curation pipeline
- Task 4: Policy evaluation design

## Structure

```text
openarm-data-pipeline/
├── configs/curation.yaml
├── docs/
├── openarm_pipeline/
├── scripts/
├── outputs/
├── requirement.txt
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Inspect dataset keys:

```bash
python scripts/inspect_dataset.py --repo-id lerobot/libero_object_image
```

Run audit:

```bash
python scripts/audit_dataset.py --config configs/curation.yaml
```

Run curation:

```bash
python scripts/curate_dataset.py --config configs/curation.yaml
```

Visualize wrist-camera frames:

```bash
python scripts/visualize_episode.py --config configs/curation.yaml --episode-id 0
```

Export Label Studio task stubs:

```bash
python scripts/export_label_studio_tasks.py --config configs/curation.yaml --manifest outputs/curated_manifest.json
```

## Outputs

Audit:

```text
outputs/audit_report.json
outputs/audit_summary.csv
outputs/audit_metadata.json
outputs/figures/
```

Curation:

```text
outputs/curated_manifest.json
outputs/curated_arrays/episode_*.npz
```

Each curated `.npz` stores `states`, `states_smooth`, `actions`, `timestamps`, and `original_indices`.

The manifest stores `episode_id`, `kept`, `drop_reasons`, `original_indices`, and `valid_alignment_mask`.

## Design decisions

Teleoperation filters are numerical and kinematic: episode length, NaN/Inf, timestamp gaps, and velocity spikes.

Egocentric filters are perceptual: blur score, exposure score, and frozen-frame score. These thresholds should be calibrated from score distributions and manual frame inspection.

The pipeline never drops video frames and joint states independently without tracking alignment. It preserves:

```text
video frame -> dataset index -> timestamp -> robot state/action
```

## What to do next

- run full audit over all episodes
- calibrate video thresholds
- export actual MP4 clips for Label Studio
- train a simple wrist-camera success detector
- benchmark ACT / Diffusion Policy / VLA on the same curated split
