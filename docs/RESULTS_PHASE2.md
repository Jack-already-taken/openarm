# Phase 2 Results

## Dataset

- Dataset: `lerobot/libero_object_image`
- Audited episodes: 50
- Selected wrist/egocentric stream: `observation.images.wrist_image`
- State key: `observation.state`
- Action key: `action`
- Timestamp key: `timestamp`

## Teleoperation audit

The audited episodes have trajectory lengths between 115 and 224 frames, with an average length of 146.62 frames.

The maximum absolute joint/state velocity across episodes ranged from 0.1242 to 0.3960 under the current finite-difference estimate.

The audit checks for:

- NaN/Inf state values
- per-dimension state min/max/mean/std
- timestamp gaps
- finite-difference velocity spikes

## Egocentric / wrist-camera audit

The wrist-camera audit sampled frames with stride `5` and computed:

- blur score using variance of Laplacian
- exposure saturation ratio
- adjacent-frame difference for frozen-frame detection

Mean bad-frame ratio: 0.0

Video rejection reason counts:

```text
{}
```

## Curation result

Candidate episodes kept by the current filters: 50 / 50

Filter reason counts:

```text
{nan: 50}
```

The curation step saves a manifest with `original_indices` and `valid_alignment_mask`, so downstream training can preserve the mapping between robot state, action, timestamp, and wrist-camera frame.

## Interpretation

Teleoperation filtering is mostly numerical and kinematic: short episodes, invalid states, timestamp gaps, and unrealistic velocity spikes.

Egocentric filtering is perceptual and threshold-sensitive. Blur/exposure/frozen-frame thresholds should be calibrated using score distributions and manual visualization before dropping full episodes.

## Next steps

- run the audit on all episodes by setting `max_episodes: null`
- manually inspect frames near the blur/exposure thresholds
- export real video clips for Label Studio rather than task stubs
- add object visibility and task-success labels
- train a small wrist-camera success detector
