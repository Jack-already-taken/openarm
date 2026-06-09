# OpenArm 2.0 — Egocentric Data & Teleoperation Pipeline Write-up

## Overview

This project builds a reproducible data pipeline for OpenArm-style robot learning demonstrations. The pipeline handles two synchronized modalities: teleoperation data and egocentric/wrist-camera data.

**Teleoperation data** includes robot state, action, timestamps, trajectory length, numerical validity, and kinematic consistency.

**Egocentric / wrist-camera data** includes wrist-camera frames, blur, exposure, frozen-frame checks, visibility/failure reasoning, and temporal alignment with robot states.

The implementation is split into two development phases. Phase 1 is a Colab prototype used to inspect the dataset and validate the audit logic. Phase 2 turns that prototype into a reusable repository with modules, scripts, configs, curation outputs, labeling design, and evaluation design.

The final implementation uses `lerobot/libero_object_image` because it contains both robot trajectory data and a wrist-camera stream.

Final run summary:

```text
Dataset: lerobot/libero_object_image
Total frames: 66,984
Total episodes: 454
Audited episodes: 50
State key: observation.state
Action key: action
Timestamp key: timestamp
Wrist-camera key: observation.images.wrist_image

Kept episodes: 50 / 50
Trajectory length range: 115 to 224 frames
Average trajectory length: 146.62 frames
Max absolute joint/state velocity range: 0.1242 to 0.3960
Bad wrist-frame ratio: 0.0
Curated array files: 50
Label Studio task stubs: 50
```

---

# Task 1 — Dataset Exploration & Quality Audit

## Responsibility

Task 1 is responsible for understanding the dataset before training. It checks whether the teleoperation stream and the wrist-camera stream are complete, usable, and properly synchronized.

This task is handled by:

```text
notebooks/OpenArm_Phase1_Colab.ipynb
scripts/inspect_dataset.py
scripts/audit_dataset.py
openarm_pipeline/loaders.py
openarm_pipeline/audit.py
openarm_pipeline/video_quality.py
openarm_pipeline/plots.py
```

## Teleoperation Audit

The teleoperation stream contains robot states, actions, and timestamps. The audit computes:

```text
- number of episodes
- trajectory length distribution
- state/action dimensions
- per-dimension state min/max/mean/std
- NaN / Inf state checks
- timestamp gap checks
- finite-difference velocity spike checks
- short-episode checks
```

Joint value anomalies are handled through invalid-value checks and kinematic checks. The code checks whether state values are finite:

```python
np.isfinite(states).all()
```

It also estimates velocity using finite differences:

```python
velocity = np.diff(states, axis=0) / dt[:, None]
```

If the maximum absolute velocity exceeds the configured threshold, the episode is flagged as a possible discontinuity or unrealistic motion.

Sensor drop-outs are mainly detected through timestamp continuity:

```python
timestamp_gaps = dt > max_timestamp_gap_ratio * median_dt
```

If robot state samples are missing, the time gap between consecutive samples becomes abnormally large.

## Egocentric / Wrist-Camera Audit

The wrist-camera stream is audited separately because video has different failure modes from joint states. The video audit checks:

```text
- motion blur
- overexposure / underexposure
- frozen or duplicate frames
- bad-frame ratio per episode
```

The implemented metrics are:

```text
Blur score:
    variance of Laplacian

Exposure score:
    fraction of pixels near black or near white

Frozen-frame score:
    mean absolute difference between adjacent sampled frames
```

These checks are implemented in `openarm_pipeline/video_quality.py`.

## Result

For the final Phase 2 run on the first 50 episodes:

```text
NaN / Inf states: none detected
Timestamp gaps: none detected
Velocity spikes: none detected
Bad wrist-camera frames: none detected
Episodes kept: 50 / 50
```

This does not prove the entire dataset is perfect, but it shows that the audit pipeline works and that the audited subset is clean under the current thresholds.

## Strategy Choice and Alternatives

The chosen strategy is to use simple, transparent checks first. This makes the pipeline easy to debug and explain.

Potential alternatives:

```text
- add OpenArm-specific joint limits
- use z-score or isolation-forest anomaly detection
- check expected FPS against actual FPS
- detect failed image decoding explicitly
- use object detectors to check visual task relevance
- use learned visual quality models
```

These alternatives would be useful in a production system, but the current version prioritizes reproducibility and interpretability.

---

# Task 2 — Data Labeling Design

## Responsibility

Task 2 defines a practical annotation scheme for both robot motion and egocentric visual observations.

This task is handled by:

```text
docs/LABELING_SCHEMA.md
openarm_pipeline/label_alignment.py
scripts/export_label_studio_tasks.py
outputs/label_studio_tasks.json
```

## Teleoperation Labeling Schema

For pick-and-place demonstrations, I would use a hierarchical schema with episode-level, segment-level, and frame-level labels.

### Episode-Level Labels

Episode-level labels describe the whole rollout:

```text
task_name
object_type
target_type
success
failure_reason
scene_id
operator_id
difficulty
```

Example:

```json
{
  "episode_id": 12,
  "task_name": "pick_and_place",
  "object_type": "block",
  "target_type": "bin",
  "success": true,
  "failure_reason": "none",
  "difficulty": "medium"
}
```

These labels support filtering, train/test split construction, and task-level evaluation.

### Segment-Level Labels

Segment-level labels describe motion phases:

```text
approach_object
pre_grasp_align
grasp
lift
transport
place
release
retract
recovery
```

Each segment is represented as a time interval:

```json
{
  "episode_id": 12,
  "label": "grasp",
  "start_time": 2.35,
  "end_time": 3.10
}
```

These labels help diagnose which part of the task a policy fails on.

### Frame-Level Labels

Frame-level labels describe instantaneous events:

```text
gripper_open
gripper_closed
object_in_gripper
contact
collision
slip
human_intervention
unstable_motion
```

Frame-level labels are expensive, so I would only use them for critical events such as contact, slip, collision, and failure moments.

## Tool Choice

I would use **Label Studio** as the primary tool because it supports video timeline labels, episode-level metadata, and JSON export.

I would use **CVAT** only if precise visual labels are needed, such as bounding boxes, segmentation masks, or object tracking.

A future custom viewer would be useful because robotics annotators may need to inspect wrist video, state plots, action plots, gripper commands, and timestamps together.

## Inter-Annotator Agreement

For episode-level categorical labels:

```text
Cohen's kappa for two annotators
Fleiss' kappa for more than two annotators
```

For segment-level motion labels:

```text
Temporal IoU between annotated segments
Boundary F1 with tolerance, such as ±0.25s or ±0.5s
Edit distance between phase sequences
```

For frame-level event labels:

```text
precision / recall / F1 after timestamp alignment
```

This is better than exact frame matching because motion boundaries are often subjective.

## Egocentric / Wrist-Camera Labels

Egocentric labels should describe what the robot sees and whether the visual stream supports the task.

Object interaction labels:

```text
object_visible
target_visible
gripper_visible
gripper_object_contact
object_grasped
object_lifted
object_slipped
object_placed
```

Hand-eye coordination phase labels:

```text
search
approach
align
contact
manipulate
verify_success
recover
```

Failure labels:

```text
view_lost
object_occluded
wrong_object
missed_grasp
slip
collision
target_not_visible
placement_failed
```

Gaze-proxy or attention-region labels:

```text
object_region
target_region
gripper_region
contact_region
occluded_region
```

These labels capture failures that joint states alone may miss.

## Temporal Alignment

The key rule is to use timestamps as the source of truth.

Each annotation is stored with episode-relative time:

```json
{
  "episode_id": 12,
  "modality": "wrist_camera",
  "label": "object_grasped",
  "start_time": 3.20,
  "end_time": 3.85
}
```

The alignment process is:

```text
annotation time
→ nearest video frame timestamp
→ nearest joint-state timestamp
→ action index
```

The module `openarm_pipeline/label_alignment.py` maps temporal labels to nearest state/action indices. The curation pipeline also saves `original_indices`, `valid_alignment_mask`, and `timestamps`, so annotations remain aligned after filtering.

## Strategy Choice and Alternatives

The chosen strategy is hierarchical labeling because different label levels support different tasks:

```text
episode labels → filtering and evaluation
segment labels → phase analysis
frame labels → event localization
egocentric labels → visual grounding
```

Alternatives include success/failure-only labels, which are cheap but not diagnostic, or dense object masks for every frame, which are powerful but expensive. The chosen schema balances annotation effort and usefulness.

---

# Task 3 — Data Curation Pipeline

## Responsibility

Task 3 is the main engineering component. It turns raw LeRobot episodes into a curated subset suitable for downstream policy training.

This task is implemented by:

```text
scripts/curate_dataset.py
openarm_pipeline/curation.py
openarm_pipeline/audit.py
openarm_pipeline/video_quality.py
openarm_pipeline/loaders.py
```

## Pipeline Steps

The curation pipeline performs:

```text
1. Load the LeRobot dataset.
2. Detect state/action/timestamp/wrist-camera keys.
3. Group frames by episode.
4. Audit teleoperation data.
5. Audit wrist-camera data.
6. Decide which episodes to keep.
7. Build a valid alignment mask.
8. Lightly smooth state trajectories.
9. Save curated arrays.
10. Save a curated manifest.
```

## Teleoperation Cleaning

The implemented filters are:

```text
minimum episode length
NaN / Inf state check
timestamp gap check
velocity spike check
```

The pipeline also saves a lightly smoothed version of the state trajectory using Savitzky-Golay smoothing. The smoothing is conservative because contact-rich manipulation can contain meaningful abrupt changes.

## Egocentric Cleaning

The wrist-camera stream is checked for:

```text
blur
bad exposure
frozen or duplicate frames
```

Rather than deleting frames independently, the pipeline records frame validity through a mask.

## Temporal Alignment

The most important curation decision is to preserve alignment explicitly.

Each curated episode stores:

```text
states
states_smooth
actions
timestamps
original_indices
```

The manifest stores:

```text
episode_id
kept
drop_reasons
num_original_steps
num_valid_aligned_steps
original_indices
valid_alignment_mask
array_path
```

This preserves the mapping:

```text
curated timestep
→ original dataset index
→ timestamp
→ robot state
→ action
→ wrist-camera frame
```

This avoids silently misaligning observations and actions.

## Result

The final run produced:

```text
curated_manifest.json entries: 50
kept episodes: 50 / 50
curated .npz files: 50
Label Studio task stubs: 50
```

A sample curated array contains:

```text
states: (143, 8)
states_smooth: (143, 8)
actions: (143, 7)
timestamps: (143,)
original_indices: (143,)
```

## Strategy Choice and Alternatives

The chosen strategy separates audit from curation and uses masks instead of irreversible deletion. This makes the pipeline auditable and reversible.

Alternatives:

```text
- one script that immediately filters and saves data
- physically delete bad frames
- interpolate missing frames or states
- resample all streams to a fixed timeline
```

Those alternatives can work, but they increase the risk of silent alignment bugs. The mask-based approach is safer for multimodal robot data.

---

# Task 4 — Policy Evaluation Design

## Responsibility

Task 4 is a design task. It defines how a trained ACT or Diffusion Policy would be evaluated using both robot-state and wrist-camera evidence.

This task is covered by:

```text
docs/EVALUATION_DESIGN.md
docs/RESULTS_PHASE2.md
README.md
```

## Teleoperation Evaluation

For an ACT or Diffusion Policy trained on the curated dataset, I would evaluate with:

```text
50 simulation rollouts per task variation
20 real-robot rollouts if hardware is available
held-out object poses
held-out object types
held-out lighting/backgrounds
held-out initial robot states
```

Metrics:

```text
task success rate
completion time
final object pose error
grasp success rate
drop/slip rate
collision rate
trajectory smoothness
action latency
number of recovery behaviors
```

Pick-and-place success requires:

```text
1. object is contacted and grasped
2. object is lifted
3. object is transported to the target
4. object is released in the target region
5. object remains stable after release
6. final pose error is below threshold
```

## Sim-to-Real Diagnosis

If a policy succeeds in simulation but fails on the real robot, I would check:

```text
camera calibration mismatch
image appearance gap
lighting mismatch
object geometry or texture mismatch
action latency
joint encoder noise
control frequency mismatch
contact dynamics mismatch
gripper friction mismatch
policy overfitting to clean simulation states
```

The same audit pipeline can be used to compare simulation and real logs.

## Egocentric Evaluation

Egocentric video reveals failures that joint states alone can miss:

```text
object not visible
target not visible
gripper blocking the camera
wrong object approached
object slips during transport
object dropped outside the camera view
kinematically smooth trajectory but visually failed grasp
```

Additional egocentric metrics:

```text
object visibility ratio
target visibility ratio
visual grasp confirmation
visual placement confirmation
occlusion during critical phases
failure-frame classification accuracy
```

## Simple Egocentric Success Detector

A simple success detector can be trained on final wrist-camera frames:

```text
input: final N wrist-camera frames
output: task_complete / not_complete
```

A stronger version could use a CNN, ViT, or CLIP-style encoder with labels such as:

```text
object_in_gripper
object_placed_in_target
empty_gripper_after_release
target_visible
```

This detector supplements robot-state metrics when joint states suggest success but the visual outcome disagrees.

## Strategy Choice and Alternatives

The chosen strategy combines success metrics with diagnostic metrics. This is more useful than success rate alone.

Alternatives include learned reward models, human preference scoring, or visual-language success classifiers. Those can be powerful, but the proposed protocol is practical without requiring extra model training.

---

# Task 5 — Model Adaptation Bonus: VLA

## Responsibility

Task 5 is addressed as a design extension. I chose the VLA option because the curated dataset naturally fits instruction-conditioned visuomotor learning.

## Teleoperation Format for VLA Fine-Tuning

A VLA sample should contain:

```text
language instruction
wrist image or multi-view image
proprioceptive robot state
action target
timestamp or action-horizon metadata
```

Example:

```json
{
  "instruction": "pick up the object and place it in the target region",
  "image": "observation.images.wrist_image",
  "proprio": "observation.state",
  "action": "action"
}
```

The action target could be:

```text
joint action
end-effector delta pose
gripper command
action chunk for ACT-style prediction
```

Key hyperparameters:

```text
learning rate
batch size
image resolution
action horizon
history length
number of frozen vision/language layers
LoRA rank
proprioception fusion method
loss weighting for gripper vs arm motion
```

## Egocentric Preprocessing for VLA

For wrist-camera input, I would:

```text
resize frames to the VLA image resolution
normalize using the model's expected image statistics
sample frames at the control frequency
align each frame with the nearest action timestamp
optionally stack recent frames for temporal context
preserve original_indices for traceability
```

The current curation pipeline already supports this by saving timestamps and original indices.

## Egocentric Failure Modes

A VLA pretrained mostly on third-person data may struggle with wrist-camera input because:

```text
the gripper dominates the image
objects move rapidly relative to the camera
motion blur is more common
the target may leave the field of view
spatial priors differ from third-person views
partial object views may confuse language grounding
```

Mitigations:

```text
fine-tune on wrist-camera data
mix wrist and third-person views when available
add proprioceptive state
use temporal frame history
augment with blur, crops, lighting changes, and occlusion
use task-phase labels for analysis or auxiliary supervision
```

## Evaluation for VLA Adaptation

I would evaluate the adapted VLA using the rollout metrics from Task 4, plus visual grounding metrics:

```text
does the object remain visible during approach?
does the gripper align with the correct object?
does visual success match task success?
does the model recover from occlusion?
```

## Strategy Choice and Alternatives

I chose VLA rather than WAM because the current pipeline already produces VLA-style ingredients: image, proprioceptive state, action, timestamp, and eventually language/task labels.

A WAM approach would model future world states or latent dynamics. That could be useful for planning, but it is more complex and less directly connected to the implemented curation pipeline. For this take-home, VLA is the more straightforward bonus extension.

---

# Final Summary

The main strategy is to prioritize data infrastructure before model training. The pipeline demonstrates that synchronized LeRobot episodes can be loaded, audited, curated, aligned, labeled, and prepared for downstream policy learning.

The strongest design choices are:

```text
- using a dataset with both state/action and wrist-camera data
- separating audit from curation
- auditing teleoperation and egocentric streams with modality-specific checks
- preserving original indices and alignment masks
- using hierarchical labels
- evaluating with both robot-state and visual success criteria
```

The main next steps would be to run the audit over all 454 episodes, add OpenArm-specific joint limits, export real video clips for Label Studio, and train a simple wrist-camera success detector or VLA fine-tuning baseline.
