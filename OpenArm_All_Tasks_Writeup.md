# OpenArm 2.0 — Egocentric Data & Teleoperation Pipeline Write-up

## Summary

This project implements the front half of an OpenArm-style robot learning data pipeline. The goal is to audit, label, curate, and evaluate multimodal demonstrations that contain both teleoperation data and egocentric/wrist-camera observations.

The code is organized in two phases:

- **Phase 1:** a Colab prototype for fast dataset exploration and quality auditing.
- **Phase 2:** a cleaned repository with reusable modules, config-driven scripts, curation outputs, labeling design, and evaluation design.

The final pipeline uses `lerobot/libero_object_image` because it provides both low-dimensional robot trajectory data and image observations, including a wrist-camera stream. In the final Colab run, the pipeline loaded the dataset successfully, audited the first 50 episodes, detected the correct state/action/timestamp/wrist-camera keys, generated audit plots, created a curated manifest, exported 50 curated `.npz` arrays, and generated Label Studio task stubs.

## Completed Tasks

I completed the following tasks:

- **Task 1: Dataset Exploration & Quality Audit**
- **Task 2: Data Labeling Design**
- **Task 3: Data Curation Pipeline**
- **Task 4: Policy Evaluation Design**
- **Task 5 Bonus: VLA Adaptation Design**

Tasks 1 and 3 are implemented in code. Tasks 2, 4, and 5 are design components supported by documentation and helper utilities.

## Dataset and Run Configuration

The final Phase 2 run used:

```text
Dataset: lerobot/libero_object_image
Total frames: 66,984
Total episodes: 454
Audited episodes: 50
State key: observation.state
Action key: action
Timestamp key: timestamp
Image streams:
  - observation.images.image
  - observation.images.wrist_image
Selected egocentric stream: observation.images.wrist_image
```

The audit configuration used:

```yaml
min_episode_len: 20
max_joint_velocity: 20.0
max_timestamp_gap_ratio: 3.0

blur_threshold: 30.0
exposure_threshold: 0.50
frozen_frame_threshold: 0.5
max_bad_frame_ratio: 0.25
video_frame_stride: 5
```

The final run processed 50 episodes. All 50 passed the current audit filters.

```text
Audited episodes: 50
Kept episodes: 50 / 50
Trajectory length range: 115 to 224 frames
Average trajectory length: 146.62 frames
Max absolute joint/state velocity range: 0.1242 to 0.3960
Bad wrist-frame ratio: 0.0
Video rejection reason counts: {}
Curated array files: 50
Label Studio task stubs: 50
```

These results do not prove the full dataset is perfect; they show that the implemented pipeline runs end-to-end and that the first 50 audited episodes do not show obvious numerical, timestamp, or wrist-camera quality failures under the current thresholds.

---

# Task 1 — Dataset Exploration & Quality Audit

## Goal

The goal of Task 1 is to profile both teleoperation and egocentric data quality before training. Teleoperation data is structured numerical time-series data, while egocentric data is perceptual image/video data. Because the modalities fail differently, the pipeline audits them differently.

## Teleoperation Audit

The teleoperation stream consists of robot state, action, and timestamp data. The pipeline loads each episode and computes:

- episode count
- trajectory length distribution
- state dimensionality
- per-dimension state min/max/mean/std
- NaN/Inf checks
- timestamp continuity
- finite-difference velocity spikes
- short-episode filtering

The main teleoperation audit logic is implemented in:

```text
openarm_pipeline/audit.py
scripts/audit_dataset.py
```

The pipeline detects joint/state anomalies using two types of checks.

First, it checks for invalid values:

```python
np.isfinite(states).all()
```

This catches NaN and Inf values.

Second, it estimates velocity with finite differences:

```python
velocity = np.diff(states, axis=0) / dt[:, None]
```

If the maximum absolute velocity exceeds the configured threshold, the episode is flagged for a possible state discontinuity or unrealistic jump.

The pipeline also profiles the state distribution using per-dimension min, max, mean, and standard deviation. This is useful for identifying unusual values even if the current dataset does not provide explicit robot joint limits.

### Joint Value Anomalies

In this implementation, joint value anomalies are accounted for by:

```text
- NaN / Inf checks
- per-dimension state min/max/mean/std
- finite-difference velocity spike detection
```

A future OpenArm-specific version should add true robot joint limits, such as per-joint lower and upper bounds. Since this take-home uses public LeRobot data rather than the real OpenArm hardware, I used generic numerical and kinematic anomaly checks instead of hardware-specific joint bounds.

### Sensor Drop-outs

Sensor drop-outs are accounted for through timestamp continuity:

```python
dt = np.diff(timestamps)
timestamp_gaps = dt > max_timestamp_gap_ratio * median_dt
```

If a robot state sample is missing, the timestamp gap becomes larger than expected. The audit records the number of timestamp gaps and the maximum timestamp interval per episode.

For visual data, the pipeline also detects frozen or duplicate frames by comparing adjacent sampled frames. A near-zero frame difference can indicate a visual sensor freeze or repeated frame.

## Egocentric / Wrist-Camera Audit

The egocentric stream is audited separately because wrist-camera data has different failure modes from joint states. The code checks:

- motion blur
- overexposure / underexposure
- frozen or duplicate frames
- bad-frame ratio per episode

This logic is implemented in:

```text
openarm_pipeline/video_quality.py
```

The metrics are:

```text
Blur score:
    variance of Laplacian

Exposure score:
    fraction of pixels near black or near white

Frozen-frame score:
    mean absolute difference between adjacent frames
```

A frame is marked bad if it fails any configured threshold. An episode is marked suspicious if its sampled bad-frame ratio exceeds `max_bad_frame_ratio`.

## What I Would Filter Before Training

Before training a policy, I would filter or flag:

1. **Short or incomplete episodes**
   - These may not contain a full task demonstration.

2. **Invalid or discontinuous robot states**
   - NaN/Inf values, timestamp gaps, or large velocity spikes can corrupt action prediction.

3. **Bad wrist-camera frames or episodes**
   - Severe blur, overexposure, frozen frames, or occlusion can make visual policy learning unreliable.

4. **Poor alignment between video and state/action**
   - Multimodal training depends on synchronized visual observations and robot actions.

5. **Failure episodes**
   - I would not necessarily delete all failures. Instead, I would label them and decide whether to use them for imitation training, evaluation, or recovery learning.

## Phase 2 Audit Result

For the final Phase 2 run on the first 50 episodes of `lerobot/libero_object_image`, the audited data was clean under the current thresholds:

```text
NaN / Inf states: none detected
Timestamp gaps: none detected
Velocity spikes: none detected
Bad wrist-camera frames: none detected
Episodes kept: 50 / 50
```

This result is expected for a curated public dataset. The main contribution of the task is not finding severe corruption; it is building a reusable audit pipeline that would catch these issues on real OpenArm logs.

---

# Task 2 — Data Labeling Design

## Teleoperation Labeling Schema

For pick-and-place episodes, I would use a hierarchical labeling schema with three levels: episode-level, segment-level, and frame-level labels.

### Episode-Level Labels

Episode-level labels describe the entire rollout:

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

These labels are useful for dataset filtering, train/test split construction, task balancing, and high-level evaluation.

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

These labels are useful for phase-conditioned analysis and failure localization. For example, a policy might succeed at approaching the object but fail during grasp alignment.

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

Frame-level annotation is more expensive, so I would only require it for critical events such as contact, slip, collision, or failure moments.

## Labeling Tool Choice

I would use **Label Studio** as the primary annotation tool. It supports video timeline labels, episode-level metadata, and flexible JSON export. This makes it suitable for temporal phase labels such as `approach`, `grasp`, `transport`, and `place`.

I would use **CVAT** as a secondary tool only if precise visual labels are needed, such as object bounding boxes, segmentation masks, or gripper/object tracking.

For later development, a custom viewer would be valuable because annotators may need to see synchronized wrist-camera video, robot state plots, gripper commands, and action trajectories in one interface.

## Inter-Annotator Agreement

For episode-level categorical labels such as success/failure, I would use:

```text
Cohen's kappa for two annotators
Fleiss' kappa for more than two annotators
```

For segment-level motion labels, exact boundary agreement is too strict. Two annotators may agree that a grasp occurred but choose slightly different start or end frames. I would therefore use:

```text
Temporal IoU between labeled segments
Boundary F1 with tolerance, such as ±0.25s or ±0.5s
Edit distance between phase sequences
```

For frame-level event labels such as slip or contact, I would compute precision, recall, and F1 after aligning labels to a shared timestamp grid.

## Egocentric / Wrist-Camera Labels

Egocentric video requires labels that describe what the robot sees, not only what its joints are doing.

I would label object interaction states:

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

I would also label hand-eye coordination phases:

```text
search
approach
align
contact
manipulate
verify_success
recover
```

These labels capture whether the wrist camera has enough task-relevant information for visuomotor learning.

Failure moment labels are also important:

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

These failures may not be obvious from joint states alone. A trajectory can look smooth kinematically while the wrist camera shows that the object was missed, occluded, or dropped.

## Gaze-Proxy Regions

A robot does not have human gaze, so I would define gaze-proxy labels as task-relevant visual regions:

```text
object_region
target_region
gripper_region
contact_region
occluded_region
```

These regions can be annotated as coarse boxes or masks when needed. They can later support attention analysis, visual debugging, and success detection.

## Temporal Alignment Between Labels and Robot States

The key rule is to use timestamps as the source of truth.

Each label should be stored using episode-relative time:

```json
{
  "episode_id": 12,
  "modality": "wrist_camera",
  "label": "object_grasped",
  "start_time": 3.20,
  "end_time": 3.85
}
```

The pipeline maps labels to the closest robot-state/action index:

```text
annotation time
→ nearest video frame timestamp
→ nearest joint-state timestamp
→ action index
```

The helper module:

```text
openarm_pipeline/label_alignment.py
```

implements this mapping by converting temporal labels into nearest state indices.

The curation pipeline also preserves:

```text
original_indices
valid_alignment_mask
timestamps
```

This makes it possible to keep labels synchronized even after filtering or masking invalid frames.

---

# Task 3 — Data Curation Pipeline

## Goal

Task 3 asks for a script that loads episodes, applies filtering/cleaning, saves a curated subset, handles egocentric video quality, and maintains temporal alignment.

This is the main engineering portion of the project.

## Implementation

The main curation script is:

```text
scripts/curate_dataset.py
```

It uses these modules:

```text
openarm_pipeline/loaders.py
openarm_pipeline/audit.py
openarm_pipeline/video_quality.py
openarm_pipeline/curation.py
```

The pipeline stages are:

```text
1. Load LeRobot dataset
2. Detect state/action/timestamp/wrist-camera keys
3. Group frames by episode
4. Audit teleoperation data
5. Audit wrist-camera data
6. Decide which episodes to keep
7. Build a valid alignment mask
8. Smooth state trajectories lightly
9. Save curated arrays
10. Save curated manifest
```

## Teleoperation Cleaning Steps

The implemented teleoperation filters include:

```text
minimum episode length
NaN / Inf state check
timestamp gap check
velocity spike check
```

The cleaning step applies light smoothing to state trajectories:

```text
states_smooth = Savitzky-Golay smoothed states
```

The smoothing is intentionally conservative. In manipulation tasks, contact events can be important, so aggressive smoothing could erase meaningful behavior.

## Egocentric Cleaning Steps

The wrist-camera stream is checked for:

```text
blur
bad exposure
frozen or duplicate frames
```

The frame-level quality results are used to create an alignment mask. The current implementation does not blindly delete video and state samples independently. Instead, it records which original indices are valid.

This is important because robot learning samples require synchronized observations and actions.

## Maintaining Temporal Alignment

The most important design decision is to preserve alignment explicitly. Each curated episode stores:

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

The mapping is:

```text
curated timestep
→ original dataset index
→ timestamp
→ robot state
→ action
→ wrist-camera frame
```

This prevents a common multimodal curation bug: deleting bad video frames while leaving the action sequence unchanged, which would silently misalign observations and labels.

## Phase 2 Curation Result

The final Phase 2 run produced:

```text
curated_manifest.json entries: 50
kept episodes: 50 / 50
curated .npz files: 50
Label Studio task stubs: 50
```

A sample curated array contains:

```text
states
states_smooth
actions
timestamps
original_indices
```

Example shape from the run:

```text
states: (143, 8)
states_smooth: (143, 8)
actions: (143, 7)
timestamps: (143,)
original_indices: (143,)
```

This confirms that the pipeline saves training-ready state/action/timestamp arrays while preserving the mapping back to the original dataset.

---

# Task 4 — Policy Evaluation Design

## Teleoperation Evaluation Protocol

Assume an ACT or Diffusion Policy has been trained on the curated dataset.

I would evaluate with:

```text
50 simulation rollouts per task variation
20 real-robot rollouts if hardware is available
held-out object poses
held-out object types
held-out lighting or backgrounds
held-out initial robot states
```

The main metrics would be:

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

For pick-and-place, I would define success as:

```text
1. object is contacted and grasped
2. object is lifted
3. object is transported to the target
4. object is released in the target region
5. object remains stable after release
6. final pose error is below threshold
```

## Diagnosing Sim-to-Real Failures

If the policy succeeds in simulation but fails on the real robot, I would check:

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

I would compare simulation and real logs using the same audit pipeline. If real episodes show higher blur, timestamp gaps, state noise, or visual occlusions, then the issue may be in the data distribution rather than only the policy architecture.

## Egocentric Evaluation

Egocentric video changes evaluation because it reveals failures that joint states alone can miss:

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

As a simple bonus detector, I would train a binary classifier over the final wrist-camera frames:

```text
input: final N wrist-camera frames
output: task_complete / not_complete
```

A stronger version could use a small CNN/ViT or CLIP-style image encoder with labels such as:

```text
object_in_gripper
object_placed_in_target
empty_gripper_after_release
target_visible
```

This success detector would supplement robot-state metrics. It is useful when joint states suggest the robot completed the motion but the camera shows the object was missed or dropped.

---

# Task 5 Bonus — VLA Adaptation Design

I chose the VLA option because the curated dataset naturally supports instruction-conditioned visuomotor learning.

## Teleoperation Format for VLA Fine-Tuning

A VLA training sample should contain:

```text
language instruction
wrist image or multi-view image
proprioceptive robot state
action target
timestamp or action horizon metadata
```

Example sample:

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

The key hyperparameters would include:

```text
learning rate
batch size
image resolution
action horizon
history length
number of frozen vision/language layers
LoRA rank if using parameter-efficient fine-tuning
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

The Phase 2 pipeline already supports this by saving timestamps and original indices for each curated episode.

## Egocentric Failure Modes for VLA

A VLA pretrained mostly on third-person internet or robot data may struggle with wrist-camera input because:

```text
the gripper dominates the image
objects move rapidly relative to the camera
motion blur is more common
the target may leave the field of view
spatial priors differ from third-person views
the model may not understand partial object views
language grounding can fail when the object is temporarily occluded
```

Mitigations include:

```text
fine-tune with wrist-camera data
mix wrist and third-person views when available
add proprioceptive state
use temporal frame history
augment with blur, crops, lighting changes, and occlusion
use task-phase labels for analysis or auxiliary supervision
```

## VLA Evaluation

I would evaluate VLA performance using the same rollout metrics from Task 4, plus visual grounding metrics:

```text
does the object remain visible during approach?
does the gripper align with the correct object?
does visual success match task success?
does the model recover from occlusion?
```

This connects model adaptation back to the audit and labeling pipeline.

---

# Architecture and Design Decisions

## Repository Architecture

The repository is structured as:

```text
configs/
  curation.yaml
  curation_colab.yaml

openarm_pipeline/
  loaders.py
  audit.py
  video_quality.py
  curation.py
  label_alignment.py
  plots.py
  utils.py

scripts/
  inspect_dataset.py
  audit_dataset.py
  curate_dataset.py
  visualize_episode.py
  export_label_studio_tasks.py

docs/
  LABELING_SCHEMA.md
  EVALUATION_DESIGN.md
  PHASE2_SUMMARY.md
  RESULTS_PHASE2.md

outputs/
  audit_report.json
  audit_summary.csv
  audit_metadata.json
  curated_manifest.json
  curated_arrays/
  figures/
  label_studio_tasks.json
```

## Why Separate Audit and Curation?

I separated audit from curation because quality metrics should be inspectable before data is deleted or filtered. This is especially important for visual thresholds. A blur threshold that works on one camera may be too strict for another.

The intended workflow is:

```text
run audit
inspect plots and sample frames
adjust thresholds
run curation
save curated subset
```

## Why Preserve Alignment Masks?

The main multimodal risk is silent misalignment. If bad video frames are deleted without deleting the corresponding robot states and actions, the policy may train on the wrong action for a visual observation.

To avoid this, the pipeline saves:

```text
original_indices
valid_alignment_mask
timestamps
```

This makes filtering reversible and auditable.

## Teleoperation vs Egocentric Trade-off

Teleoperation data is structured and can be checked with numerical rules. Egocentric video is perceptual and context-dependent. A joint velocity spike is usually objectively suspicious, but a blurry wrist frame may still be useful if the task state is visible. Therefore, visual filtering should be more conservative and should be calibrated with manual inspection.

---

# Limitations

The current implementation is a strong take-home prototype, but it has limitations:

```text
- It audits the first 50 episodes by default, not the full 454 episodes.
- It does not enforce OpenArm-specific joint limits because the public dataset does not provide OpenArm hardware limits.
- Label Studio export currently creates task stubs, not full hosted video annotation tasks.
- The curation pipeline saves state/action arrays but does not yet export synchronized video clips.
- The evaluation design is not run on an actual trained ACT, Diffusion Policy, or VLA model.
- VLA adaptation is described as a design, not implemented as training code.
```

These are reasonable trade-offs for a 1–2 hour take-home. The implementation focuses on correctness, reproducibility, and clear reasoning rather than superficial coverage of every possible extension.

---

# What I Would Do Next

With more time or hardware access, I would:

```text
1. Run the audit over all 454 episodes.
2. Add robot-specific joint limits for OpenArm.
3. Export synchronized wrist-camera MP4 clips for Label Studio.
4. Add explicit object visibility labels and success/failure labels.
5. Calibrate blur/exposure thresholds with manual review.
6. Train a simple wrist-camera success detector.
7. Add train/validation/test split generation.
8. Benchmark ACT, Diffusion Policy, and VLA models on the same curated split.
9. Compare simulation and real-robot logs using the same audit metrics.
10. Add a custom annotation viewer that shows wrist video, state plots, action plots, and timestamps together.
```

---

# Conclusion

This project builds a reproducible multimodal data pipeline for OpenArm-style robot learning. Phase 1 demonstrates the audit logic in Colab. Phase 2 turns that logic into a reusable repository with configurable scripts, modular code, curation outputs, labeling design, and evaluation design.

The key idea is to treat teleoperation and egocentric data as synchronized but different modalities. Robot states and actions are audited with numerical and kinematic checks. Wrist-camera frames are audited with perceptual quality checks. The curation pipeline preserves timestamps, original indices, and alignment masks so that downstream labels and training samples remain synchronized.

The final result is a working data infrastructure prototype that can load LeRobot episodes, audit both modalities, curate clean subsets, prepare annotation tasks, and support future policy training or VLA fine-tuning.
