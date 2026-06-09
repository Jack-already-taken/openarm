# Labeling Schema

Primary tool: **Label Studio**. It supports temporal segment labels, episode-level labels, and flexible metadata export.

Secondary tool: **CVAT**. Use it only when precise object boxes or masks are required.

## Teleoperation labels

Episode-level: `task_name`, `object_type`, `scene_id`, `success`, `failure_reason`, `operator_id`, `difficulty`.

Segment-level: `approach_object`, `pre_grasp_align`, `grasp`, `lift`, `transport`, `place`, `release`, `retract`, `recovery`.

Frame-level: `contact_state`, `gripper_open`, `gripper_closed`, `object_in_gripper`, `collision`, `slip`, `human_intervention`.

## Egocentric / wrist-camera labels

`object_visible`, `target_visible`, `gripper_object_contact`, `object_grasped`, `object_slipped`, `placement_complete`, `occlusion`, `failure_moment`, `view_lost`, `visual_verification`.

## Temporal alignment

Use timestamps as the source of truth:

```text
video time -> nearest video frame index -> nearest joint-state timestamp -> action index
```

The curation pipeline saves `original_indices` and `valid_alignment_mask` for traceability.
