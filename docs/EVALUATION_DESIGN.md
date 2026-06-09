# Policy Evaluation Design

## Teleoperation evaluation

For ACT or Diffusion Policy:

- 50 simulated rollouts per task variation
- 20 real-robot rollouts if hardware is available
- held-out object poses, object types, lighting, and initial robot states

Metrics:

- success rate
- completion time
- final object pose error
- grasp success rate
- drop/slip rate
- collision rate
- trajectory smoothness
- action latency
- number of recovery behaviors

## Sim-to-real diagnosis

Check camera calibration, appearance gap, lighting gap, action latency, joint encoder noise, contact dynamics, gripper friction, and overfitting to simulation.

## Egocentric evaluation

Wrist-camera data catches failures that joint states alone can miss: wrong object, object leaving view, occlusion, slip during transport, visual failure despite kinematic success.

A simple success detector can classify the final N wrist-camera frames as `task_complete` or `not_complete`.
