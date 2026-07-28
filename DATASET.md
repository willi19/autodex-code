# AutoDex Grasp Dataset

Real-robot dexterous grasp executions with multi-view video, calibrated cameras,
6-DOF object poses, and the executed grasp for each trial.

## Datasets

| Name | Hand | Object-pose source | Trials (main) |
|------|------|--------------------|---------------|
| `selected_100` | xArm6 + Allegro (16-DOF) | legacy SAM3 + FoundationPose (per-frame silhouette-refined) | 1656 |
| `corl_selected_100` | xArm6 + Allegro (16-DOF) | MV-GoTrack multi-view tracking | 574 |
| `selected_100_inspire` | xArm6 + Inspire (6-DOF) | MV-GoTrack multi-view tracking | 384 |

Each dataset is a tree of **episodes**:

```
{dataset}/{object_name}/{timestamp}/     # one grasp trial
```

Trials whose object pose failed the quality gate (see *Pose quality* below) are
moved to `{dataset}_pose_outlier/`; incomplete captures to `{dataset}_error/`.

## Per-episode core files

```
{object}/{timestamp}/
├── pose_world.npy          # (4,4) float — object 6D pose in the world frame
├── C2R.npy                 # (4,4) float — T_world_robot (world ← robot base)
├── cam_param/
│   ├── intrinsics.json     # per-camera-serial intrinsics
│   └── extrinsics.json     # per-camera-serial (3,4) world → camera
├── videos/{serial}.avi     # 24 synchronized camera videos of the execution
├── init_capture/images/{serial}.png   # frame-0 still image per camera
├── arm/                    # executed arm trajectory
│   ├── action.npy          # (T,6)  commanded joint targets (rad)
│   ├── action_qpos.npy     # (T,6)  commanded joint positions
│   └── state.npy           # measured arm state
├── hand/                   # executed hand trajectory
│   ├── action.npy          # (T,16 Allegro | 6 Inspire) commanded finger targets
│   └── state.npy           # measured hand state
├── raw/                    # unsynchronized raw sensor logs
│   ├── arm/   {action,action_qpos,position,velocity,torque,time}.npy
│   ├── hand/  {action,position,time}.npy
│   ├── images/{serial}.png
│   └── timestamps/{frame_id,timestamp}.npy
├── executed_grasp/         # the grasp the robot actually performed
│   ├── wrist_se3.npy       # (4,4) wrist pose in the OBJECT frame
│   ├── grasp_pose.npy      # (16|6) finger joint configuration at grasp
│   └── meta.json           # obj, hand, pose_id, grasp_time, success, states…
├── object_tracking/        # 6D object trajectory (gotrack datasets only)
│   ├── gotrack_output/
│   │   ├── world_pose_records.json   # fused object pose per frame (world)
│   │   ├── frame_poses/{serial}.json # per-camera per-frame pose
│   │   └── summary.json              # tracker config + mesh_path used
│   └── overlay_videos/{serial}.mp4   # mesh rendered onto each view
└── recompute_pose.json     # {sil_loss, reject} — pose-quality record
```

### Coordinate conventions

- **`pose_world`** is `T_world_object`: it maps object-local points to the
  world (charuco) frame. `world_point = pose_world @ [x,y,z,1]ᵀ`.
- **`extrinsics[serial]`** is `T_camera_world` (world → camera). To render/project
  the object into camera `s`:
  `pose_cam = extrinsics[s] @ pose_world`, then `pixel = K @ pose_cam @ point`.
- **`C2R`** is `T_world_robot`. Robot-base points → world: `C2R @ p`.
  World → robot: `inv(C2R) @ p`.
- **`executed_grasp/wrist_se3`** is the executed wrist expressed in the **object
  frame** (`inv(inv(C2R) @ pose_world) @ FK(arm_qpos_at_grasp)`), so a grasp can
  be replayed relative to the object regardless of where the object sat.

### `cam_param/intrinsics.json`

Keyed by camera serial. Each value has:
`original_intrinsics` (3×3, raw), `intrinsics_undistort` (3×3, use this with the
undistorted video), `dist_params`, `width`, `height`. Videos are already
undistorted — use `intrinsics_undistort`.

### `executed_grasp/meta.json`

`hand`, `pose_id` (canonical tabletop orientation), `grasp_frame` /
`grasp_time` (when the grasp closed), `finger_L2` (match distance to the source
grasp candidate), `success` (object lifted >5 cm after grasp, or `null` if
untracked), `execution_states` (init → approach → grasp → squeeze → lift with
timestamps), `wrist_source` (`fk_executed` = FK of the real arm at grasp).

## Pose quality

`pose_world` is validated by rendering the object mesh at that pose and measuring
the silhouette MSE against the per-view object masks (antialiased, full
resolution, using the object's raw mesh). Trials with `sil_loss ≤ 0.003` are in
the main set; trials above are quarantined in `{dataset}_pose_outlier/`.
`recompute_pose.json` records each trial's `sil_loss` and `reject` flag.

`object_tracking` gives the full object trajectory; its first frame may differ
from `pose_world` (e.g. a symmetry rotation for rotationally-symmetric objects) —
`pose_world` is the authoritative pose for grasp definition.

## Loading example

```python
import numpy as np, json, cv2, os

trial = "corl_selected_100/attached_container/20260330_164351"

pose_world = np.load(f"{trial}/pose_world.npy")          # (4,4) object → world
C2R        = np.load(f"{trial}/C2R.npy")                 # (4,4) world ← robot
intr = json.load(open(f"{trial}/cam_param/intrinsics.json"))
extr = json.load(open(f"{trial}/cam_param/extrinsics.json"))

# executed grasp in the object frame
wrist_obj = np.load(f"{trial}/executed_grasp/wrist_se3.npy")   # (4,4)
fingers   = np.load(f"{trial}/executed_grasp/grasp_pose.npy")  # (16,) Allegro
meta      = json.load(open(f"{trial}/executed_grasp/meta.json"))

# project the object origin into camera `s`
s = list(intr)[0]
K  = np.array(intr[s]["intrinsics_undistort"])           # (3,3)
E  = np.array(extr[s])                                    # (3,4) world → cam
p_cam = E @ pose_world @ np.array([0, 0, 0, 1.0])
uv = (K @ p_cam[:3]); uv = uv[:2] / uv[2]                 # pixel

# object trajectory (gotrack datasets)
recs = json.load(open(f"{trial}/object_tracking/gotrack_output/world_pose_records.json"))
poses = [np.array(r["pose_world"]) for r in recs if r["status"] == "ok"]
```

## Camera rig

24 synchronized FLIR cameras (serials are the JSON keys). All per-view files —
`videos/`, `init_capture/images/`, `cam_param/*`, `raw/images/`,
`object_tracking/overlay_videos/` — are keyed by the same serial.
