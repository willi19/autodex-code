# AutoDex

Dex manipulation pipeline: perception → plan → execute.

## Repository Structure

```
autodex/                    # Core library (importable package)
├── perception/             # Mask, depth, pose (has its own CLAUDE.md)
│   ├── mask.py             # YOLOE + SAM3 segmentation
│   ├── depth.py            # FoundationStereo + Depth-Anything-3
│   ├── pose.py             # FoundationPose tracking
│   ├── stereo_video_depth.py  # CLI batch stereo depth (TRT)
│   └── thirdparty/         # External model repos + weights
│       └── weights/        # yoloe-26x-seg.pt, mobileclip2_b.ts
├── planner/                # Motion planning (has its own CLAUDE.md)
│   └── planner.py          # GraspPlanner: IK → plan_single_js
├── executor/               # Robot execution
│   └── real.py             # Real robot executor
└── utils/
    └── file_io.py          # (WIP) Cache management, download/upload utilities

src/                        # Scripts & CLI wrappers
├── process/                # Batch processing pipelines
│   ├── batch_mask.py       # SAM3 video segmentation
│   ├── batch_mask_yoloe.py # YOLOE single-frame segmentation
│   ├── batch_mask_all.py   # Run mask on all cameras
│   ├── batch_depth.py      # Stereo depth (fixed camera pair, TRT)
│   ├── batch_depth_auto.py # Stereo depth (auto pair selection, all cameras, TRT)
│   ├── batch_pose.py       # FoundationPose batch tracking
│   ├── batch_pose_overlay.py # Per-camera pose tracking + mesh overlay video
│   ├── download_videos.py  # Network FS → local cache
│   └── upload_results.py   # Local cache → network FS
├── demo/                   # Demo scripts
│   ├── real.py             # Real robot demo
│   ├── perception_exp.py   # Perception experiment runner
│   └── run_perception.py   # Perception pipeline demo
├── visualization/
│   ├── mesh_process/       # Mesh viewers (object, scene, table_top)
│   └── turntable_grasp.py  # Turntable video renderer for grasp candidates
├── grasp_generation/
│   ├── BODex/              # Dexterous grasp generation (has its own CLAUDE.md)
│   │   ├── generate.py     # Main entry point
│   │   ├── run.sh          # Batch runner — edit object list here
│   │   └── src/curobo/     # Forked cuRobo library
│   └── sim_filter/         # MuJoCo validation + set cover selection (has its own CLAUDE.md)
└── validation/             # Validation & comparison scripts
    └── perception/         # Perception pipeline validation (has its own CLAUDE.md)
        ├── scene.py        # Single-scene overlay validation
        ├── stereo_rectify.py  # Stereo rectification visualization
        ├── viz_stereo_pairs.py # Visualize auto-selected stereo pairs
        └── multiobject/    # Multi-object combinatorial validation pipeline

bodex_outputs/              # BODex grasp generation results (gitignored)
logging/                    # Run logs (grasp_generation/generate.log)

Visualization/              # Scene visualization & evaluation
├── scene.py                # Viser-based scene viewer
└── ...                     # Evaluation, paper figures
```

## Key Conventions

- **Local cache**: `~/video_cache/` mirror network FS. Map: strip `/home/mingi/paradex1/capture/` prefix.
- **Video format**: `.avi` throughout. Mask=MJPG. Depth=FFV1 (lossless, uint16 mm as BGR).
- **Camera params**: `cam_param/intrinsics.json` + `extrinsics.json` per capture dir. Key=serial string.
  - `intrinsics.json` vals=dict w/ `intrinsics_undistort`, `original_intrinsics`, `dist_params`, `width`, `height`.
- **Capture dir layout**: `{base}/{obj_name}/{idx}/` w/ `videos/`, `cam_param/`, `depth/`, `obj_mask/`, etc.
- **Model weights**: all in `autodex/perception/thirdparty/weights/`. Ref via `YOLOE_WEIGHTS` from `autodex.perception.mask`.

## Stereo Depth Pipeline

Two depth scripts:

- **`batch_depth.py`**: fixed cam pair (manual `--left_serial` / `--right_serial`). Proven, simple.
- **`batch_depth_auto.py`**: auto pair select all cams. Rig-based adjacency (focal_group × z_level, angle-sorted, MAX_ANGLE_GAP=40°).

### Stereo Rectification

Both use `cv2.stereoRectify` → R1/R2 rot mat.
`src/process/depth.py` use **validation approach** (same as `stereo_rectify.py`):
- Use `f_orig = max(K_left[0,0], K_right[0,0])` not stereoRectify `f_rect` (degenerate for wide-baseline).
- Oversized canvas → valid region → workspace crop → final P matrix.
- Both L/R use same P (same f, cx, cy), preserve epipolar align.

### Stereo Rectification Cropping Rules (IMPORTANT)

1. **Valid region**: UNION (`valid_l | valid_r`) — NEVER intersection, cut right cam content.
2. **Workspace crop**: crop TO fixed robot-frame bbox, baked into P via `initUndistortRectifyMap` (one remap, no intermediate full-size).
   - Fixed bounds robot frame: `ws_min=[0.35, -0.30, 0.0]`, `ws_max=[0.80, 0.21, 0.4]`
   - Same const every capture — set once from charuco triangulation, NOT recomputed.
   - Project 8 bbox corners via `C2R.npy` + extrinsics + R1/R2 to rectified space.
   - UNION of both cams' projections so full bbox visible in both views.
3. **Same cx** both cams — no per-cam cx offset, no disparity correction needed.
4. **Aspect ratio filter**: skip pair ratio > 2.5:1 (degenerate wide-baseline).
5. **Object NEVER cut off** — bbox must fully contain workspace.

### Disparity-to-Depth: Rectified Z vs Original Z

**Critical**: stereo `depth = f * B / disparity` give Z in **rectified** cam frame, not original. When un-rectify depth back to original pixel coords, divide by `rz` — Z component of `R1 @ K_inv @ [u, v, 1]` per original pixel `(u, v)`:

```
Z_orig = Z_rect / rz
```

Without this, cams w/ big R1 rot (e.g. 65° wide-baseline) get ~30-50% depth err, big cross-view reproj misalign. Cams w/ small R1 (~20°) look fine cuz `rz ≈ 1`.

Bug subtle cuz:
- Per-cam depth colormap look OK (relative depth order preserved)
- Self-reproj (pixel → 3D → same pixel) trivially perfect any depth val
- Only cross-view reproj reveal err, magnitude depend on R1 angle

### Depth Debugging Checklist

Stereo depth wrong → use **cross-view reproj** to validate — NOT per-cam colormap or self-reproj (both hide err). Steps:
1. Pick src cam w/ depth, backproject 3D world via `K_src`, `T_src`
2. Reproject 3D to diff cam via `K_tgt`, `T_tgt`
3. Overlay reproj on target image — features (checkerboard, obj) must align
4. Misalign → check `rz` correction, stereo pair quality (R1 angle), baseline/focal

`batch_depth_auto.py --overlay_only` gen cross-view reproj grids in `depth_overlay/`.

### Depth Encoding

FFV1 codec, uint16 mm as BGR: `B = low_byte, G = high_byte, R = 0`.
Use `encode_depth_uint16()` / `decode_depth_uint16()` from `autodex.perception.depth`.

## Conda Environments

- `foundation_stereo`: FoundationStereo TRT depth (`tensorrt` + `pycuda` here)
- `foundationpose`: FoundationPose, YOLOE
- `sam3`: SAM3 seg, Depth-Anything-3

## Grasp Candidate Visualization

`src/visualization/turntable_grasp.py` render turntable video of grasp candidates (obj + Allegro hand) via Open3D offscreen renderer (EGL headless).

### Data Sources

- **Candidates**: `{candidate_path}/{version}/{obj_name}/{scene_type}/{scene_id}/{grasp_name}/` — has `wrist_se3.npy`, `grasp_pose.npy`, `pregrasp_pose.npy`
- **Setcover order**: `{code_path}/order/{version}/{obj_name}/setcover_order.json` — ranked grasp list (greedy set cover)
- **Object meshes**: `{obj_path}/{obj_name}/raw_mesh/{obj_name}.obj`
- **Object pose**: `get_scene_dir(hand, obj_name, "table")/4.json` (= `~/shared_data/AutoDex/scene/{hand}/{obj_name}/table/4.json`; scenes are hand-specific)
- **Robot URDF**: `{urdf_path}/allegro_hand_description_right.urdf`

Paths from `rsslib.path`: `candidate_path=/home/mingi/RSS_2026/candidates`, `code_path=/home/mingi/RSS_2026`, `obj_path=/home/mingi/shared_data/RSS2026_Mingi/object/paradex`.

### Setcover Versions (no dup except attached_container → use revalidate)

- `revalidate`: 33 obj
- `v2`: 21 obj (20 unique after dedup)
- `v3`: 45 obj
- Total: 98 unique obj

### Output Layout (episode-wise for HuggingFace/GitHub Pages)

```
data/{obj_name}/{rank:03d}/turntable.mp4
```

### Commands

```bash
# Single grasp
python src/visualization/turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11

# Top N from setcover
python src/visualization/turntable_grasp.py --version revalidate --obj soap_dispenser --top 100

# All 98 objects × top 100
python src/visualization/turntable_grasp.py --batch-all --top 100
```

### Camera Auto-framing

Use bounding sphere of combined obj+robot mesh. Cam dist = `sphere_radius * padding / sin(effective_half_fov)`. No clipping any turntable angle.

## Planning Validation

### Reachability Grid Search (`src/validation/planning/reachability_set.py`)

IK-only check over grid (x_offset, z_rotation, tabletop_pose) per obj.
- Grid: x_offset [0.2–0.5, step 0.05] × z_rotation [0°–330°, step 30°] × all tabletop poses × 10 trials/point
- Output: `outputs/reachability/{obj_name}/reachability_selected_100.json` (grid) + `*_viz.json` (IK sol viz)
- 14 obj done. IK ~97% deterministic (all-or-nothing), ~3% partial at boundary configs.

### Reachability Viewer (`src/validation/planning/reachability_viewer.py`)

Interactive viser viewer for reachability. Two modes:
- **Single**: robot at IK qpos + obj mesh + 5 grasp candidate hands (green=fwd, red=bwd). Sliders for pose, x_offset, z_rotation, IK sol #.
- **Heatmap**: 1D row color-coded spheres along x_offset (green=reach, red=unreach, yellow=partial). Z_rotation via slider. Robot at INIT_STATE default pose. Obj mesh shown offset in y for ref.
- **Filter/Navigate**: jump between Reachable/Unreachable/Partial points.

```bash
python src/validation/planning/reachability_viewer.py --port 8080
```

### Planning Success Rate (`src/validation/planning/success_rate.py`)

Compare IK reachability vs full planning success. Load IK-reachable points from `outputs/reachability/`, run `planner.plan()` only on those. Break early on success (retry only on fail). Save per-stage timing breakdown (all/success/fail) to JSON.

```bash
python src/validation/planning/success_rate.py --obj attached_container --version selected_100 --n_trials 1
```

Output: `outputs/planning_success_rate/{obj_name}/plan_vs_ik_{version}.json`

### Key Constants

- `TABLE_POSE_XYZ = [1.1, 0, -0.1]`, `TABLE_DIMS = [2, 3, 0.2]`
- `INIT_STATE`: from `autodex.utils.robot_config` — xarm6 + allegro default joint config
- Grasp candidates: `selected_100` version, 100 per obj via `load_candidate()`
- Backward filter: `wrist_se3[:, 0, 2] < 0.3`

## External Dependency: `rsslib` (legacy, being replaced)

Old shared lib `~/RSS_2026/rsslib/`. Still imported by `src/`, `Visualization/`, BODex scripts. **All core fn already ported to `autodex/`:**

- `rsslib.path` → `autodex.utils.path` (same fn, `project_dir` → `~/shared_data/AutoDex`)
- `rsslib.conversion` → `autodex.utils.conversion` (same: `cart2se3`, `se32cart`, `se32action`)
- `rsslib.robot_config` → `autodex.utils.robot_config` (same: `INIT_STATE`, `LINK6_TO_WRIST`)
- `rsslib.scene` → `autodex.utils.scene` (partial: `overlay_scene`, `get_scene_image_dict_template`)
- `rsslib.curobo_util` → `autodex.planner.planner.GraspPlanner` (fully replace — `CuroboPlanner`/`CuroboIkSolver`/`filter_collision`/`get_traj` all methods on `GraspPlanner`)
- `rsslib.planner` → `autodex.planner.planner` (replaced)
- `rsslib.visualizer`, `rsslib.gui_player` — not yet ported (Viser GUI helper)

Migration: mech-replace `from rsslib.xxx import` → `from autodex.utils.xxx import` in `src/`. BODex internal import need separate handle (forked cuRobo w/ own `rsslib` ref).

## Grasp Generation: BODex (`src/grasp_generation/BODex/`)

GPU-accel dex grasp gen on **forked NVIDIA cuRobo**. Optimize Allegro joint angles + wrist pose for force-closure grasp w/ collision avoid. See `src/grasp_generation/BODex/CLAUDE.md` for arch detail.

## Grasp Pipeline: BODex → Sim Filter → Candidate → Selection

BODex raw output → sim validate → select before plan. See `src/grasp_generation/sim_filter/CLAUDE.md`.

```
bodex_outputs/ → MuJoCo sim eval → candidates/ → set cover selection → selected_100/
```

Key data (now at `~/RSS_2026/`): `candidates/{version}/`, `order/{version}/`, `candidates/selected_100/`.

## Perception Evaluation Pipeline (`src/validation/execution/eval_perception/`)

Eval per-view 6D pose quality to pick best cam views.
See `src/validation/execution/eval_perception/CLAUDE.md`.

### CRITICAL: Reference Implementation

**`/home/mingi/shared_data/_object_6d_tracking/`** = ref (ground truth) pipeline. Write perception code → ALWAYS read+follow ref first. Do NOT improvise / write from scratch.

Key ref files:
- `run/models/depth_server.py` — DA3 depth: `DepthAnything3.from_pretrained()`, intrinsics + extrinsics, fallback on exception
- `run/models/foundationpose_server.py` — FPose: `trimesh.load(process=False)`, downscale=0.5, `mask.astype(bool)`
- `run/models/silhouette_server.py` — Differentiable sil opt (200 iter, MSE + IoU loss, rot 6d param)
- `run/run_object_6d_pipeline_distributed.py` — Full orchestration, NMS, viz w/ `nvdiffrast_render`

### HARD RULES (learned from mistakes)

1. **NEVER `DepthAnything3(model_name=...)` — ALWAYS `DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")`**. Constructor = random-init weights. `from_pretrained` loads real trained weights.
2. **NEVER omit extrinsics from DA3** — multi-view align w/ extrinsics → correct metric depth. No extrinsics → wrong scale.
3. **NEVER `trimesh.load(force="mesh")`** for FoundationPose — use `process=False`. `force="mesh"` merges/dedup vertices (7944 vs 22743), change mesh geom.
4. **NEVER rewrite render code** — use `Utils.py`'s `nvdiffrast_render` + `make_mesh_tensors` direct. Import need pytorch3d in env.
5. **Thing not work → read ref code first** — no blame external (DA3, extrinsics, calib, xformers).

## Conda Environments (Updated)

- `foundation_stereo`: FoundationStereo TRT (`tensorrt` + `pycuda`)
- `foundationpose`: FoundationPose, YOLOE, pytorch3d, nvdiffrast
- `sam3`: SAM3 seg
- `dav3`: Depth-Anything-3 (separate env w/ all DA3 dep)

## Daemon Setup (Perception Pipeline)

### Current pipeline (FoundPose init only)

`src/execution/run_auto.py` use **`init_daemon`** on capture1-3, 5, 6 (gotrack_cu128 env).
Start/stop/status all wrap by `scripts/init_daemons.sh`:

```bash
bash scripts/init_daemons.sh start
bash scripts/init_daemons.sh status
bash scripts/init_daemons.sh stop
bash scripts/init_daemons.sh log capture1   # tail one PC's log
```

Daemon: `src/execution/daemon/init_daemon.py`. Ports: 5006 (mask PUB),
5007 (pose PUB), 6893 (control). Orchestrator: `autodex.perception.init_orchestrator.InitOrchestrator`.

### Legacy SAM3+FPose daemon setup (for `src/execution_prev/`)

Only need when run old `src/execution_prev/run_auto.py` / `run_debug.py` /
`run_demo.py`. New pipeline don't use.

### Architecture (legacy)

- **Main PC** (mingi, RTX 3090): DA3/stereo depth + sil match + plan
- **capture1, 2, 3**: SAM3 daemon (ZMQ, port 5001)
- **capture4, 5, 6**: FPose daemon (ZMQ, port 5003)

### First-time setup on each capture PC

```bash
# 1. Clone repo
git clone https://github.com/willi19/AutoDex.git ~/AutoDex
cd ~/AutoDex

# 2. Download weights from NAS
bash scripts/setup_weights.sh

# 3. For FPose PCs: copy mycpp build (python 3.9 required)
mkdir -p ~/AutoDex/autodex/perception/thirdparty/FoundationPose/mycpp/build
cp ~/shared_data/AutoDex/weights/foundationpose/mycpp_build/mycpp*.so \
   ~/AutoDex/autodex/perception/thirdparty/FoundationPose/mycpp/build/
```

### SAM3 daemon (capture1, 2, 3) — legacy

```bash
# Conda env setup (once)
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyzmq ftfy regex psutil pycocotools einops iopath hydra-core timm tqdm pillow scipy huggingface_hub opencv-python numpy
python -c "from huggingface_hub import login; login(token='<HF_TOKEN>')"

# Run daemon
conda activate sam3
cd ~/AutoDex
python src/execution_prev/daemon/perception_daemon.py --model sam3 --port 5001
```

### FPose daemon (capture4, 5, 6) — legacy

```bash
# Conda env setup (once)
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyzmq opencv-python numpy trimesh nvdiffrast omegaconf
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Run daemon
conda activate foundationpose
cd ~/AutoDex
python src/execution_prev/daemon/perception_daemon.py --model fpose --port 5003 \
    --mesh ~/shared_data/object_6d/data/mesh/attached_container/attached_container.obj
```

### Update on capture PCs

```bash
cd ~/AutoDex && git fetch origin && git reset --hard origin/main
```

### NAS Weight Structure

```
~/shared_data/AutoDex/weights/
├── foundationpose/
│   ├── 2023-10-28-18-33-37/   # RefinePredictor
│   ├── 2024-01-11-20-02-45/   # ScorePredictor
│   └── mycpp_build/           # Pre-built C++ extension
├── sam3/
│   └── sam3.pt                # SAM3 checkpoint (3.3GB)
├── da3/
│   └── model.safetensors      # DA3-LARGE (1.6GB)
└── yoloe/
    ├── yoloe-26x-seg.pt       # YOLOE (164MB)
    └── mobileclip2_b.ts       # MobileCLIP (243MB)
```

### Quick Start: Run Legacy Perception Pipeline (`src/execution_prev/`)

```bash
# 1. Start SAM3 daemons (capture1, 2, 3)
ssh capture1 "cd ~/AutoDex && conda activate sam3 && python src/execution_prev/daemon/perception_daemon.py --model sam3 --port 5001"
ssh capture2 "cd ~/AutoDex && conda activate sam3 && python src/execution_prev/daemon/perception_daemon.py --model sam3 --port 5001"
ssh capture3 "cd ~/AutoDex && conda activate sam3 && python src/execution_prev/daemon/perception_daemon.py --model sam3 --port 5001"

# 2. Start FPose daemons (capture4, 5, 6)
ssh capture4 "cd ~/AutoDex && conda activate foundationpose && python src/execution_prev/daemon/perception_daemon.py --model fpose --port 5003"
ssh capture5 "cd ~/AutoDex && conda activate foundationpose && python src/execution_prev/daemon/perception_daemon.py --model fpose --port 5003"
ssh capture6 "cd ~/AutoDex && conda activate foundationpose && python src/execution_prev/daemon/perception_daemon.py --model fpose --port 5003"

# 3. Run pipeline (robot PC)
ssh robot
conda activate autodex
cd ~/AutoDex
python src/execution_prev/run_perception.py \
    --capture_dir ~/shared_data/mingi_object_test/attached_container/20260317_172712 \
    --obj attached_container --depth da3
```

### Grasp Selection (Set Cover)

```bash
conda activate mingi
python src/grasp_generation/order/compute_order.py --hand allegro --version v3
python src/grasp_generation/order/compute_order.py --hand inspire --version v3
```

Output: `~/AutoDex/candidates/{hand}/v3_order/{obj}/setcover_order.json`

## Execution Pipeline (`src/execution/`)

Current pipeline run **FoundPose distributed init** (`InitOrchestrator` →
capture1-3,5,6 `init_daemon`) for perception, then plan + exec on robot.
GoTrack track = **out of scope** here — perception init-only, one shot per trial.

Legacy SAM3+FPose distrib pipeline (w/ own `perception_pipeline.py` +
`perception_daemon.py` + `gotrack_daemon.py`) live under `src/execution_prev/` for
ref. Imports rewritten (`src.execution.*` → `src.execution_prev.*`) so
can still run side-by-side if needed.

```
src/execution/
├── run_auto.py        # automated FoundPose init → plan → execute → label loop
├── scene_cfg.py       # pose_world → planner scene_cfg (cylinder/sphere snap, table cuboid)
├── label.py           # human-in-the-loop trial label prompt
└── daemon/
    └── init_daemon.py # runs on capture PCs (gotrack_cu128 env)
```

### Prerequisites

```bash
bash scripts/init_daemons.sh start    # launches init_daemon on capture1-3, 5, 6
bash scripts/init_daemons.sh status   # expect "1" per PC
```

### run_auto.py — Automated grasp evaluation loop

```bash
# Basic (table, all candidates)
python src/execution/run_auto.py --obj wood_organizer

# Success-only candidates (retest proven grasps)
python src/execution/run_auto.py --obj brown_ramen --success_only

# Scene modes
python src/execution/run_auto.py --obj brown_ramen --scene wall --wall_angle 0 --wall_gap 0.04 --success_only
python src/execution/run_auto.py --obj brown_ramen --scene shelf --success_only
python src/execution/run_auto.py --obj brown_ramen --scene cluttered --clutter_seed 42 --clutter_min_dist 0.12 --success_only
python src/execution/run_auto.py --obj brown_ramen --viz                # launch viser visualizer
python src/execution/run_auto.py --obj brown_ramen --hand inspire_left  # left inspire hand
```

`--hand` accept `allegro` (default), `inspire`, `inspire_left`. `inspire_left` use `xarm_inspire_left.yml` (copy to `~/shared_data/AutoDex/content/configs/robot/` from `~/mcc_minimal/`).

Each trial keep capture-PC stream running for init pipeline SHM access; toggle off → vid record → back on around executor exec window.

### run_debug.py — Single trial, viser preview, GUI executor

Same init pipeline as `run_auto.py`, but stop at visualizer after plan
so can inspect traj before commit. Press `y` → traj run via `RealExecutor(mode="gui")` (step-through GUI); `q` skip exec. Single trial only. No video / sync_generator / timestamp_monitor — stream stay on whole time for init pipeline. Save to `~/shared_data/AutoDex/experiment/debug/...` default (`--exp_name`).

```bash
python src/execution/run_debug.py --obj attached_container
python src/execution/run_debug.py --obj brown_ramen --hand inspire_left
python src/execution/run_debug.py --obj brown_ramen --scene wall --wall_angle 0
```

### Legacy `src/execution_prev/`

`run_auto.py`, `run_debug.py`, `run_demo.py`, `run_perception.py` from old
SAM3+FPose pipeline still live there. Need legacy daemon (see Daemon Setup below).

### Key Design Decisions

- **Candidate result tracking**: `result.json` saved in candidate dir (`candidates/allegro/selected_100/{obj}/{scene}/{id}/{grasp}/`). `load_candidate` skip candidate w/ existing result (table mode). Other scene (`--success_only`, wall, shelf, cluttered) don't skip/save to candidates.
- **Cylinder symmetry**: obj w/ y-axis symmetry (in `CYLINDER_OBJECTS`: pepper_tuna, pepper_tuna_light, pepsi, pepsi_light) → rot snap to tabletop pose whose y-axis best match est pose. Only rotmat replaced, trans preserved. Tabletop poses at `{obj_path}/{obj}/processed_data/info/tabletop/*.npy`. NOTE: cylinder snap had multi bug (wrong frame, wrong tabletop sel → standing→lying). Some early cylinder exp (pepper_tuna, pepper_tuna_light success_only) may have bad data from buggy snap.
- **Table surface snap**: `_snap_z_to_table` ensure mesh bottom ≥ TABLE_SURFACE_Z (0.039m). Prevent hand below table. NOTE: changed 0.037→0.043→0.045→0.042→0.039 on 2026-03-27. Higher val → plan fail (too much lift), lower → table scratch. 0.039 works — revisit if issue recur.
- **Lift speed**: `_move_cartesian` lift use `vel_scale=1/1.5` (slower than default). Changed 2026-03-30 — default too fast, cause drop.
- **Sil loss threshold**: perception return None if sil match loss > 0.003 (unreliable pose).
- **IK retract_config**: IK solver use `retract_config=INIT_STATE` so joint sol stay near start config. Fix joint 6 wrap issue (IK return val in [-2π, 2π]).
- **Trajectory smoothing**: use `get_interpolated_plan()` not raw `optimized_plan` (64 waypoint → dense interp traj). Allegro=CUBIC interp (jerk min), Inspire=LINEAR_CUDA. Changed 2026-04-03 — raw optimized_plan had jerky motion.
- **Per-hand planner config**: `GraspPlanner(hand=)` select YAML config, collision_activation_distance, num_trajopt_seeds, interp_type per hand. Allegro: `xarm_allegro.yml`, act_dist=0.01, 32 seeds, CUBIC. Inspire (right): `xarm_inspire.yml`, act_dist=0.002, 32 seeds, LINEAR_CUDA. Inspire_left: `xarm_inspire_left.yml` + `inspire_left_floating.yml`, same numerics as inspire. Joint order right/left-mirror only in *names*, so `INSPIRE_INIT` shared btw inspire + inspire_left.
- **Inspire hand conversion**: `_convert_inspire` map rad → 0-1000 controller unit w/ joint reorder (cuRobo order: thumb_yaw/pitch/index/middle/ring/pinky → ctrl order: pinky/ring/middle/index/thumb_pitch/thumb_yaw). Joints norm by per-joint limit `[1.15, 0.55, 1.6, 1.6, 1.6, 1.6]`, invert (1000=open, 0=closed).
- **retract_config fix**: `xarm_allegro.yml` retract_config finger joints → `ALLEGRO_INIT` vals — old vals had thumb base violate joint limit [0.263, 1.396]. Code use `robot_config.py` INIT_STATE not YAML retract_config for `plan_single_js` start state.

### Experiment Storage Layout

```
~/shared_data/AutoDex/experiment/{exp_name}/
├── allegro/{obj}/{timestamp}/              # table (default)
├── success_only/allegro/{obj}/{timestamp}/ # --success_only
├── wall/allegro/{obj}/{timestamp}/         # --scene wall
├── wall_success_only/allegro/{obj}/{timestamp}/
├── shelf/allegro/{obj}/{timestamp}/
├── shelf_success_only/allegro/{obj}/{timestamp}/
├── cluttered/allegro/{obj}/{timestamp}/
└── cluttered_success_only/allegro/{obj}/{timestamp}/
```

Each exp dir has: `raw/`, `images/`, `cam_param/`, `pose_world.npy`, `pose_overlay/`, `plan/`, `result.json`.

### Scene Obstacles (`autodex/planner/obstacles.py`)

- **table**: Table cuboid only
- **wall**: Single wall around object. `--wall_gap` (meters), `--wall_angle` (degrees, 0=+y)
- **shelf**: Open-front shelf. `--shelf_width/depth/height/gap`, `--no_shelf_back/sides/top`
- **cluttered**: Random cubes. `--clutter_seed`, `--clutter_n`, `--clutter_min_dist/max_dist`

### Known Issues / Fixes Applied

- **URDF joint 6 limits**: `xarm_allegro.urdf` has ±2π, real xarm6 = ±π. IK can return val outside ±π. Fixed via `retract_config=INIT_STATE` in IK solver (no URDF change).
- **Allegro collision sphere**: `spheres/allegro.yml` base_link had rad 0.5 (typo, should be 0.015). Fixed.
- **moviepy import**: DA3 `gs.py` import `moviepy.editor` not in moviepy 2.x. Fixed w/ try/except.
- **PySpin version**: must match Spinnaker SDK ver (4.3.0.189). PySpin 4.2 → symbol err.
- **numpy version**: PySpin 4.3 need numpy<2.
- **FPose daemon mesh**: pipeline `__init__` must send mesh/obj_name to FPose daemon. Daemon support `obj_name` lookup from NAS (`~/shared_data/object_6d/data/mesh/{obj}/`).

### Reference

`~/RSS_2026/planner/inference/train/run_auto_v2.py` = ref impl. All exec seq (init → approach → pregrasp → grasp → squeeze → lift → release) match this ref.

## Ongoing Refactoring

`src/process/` scripts have heavy code dup w/ `autodex/perception/`.
Plan: consolidate core logic into `autodex/`, make `src/process/` thin CLI wrapper.
See `autodex/perception/CLAUDE.md` for detailed plan.