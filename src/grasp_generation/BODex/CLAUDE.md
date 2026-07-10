# BODex — Dexterous Grasp Generation

GPU-accelerated dexterous grasp generation built on a **forked NVIDIA cuRobo**. Optimizes Allegro hand joint angles + wrist pose for force-closure grasps with collision avoidance.

## Directory Structure

```
BODex/
├── generate.py              # Main entry point (run grasp generation)
├── run.sh                   # Batch runner — edit object list here, then `bash run.sh`
├── src/curobo/              # Forked cuRobo library
│   ├── content/
│   │   ├── assets/robot/    # URDF + meshes (allegro, shadow, xarm_allegro)
│   │   └── configs/
│   │       ├── robot/       # Robot configs (allegro.yml, spheres/, hand_pose_transfer/)
│   │       └── manip/sim_allegro/  # Experiment configs (paradex_box/shelf/wall.yml)
│   ├── wrap/reacher/
│   │   └── grasp_solver.py  # GraspSolver: batch grasp optimization
│   ├── rollout/cost/
│   │   ├── grasp_cost.py    # Contact-based grasp cost, multi-stage optimization
│   │   └── grasp_energy/    # QP force closure (qp.py), DFC, CHQ1, TDG
│   ├── util/
│   │   ├── world_cfg_generator.py  # ParadexDataset: loads scenes, skip logic
│   │   └── sample_grasp.py  # HeurGraspSeedGenerator
│   └── curobolib/           # CUDA kernels (JIT compiled on first run)
├── collision_mesh.obj        # Debug meshes
├── collision_scene.obj
├── robot_spheres.obj
└── setup.py                  # `pip install -e .` for curobo package
```

## Conda Environment

**`bodex`** — cuRobo + `coal_openmp_wrapper` + torch + warp. Installed as editable:

```bash
cd src/grasp_generation/BODex
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 /home/mingi/miniconda3/envs/bodex/bin/pip install -e .
```

IMPORTANT: **Do NOT install BODex curobo into the `mingi` env** — it will break `autodex.planner` which uses a different cuRobo (from `~/RSS_2026/planner`).

- `bodex` env curobo → `AutoDex/src/grasp_generation/BODex/src/curobo/`
- `mingi` env curobo → `~/RSS_2026/planner/src/curobo/` (different fork, no `coal_openmp_wrapper`)

## Pipeline

1. Load scene configs (object mesh + placement + environment: table/box/shelf/wall)
2. Sample initial hand configurations (200 seeds/scene)
3. Multi-stage optimization: far approach → close contact → force closure
4. Contact points: 4 Allegro fingertips (`link_{3,7,11,15}.0_tip`)
5. QP solver evaluates grasp quality (friction coeff 0.1)
6. Save per-seed results to `bodex_outputs/{version}/`

## Output

`bodex_outputs/{version}/{obj_name}/{scene_type}/{scene_id}/{seed}/`:
- `wrist_se3.npy` — 4×4 SE3, **in object frame** (object-relative)
- `pregrasp_pose.npy` — 16 Allegro joint angles (pre-grasp)
- `grasp_pose.npy` — 16 Allegro joint angles (grasp)
- `bodex_info.npy` — dict: `contact_point`, `contact_frame`, `contact_force`, `grasp_error`, `dist_error`, `success`

Logging goes to `logging/grasp_generation/generate.log` (append mode, timestamps + object names + success rates).

## Running

Edit `run.sh` object list, then:

```bash
cd src/grasp_generation/BODex && bash run.sh
```

Or directly:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python generate.py \
    -c sim_allegro/paradex_shelf.yml -w 10 \
    --obj_list_file /tmp/bodex_obj_list.txt
```

### Key flags

- `-c` — Config file (paradex_box/shelf/wall.yml)
- `-w` — Parallel worlds on GPU. **`-w 10` is safe for all scene types**. `-w 35` works for box/wall but OOMs on shelf when other programs use GPU.
- `--obj_list_file` — Text file with object names (one per line, `#` for comments). Overrides config's `obj_list`.
- `-o` — Override output directory (default: `{repo}/bodex_outputs/{exp_name}`)

### Skip logic

`ParadexDataset` skips scenes where all seeds already have `grasp_pose.npy` in the output directory. Safe to interrupt and resume.

### Success field

`success` in `bodex_info.npy` is always 0 — the QP force closure threshold is overly strict. The grasp candidates are still usable for downstream planning/selection.

## Object Data

Object meshes live under the object root (`--obj_root_dir`, e.g. `~/shared_data/object_processing/{obj_name}/`):
- `raw_mesh/{obj_name}.obj` — Object mesh
- `processed_data/info/simplified.json` — OBB, gravity center
- `processed_data/mesh/simplified.obj` — Simplified mesh for planning
- `processed_data/urdf/coacd.urdf` — Convex decomposition for collision

Scene placement configs are **hand-specific** (the obstacle gap is adapted per hand) and live SEPARATELY under the AutoDex NAS, resolved via `autodex.utils.path.get_scene_dir(hand, obj, scene_type)`:
- `~/shared_data/AutoDex/scene/{hand}/{obj_name}/{box,shelf,wall,table,float}/*.json`

`generate.py` derives `hand` from the config's `robot_file` (e.g. `sim_allegro` → `allegro`) and `ParadexDataset` loads scenes from that per-hand path; mesh/OBB info still comes from `--obj_root_dir`.

## Adding a New Robot Hand (e.g. Inspire)

Add files in 4 locations under `src/curobo/content/`:

1. `assets/robot/inspire_description/` — URDF + meshes
2. `configs/robot/inspire.yml` — Kinematics, joint names, ee_link, collision links
3. `configs/robot/spheres/inspire.yml` — Collision sphere definitions
4. `configs/robot/hand_pose_transfer/inspire.yml` — Fingertip contact mapping
5. `configs/manip/sim_inspire/*.yml` — Experiment configs (contact strategy, friction, seed pose)

Currently supported: **Allegro**, **Shadow Hand**, **UR10e+Shadow**, **xArm+Allegro**.

