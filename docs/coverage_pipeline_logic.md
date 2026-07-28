# v8 Grasp-Gen → Scene → Coverage : current logic

Written for review. This describes **how it works now**, and flags the messy/wrong parts so you can decide fixes.

> **⚠️ IMPORTANT — the coverage work done so far is a MESSY PREVIOUS VERSION.**
> All the executed-grasp / pose_id / canonicalize work on
> `~/shared_data/autodex_dataset/{selected_100 (RSS), corl_selected_100 (corl)}` was
> the process of a messy *previous* version, NOT the path forward. **Do not continue
> building on it.** The forward-looking, correct pieces are: (a) hand-specific scenes
> at `get_scene_dir(hand, obj)` with per-hand min gaps, and (b) the symmetry fix in
> grasp generation (below). The autodex_dataset coverage section is kept only to
> record what was done, not as a spec to keep extending.
>
> **Definition of "cover":** a grasp *covers* a scene iff **(1)** the grasp has **no
> collision** in that scene (obstacles + table) AND **(2)** the object's **tabletop
> pose is the same** as the scene's. Both conditions are required.
>
> **Scene visualization rule:** to visualize a scene, ALWAYS use the dedicated
> visualizer (`src/visualization/grasp_generation/view_bodex.py`, which reads
> `get_scene_dir(hand, obj)`). Never improvise scene building for viewing.

---

## 1. Object source
- All v8 grasp gen + coverage takes objects from **`~/shared_data/object_processing/{obj}/processed_data/`** (passed as `--obj_root`).
  - `mesh/simplified.obj`, `info/simplified.json` (obb), `info/tabletop/*.npy` (resting poses), `symmetry.json` (per-object, currently **empty** for most), `urdf/`.
- **NOT** `autodex.utils.path.obj_path` (= `~/shared_data/AutoDex/object/paradex`). That is an older, DIFFERENT object set with different tabletop poses. The stock `classify_tabletop_pose` reads from there → mislabels pose_id for object_processing scenes (the classifier bug).

## 2. Grasp generation (`adaptive_orchestrator.py`)
For each `(obj, scene_type ∈ {wall, shelf, box}, scene_id)`:
- **scene_id = (pose_idx × z_rot)**, enumerated by `enumerate_scene_configs`.
  - z_rots per pose from `z_rots_for_pose(obj, pose, symmetry_reg)`:
    - revolute object (in global `src/scene_generation/symmetry.json`) with vertical axis → `[0]` (folded).
    - else → default 5 z_rots `[0,72,144,216,288]`.
- **Adaptive gap search (the core principle):** escalate gap `[0.02,0.04,0.06,0.08]` × seed_num `[200,1000]` until ≥ **SUCCESS_THRESHOLD (5)** grasps pass the MuJoCo sim filter. On success at 0.02, a **bonus** phase tries gap `0.0`. Keep the **tightest (minimum) gap** that succeeded → that is the scene's **final gap**.
- Output:
  - **Scenes → `get_scene_dir(hand, obj, type)` = `~/shared_data/AutoDex/scene/{hand}/{obj}/{type}/{id}.json`** — HAND-SPECIFIC, at the final gap. (First run backs up prior to `{type}_prev/`.)
  - **Grasps (candidates) → `~/AutoDex/candidates/{hand}/v8/{obj}/{type}/{id}/`** (`wrist_se3.npy`, `grasp_pose.npy`, ...).
  - **Summary → `~/AutoDex/logging/adaptive/{hand}/v8/{obj}.json`** — per scene: `final.gap`, `final.status`, history.

### Key consequences
- **The gap is per-HAND.** Same (obj, pose, z_rot) settles at a different final gap for allegro vs inspire (escalation depends on that hand's grasp success). So there is no single "the scene" — allegro's scene ≠ inspire's scene.
- Authoritative per-hand scene = `get_scene_dir(hand, obj, type)` (its `meta.param.gap` == summary `final.gap`).
- **`object_processing/{obj}/scene/` is STALE / not written by the current orchestrator** — do not read it.

## 2.5 Data processing — how `executed_grasp` was generated (MESSY PREVIOUS VERSION, for record)
Scripts at `src/dataset/exec_grasp/` (+ `src/dataset/`). A capture trial only stored raw robot logs; the grasp was reconstructed. Pipeline per trial:

**RSS (`selected_100`):**
1. `fix_c2r.py` — the trial's C2R was wrong; match the trial to its `handeye_calibration` episode and write a corrected `C2R.npy`.
2. `copy_raw.py` — copy raw (`raw/arm`, `raw/hand`, `videos/`, `cam_param/`) into `autodex_dataset/selected_100/{obj}/{ts}/` (real copy, not symlink).
3. `recompute_pose.py` — re-estimate the object's init pose with FoundPose → `pose_world.npy` (+ `recompute_pose.json`; reject if sil_loss > 0.003).
4. `extract_exec.py` — find the grasp frame in the trajectory (near min wrist-z, grasped).
5. `build_exec_grasps.py` — identify the executed candidate by **finger-pose matching** (`raw/hand/action` vs candidate `grasp_pose`, with `_convert_allegro` thumb reorder, L2 ≤ 0.05) → save `grasp_pose.npy`.
6. `refine_exec.py` — FK the executed arm qpos with the **new** `xarm_allegro.urdf` → `link6 @ ALLEGRO_LINK6_TO_WRIST` → wrist in robot frame → `wrist_se3 = inv(obj_robot) @ wrist` (object frame). `obj_robot = inv(C2R) @ pose_world`.
7. `save_states.py` (`detect_states.py`) — parse `execution_states` (init/approach/pregrasp/grasp/squeeze/lift start timestamps).
8. `harmonize_merge.py` — write the unified **17-key** `meta.json` (`note` tags RSS/corl). NOTE: harmonize STRIPS any non-17-key field (so `coverable` etc. can't be stored).

**corl (`corl_selected_100`):** `corl_exec.py` — same idea but uses `plan/traj[-1]` as the grasp config (no external candidate), FK wrist, states, success from object-tracking 6D. Then copied into `autodex_dataset/corl_selected_100/`.

**Post-processing (this session):** `reclassify_plain.py` (pose_id vs object_processing tabletop, initial-frame) + `canonicalize.py` (symmetry-variant grasps re-expressed in canonical frame). `coverable` is DERIVED from `pose_id`, never stored.

Output per trial: `{trial}/executed_grasp/{wrist_se3.npy, grasp_pose.npy, meta.json}` (+ `wrist_se3_orig.npy` backup if canonicalized).

## 3. Coverage (executed real-robot grasps vs v8 scenes)
Executed grasps live in `~/shared_data/autodex_dataset/{selected_100(RSS) | corl_selected_100(corl)}/{obj}/{ts}/executed_grasp/` (all **allegro**):
- `wrist_se3.npy` (executed wrist in object frame, from FK of qpos), `grasp_pose.npy` (fingers), `meta.json`.
- **pose_id** = which tabletop pose it rests at, via `reclassify_plain` (plain z-aligned geodesic vs **object_processing** tabletop, residual >25° ⇒ `pose_id=None` = uncoverable). `coverable` is DERIVED from `pose_id` (not stored — harmonize strips extra keys).
- **Symmetry canonicalize:** an uncoverable capture that is a mesh-symmetry variant of a tabletop pose is re-expressed in the canonical frame (`wrist6d_new = inv(S) @ wrist6d`, physical grasp unchanged → no table penetration), so it becomes coverable at that pose.

**Coverage** = for each v8 scene (per-hand, at its final gap): is there an executed grasp at the matching resting pose whose hand clears all obstacles **incl. the table**? Should read scenes from `get_scene_dir(allegro, obj, type)`.

---

## 4. What is messy / wrong (decide fixes)
1. **Symmetry registry incomplete.** Scene gen folds z_rots only for the global `src/scene_generation/symmetry.json` (9 entries: 8 revolute cylinders + 1 discrete). **Spheres and boxes are NOT registered** → default 5 z_rots → redundant scenes (e.g. baseball = sphere, 1 tabletop pose but 21 scenes; identical scenes even get different final gaps from seed randomness). A mesh-based symmetry detector exists (used for grasp canonicalize) but was never fed back into the registry.
2. **Two+ disconnected symmetry sources:** global `src/scene_generation/symmetry.json`, per-object `object_processing/{obj}/symmetry.json` (empty), `pose_symmetry.json` (via `get_cyl_axis_local`), and the ad-hoc mesh detector. They don't agree.
3. **Stale `object_processing/{obj}/scene/`** left around from an older pipeline; reading it (as I mistakenly did) gives wrong/mismatched scenes.
4. **My past improvisations (already reverted/being fixed):** rebuilt scenes at a fixed gap 0.02 instead of the per-hand final gap; read the shared stale scene dir; reconstructed from summary when I could just read `get_scene_dir(hand,...)`.

## 5. Open question for you
- Populate symmetry (spheres/boxes) so scene gen folds redundant z_rots — using which source? (global symmetry.json hand-curated, or auto mesh detection, or per-object symmetry.json)
- Coverage viewer: switch to reading `get_scene_dir(allegro, obj, type)` directly (simplest, matches view_bodex).
