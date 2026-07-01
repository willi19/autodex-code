# exp.py — Scene-coverage campaign + caching (handoff notes)

File: `src/visualization/exp.py`  (run in conda env `mingi`)
Also: `src/validation/planning/viz_graph_seed.py` (graph-seed / trajopt viz).

## Goal
For ONE object, autonomously **cover every deployment scene** (`scene/{shelf,wall,box}/*.json`)
by executing grasps, using **reset (reorient) + reposition** to move the object between
the placements/poses each grasp needs. End when every scene is covered. It's an
**offline plan + viser** tool (no robot), to verify the whole sequence is executable.

A scene is "covered" when a grasp that is **collision-free with that scene** (coverage)
is **actually graspable (IK + trajopt succeeds)** at a placement we can transport the
object to. Selection rule (user-defined "best"): among feasible grasps at the current
pose, pick the one covering the **most still-uncovered scenes** (greedy set-cover).

## Run
```
conda activate mingi
python src/visualization/exp.py --obj <obj> --hand inspire_left --port 8080
```
- **Plan path** button: single grasp — reset route (BFS) INIT→grasp + draw target ghost. ~15s.
- **Cover all scenes** button: full campaign. No cache → BUILD (~6–13 min) + save cache;
  cache present → **instant replay** (no planner/GPU).
- **rebuild cache** checkbox: force rebuild.
- Sliders/dropdowns: object, hand, scene type/id, grasp, start pose, initial x/θ (Plan path).

Only `inspire_left` has reset cells. Objects = those with BOTH reset cells and v7 grasps
(`objects_for_hand`): 12 objs (attached_container, pepsi, donut, ...).

## ===== CACHE LOGIC (what you asked) =====
Three cached artifacts, all under `{repo_dir}/order/{hand}/v7/{obj}/`
(`repo_dir` = `~/AutoDex`; NOTE candidates live under `project_dir`=`~/shared_data/AutoDex`):

1. **`coverage_current.npy` + `coverage_current.json`** — collision coverage.
   - `valid_array` (S scenes × G grasps) bool: grasp g collision-free with scene s.
   - `scene_meta`: list of `(scene_type, scene_id, pose_idx)` per row.
   - Built in `compute_coverage_data`: over CURRENT_SCENE_TYPES = ("shelf","wall","box")
     ONLY (NOT `_prev`/reorient/float/table). Uses experiment `load_grasp_data` +
     `planner._check_collision`. Cached; reused if present.

2. **`reach_current.npy`** — reachability `reach[g, xi, yi]` bool: grasp g IK-reachable at
   placement (X_GRID[xi], YAW_GRID[yi]) for g's pose. Built in `_compute_reachability`
   (one batch-IK sweep over all placements). Cached. Used so placement selection is a
   **lookup** (no live IK) via `rank_placements`.

3. **`campaign_cache.pkl`** — the WHOLE successful campaign trajectory.
   - `pickle` dict: `{"timeline": [(name, robot_traj(T,ndof), obj_traj(T,4,4)), ...],
     "init_T": 4x4 object start pose, "status": readout text}`.
   - **Save**: end of `_run_campaign` after a build.
   - **Replay**: top of `_run_campaign` — if exists and not `req["rebuild"]`: load,
     `_ensure_scene` (mesh+urdf only, **no planner warmup / no GPU**), `clear_traj`,
     set object transform=`init_T`, `add_traj` each phase, play. Return. → instant.

Rationale: trajopt/reset are flaky+slow (stochastic seeds). Do the flaky work ONCE
offline (with retries), store only successful trajectories, replay deterministically.

## Campaign build flow (`App._run_campaign`)
```
data = compute_coverage_data(...)        # valid_array, scene_meta, grasp_meta, wrist_obj, pregrasp, reach
for pose p in sorted(poses of scenes):
  (a) x_ranked = rank_placements(reach, grasps@p)        # grasp-good x, yaw=0, lookup
      reset object current_pose -> p at a grasp-good x:
        plan_transition(x_grid=x_ranked, yaw=[0]); fallback full grid; RETRY x4.
        if still fails -> REPOSITION-FOR-RESET: move within current pose to another x
           (plan_reposition) then retry the reset. (reset robustness — was the main flaky point)
  (b) plan_grasps_at_placement(...)      # batch-IK all grasps@p once; greedily _refine_fingers
      the reachable ones covering most-uncovered scenes (fallback to next on trajopt fail)
  (c) REPOSITION loop for leftover scenes: iterate candidate placements; transport object
      there (plan_reposition), then actually try grasps; first placement that covers new
      scenes wins. ("move so the grasp solves")
build timeline (reset/grasp/reposition phases) -> save campaign_cache.pkl
```
- **reset** = pose change via reorient candidates (`reset/reorient_{h_cm}/{i}_{j}/`),
  full chain approach→grasp→lift→reorient→descent→release→retract.
- **reposition** = same pose, different x: pick (any graspable transport grasp at current
  placement) → cartesian carry up/over/down → place (`plan_reposition`,`cartesian_object_path`).
- **lift/descent** = STRAIGHT world-z cartesian (`cartesian_move_z`), mirrors executor
  `real.py:_move_cartesian`. (NOT plan_obj_placement which curves.)

## Key functions
- `compute_coverage_data(planner,obj,hand)` -> valid_array/reach/meta/wrist_obj/pregrasp (cached).
- `rank_placements(reach,cand,restrict,yaw_idxs=[0])` -> placements ranked by #reachable grasps.
- `plan_transition(...)` -> one reset (pose i->j), reuses reorient seeds + ik_check_seeds + cartesian lift/descent.
- `plan_grasps_at_placement(...)` -> cover scenes at a fixed placement (batch IK + refine selected).
- `plan_reposition(...)` -> same-pose transport (pick+carry+place), tries multiple transport grasps.
- `cartesian_move_z` / `cartesian_object_path` -> straight cartesian, IK-per-waypoint + `_unwrap_arm`.

## Results (verified)
- **attached_container**: 105 scenes (shelf75+wall25+box5), 5 poses → **100%**, 4 resets, ~366s build.
- **pepsi**: 33 scenes (shelf+wall, NO box), 3 poses → **100%**, ~800s build.
- Replay: **0.0s, no GPU**.

## Known issues / TODO (next agent)
1. **Cartesian IK discontinuities (PARTIALLY fixed)**: per-waypoint IK in
   `cartesian_move_z`/`cartesian_object_path` can return a different branch than the
   previous waypoint → visual jump in lift/descent/carry, even though the WRIST is
   smooth (object barely moves).
   - **2π wraps FIXED** via `_unwrap_arm` (unwrap all 6 arm joints to nearest prev
     within limits). e.g. pepsi reset lift frame ~924 was 6.27rad(=2π), now ~0.02.
   - **STILL REMAINS: ~π branch flips** (elbow up/down, symmetric-gripper roll). NOT a
     2π wrap so `_unwrap_arm` misses it. e.g. after pepsi rebuild, `p2_g107_lift` still
     has a 3.141rad(=π) frame jump. FIX OPTIONS for next agent:
       (a) reject a per-waypoint IK solution whose joint delta from prev > threshold
           (e.g. >0.5rad), and re-solve (more seeds) or interpolate from prev;
       (b) stronger retract bias toward prev in the IK (keep on-branch);
       (c) for the lift specifically, since it's a tiny straight translation, seed IK
           from prev hard / accept only near-prev solutions.
   - **ACTION**: after fixing, rebuild caches (`attached_container` cache is STALE = built
     before any fix; pepsi rebuilt but still has the π jump). rebuild-cache checkbox or
     delete `campaign_cache.pkl`.
2. **Determinism + speed (REQUESTED, not done)**: trajopt/IK are stochastic (random seeds).
   - Make deterministic: `torch.manual_seed/np.random.seed` before each plan; for retries
     iterate seeds 1,2,3… (deterministic) instead of relying on randomness.
   - Speed up `cartesian_move_z`/`cartesian_object_path`: currently solve IK **per waypoint**
     (40 calls, each tiles ONE pose to BATCH_SIZE = wasteful). Batch ALL waypoints into
     ceil(n/BS) solve_batch calls (per-row retract = linear-interp seed), then `_unwrap_arm`.
     → big speedup for lift-up etc.
   - cuRobo knobs: `use_cuda_graph=True`, `num_trajopt_seeds`/`grad_trajopt_iters` (speed↔success trade-off).
3. **Held-object collision NOT considered during reorient** (`scene_lift` strips object mesh,
   object never `attach_objects_to_robot`'d). cuRobo sphere-fit attach is poor, so it was
   skipped on purpose. Proper fix = **post-hoc mesh collision check** of the held object
   mesh along the trajectory vs table/scene cuboids, reject colliding plans. BLOCKED: no
   `python-fcl` in env → use vertex-vs-oriented-cuboid test (or install fcl).
4. **reposition transport grasp source is WRONG (design fix)**: reposition is a pick-and-
   place on an OPEN TABLE, so the transport grasp should come from **tabletop grasps**
   `candidates/{hand}/table_only/{obj}/table/{sid}/` (sid -> pose via
   `scene/table/{sid}.json` meta.pose_idx; for attached_container the PADDED sids
   `000/001/006/009/016` map to poses 0/1/6/9/16, ignore the unpadded duplicates).
   CURRENT CODE wrongly reuses the **v7** (shelf/wall/box) grasps as transport
   (`plan_reposition`/`plan_grasps_at_placement` pull from `compute_coverage_data`'s v7
   pool). FIX: load table_only grasps separately (own wrist_obj/pregrasp/grasp arrays,
   per pose) and use THOSE for reposition + reposition-for-reset transport. (Coverage
   testing still uses v7; only the *transport* grasp should be tabletop.) table grasps
   are more numerous/reliable (no shelf/wall constraint) -> repositions succeed more.
5. **reposition yaw**: currently yaw=0 only (resets to non-zero yaw are unreliable). Could
   open yaw for repositions to recover more scenes.
6. **box has no grasp candidates** for some objects (v7 = shelf/wall for many); auto-skipped.

## cuRobo facts learned (context)
- Plan = graph search (PRM, collision-free coarse path = trajopt SEED) → trajopt (optimize
  full 64-step traj jointly, MPPI+L-BFGS, many seeds). `MotionGenResult.graph_plan` is
  (1, 60, 12) = (graph_seed, steps, dof) — squeeze batch dim 0.
- Timing: graph ~0.05–0.14s, trajopt ~1.3–2.3s (trajopt dominates → precomputing graph
  seed saves little; caching the TRAJOPT RESULT is what helps → campaign_cache).
- Failures: `GRAPH_FAIL` (no coarse path), `TRAJOPT_FAIL` (opt didn't converge, seed-dependent
  → retry helps), `INVALID_START_STATE_WORLD_COLLISION` (goal/start in collision).
- reorient harder than approach: held object enlarges collision body + 6-DOF goal constraint
  + near joint limits, vs approach = empty hand, free space, single goal.

## Data paths
- candidates: `~/shared_data/AutoDex/candidates/{hand}/{v7|reset}/{obj}/...`
- scenes: `~/shared_data/AutoDex/object/paradex/{obj}/scene/{type}/{id}.json` (meta.pose_idx)
- tabletop poses: `{obj}/processed_data/info/tabletop/*.npy`
- order/caches: `~/AutoDex/order/{hand}/v7/{obj}/`
- experiment set-cover ref: `src/grasp_generation/order/compute_order.py` (load_grasp_data, setcover_order)
- executor ref (straight lift/descent): `autodex/executor/real.py:_move_cartesian` (592, 626)

---

## 2026-06-21 — seed-cache integration plan (for "plan the entire exp" reliably)

Context: the (B)-1 seed cache (`seed_cache.py`) now produces CLEAN, reliable
INIT->grasp seeds (Cartesian-clean, deterministic reachability — see
`HANDOFF_seed_cache.md`). The campaign (`_run_campaign`) is still stochastic: reset
(`plan_transition`) retries 4× + reposition fallback, and grasp planning re-plans
from scratch each run. Integrating the cache should make the campaign reliable + clean.

DONE:
- `_compute_reachability` now `torch.manual_seed(0)` -> deterministic/predictable
  resettable area (the reach map is reproducible run-to-run).

PROPOSED (not yet done — needs care, it's the main tool):
1. **Reachability source**: the seed cache's "cell has a seed" already == IK-reachable
   (and a clean seed exists). Could replace/cross-check `reach[g,xi,yi]` with the seed
   cache so the campaign only places where a CLEAN grasp seed exists, not just IK.
2. **Grasp planning in `plan_grasps_at_placement`**: instead of `_refine_fingers` from
   scratch (stochastic, ugly arcs), load the cached seed for (grasp, nearest cell) and
   `plan_with_seed` (adjust to the actual placement). Fast (~50ms) + clean + reliable.
3. **Reset (`plan_transition`)**: the approach phase (INIT->reset-grasp) can use the
   cached seed the same way. The lift/reorient/descent stay cartesian/plan_obj_placement.
4. **Determinism**: `torch.manual_seed` before each stochastic plan so a cached campaign
   reproduces. Campaign already caches the full timeline to `campaign_cache.pkl`.

WHY this matters (user's two requirements):
- (1) reachable-but-traj-fail is the campaign's worst annoyance (place object, then
  "can't grasp"). The seed cache has 0% of that on-grid -> use it as the gate.
- (2) no z+/sideways/rotation waste: cached seeds are Cartesian-clean by construction.

RISK: `_run_campaign` is GUI-driven and stochastic; do NOT rewrite it blindly. Change
`plan_grasps_at_placement` first (smallest, highest value), verify on one object, then
extend to reset.
