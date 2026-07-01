# Handoff — (B)-1 seed cache + wrist z-rise + lift

Written for the next agent. Read this fully before touching `src/visualization/seed_*`.

## RESOLVED (2026-06-21) — read this first, it supersedes the z-rise section below

The z-rise / "stupid movement" is solved. Current `seed_cache.py`:
- `build_one` returns ALL candidates (base-angle sweep {0,±30,±60,±90}: rotate object
  by θ → re-IK grasp → plan → rotate seed j0 back by -θ). This == "change the init base
  angle so the hand approaches easier" (object-rotate(-θ) ≡ start-j0(+θ)).
- **THE FIX**: the skip short-circuit uses `_cart` (Cartesian wrist-path waste), NOT
  `_redundancy`. `_redundancy`=0 (monotonic joints) can still loop the wrist 2.2 m /
  341°, so redundancy-skip stranded the worst cells at 1 bad candidate. Skip→`_cart`:
  z-rise>15cm 12→1, pos-excess max 2.23→0.66 m, cands 2.6→4.8/cell.
- Metric to SELECT by: `_cart` (catches lift+sideways+far+rotation). `_redundancy` is
  meaningless; `_zrise` only sees height.
- Candidates saved to `*_cands.pkl`; re-pick FREE with `--reselect cart|zrise|red|cost`
  or `select()`. NEVER rebuild just to change the metric.
- Cache cell has-seed ⇔ IK-reachable (0 IK-ok-but-traj-fail at build) = deterministic
  reachability gate.
- `outputs/seed_sweep/cache_view.pkl` + `seed_replay_viz.py` show it; filters
  worst-zrise / worst-redund, status shows z-rise + redundancy.

## What this is

Mechanism (B)-1: precompute one `INIT -> grasp` trajectory per grid cell
`(grasp gi, radius xi, yaw yi)`, freeze to disk. At runtime, adjust the cached
trajectory to the actual off-grid object pose and feed it to trajopt as a SEED
(`plan_with_seed`). Goal = kill stochastic trajopt failures + be fast (~50ms vs
~1s scratch).

- Runtime strategy (DONE, works): **joint -> task fallback** = `adjust_and_plan`
  in `seed_plan_viz.py`. joint adjust first (cheap); if its plan fails, task adjust.
  Union success ≈ **95.5%** on pepsi/pose2.
- `plan_with_seed` (planner.py) preserves the seed SHAPE almost exactly
  (z-rise corr seed<->out ≈ 1.0). So whatever the cached seed looks like, the
  runtime plan looks like that.

## Files

- `seed_cache.py` — builds + freezes the cache. **`build_one` is the thing in flux**
  (see below). `build_all` sets the motion_gen world once per cell.
- `seed_sweep.py` — headless off-grid success sweep. Flags: `--task`, `--lift`,
  `--max_grasps N` (fast subset), `--save_results PATH`.
- `seed_plan_viz.py` — `adjust_seed` (joint), `adjust_seed_task` (task, SE3-ramp +
  no per-waypoint fallback), `adjust_and_plan` (the joint->task strategy),
  `plan_lift` (experiment-side).
- `seed_replay_viz.py` — replay viz. Filters: fails / success / worst-zrise /
  j0-benefit. Has a frame scrubber + per-joint sliders + an OLD-cache ghost overlay.
- `autodex/planner/planner.py` — `plan_with_seed`, `plan_lift` (production),
  `_update_world` / `_update_target_pose_only` (the latter calls
  `world_coll_checker.update_obstacle_pose` — it DOES move the obstacle in cuRobo).
- `autodex/executor/real.py` — `execute_lift(lift_traj, hold_hand)` runs a planned
  qpos lift via `_move_joints` (NOT cartesian).

## THE UNRESOLVED THING: wrist z-rise ("stupid movement")

The approach sometimes arcs the wrist (end-effector) UP to ~45 cm then back down
to the grasp, instead of going straight in. Measured as wrist Cartesian z-rise via
FK = `ee_z.max() - max(ee_z[0], ee_z[-1])`. **Use FK z-rise, NOT per-joint hump**
(a 17 cm wrist rise can show joint-1/2 hump = 0; it comes from the joint combo).

Honest findings (this took hours and went in circles — don't repeat it):
- The arc comes from the BUILD-TIME planner (graph search) and is **heavily
  stochastic**: the SAME goal plans to 1–45 cm depending on process state
  (deterministic within one process, varies across runs). `torch.manual_seed`
  barely changes it.
- The arc is **NOT** caused by object/table collision. Object sits on a flat table
  either way; moving it does not change object-table contact.
- Fix = **generate several candidate plans per cell and keep the flattest.** What
  matters is how DIVERSE the candidates are.

What was tried (z-rise mean / outliers >15cm / coverage, pepsi pose2):
- OLD (no sweep, default IK + 1 plan): mean 2.1, 16 outliers, 419 cells.
- **OBJECT-ROTATION sweep (BEST)**: for θ in {0,±30,±60}: rotate object by θ about
  base-z, re-IK the grasp, plan INIT->that goal, then rotate the seed's j0 back by
  -θ; keep flattest. -> mean **1.2, 0 outliers, max 14.4, 434 cells.** Clean win.
- retract-vary (vary IK retract pose, 7 goals): mean 3.1, ~10 outliers, ~210. Weak
  (candidates too similar -> plans correlated -> all arc together).
- start-only-j0 (rotate START j0 only, keep goal): same as retract-vary, weak.
- flat straight-interp seed: fixes extremes but RAISES the median; rejected.
- IK `return_seeds` branch sampling: this grasp has only ~2 IK branches; not enough.

USER'S ACTUAL IDEA (record it correctly): change the INIT / START config (e.g. its
j0) so the hand approaches the grasp more easily — start only, goal unchanged. This
WAS tested = "method 2 / start-j0-vary": for off in {-60..60} plan (INIT with j0+off)
-> same goal, keep flattest. Result: covered 214/300, mean 3.1, max 45.9, **13
outliers remain**. It helps most cells but cannot escape a goal config that forces an
arc (only the start changed, the goal didn't). So it is OKAY but leaves outliers.

Why object-rotation beats it: object-rotation re-solves the IK at each angle, so it
gets DIFFERENT GOAL configs (5 of them) -> can escape an arced goal. The diversity
that removes outliers comes from varying the GOAL, which start-only-j0 does not do.
(The object/table collision is NOT involved; object on a flat table.)

### RECOMMENDATION
Use the object-rotation method (= full-problem j0 rotation = rotate-IK-target-pose).
It gave 0 outliers. It is a BUILD-TIME trick to get diverse goal candidates; nothing
moves at runtime. Implement `build_one` as: for θ in {0,±30,±60}: `Tw = Rz(θ)@Tw0`;
`q = ik(Tw)`; `ok,tr = _refine_fingers(INIT, q)`; `z = zrise(tr)`; keep flattest;
then `tr[:,0] -= θ` (rotate seed back to azimuth 0). Short-circuit θ=0 if already
flat (<=10cm). Rebuild + sweep to confirm (expect ~0 outliers, ~83% runtime success
unchanged — z-rise is cosmetic, success is handled by the joint->task fallback).

NOTE: the cache file `order/inspire_left/v7/pepsi/seed_cache/pose2_y12.pkl` is
currently from a WEAK build (branch/retract experiments). `.arcbak` = OLD. Rebuild
with the recommended method. `build_one` in the repo right now = retract-vary (weak).

## Lift (the OTHER real task, less explored)

Problem: cartesian-servo lift (`_move_cartesian`) throws UNRECOVERABLE xarm
kinematic errors -> forces a full program/robot/camera restart. Fix = give the robot
a PLANNED qpos lift.
- DONE: `GraspPlanner.plan_lift(grasp_qpos, grasp_wrist_world, scene_lift, lift_h)`
  -> trajopt grasp->(wrist +z 10cm), held object stripped from world, returns
  (ok, traj). ok=False (joint-limit unreachable) -> caller skips, no crash.
- DONE: `RealExecutor.execute_lift(lift_traj, hold_hand)` -> joint-space lift.
- PENDING (user's spec): wire lift into `plan()` candidate acceptance — a grasp whose
  lift can't plan is dropped, move to next candidate. Then `run_auto.py`:
  `execute(skip_lift=True)` -> `plan_lift` -> `execute_lift`.
- +z lift moves the object UP/away -> NO table collision; table-scene lift failures
  are KINEMATIC (lifted config near a joint limit), not collision.

## Gotchas
- `/tmp` is wiped on reboot. Save results to `outputs/` (e.g.
  `outputs/seed_sweep/sweep_p2.pkl`).
- viser robot `change_color` needs **0-255** ints (objects need 0-1).
- IK `solve_batch` uses random seeds internally but returns the single BEST, which
  is consistent per (pose, retract). Diversity must come from varying the input
  POSE, not from IK randomness. Do NOT regularize toward INIT when you want diverse
  branches (it collapses them).
- conda env: `mingi`. Run with `PYTHONPATH=/home/mingi/AutoDex`.
