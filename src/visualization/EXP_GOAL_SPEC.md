# exp.py — GOAL & SPECIFICATION (what it must do)

This is the **requirements spec** (the "what" and "why"). For the current
implementation details / cache file formats / known bugs, see
`EXP_CAMPAIGN_NOTES.md` next to this file.

---

## 1. Motivation / problem
On the real robot we grasp one object across many **scenes** (the object sitting
in a shelf, against a wall, in a box, etc.). To grasp it in the next scene the
object must first be moved into the **pose and position that next grasp needs**.
In past experiments this "**reset & reposition for the next grasp**" step failed,
so we could not chain grasps across scenes.

We want a tool that, **for one object, plans and verifies the entire sequence of
"move the object → grasp it" that covers every scene**, end to end, offline, and
caches the successful trajectories so it can be replayed instantly.

## 2. Final goal (one sentence)
> For ONE object, **autonomously produce and verify a single continuous motion
> sequence that covers every deployment scene** — using **reset (reorient)** and
> **reposition (move on the table)** between grasps to put the object where each
> grasp can actually be executed — and **cache** the result for instant replay.

"Covered" must hold for **every** scene at the end (target = 100%), or the tool
must clearly report which scenes are uncoverable and why.

## 3. Definitions
- **object**: one rigid object (e.g. `attached_container`, `pepsi`). Has a set of
  stable **tabletop poses** (orientations it can rest in on a flat table),
  enumerated by `processed_data/info/tabletop/*.npy` (pose id = filename int).
- **scene**: one deployment configuration `scene/{type}/{id}.json`, where
  `type ∈ {shelf, wall, box}` (current deployment types). Each scene contains the
  object mesh pose + obstacle cuboids, and `meta.pose_idx` = the tabletop pose the
  object rests in for that scene. Many scenes share one pose.
- **grasp candidate**: a pre-generated grasp (BODex) = `wrist_se3` (object frame),
  `pregrasp_pose`, `grasp_pose` (finger configs). We do NOT generate/optimize new
  grasps; we only select among existing candidates.
- **placement**: where the object rests in the world for a given pose — its (x, y,
  yaw) on the table. (y fixed = 0 in current setup; x and yaw are the knobs.)
- **reset (reorient)**: change the object's **tabletop pose** (orientation), by
  grasping it, lifting, rotating mid-air, and placing it back in the new pose.
- **reposition**: keep the same pose, **change the placement (x / yaw)** by a
  pick-and-place on the table — used so a grasp becomes physically executable.
- **covered**: a scene S is covered when there EXISTS a grasp G such that
  (a) **coverage**: G's hand is collision-free with S's obstacles (object in S's
      pose), AND
  (b) **graspable**: the robot can actually execute G (IK + collision-free smooth
      trajectory) at a placement we can transport the object to.

## 4. Inputs & data (paths)
- candidates root: `~/shared_data/AutoDex/candidates/{hand}/...`
  - **coverage / deployment grasps**: `v7/{obj}/{shelf|wall|box}/{scene_id}/{grasp}/`
  - **reset/reorient grasps**: `reset/{obj}/reorient_{h_cm}/{i}_{j}/{grasp}/`
    (pick at pose i → place at pose j; collision-checked at generation vs
    table_i / pillars / table_j). `h_cm` = lift height cm; use smallest available.
  - **tabletop grasps**: `table_only/{obj}/table/{scene_id}/{grasp}/`
    (object on a plain table at a tabletop pose; no shelf/wall constraints).
- scenes: `~/shared_data/AutoDex/object/paradex/{obj}/scene/{type}/{id}.json`
- tabletop poses: `{obj}/processed_data/info/tabletop/*.npy` (4x4 robot frame)
- mesh: `{obj}/raw_mesh/{obj}.obj`
- planner: `autodex.planner.GraspPlanner` (cuRobo). hand currently `inspire_left`
  (only hand with reset cells). object set = objects having BOTH reset cells and
  v7 grasps.

## 5. Required behavior

### 5.1 Which grasp set is used WHERE (important — keep these separate)
| purpose | grasp source |
|---|---|
| decide coverage / what to TEST in each scene | **v7** (`v7/{obj}/{type}/{id}`) |
| transport for **reset** (pose change) | **reorient** (`reset/.../{i}_{j}`) |
| transport for **reposition** (same-pose move) | **tabletop** (`table_only/{obj}/table/{pose}`) |

Rationale: reposition is a plain pick-and-place on an open table, so its transport
grasp should be a tabletop grasp (more candidates, no shelf/wall constraint),
**not** a v7 shelf/wall grasp.

### 5.2 Grasp selection rule ("best")
At a given object pose, among the grasps that are **feasible right now**
(IK-reachable + their motion plans), pick the one that **covers the most
still-uncovered scenes**. (Greedy set-cover by remaining coverage. We are NOT
optimizing a grasp per scene; we pick the best existing candidate.)

### 5.3 Coverage determination
A grasp covers scene S iff its hand is **collision-free with S's obstacles** when
the object is in S's pose. This is a geometric check (one-shot, against the scene
cuboids), precomputed into a `valid_array` (scene × grasp). NOTE: the scene
obstacles are used ONLY for this coverage check — the **execution/motion planning
is done in a table-only world** (we do NOT impose the shelf/wall during the arm
motion; the scene was a generation-time constraint, not a runtime obstacle to
re-plan against).

### 5.4 The campaign (end-to-end loop)
Goal: cover all scenes with minimal **reset/reposition** moves.
```
covered = {}                                  # scene -> grasp that covered it
object starts at some pose/placement
group scenes by required pose
for each pose p (visited in an order that needs few resets):
    # get the object into pose p
    RESET object current_pose -> p  (reorient grasp).  RESET is stochastic, so:
        retry; and if it keeps failing, REPOSITION within the current pose to a
        placement from which the reset DOES plan, then retry (reposition-for-reset).
    # cover scenes at pose p
    repeat:
        pick the best feasible grasp (5.2) and execute it (approach->grasp->lift);
        mark the scenes it covers (5.3) as covered.
        for scenes still uncovered at pose p whose grasp won't plan HERE:
            REPOSITION (same pose, new x/yaw) to a placement where that grasp
            actually PLANS (try-plan at candidate placements until it solves),
            then execute. ("move so the grasp solves" — not just IK-reachable.)
    until pose p's scenes are covered or no progress is possible.
stop when every scene covered (or report uncoverable scenes + reason).
```
Minimize the number of resets (≈ number of distinct poses) and repositions.

### 5.5 Motion correctness requirements
- **Straight lift/descent**: lifting/lowering the held object must be a **straight
  vertical (world-z) cartesian** motion (mirror the real executor
  `autodex/executor/real.py:_move_cartesian`), NOT a joint-space plan that curves.
- **Continuity**: the concatenated trajectory must be continuous — no joint jumps
  at phase boundaries or within cartesian segments (per-waypoint IK can flip
  branches / wrap a joint by 2π or π; must enforce continuity).
- **Held-object collision (SHOULD)**: while the object is held (lift/reorient/
  descent/carry), its collision with the table/environment SHOULD be considered.
  cuRobo's sphere-fit attach is too crude, so prefer a **post-hoc mesh collision
  check** of the held object along the trajectory and reject colliding plans.
  (Currently NOT done — see NOTES.)

### 5.6 Determinism & speed (desired)
- Planning (trajopt/IK) is stochastic; runs should be **reproducible** (fix random
  seed; for retries, iterate seeds deterministically).
- Should be reasonably **fast**, especially the cartesian lift/descent (batch the
  per-waypoint IK instead of one solve per waypoint).

## 6. Output
1. **viser visualization**: the entire `reset → grasp → reposition → grasp → …`
   sequence as one playable timeline (robot + object move together), so a human
   can confirm it executes from beginning to end. Show a coverage readout
   (covered N / total, #resets, #repositions, per-pose breakdown, any uncovered).
2. **cache**: the successful trajectories saved to disk so a later run **replays
   instantly with no planning / no GPU**. Build once (offline, with retries),
   replay deterministically. Provide a "rebuild" option.

## 7. Success criteria
- For a given object: **every scene covered (100%)** by an executable
  reset/reposition/grasp sequence, OR a clear report of which scenes are
  uncoverable and why.
- The full sequence is **continuous and physically plausible** (straight
  lift/descent, no joint jumps, held object doesn't pass through the table).
- A second run **replays from cache instantly**.

## 8. Non-goals / assumptions
- Not running on the real robot here — offline planning + visualization to VERIFY
  feasibility (it should mirror what the real executor would do).
- Not generating/optimizing new grasps — only selecting among existing candidates.
- Not imposing shelf/wall obstacles during the arm motion planning (only for the
  coverage collision check).
- One object at a time; one hand (`inspire_left`).

## 9. What "done" looks like
Pressing **Cover all scenes** for an object: builds (first time) a 100%-coverage
sequence with reset (reorient grasps) + reposition (tabletop grasps) + grasps
(v7), straight and continuous motions, held-object collisions respected; caches
it; and thereafter replays it instantly in viser. The number of resets/repositions
is small, and the readout shows full coverage.
