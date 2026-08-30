# Continuous multi-object basket demo — first implementation plan

## Objective and acceptance take

One known object at a time is placed in the robot-comfortable pick region. The
robot identifies which object from a fixed catalogue, picks it, drops it into a
basket, and continues until at least 11 successes are recorded in one external,
uncut video. The operator may take a previously placed object from the basket
and place it back in another stable pose; it must be treated as a new normal
cycle, not a special reset routine.

The first implementation is deliberately constrained to a pre-onboarded
catalogue of 3–6 objects with known successful `v8` grasps. It is not a claim
that arbitrary unseen objects can be grasped.

## What the current banana path does poorly

`src/demo/banana_test/run_demo.py` is an excellent one-object safety/debug
script, but it is a poor continuous-demo loop:

| Current behavior | Why it is costly for this demo | Replacement |
| --- | --- | --- |
| One hard-coded `--obj` and object-specific FoundPose init | Cannot choose among a fixed set of objects | YOLO-E catalogue scan selects one object before FoundPose is run. |
| Per trial SAM3 + FoundPose + cross-view IoU + 100-iteration silhouette refine | The distributed-path notes measure the old full path at about 10 s after FoundPose init; silhouette refine alone is about 4.6 s | `selection_mode=quality` chooses the strongest FoundPose view and performs neither rendered IoU nor silhouette optimization. |
| Human y/n/c label prompt | Breaks the uncut take and makes retry policy manual | Lift and drop are re-observed automatically in robot coordinates. |
| Candidate failure ends a trial; final retract uses reset/home paths | A miss unnecessarily returns to the initial configuration | Measured current joints become the planner start state; the failed candidate is removed and the next one is planned over the object still on the table. |
| Fixed table-marker target | Does not express an actual container | Explicit robot-frame basket release reference, approached from above at the lifted height. |
| A new video sink per trial | Produces cut artifacts, not a single take | Use an external continuous video for the first take. ParaDex snapshots are only internal perception artifacts. |

The current distributed notes are the relevant performance baseline:

- FoundPose + SAM3 initialization is reported near 10 s, versus roughly 31–37
  s for the old centralized `PerceptionPipeline`.
- The reported GoTrack distributed design is around 0.4 s/frame for 12 camera
  centralized-equivalent work. Its live daemon integration is still documented
  as pending validation, so this first implementation does not make it a
  safety-critical dependency.

## Implemented first slice

`src/demo/continuous_basket/run_demo.py` now implements this loop:

```text
ParaDex snapshot
  -> YOLO-E catalogue agreement (>= 2 views)
  -> selected class only: distributed FoundPose constrained to the pick workspace
     (so accumulated basket contents are not treated as the next instance)
     -> quality selection, no silhouette
  -> select successful pose-compatible grasp + preflight lift/carry
     -> preflight failure: remove that candidate and try the next one in place
  -> start GoTrack from the initial FoundPose pose (once)
  -> execute lift
  -> fresh GoTrack pose
       -> still on table: remove attempted grasp, replan from measured raised pose
       -> lifted: live carry to basket, drop, retreat up
       -> ambiguous: stop raised for a manual safety check
  -> fresh GoTrack basket pose and JSON record
       -> missed basket but object is back in pick workspace: re-grasp in place
```

Three supporting pieces are intentionally independent of hardware:

- `catalog.py` collects cross-view YOLO-E evidence and prevents a
  single-camera false detection from selecting a mesh.
- `policy.py` only retries after positive evidence that the object is still at
  the original table location. It never retries on an ambiguous held-object
  observation.
- `GraspPlanner.set_start_state()` updates only the current joint seed, keeping
  the cached cuRobo world/roadmap. This removes the old home-reset dependency.
- The FR3 and XArm executors now expose the opt-in
  `execute(..., start_from_current=True)` counterpart. Existing callers retain
  the legacy init move; the new runner uses the measured-state route.

`InitOrchestrator` retains its legacy default (`selection_mode="iou"` with
silhouette refinement); the new `quality` mode is opt-in and explicitly logs
`sil_skipped: true`. Existing experiments are therefore unchanged.

The runner defaults to `--verification-mode gotrack`: normal cycles use one
FoundPose initialization plus low-latency GoTrack poses for lift/drop/retry
checks. This is the path intended to stay below the 20-second inference
budget. `--verification-mode foundpose` remains an intentionally slower
bring-up fallback if the GoTrack daemons are not available.

## Bring-up checklist

Before turning on the robot, do all of the following.

1. Pick 3–6 objects with `v8` successful grasps for the intended hand and
   create FoundPose assets plus GoTrack anchor banks for every one. Run the
   offline readiness check first; it fails before robot motion when an asset,
   successful grasp record, or anchor is absent:

   ```bash
   python src/demo/continuous_basket/preflight.py \
     --objects banana wood_organizer='wood organizer' beige_brush='beige brush' \
     --hand inspire --version v8
   ```

   Start `gotrack_daemon.py` on every capture PC before the default run; the
   runner configures it dynamically per selected catalogue object.
2. Place a shallow basket at a comfortable robot-frame center. Set
   `--basket-center X Y Z` to the release reference above its open interior;
   start conservatively with `--drop-height 0.05` and a clear vertical path.
   Confirm `--pick-workspace` excludes the basket and its contents; the runner
   uses this 3D region to disambiguate repeated object classes.
3. Test each object separately with the new runner using `--max-successes 1`.
   Check its `result.json` pose evidence, lift verification and drop check.
4. Require multi-camera detector agreement (`--catalog-min-views 2` or more),
   then run 3 continuous successes. Do not use a catalogue item whose detector
   prompt frequently matches the basket or another item.
5. Record the actual demo with one external camera. Keep the ParaDex
   controller alive throughout; it is never intentionally reset between
   successful cycles or verified empty-grasp retries.

Example command (replace the calibrated basket coordinates):

```bash
python src/demo/continuous_basket/run_demo.py \
  --objects banana=banana wood_organizer='wood organizer' beige_brush='beige brush' \
  --arm franka --hand inspire --grasp-version v8 \
  --basket-center 0.52 -0.23 0.20 --max-successes 12
```

## Next implementation gates

1. **Hardware dry run:** verify the real basket geometry. Add collision cuboids
   for the rim/walls once measured; this initial version uses a high, vertical
   release and treats the basket as a validated drop zone rather than a motion
   obstacle.
2. **Evidence calibration:** collect 20 lift/drop traces, select per-object
   pose-quality and lift-z thresholds, then record false-positive/false-
   negative rates. Current thresholds are intentionally conservative.
3. **GoTrack hand-off:** after the daemon integration tests documented in
   `docs/distributed_tracking.md` pass, replace the two post-action FoundPose
   checks with GoTrack. Keep FoundPose as relocalization when tracking quality
   falls below threshold.
4. **Unseen-object extension:** build a mesh/onboarding service and candidate
   grasp fallback separately. It is a different promise from this fixed
   `seen object` demonstration and must not silently fall back during the
   recorded take.
