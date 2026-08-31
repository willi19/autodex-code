# Franka + Inspire: unseen-catalogue → continuous basket demo

The continuous demo can select an object that is *unseen in prior real Franka
trials*, but it is not mesh-free open-set grasping. Before a new object may
move the FR3, it must have a scan mesh, 6D pose assets, simulated candidates,
and at least one physical FR3+Inspire success record. This is intentional:
the demo's runtime is fast because it does not generate grasps, silhouette
match, or explore the object during the uncut take.

The first practical catalogue is:

```text
apple  banana  pepsi  toothbrush_holder
```

On 2026-08-31, all four have object-processing, FoundPose, GoTrack, and v8
candidate assets. `banana` has seven Franka+Inspire successes; the other three
need real-robot collection. This gives a small 4-class first demo without
waiting for arbitrary-object scanning.

## 1. Audit before any robot connection

```bash
/home/robot/anaconda3/envs/planner/bin/python \
  src/demo/continuous_basket/prepare_franka_catalog.py \
  --objects apple banana pepsi toothbrush_holder --stage audit
```

`FR3 ok` must be nonzero for every object before the continuous runner is
allowed to move. The audit also checks new-object scan products:

- `raw_mesh/{object}.obj` — FoundPose mesh;
- `processed_data/mesh/simplified.obj` and `urdf/coacd.urdf` — BODex/planning;
- `processed_data/info/simplified.json` and `tabletop/*.npy` — stable-pose
  table candidate generation.

For an entirely new scan, put those products under
`~/shared_data/object_processing/{object}/` first. The tool deliberately
stops there if any are missing; do not substitute a differently framed legacy
mesh from `object/paradex` for v8.

## 2. Onboard perception and tracking assets

This is GPU-only work, with no robot or camera command. Review commands first
by omitting `--execute`, then run:

```bash
/home/robot/anaconda3/envs/planner/bin/python \
  src/demo/continuous_basket/prepare_franka_catalog.py \
  --objects new_object --stage assets --execute --onboard-workers 1
```

It generates the FoundPose representation at
`~/shared_data/AutoDex/foundpose_assets/{object}/` and the 256-anchor GoTrack
bank at `autodex/perception/thirdparty/MV-GoTrack/anchor_banks/{object}.npz`.
One worker is the conservative first setting; FoundPose onboarding is normally
the longest offline stage.

## 3. Generate table-only simulated candidates

```bash
/home/robot/anaconda3/envs/planner/bin/python \
  src/demo/continuous_basket/prepare_franka_catalog.py \
  --objects new_object --stage candidates --execute --bodex-parallel 4 --seed-num 200
```

This writes table scenes using the same `object_processing` paths as v8 and
uses the new `sim_inspire/paradex_table.yml`. Passing candidates go directly to
`~/shared_data/AutoDex/candidates/inspire/v8/{object}/`; there is no manual
copy from the source checkout. Existing scene files are never replaced unless
`--overwrite-scenes` is explicitly supplied.

The simulation models the Inspire hand, so it only supplies safe starting
candidates. FR3 reachability and real success are established next.

## 4. Collect Franka successes, one object at a time

First print the commands:

```bash
/home/robot/anaconda3/envs/planner/bin/python \
  src/demo/continuous_basket/prepare_franka_catalog.py \
  --objects apple pepsi toothbrush_holder --stage collect --auto-label
```

For each printed command, stage only that object at a comfortable table
location and run it under supervision. The essential command shape is:

```bash
/home/robot/anaconda3/envs/planner/bin/python src/execution/run_auto.py \
  --obj apple --grasp_version v8 --hand inspire --arm franka --scene table \
  --candidate-scene-type table --max_trials 8 --auto
```

`--candidate-scene-type table` is important: it selects the newly generated
table candidates without requiring a coverage JSON, while still retaining
skip-done behavior. Each completed real trial writes its result into that
candidate's `result.json` with `"arm": "franka"`; this is exactly what the
continuous-demo preflight reads. `--auto` uses the Charuco lift check; omit it
when that board is not configured and label the collection trial manually.

Collect at least one success for each object, then repeat successful objects
in a second stable tabletop pose. The latter is what supports the visible
"put it back in a different pose" part of the demo. Collection may home/reset
between trials; only the final continuous take avoids resets.

Run the audit again after each session. Do not proceed while it reports
`successful_grasp` missing.

## 5. Start the continuous take

Keep the GoTrack daemons running on capture1/2/3/5/6 and attach one
standalone (non-Charuco) `6X6_1000` ArUco marker to the basket. The runner
triangulates it before it connects to the arm. `--basket-marker-offset` is in
the marker frame and points from that marker centre to the safe release point
over the basket interior. First run a one-success check for each class, then
the uncut sequence:

```bash
/home/robot/anaconda3/envs/planner/bin/python \
  src/demo/continuous_basket/run_demo.py \
  --objects apple banana pepsi toothbrush_holder \
  --hand inspire --arm franka --grasp-version v8 \
  --basket-marker-id 42 --basket-marker-offset 0 0 0.08 \
  --max-successes 12 --max-cycles 40
```

Set `--pick-workspace` so it includes the placement region but excludes the
basket. Pass `--basket-center X Y Z` instead only when using an independently
measured manual reference. The runner uses multi-prompt YOLO-E only to choose among these fixed
classes; it then runs FoundPose once for that selected class, skips silhouette
matching, tracks with GoTrack, replans retries from the current raised state,
and records automatic lift/drop verification. It does not home-reset during a
normal retry.
