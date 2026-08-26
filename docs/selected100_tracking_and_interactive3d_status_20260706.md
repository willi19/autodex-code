# AutoDex Selected 100 Object Tracking and Interactive 3D Status

Date: 2026-07-06

This note summarizes the current state of the `selected_100` object tracking run and the later interactive 3D GLB export pass.

## Scope

- Experiment root:
  `/home/robot/shared_data/AutoDex/experiment/selected_100`
- Object tracking schedule:
  `/home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_selected100_all_20260701_164328`
- Overlay video root:
  `/home/robot/shared_data/AutoDex/object_overlay_video`
- Interactive 3D output root:
  `/home/robot/shared_data/AutoDex/interactive_3d`

## Object Tracking Status

The object tracking schedule is almost complete, but it is not 100% complete.

| Item | Count |
|---|---:|
| Scheduler tasks | 1,141 |
| Done in this schedule | 844 |
| Skipped because already done | 276 |
| Failed | 20 |
| Still pending | 1 |
| Tasks with GoTrack + overlay outputs complete | 1,120 |

Current output counts:

| Output | Count |
|---|---:|
| `world_pose_records.json` files | 1,120 |
| `overlay_*.mp4` files | 26,969 |

The 1,120 completed episodes are usable for downstream object-pose based processing. They include tasks newly processed in the scheduler and tasks that were detected as already complete from earlier runs.

## Remaining Object Tracking Work

There are 21 incomplete scheduler tasks.

| Type | Count | Pattern |
|---|---:|---|
| Failed | 20 | all `default/wood_tray_small` |
| Pending | 1 | `default/allegro/brown_ramen/20260327_000807` |

The failed tasks are:

- `allegro/wood_tray_small/20260405_041521`
- `allegro/wood_tray_small/20260405_042529`
- `allegro/wood_tray_small/20260405_042651`
- `allegro/wood_tray_small/20260405_042805`
- `allegro/wood_tray_small/20260405_042911`
- `allegro/wood_tray_small/20260405_043023`
- `allegro/wood_tray_small/20260405_043150`
- `allegro/wood_tray_small/20260405_043320`
- `allegro/wood_tray_small/20260405_043439`
- `allegro/wood_tray_small/20260405_043550`
- `allegro/wood_tray_small/20260405_043810`
- `allegro/wood_tray_small/20260405_043917`
- `allegro/wood_tray_small/20260405_044021`
- `inspire/wood_tray_small/20260406_050653`
- `inspire/wood_tray_small/20260406_050831`
- `inspire/wood_tray_small/20260406_050940`
- `inspire/wood_tray_small/20260406_051043`
- `inspire/wood_tray_small/20260406_051217`
- `inspire/wood_tray_small/20260406_051333`
- `inspire/wood_tray_small/20260406_051502`

The pending task is:

- `allegro/brown_ramen/20260327_000807`

## Object Tracking Failure Pattern

The `wood_tray_small` failures are not missing-data failures. The logs show that GoTrack reached the frame loop, loaded all 24 cameras, loaded initial poses, and then failed at the renderer stage on frame 0.

Representative error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.61 GiB.
...
renderer_nvdiffrast.py, _render_mesh_batch
tex_image = texture.expand(batch_size, -1, -1, -1).contiguous()
```

Observed characteristics:

- Object: `wood_tray_small`
- Cameras: 24 views
- Failure point: GoTrack render/refinement input preparation
- Hardware seen in logs: 16 GB GPU class, with about 5.86 GiB free at failure time
- Immediate cause: large texture/render batch memory allocation
- Likely fix: rerun only these 20 episodes with lower memory settings, such as smaller render/refinement batch size, reduced texture resolution, fewer simultaneous camera render inputs, or a GPU with more free VRAM.

The single pending `brown_ramen` task has not produced a worker log. Its task JSON is still `pending`, but a stale claim lock exists:

```text
/home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_selected100_all_20260701_164328/claims/allegro__brown_ramen__20260327_000807.lock
```

The claim was created by `capture5` on 2026-07-01, but the task stayed pending. Because `claim_next()` skips tasks with an existing claim lock, later workers reported `no_claimable_tasks`. This is a scheduler bookkeeping issue, not an object tracking model failure.

## Interactive 3D GLB Export Status

The later interactive 3D pass uses object tracking outputs plus synced robot qpos to build animated GLB files.

| Item | Count |
|---|---:|
| GLB export batch tasks | 1,117 |
| GLB export success | 939 |
| GLB export failed | 178 |
| Existing generated GLBs included in output root | 942 |

Output location:

```text
/home/robot/shared_data/AutoDex/interactive_3d/<relative_episode_path>/animated.glb
```

Gallery-facing manifests:

```text
/home/robot/shared_data/AutoDex/interactive_3d/_site/index.json
/home/robot/shared_data/AutoDex/interactive_3d/_site/assets3d.json
```

## Interactive 3D Failure Pattern

The 178 interactive 3D export failures are different from the 20 object tracking failures above.

For all 178 failed GLB exports:

- `object_tracking/gotrack_output/world_pose_records.json` exists.
- Object trajectory is available.
- Robot arm/hand trajectory is missing.
- Therefore robot + hand + object animated GLB cannot be produced.

Failure message:

```text
FileNotFoundError:
Synced qpos missing. Expected episode/arm/state.npy and episode/hand/state.npy.
```

Breakdown:

| Missing data type | Count |
|---|---:|
| `arm/` and `hand/` folders exist, but `state.npy` is missing | 48 |
| `arm/` and `hand/` folders are missing entirely | 130 |

Scene-level pattern:

| Scene prefix | Total in GLB batch | GLB failed | Failure rate |
|---|---:|---:|---:|
| `default` | 1,033 | 94 | 9.1% |
| `success_only` | 34 | 34 | 100% |
| `shelf` | 13 | 13 | 100% |
| `shelf_success_only` | 13 | 13 | 100% |
| `wall` | 7 | 7 | 100% |
| `wall_success_only` | 7 | 7 | 100% |
| `cluttered` | 5 | 5 | 100% |
| `cluttered_success_only` | 5 | 5 | 100% |

The non-default scene subsets have object pose tracking results, but they do not have the synced robot qpos files required by the current animated GLB exporter.

Most common objects among the 178 GLB failures:

| Object | GLB failed |
|---|---:|
| `brown_ramen` | 37 |
| `plant_mister` | 26 |
| `pepsi_light` | 18 |
| `icecream_scoop` | 14 |
| `organizer_beige` | 10 |
| `lemon_squeezer` | 8 |
| `pepper_tuna_light` | 8 |
| `wateringcan` | 7 |
| `blue_alarm` | 7 |
| `soaptray` | 6 |

## Interpretation

Object tracking and interactive 3D export should be treated as separate stages.

Object tracking status:

- 1,120 episodes currently have object tracking output.
- 21 object tracking tasks remain unresolved.
- The unresolved tracking tasks are narrowly scoped: 20 `wood_tray_small` OOM failures and 1 stale-lock pending `brown_ramen`.

Interactive 3D status:

- 942 animated GLB files currently exist.
- 178 object-tracked episodes cannot become robot + hand + object animated GLBs unless synced robot qpos files are recovered or regenerated.
- These 178 are still valid for object-only trajectories, but not for the current full robot/hand/object animated GLB format.

## Recommended Next Actions

1. Restart the GoTrack episode dashboard when inspection is needed:

   ```bash
   python scripts/gotrack_episode_dashboard.py \
     --schedule-dir /home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_selected100_all_20260701_164328 \
     --host 127.0.0.1 \
     --port 8771
   ```

2. For the single pending `brown_ramen` task, clear only its stale claim lock and rerun that one task.

3. For the 20 `wood_tray_small` failures, rerun only those episodes with lower-memory GoTrack settings. Do not rerun the full selected_100 schedule.

4. For the 178 interactive 3D failures, decide whether to:
   - recover/generate `episode/arm/state.npy` and `episode/hand/state.npy`, then rerun GLB export, or
   - generate object-only animated GLBs for these episodes.

