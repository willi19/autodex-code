# P2 — generic fruit/non-fruit semantic routing

P2 reuses the existing FoundPose → silhouette optimization → v8 Inspire grasp
planning → grasp → 15 cm lift pipeline. It adds only a semantic side channel.

The four protocol objects (`apple`, `banana`, `pringles`, `spam_can`) retain
automatic semantic ground-truth C.  Other object asset names are also allowed:
they use exactly the same generic FRUIT/NON_FRUIT routing and grasp pipeline,
but automatic semantic C is stored as unscored because the benchmark's expected
class is not defined for them.

1. For P2 only, each capture PC processes cameras in ascending serial order and
   sends the first SAM3 crop whose foreground bbox is at least 16 pixels from
   every image edge.
2. The crop is a `1.5 × max(bbox_w, bbox_h)` square, foreground RGB over
   neutral gray, resized to `448 × 448`. No area/IoU/pose-quality ranking or
   post-selection validation is performed.
3. The first three distinct PCs to publish a crop are passed together to local
   4-bit NF4 `Qwen/Qwen2.5-VL-3B-Instruct`. The prompt contains neither object
   names nor P2's object list, only the binary `FRUIT` / `NON_FRUIT` question.
4. `FRUIT` routes to left / final object bearing `+50°`; `NON_FRUIT` routes to
   right / `-30°`. Existing inference reads the post-lift joint state, uses FK,
   then moves only J0 to the selected final bearing before opening the hand.

The normal AutoDex mask and FoundPose pose PUB payloads are unchanged. The
JPEG crop uses optional port `5010` only for a `p2_semantic_enabled` request.

## One-time setup

```bash
/home/robot/anaconda3/envs/autodex/bin/python -m pip install -r src/demo/p2/requirements-vlm.txt
hf download Qwen/Qwen2.5-VL-3B-Instruct
bash scripts/init_daemons.sh start --p2-semantic
```

Install these in the Robot PC's `autodex` environment (the capture-PC
`gotrack_cu128` daemon only creates JPEG crops). The source checkpoint lives
in the authenticated Hugging Face cache and is loaded in NF4 4-bit mode on the
robot GPU; it is not stored in the repository or NAS episode directories.
Without `--p2-semantic`, normal AutoDex stays unchanged
and a P2 run exits as `semantic_timeout` before any pick/approach motion (after
the normal initial clear-view home, if requested).

## Run

```bash
python src/demo/p2/run_demo.py --obj apple --arm franka --execute
python src/demo/p2/run_demo.py --obj banana --arm franka --execute
python src/demo/p2/run_demo.py --obj pringles --arm franka --execute
python src/demo/p2/run_demo.py --obj spam_can --arm franka --execute
```

P2 rejects a non-FoundPose backend, Cartesian transfer, marker target, or any
lift height other than 0.15 m. Episodes remain under the normal
`~/shared_data/AutoDex/experiment/v8_demo/inspire/<object>/<timestamp>/`
hierarchy. `semantic/` holds the three JPEG inputs and `semantic_result.json`;
the episode `result.json` stores the selected route and semantic correctness C.

## Execution AVI recording

P2 now uses the same capture-PC recording layout as AutoDex collection.  The
single-trial runner records only after planning has succeeded and immediately
before the first grasp motion:

```bash
python src/demo/p2/run_demo.py --obj apple --arm franka --execute
```

Each capture PC writes a task-only AVI (grasp/lift, transfer and release;
retreat/reset are intentionally excluded):
`AutoDex/experiment/v8_demo/inspire/<object>/<timestamp>/raw/exec/videos/<serial>.avi`.
The episode's `recording.json` records the matching NAS destination and robot
state/timestamp sidecars.  Use `--no-video` only when an AVI is not wanted.
After the take has fully stopped, the usual ParaDex uploader can upload every
pending local recording (do not run it while a take is being captured):

```bash
cd ~/paradex
conda activate flir_env
python src/util/upload_video/main.py
```

It retains the existing hierarchy and produces
`~/shared_data/AutoDex/experiment/v8_demo/inspire/<object>/<timestamp>/videos/exec/<serial>.avi`.

## Consecutive P2 trials

```bash
python src/demo/p2/run_auto.py --arm franka --execute
```

The runner opens camera control/calibration, planner, robot executor, Qwen and
the video-trigger resources once.  At every trial it asks for the object and
the location (`0=upper_right`, `1=center`, `2=lower_left`); Enter repeats the
preceding answer.  It reloads FoundPose templates only when the selected object
changes, then runs the same `run_demo.py` inference pipeline per episode.

When motion completes (or the pipeline stops safely), choose the furthest
verified outcome: `f` fail, `g` grasp, `p` place, or `c` correct bin.  Use
`a` aborted for a deliberately unscored take; it is saved separately and is
not counted as a failure.  Then enter an optional memo.  `result.json` and
`p2_trial.json` in that episode store the
object/location, automated semantic route, operator G/P/C label, memo, video
manifest, and total/sub-step timing for perception and planning.  Execution is
stored separately as `execution.task` (grasp/lift, transfer, release) and
`execution.reset` (retreat and return home), plus their combined duration.
`--obj apple --location 0` can seed the first two prompts, and
`--max-trials N` ends a bounded session after N takes.
