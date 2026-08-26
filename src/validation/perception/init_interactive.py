#!/usr/bin/env python3
"""Interactive end-to-end FoundPose init test.

Launches the distributed init pipeline against capture1-6 (which must already
be running init_daemon.py in `gotrack_cu128`). Each Enter triggers one full
init across all 6 PCs and prints per-stage latency.

The capture step is simulated from a saved episode dir (until cameras are
running live). Cycle through episodes with `n` (next) / use one fixed dir.

Usage:
    # On capture1-6: ensure init_daemon.py is running.
    # On robot PC:
    python src/execution/run_init_interactive.py --obj attached_container

Trial output: ~/shared_data/AutoDex/experiment/object6d_test_foundpose/{obj}/{trial:02d}/
              pose_world.npy
              overlay/{serial}.png + grid.png
              timing.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture arms in 'acquire' and its
    outputs are toggled as SINKS (set_stream / set_record). The capture PCs
    reject the old names outright, and the rejection LATCHES an error on every
    camera that only a daemon-side reload clears -- so a single call from one
    stale script leaves the next run with no frames at all.
    """
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
    elif mode == "full":
        # "full" was video AVI + SHM stream at once (snapshot_daemon reads the
        # stream while the AVI records). Both are just sinks now.
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
        rcc.set_stream(True)
    elif mode == "video":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
    else:                       # 'image' is still a real capture mode
        rcc.start(mode, sync_mode, save_path, fps=fps)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_FP_ROOT = REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
EXP_SRC = Path.home() / "shared_data/AutoDex/experiment/selected_100/allegro"
EXP_OUT = Path.home() / "shared_data/AutoDex/experiment/object6d_test_foundpose"
DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]  # capture4 out


EXP_SRC_ALT = Path.home() / "shared_data/AutoDex/experiment/allegro/selected_100_prev"


def _list_episodes(obj: str, exp_root: Optional[Path] = None) -> List[Path]:
    roots = [exp_root] if exp_root else [EXP_SRC, EXP_SRC_ALT]
    for root in roots:
        if root is None:
            continue
        obj_dir = root / obj
        if not obj_dir.exists():
            continue
        out = []
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            if (ep / "images").exists() and (ep / "cam_param/intrinsics.json").exists() \
                    and (ep / "cam_param/extrinsics.json").exists():
                out.append(ep)
        if out:
            return out
    return []


def _load_calib(ep: Path):
    intr_path = ep / "cam_param/intrinsics.json"
    extr_path = ep / "cam_param/extrinsics.json"
    if not intr_path.exists() or not extr_path.exists():
        # Support direct calib dir layout: <calib>/intrinsics.json, extrinsics.json
        intr_path = ep / "intrinsics.json"
        extr_path = ep / "extrinsics.json"
    with open(intr_path) as f:
        intr_raw = json.load(f)
    with open(extr_path) as f:
        extr_raw = json.load(f)
    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a
    H = next(iter(intrinsics_full.values()))["height"]
    W = next(iter(intrinsics_full.values()))["width"]
    return intrinsics_full, extrinsics_full, H, W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--mode", choices=["live", "disk"], default="live",
                        help="live: read from capture-PC SHM (cameras must be running). "
                             "disk: read images from saved episode dir.")
    parser.add_argument("--ep", type=str, default=None,
                        help="(disk only) Episode dir name under {exp_root}/{obj}/.")
    parser.add_argument("--exp-root", type=str, default=None,
                        help="(disk only) Experiment root. Default: tries selected_100/allegro then "
                             "allegro/selected_100_prev.")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="(live only) cam_param dir. Default: latest under "
                             "~/shared_data/cam_param/.")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--pc-list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=str, default=str(EXP_OUT))
    parser.add_argument("--no-auto-start-stream", dest="auto_start_stream",
                        action="store_false",
                        help="(live only) Skip starting camera stream — assume it's already running.")
    parser.set_defaults(auto_start_stream=True)
    parser.add_argument("--stream-fps", type=int, default=10,
                        help="(live + auto-start-stream) Stream FPS.")
    parser.add_argument("--stream-warmup-s", type=float, default=2.0,
                        help="(live + auto-start-stream) Seconds to wait after starting stream.")
    args = parser.parse_args()

    from paradex.utils.system import get_pc_ip, get_camera_list
    from autodex.perception.init_orchestrator import InitOrchestrator

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    out_root = Path(args.out) / args.obj
    out_root.mkdir(parents=True, exist_ok=True)

    ep: Optional[Path] = None
    if args.mode == "disk":
        exp_root = Path(args.exp_root).expanduser() if args.exp_root else None
        eps = _list_episodes(args.obj, exp_root=exp_root)
        if not eps:
            searched = [exp_root] if exp_root else [EXP_SRC, EXP_SRC_ALT]
            sys.exit(f"No episodes for {args.obj} under any of: {searched}")
        if args.ep:
            ep = eps[0].parent / args.ep
            if not ep.exists():
                sys.exit(f"episode not found: {ep}")
        else:
            ep = eps[0]
        print(f"mode=disk  obj={args.obj}  episode={ep.name}")
        intrinsics_full, extrinsics_full, H, W = _load_calib(ep)
    else:
        # live: latest cam_param/<ts>/
        if args.calib_dir:
            calib = Path(args.calib_dir).expanduser()
        else:
            cam_root = Path.home() / "shared_data/cam_param"
            calib = sorted(cam_root.iterdir())[-1]
        print(f"mode=live  obj={args.obj}  calib={calib.name}")
        intrinsics_full, extrinsics_full, H, W = _load_calib(calib)
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"calib: {len(intrinsics_full)} cams active ({len(args.pc_list)} PCs)  {H}x{W}")

    rcc = None
    if args.mode == "live" and args.auto_start_stream:
        from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
        print(f"[stream] starting camera stream on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
        rcc = remote_camera_controller("init_interactive", pc_list=args.pc_list)
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if args.stream_warmup_s > 0:
            time.sleep(args.stream_warmup_s)
        print("[stream] started")

    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )
    try:
        print("\n[init] sending init to capture PCs (FoundPose load ~3s/PC for new object)...")
        t0 = time.perf_counter()
        orch.init_object(
            obj_name=args.obj,
            mesh_path=str(mesh_path), assets_root=str(assets_root),
            intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
            image_hw=(H, W),
            mode=args.mode, pc_serials=pc_serials,
        )
        print(f"[init] dispatched in {time.perf_counter()-t0:.1f}s "
              f"(daemons may still be loading models — first trial will reflect that)")

        records: List[Dict[str, Any]] = []
        trial = 0
        while True:
            trial += 1
            ans = input(f"\n[trial {trial}] Press Enter to run, 'q' to quit: ").strip().lower()
            if ans == "q":
                break

            trial_ts = time.strftime("%Y%m%d_%H%M%S")
            trial_dir = out_root / trial_ts
            trial_dir.mkdir(parents=True, exist_ok=True)
            t_start = time.perf_counter()
            live_capture_dir = trial_dir / "capture" if args.mode == "live" else None

            pose, timing = orch.trigger_init(
                prompt=args.prompt,
                capture_dir=str(ep) if (args.mode == "disk" and ep is not None) else None,
                save_capture_dir=str(live_capture_dir) if live_capture_dir is not None else None,
                sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                timeout_s=args.timeout_s,
            )
            wall = time.perf_counter() - t_start

            rec: Dict[str, Any] = {
                "obj": args.obj, "trial": trial, "trial_ts": trial_ts,
                "episode": ep.name if ep is not None else None,
                "mode": args.mode, "wall_s": wall, **timing,
            }

            if pose is None:
                rec["ok"] = False
                print(f"  FAILED: {timing.get('reason')}  (wall {wall:.2f}s)")
            else:
                rec["ok"] = True
                np.save(trial_dir / "pose_world.npy", pose)
                rec["total_s"] = float(timing["total_s"])
                print(f"  OK   total {rec['total_s']:.2f}s "
                      f"(collect {timing['dispatch_to_collected_s']:.2f} "
                      f"iou {timing['iou_select_s']:.2f} "
                      f"sil {timing['sil_refine_s']:.2f})  "
                      f"iou={timing.get('best_iou', 0):.3f}")
            with open(trial_dir / "timing.json", "w") as f:
                json.dump(rec, f, indent=2, default=str)
            records.append(rec)

        # Aggregate
        ok = [r for r in records if r.get("ok")]
        if ok:
            print(f"\n=== {len(ok)}/{len(records)} successful trials ===")
            keys = ["dispatch_to_collected_s", "iou_select_s", "sil_refine_s",
                    "overlay_s", "total_s"]
            print(f"{'stage':<24} {'mean':>8} {'median':>8} {'stdev':>8} {'min':>8} {'max':>8}")
            for k in keys:
                vals = [r[k] for r in ok if r.get(k) is not None]
                if not vals:
                    continue
                m = statistics.mean(vals); med = statistics.median(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"{k:<24} {m:>8.2f} {med:>8.2f} {sd:>8.2f} {min(vals):>8.2f} {max(vals):>8.2f}")
            with open(out_root / "_summary.json", "w") as f:
                json.dump(records, f, indent=2, default=str)
            print(f"\nsummary -> {out_root}/_summary.json")
    finally:
        if rcc is not None:
            try:
                print("[stream] stopping camera stream...")
                rcc.stop()
            except Exception:
                pass
            try:
                rcc.end()
            except Exception:
                pass
        orch.close()


if __name__ == "__main__":
    main()
