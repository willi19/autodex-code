#!/usr/bin/env python3
"""Interactive end-to-end GoTrack tracking validation.

Runs FoundPose init once (or loads a saved init pose), then drives
gotrack_daemon on capture1-6 to track the object in real time.

Keyboard control:
    c   start tracking
    q   stop tracking and quit

Live tracking pose is:
  - saved per-frame to {out}/{obj}/{trial_ts}/poses/{frame_id:08d}.npy
  - visualized in viser as object mesh transformed by current pose

Usage (live):
    python src/validation/perception/track_interactive.py \\
        --obj attached_container --mode live \\
        --auto-start-stream --stop-stream-on-exit \\
        --prompt "object on the checkerboard"

Pre-conditions:
  - init_daemon.py running on capture1-6 (port 6893)
  - gotrack_daemon.py running on capture1-6 (port 6892)
  - FoundPose repre.pth and GoTrack anchor_bank present per
    docs/distributed_tracking.md

Trial output:
    ~/shared_data/AutoDex/experiment/object6d_test_gotrack/{obj}/{trial_ts}/
        init_pose_world.npy
        init_timing.json
        poses/{frame_id:08d}.npy
        track_log.json
"""
from __future__ import annotations

import argparse
import json
import logging
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from concurrent.futures import ThreadPoolExecutor
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
EXP_OUT = Path.home() / "shared_data/AutoDex/experiment/object6d_test_gotrack"
EXP_SRC = Path.home() / "shared_data/AutoDex/experiment/selected_100/allegro"
EXP_SRC_ALT = Path.home() / "shared_data/AutoDex/experiment/allegro/selected_100_prev"
DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]  # capture4 out
DEFAULT_ANCHOR_BANK_REL = "autodex/perception/thirdparty/MV-GoTrack/anchor_banks"
GOTRACK_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack"


def ensure_anchor_bank(obj_name: str, mesh_path: Path,
                        num_anchors: int = 256) -> Path:
    """Generate anchor bank locally on robot PC if missing. Returns its path.

    Uses sys.executable (caller is expected to run inside gotrack_cu128 env).
    """
    bank = REPO_ROOT / DEFAULT_ANCHOR_BANK_REL / f"{obj_name}.npz"
    if bank.exists():
        return bank
    bank.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anchor] {bank.name} missing — generating (~수 초)...")
    cmd = [
        sys.executable, "scripts/generate_anchor_bank.py",
        "--mesh-path", str(mesh_path),
        "--output-path", str(bank),
        "--num-anchors", str(num_anchors),
    ]
    subprocess.run(cmd, cwd=str(GOTRACK_ROOT), check=True)
    if not bank.exists():
        raise RuntimeError(f"generate_anchor_bank.py did not write {bank}")
    print(f"[anchor] generated -> {bank}")
    return bank


def rsync_anchor_bank_to_pcs(bank_local: Path, pc_list: List[str],
                              parallel: bool = True) -> None:
    """rsync anchor bank file from robot PC to each capture PC's same ~/relative path."""
    rel = bank_local.relative_to(Path.home())
    remote_dir = f"~/{rel.parent.as_posix()}"
    remote_path = f"~/{rel.as_posix()}"

    def _one(pc: str) -> None:
        subprocess.run(["ssh", pc, f"mkdir -p {remote_dir}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run(
            ["rsync", "-az", "--checksum", str(bank_local), f"{pc}:{remote_path}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rsync to {pc} failed: {result.stderr.strip()}")
        print(f"[anchor] -> {pc}: ok")

    print(f"[anchor] rsync {bank_local.name} to {len(pc_list)} capture PCs...")
    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(pc_list), 8)) as ex:
            for fut in [ex.submit(_one, pc) for pc in pc_list]:
                fut.result()
    else:
        for pc in pc_list:
            _one(pc)


def _to_home_relative(p) -> str:
    p = str(p)
    home = str(Path.home())
    if p.startswith(home + "/"):
        return "~/" + p[len(home) + 1:]
    return p


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


class _Keyboard:
    """Background single-key listener (cbreak mode on stdin)."""

    def __init__(self):
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self._exit = threading.Event()
        self._fd: Optional[int] = None
        self._old = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread.start()

    def _loop(self) -> None:
        while not self._exit.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            try:
                ch = sys.stdin.read(1).lower()
            except Exception:
                continue
            if ch == "c":
                self.start_event.set()
            elif ch == "q":
                self.stop_event.set()
                break

    def close(self) -> None:
        self._exit.set()
        if self._fd is not None and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass


def _set_pose_handle(handle, T: np.ndarray) -> None:
    """Update viser frame/mesh handle from a 4x4 SE(3) matrix."""
    from scipy.spatial.transform import Rotation
    handle.position = tuple(np.asarray(T[:3, 3], dtype=np.float64).tolist())
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    handle.wxyz = (float(q_xyzw[3]), float(q_xyzw[0]),
                   float(q_xyzw[1]), float(q_xyzw[2]))


def _build_viser(mesh_path: Path,
                 extrinsics_full: Dict[str, np.ndarray],
                 init_pose_robot: np.ndarray,
                 port: int,
                 c2r: Optional[np.ndarray] = None):
    """viser scene rendered in robot base frame.

    extrinsics_full entries are world->cam (charuco frame). ``c2r`` = world->robot
    representation of the robot origin (i.e. robot expressed in charuco). The
    point-frame transform is ``p_robot = inv(c2r) @ p_world``.
    """
    import viser
    import trimesh
    from scipy.spatial.transform import Rotation

    if c2r is None:
        r2c = np.eye(4)
    else:
        r2c = np.linalg.inv(np.asarray(c2r, dtype=np.float64).reshape(4, 4))

    server = viser.ViserServer(port=port)
    # Robot base origin frame (longer axes so it's clearly visible).
    server.scene.add_frame("/world", show_axes=True, axes_length=0.2, axes_radius=0.005)
    # Floor grid (z=0 in robot frame == table plane).
    try:
        server.scene.add_grid(
            "/floor",
            width=2.0, height=2.0,
            width_segments=20, height_segments=20,
            plane="xy",
            cell_color=(160, 160, 160),
            section_color=(80, 80, 80),
        )
    except Exception:
        pass

    obj_frame = server.scene.add_frame("/object", show_axes=True,
                                        axes_length=0.05, axes_radius=0.002)
    obj_tm = trimesh.load(mesh_path, process=False)
    server.scene.add_mesh_simple(
        "/object/mesh",
        vertices=np.asarray(obj_tm.vertices, dtype=np.float32),
        faces=np.asarray(obj_tm.faces, dtype=np.uint32),
        color=(220, 100, 80), opacity=0.85,
    )
    _set_pose_handle(obj_frame, init_pose_robot)

    for s, T_wc in extrinsics_full.items():
        T_cw = np.linalg.inv(T_wc)        # cam-in-world (charuco)
        T_cr = r2c @ T_cw                 # cam-in-robot
        q_xyzw = Rotation.from_matrix(T_cr[:3, :3]).as_quat()
        server.scene.add_frame(
            f"/cam/{s}",
            position=tuple(T_cr[:3, 3].astype(float)),
            wxyz=(float(q_xyzw[3]), float(q_xyzw[0]),
                  float(q_xyzw[1]), float(q_xyzw[2])),
            show_axes=True, axes_length=0.03, axes_radius=0.0015,
        )
    return server, obj_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--mode", choices=["live", "disk"], default="live")
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--exp-root", type=str, default=None)
    parser.add_argument("--calib-dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--pc-list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    # FoundPose-init daemon ports.
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    # GoTrack daemon ports.
    parser.add_argument("--port-obs", type=int, default=1235)
    parser.add_argument("--port-prior", type=int, default=1236)
    parser.add_argument("--port-cmd-track", type=int, default=6892)
    parser.add_argument("--anchor-bank", type=str, default=None,
                        help="Path to anchor bank .npz (resolved on each capture PC). "
                             "Default: ~/AutoDex/" + DEFAULT_ANCHOR_BANK_REL + "/{obj}.npz")
    parser.add_argument("--num-anchors", type=int, default=256,
                        help="Number of anchors to sample (only used if generating).")
    parser.add_argument("--no-anchor-rsync", action="store_true",
                        help="Skip rsync of anchor bank to capture PCs (assume already in place).")
    parser.add_argument("--init-pose", type=str, default=None,
                        help="Skip FoundPose init; load 4x4 pose from this .npy file.")
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--min-cams-per-frame", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--out", type=str, default=str(EXP_OUT))
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--auto-start-stream", action="store_true")
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--stream-warmup-s", type=float, default=2.0)
    parser.add_argument("--stop-stream-on-exit", action="store_true")
    args = parser.parse_args()

    from paradex.utils.system import get_pc_ip, get_camera_list
    from paradex.io.capture_pc.command_sender import CommandSender
    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.perception.gotrack_tracker import GoTrackTracker

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj} at {assets_root}")

    if args.anchor_bank:
        anchor_bank_path = Path(args.anchor_bank).expanduser()
    else:
        anchor_bank_path = ensure_anchor_bank(args.obj, mesh_path, args.num_anchors)
    print(f"[track] mesh:        {mesh_path}")
    print(f"[track] anchor_bank: {anchor_bank_path}")

    if not args.no_anchor_rsync and not args.anchor_bank:
        try:
            rsync_anchor_bank_to_pcs(anchor_bank_path, args.pc_list)
        except Exception as exc:
            sys.exit(f"[anchor] rsync failed: {exc}\n"
                     f"  Either fix ssh/rsync, or run with --no-anchor-rsync after "
                     f"manually placing the file on each capture PC.")

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    out_root = Path(args.out) / args.obj
    out_root.mkdir(parents=True, exist_ok=True)
    trial_ts = time.strftime("%Y%m%d_%H%M%S")
    trial_dir = out_root / trial_ts
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr + add a file handler to root logger so every print()
    # and every logger.* call is saved to trial_dir/track.log. Offline tools
    # (video builder, post-trial analysis) read this log to correlate fid
    # with per-frame events.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try: st.write(s); st.flush()
                except Exception: pass
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass
    log_path = trial_dir / "track.log"
    log_fp = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    _file_handler = logging.FileHandler(log_path, mode="a")
    _file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_file_handler)
    print(f"[track] log → {log_path}")

    ep: Optional[Path] = None
    if args.mode == "disk":
        exp_root = Path(args.exp_root).expanduser() if args.exp_root else None
        eps = _list_episodes(args.obj, exp_root=exp_root)
        if not eps:
            searched = [exp_root] if exp_root else [EXP_SRC, EXP_SRC_ALT]
            sys.exit(f"No episodes for {args.obj} under any of: {searched}")
        ep = (eps[0].parent / args.ep) if args.ep else eps[0]
        if not ep.exists():
            sys.exit(f"episode not found: {ep}")
        print(f"[track] mode=disk  episode={ep.name}")
        intrinsics_full, extrinsics_full, H, W = _load_calib(ep)
    else:
        if args.calib_dir:
            calib = Path(args.calib_dir).expanduser()
        else:
            cam_root = Path.home() / "shared_data/cam_param"
            calib = sorted(cam_root.iterdir())[-1]
        print(f"[track] mode=live  calib={calib.name}")
        intrinsics_full, extrinsics_full, H, W = _load_calib(calib)
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"[track] {len(intrinsics_full)} cams active ({len(args.pc_list)} PCs)  {H}x{W}")

    rcc = None
    sync_generator = None
    timestamp_monitor = None
    orch: Optional[InitOrchestrator] = None
    cmd_track: Any = None
    tracker: Optional[GoTrackTracker] = None
    keyboard = _Keyboard()
    server = None
    obj_frame = None
    pose_log: List[Dict[str, Any]] = []
    init_pose_world: Optional[np.ndarray] = None

    try:
        # 1) Optional camera stream with hardware sync.
        # When the stream is enabled we run UTGE900 (signal_generator) +
        # TimestampMonitor on the robot PC. With syncMode=True the cameras
        # advance their SHM `fid` only on each signal pulse, so all cameras
        # across all PCs share the same global frame_id. The TimestampMonitor
        # reads back (frame_id, pc_time) per pulse from the trigger-wired
        # monitor camera, so the tracker can compare its received fids to
        # what the host PC thinks the *current* fid is.
        if args.mode == "live" and args.auto_start_stream:
            from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
            from paradex.io.camera_system.signal_generator import UTGE900
            from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
            from paradex.utils.system import network_info
            print(f"[stream] starting signal_generator @ {args.stream_fps} FPS...")
            sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
            timestamp_monitor.start(None)
            sync_generator.start(fps=args.stream_fps)
            print(f"[stream] starting camera stream on {len(args.pc_list)} PCs (syncMode=True)...")
            rcc = remote_camera_controller("track_interactive", pc_list=args.pc_list)
            _rcc_start(rcc, "stream", True, fps=args.stream_fps)
            if args.stream_warmup_s > 0:
                time.sleep(args.stream_warmup_s)
            print("[stream] started")

        # 2) Init pose acquisition.
        init_timing: Dict[str, Any]
        if args.init_pose:
            init_pose_world = np.load(Path(args.init_pose).expanduser())
            if init_pose_world.shape != (4, 4):
                sys.exit(f"--init-pose must be 4x4, got {init_pose_world.shape}")
            print(f"[init] loaded init pose from {args.init_pose}")
            init_timing = {"source": "file", "path": str(args.init_pose)}
        else:
            print("[init] running FoundPose init via init_orchestrator...")
            orch = InitOrchestrator(
                pc_list=args.pc_list, capture_ips=pc_ips,
                port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
            )
            orch.init_object(
                obj_name=args.obj,
                mesh_path=str(mesh_path), assets_root=str(assets_root),
                intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
                image_hw=(H, W), mode=args.mode, pc_serials=pc_serials,
            )
            t0 = time.perf_counter()
            init_pose_world, init_timing = orch.trigger_init(
                prompt=args.prompt,
                capture_dir=str(ep) if (args.mode == "disk" and ep is not None) else None,
                save_capture_dir=str(trial_dir / "init_capture") if args.mode == "live" else None,
                sil_iters=args.sil_iters, sil_lr=args.sil_lr, timeout_s=args.timeout_s,
            )
            if init_pose_world is None:
                sys.exit(f"FoundPose init failed: {init_timing.get('reason')}")
            print(f"[init] pose acquired in {time.perf_counter()-t0:.2f}s "
                  f"(iou={init_timing.get('best_iou', 0):.3f}, "
                  f"sil_loss={init_timing.get('sil_loss', 0):.4f})")
            init_timing["source"] = "foundpose_init"

        # Save C2R (world/charuco -> robot base) into trial dir + convert init pose.
        from paradex.calibration.utils import save_current_C2R, load_c2r
        save_current_C2R(str(trial_dir))
        c2r = load_c2r(str(trial_dir))
        r2c = np.linalg.inv(c2r)
        init_pose_robot = r2c @ init_pose_world
        np.save(trial_dir / "init_pose_world.npy", init_pose_world)
        np.save(trial_dir / "init_pose_robot.npy", init_pose_robot)
        np.save(trial_dir / "C2R.npy", c2r)
        with open(trial_dir / "init_timing.json", "w") as f:
            json.dump(init_timing, f, indent=2, default=str)

        # 3) Build viser scene in ROBOT base frame (optional).
        if not args.no_viser:
            try:
                server, obj_frame = _build_viser(mesh_path, extrinsics_full,
                                                  init_pose_robot, args.viser_port, c2r=c2r)
                print(f"[viser] http://0.0.0.0:{args.viser_port}")
            except Exception as exc:
                print(f"[viser] failed to start ({exc}); continuing headless")
                server, obj_frame = None, None

        # 4) Init GoTrack daemons (port 6892). Send K_orig + dist_params too so
        # the daemon can undistort SHM frames to match the K_undist the engine uses.
        intrinsics_payload = {
            s: {
                "K": intrinsics_full[s]["K_undist"].tolist(),
                "K_orig": np.asarray(intrinsics_full[s]["K_orig"]).tolist(),
                "dist_params": np.asarray(intrinsics_full[s]["dist_params"]).tolist(),
                "width": int(intrinsics_full[s]["width"]),
                "height": int(intrinsics_full[s]["height"]),
            }
            for s in intrinsics_full
        }
        extrinsics_payload = {
            s: np.asarray(extrinsics_full[s], dtype=np.float64).reshape(4, 4).tolist()
            for s in extrinsics_full
        }
        info_track = {
            "mesh_path": _to_home_relative(mesh_path),
            "anchor_bank_path": _to_home_relative(anchor_bank_path),
            "object_id": 1,
            "object_name": args.obj,
            "intrinsics": intrinsics_payload,
            "extrinsics": extrinsics_payload,
            "mesh_scale": 1.0,
            "unit_scale_mode": "auto",
            "num_iters": 1,
            "first_frame_num_iters": 5,
        }
        cmd_track = CommandSender(pc_list=args.pc_list, port=args.port_cmd_track)
        print(f"[track] sending init to gotrack daemons on port {args.port_cmd_track}...")
        cmd_track.send_command("init", wait=False, cmd_info=info_track)
        time.sleep(0.5)  # let daemons begin loading the engine

        # 5) Build robot-PC tracker.
        tracker = GoTrackTracker(
            capture_pc_ips=pc_ips,
            port_obs=args.port_obs, port_prior=args.port_prior,
            min_cams_per_frame=args.min_cams_per_frame,
        )

        # 6) Keyboard listener + control.
        keyboard.start()
        print("\n>>> Press 'c' to start tracking, 'q' to quit. <<<\n")
        while not keyboard.start_event.is_set() and not keyboard.stop_event.is_set():
            time.sleep(0.05)
        if keyboard.stop_event.is_set():
            print("[track] aborted before start")
            return

        # Save current cam_param at trial level on NAS so build_track_video.py
        # can find the calibration that was active for this exact trial.
        trial_crops_nas = (Path.home() / "shared_data/AutoDex/debug/gotrack_crops"
                           / args.obj / trial_ts)
        trial_crops_nas.mkdir(parents=True, exist_ok=True)
        cam_param_nas = trial_crops_nas / "cam_param"
        cam_param_nas.mkdir(parents=True, exist_ok=True)
        with open(cam_param_nas / "intrinsics.json", "w") as f:
            json.dump({s: {
                "K_undist": np.asarray(v["K_undist"]).tolist(),
                "K_orig": np.asarray(v["K_orig"]).tolist(),
                "dist_params": np.asarray(v["dist_params"]).tolist(),
                "width": int(v["width"]),
                "height": int(v["height"]),
            } for s, v in intrinsics_full.items()}, f, indent=2)
        with open(cam_param_nas / "extrinsics.json", "w") as f:
            json.dump({s: np.asarray(v).reshape(4, 4).tolist()
                       for s, v in extrinsics_full.items()}, f, indent=2)

        # 7) Send "start" to gotrack daemons with the trial_ts so each daemon
        # writes its crops under .../{obj}/{trial_ts}/{fid:06d}/.
        cmd_track.send_command("start", wait=False, cmd_info={"trial_ts": trial_ts})
        print("[track] tracking started")

        poses_dir = trial_dir / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)

        def _run():
            n = 0
            t_start = time.perf_counter()
            print(f"[track-worker] entered _run, calling tracker.track(...)")
            # Background watchdog: every 2s report received / success / fail
            # breakdown so failures are visible. n_yielded alone is meaningless
            # — pair it with received count + per-reason fail counts.
            # When timestamp_monitor is available it reports the host-side
            # current global fid so we can compute per-PC and tracker lag.
            def _watchdog():
                while not keyboard.stop_event.is_set():
                    time.sleep(2.0)
                    with tracker._status_lock:
                        per_pc = dict(tracker.status.get("per_pc_last_frame", {}))
                        counts = dict(tracker.status.get("counts", {}))
                        last_fid_yielded = int(tracker.status.get("frame_id", -1))
                    received = int(counts.get("received", 0))
                    success = int(counts.get("success", 0))
                    fail_by = dict(counts.get("fail_by_reason", {}))
                    n_inflight = len(tracker.sync_buffer._buf)
                    rate = (success / received * 100.0) if received else 0.0
                    fail_str = ", ".join(f"{k}={v}" for k, v in sorted(fail_by.items())) or "-"
                    cur_fid = -1
                    if timestamp_monitor is not None:
                        try:
                            cur_fid = int(timestamp_monitor.get_data().get("frame_id", -1))
                        except Exception:
                            cur_fid = -1
                    per_pc_str = {ip: v.get('frame_id') for ip, v in per_pc.items()}
                    lag_str = "-"
                    if cur_fid > 0:
                        per_pc_lag = {ip: cur_fid - int(v.get('frame_id', 0)) for ip, v in per_pc.items()}
                        tracker_lag = cur_fid - last_fid_yielded if last_fid_yielded > 0 else -1
                        lag_str = f"current={cur_fid} tracker_lag={tracker_lag} per_pc_lag={per_pc_lag}"
                    print(f"[track-worker:watchdog] received={received} success={success} "
                          f"({rate:.1f}%) fails[{fail_str}] inflight={n_inflight} "
                          f"per_pc_last_fid={per_pc_str} {lag_str}")
            threading.Thread(target=_watchdog, daemon=True).start()
            try:
                for fid, pose, info in tracker.track(init_pose_world):
                    if n == 0:
                        print(f"[track-worker] first yield: fid={fid}")
                    pose_robot = r2c @ pose
                    np.save(poses_dir / f"{int(fid):08d}.npy", pose_robot)
                    pose_log.append({
                        "frame_id": int(fid),
                        "wall_ts": time.time(),
                        "n_inliers": int(info.get("n_inliers", 0)),
                        "mean_residual_mm": float(info.get("mean_residual_mm", -1)),
                        "pose_robot": pose_robot.tolist(),
                        "pose_world": pose.tolist(),
                    })
                    if obj_frame is not None:
                        try:
                            _set_pose_handle(obj_frame, pose_robot)
                        except Exception:
                            pass
                    n += 1
                    if n % 30 == 0:
                        fps = n / max(time.perf_counter() - t_start, 1e-6)
                        print(f"[track] frame {fid}  n={n}  fps={fps:.1f}  "
                              f"inl={info.get('n_inliers', 0)}  "
                              f"resid_mm={info.get('mean_residual_mm', -1):.2f}")
                    if args.max_frames > 0 and n >= args.max_frames:
                        print(f"[track] reached --max-frames {args.max_frames}")
                        break
                    if keyboard.stop_event.is_set():
                        break
            except Exception as exc:
                print(f"[track-worker] EXCEPTION: {type(exc).__name__}: {exc}")
                import traceback; traceback.print_exc()
            print(f"[track-worker] _run exiting, n_frames={n}")

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        # 8) Wait for 'q' or worker exit.
        while worker.is_alive() and not keyboard.stop_event.is_set():
            time.sleep(0.1)

        # Diagnose exit reason.
        if not worker.is_alive():
            print(f"[track] main loop exit: worker died (n_frames={len(pose_log)})")
        elif keyboard.stop_event.is_set():
            print(f"[track] main loop exit: user pressed 'q'")

        # 9) Stop gracefully.
        if tracker is not None:
            tracker._stop.set()
        worker.join(timeout=2.0)
        try:
            print(f"[track] sending stop to gotrack daemons...")
            cmd_track.send_command("stop", wait=False, cmd_info={})
        except Exception:
            pass

        # 10) Save log.
        with open(trial_dir / "track_log.json", "w") as f:
            json.dump({
                "obj": args.obj,
                "trial_ts": trial_ts,
                "n_frames": len(pose_log),
                "init_pose_world": init_pose_world.tolist(),
                "frames": pose_log,
            }, f, indent=2, default=str)
        print(f"[track] saved {len(pose_log)} frames -> {trial_dir}")

    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        # CmdTrack: close sockets only (do NOT broadcast 'exit' to keep daemons alive).
        if cmd_track is not None:
            try:
                for s in cmd_track.sockets.values():
                    s.close()
            except Exception:
                pass
        if orch is not None:
            try:
                orch.close()
            except Exception:
                pass
        if rcc is not None:
            if args.stop_stream_on_exit:
                try:
                    print("[stream] stopping camera stream...")
                    rcc.stop()
                except Exception:
                    pass
            try:
                rcc.end()
            except Exception:
                pass
        if timestamp_monitor is not None:
            for fn in (timestamp_monitor.stop, timestamp_monitor.end):
                try: fn()
                except Exception: pass
        if sync_generator is not None:
            for fn in (sync_generator.stop, sync_generator.end):
                try: fn()
                except Exception: pass
        keyboard.close()


if __name__ == "__main__":
    main()
