#!/usr/bin/env python3
"""Run batch_object_overlay with local conda path auto-detection.

The original batch script is shared with older machines and keeps its
interpreter paths hard-coded. This wrapper lets the episode scheduler use the
same script without editing it.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = REPO_ROOT / "src" / "process" / "batch_object_overlay.py"
BF16_SITECUSTOMIZE = REPO_ROOT / "scripts" / "gotrack_bf16_sitecustomize"


def _first_existing(env_keys: tuple[str, ...], candidates: tuple[Path, ...]) -> Path:
    for key in env_keys:
        value = os.environ.get(key)
        if value and Path(value).expanduser().exists():
            return Path(value).expanduser()
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()
    missing = ", ".join(str(p.expanduser()) for p in candidates)
    raise FileNotFoundError(f"none of the candidate interpreters exist: {missing}")


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("autodex_batch_object_overlay", BATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepend_ld_library_path(*dirs: Path) -> None:
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    prepend = [str(d) for d in dirs if d.is_dir() and str(d) not in existing]
    if prepend:
        os.environ["LD_LIBRARY_PATH"] = ":".join(prepend + existing)


def _nccl_dirs_for_python(python_bin: Path) -> list[Path]:
    env_root = python_bin.parent.parent
    return [
        path
        for path in env_root.glob("lib/python*/site-packages/nvidia/nccl/lib")
        if (path / "libnccl.so.2").exists()
    ]


def _is_gotrack_tracking_command(cmd) -> bool:
    return any(str(part).endswith("run_multiview_gotrack_anchor_online.py") for part in list(cmd))


class _BatchSubprocessProxy:
    def __init__(self, module, gotrack_forward_precision: str, max_refinement_batch_size: str):
        self._module = module
        self._gotrack_forward_precision = gotrack_forward_precision
        self._max_refinement_batch_size = max_refinement_batch_size

    def __getattr__(self, name):
        return getattr(self._module, name)

    def Popen(self, cmd, *args, **kwargs):
        if _is_gotrack_tracking_command(cmd):
            env = dict(os.environ)
            env.update(kwargs.get("env") or {})
            if self._max_refinement_batch_size:
                env.setdefault(
                    "AUTODEX_GOTRACK_MAX_REFINEMENT_BATCH_SIZE",
                    self._max_refinement_batch_size,
                )
            if self._gotrack_forward_precision == "bf16":
                existing = env.get("PYTHONPATH", "")
                paths = [str(BF16_SITECUSTOMIZE)]
                if existing:
                    paths.append(existing)
                env["PYTHONPATH"] = os.pathsep.join(paths)
                env["AUTODEX_GOTRACK_BF16_XFORMERS"] = "1"
                env.setdefault("AUTODEX_GOTRACK_BF16_REQUIRED", "1")
                print("[env] enabled GoTrack BF16 xformers shim", flush=True)
            kwargs["env"] = env
        return self._module.Popen(cmd, *args, **kwargs)


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "episode"


class _ProgressPrinter:
    def __init__(self, stage: str):
        self.stage = stage
        self.total = None
        self.current = 0

    def reset(self, total=None):
        self.total = total
        self.current = 0

    def set_postfix_str(self, *_args, **_kwargs):
        return None

    def update(self, delta):
        self.current += int(delta)
        total = int(self.total or max(self.current, 1))
        pct = int((100 * self.current / total) if total else 0)
        print(f"frames: {pct}%|direct| {self.current}/{total} {self.stage}", flush=True)


def _parse_direct_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--overlay-output-dir", required=True)
    parser.add_argument("--cache-key", default=None)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--ep", required=True)
    parser.add_argument("--track-cams", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run_direct_episode(batch, argv: list[str]) -> int:
    args = _parse_direct_args(argv)
    nas_ep = Path(args.episode_dir).expanduser()
    overlay_out = Path(args.overlay_output_dir).expanduser()
    if not nas_ep.is_dir():
        raise FileNotFoundError(f"episode dir not found: {nas_ep}")
    videos_dir = nas_ep / "videos"
    serials = sorted(p.stem for p in videos_dir.glob("*.avi")) if videos_dir.is_dir() else []
    if not serials:
        raise FileNotFoundError(f"no episode videos found: {videos_dir}")
    print(f"=== direct episode {args.hand}/{args.obj}/{args.ep} cams={len(serials)} ===", flush=True)
    print(f"[direct] episode_dir={nas_ep}", flush=True)
    print(f"[direct] overlay_output_dir={overlay_out}", flush=True)
    if args.dry_run:
        print("[direct] dry_run_done", flush=True)
        return 0

    cache_key = _safe_cache_key(args.cache_key or str(nas_ep))
    local_ep = batch.LOCAL_CACHE / "direct" / cache_key
    local_gt_out = batch.LOCAL_CACHE / "gt_output" / "direct" / cache_key / "gotrack_output"
    local_overlay_out = batch.LOCAL_CACHE / "overlay_output" / "direct" / cache_key
    nas_gt_out = nas_ep / batch.GOTRACK_REL

    do_gt = not batch.gotrack_done(nas_ep)
    do_ov = not batch.overlay_done(overlay_out, serials)
    if not (do_gt or do_ov):
        print("[direct] skip existing outputs", flush=True)
        return 0

    if not batch._is_videos_cached(local_ep / "videos"):
        print(f"[direct] downloading videos: {nas_ep}", flush=True)
        batch.download_episode(nas_ep, local_ep)

    try:
        if do_gt:
            ok = batch.run_gotrack(
                local_ep,
                nas_ep,
                args.obj,
                local_gt_out,
                track_cams=args.track_cams,
                frame_pbar=_ProgressPrinter("gotrack"),
            )
            if ok:
                batch.upload_dir(local_gt_out, nas_gt_out)
                shutil.rmtree(local_gt_out, ignore_errors=True)
            else:
                return 2

        if do_ov and batch.gotrack_done(nas_ep):
            ok = batch.run_overlay(
                local_ep,
                nas_ep,
                args.obj,
                local_overlay_out,
                frame_pbar=_ProgressPrinter("overlay"),
            )
            if ok:
                batch.upload_dir(local_overlay_out, overlay_out)
                shutil.rmtree(local_overlay_out, ignore_errors=True)
            else:
                return 3

        return 0
    finally:
        shutil.rmtree(local_ep, ignore_errors=True)


def main() -> int:
    home = Path.home()
    batch = _load_batch_module()
    batch.GOTRACK_PY = _first_existing(
        ("AUTODEX_GOTRACK_PY", "GOTRACK_PY"),
        (
            batch.GOTRACK_PY,
            home / "anaconda3" / "envs" / "gotrack_cu128" / "bin" / "python",
            home / "anaconda3" / "envs" / "gotrack" / "bin" / "python",
            home / "miniconda3" / "envs" / "gotrack" / "bin" / "python",
        ),
    )
    batch.FPOSE_PY = _first_existing(
        ("AUTODEX_FPOSE_PY", "FPOSE_PY"),
        (
            home / "anaconda3" / "envs" / "gotrack_cu128" / "bin" / "python",
            home / "anaconda3" / "envs" / "paradex" / "bin" / "python",
            home / "anaconda3" / "envs" / "planner" / "bin" / "python",
            home / "anaconda3" / "envs" / "foundationpose" / "bin" / "python",
            batch.FPOSE_PY,
            home / "miniconda3" / "envs" / "foundationpose" / "bin" / "python",
        ),
    )
    _prepend_ld_library_path(*_nccl_dirs_for_python(batch.GOTRACK_PY), *_nccl_dirs_for_python(batch.FPOSE_PY))
    gotrack_forward_precision = os.environ.get("AUTODEX_GOTRACK_FORWARD_PRECISION", "bf16").strip().lower()
    max_refinement_batch_size = os.environ.get("AUTODEX_GOTRACK_MAX_REFINEMENT_BATCH_SIZE", "8").strip()
    batch.subprocess = _BatchSubprocessProxy(
        batch.subprocess,
        gotrack_forward_precision,
        max_refinement_batch_size,
    )
    print(f"[env] GOTRACK_PY={batch.GOTRACK_PY}", flush=True)
    print(f"[env] FPOSE_PY={batch.FPOSE_PY}", flush=True)
    print(f"[env] AUTODEX_GOTRACK_FORWARD_PRECISION={gotrack_forward_precision}", flush=True)
    print(f"[env] AUTODEX_GOTRACK_MAX_REFINEMENT_BATCH_SIZE={max_refinement_batch_size}", flush=True)
    print(f"[env] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
    if "--episode-dir" in sys.argv[1:]:
        return _run_direct_episode(batch, sys.argv[1:])
    batch.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
