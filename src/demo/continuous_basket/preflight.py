#!/usr/bin/env python3
"""Validate that every fixed-catalogue object is ready before robot motion.

Example:

    python src/demo/continuous_basket/preflight.py \
      --objects banana wood_organizer='wood organizer' --hand inspire --version v8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# ``python src/demo/.../preflight.py`` puts only this directory on sys.path.
# Keep the documented direct invocation usable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.demo.continuous_basket.catalog import CatalogObject, parse_catalog


DEFAULT_ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
DEFAULT_ANCHOR_ROOT = (Path(__file__).resolve().parents[3]
                       / "autodex/perception/thirdparty/MV-GoTrack/anchor_banks")


def _default_object_root(version: str) -> Path:
    # Keep this preflight dependency-light: importing autodex.utils.path pulls
    # trimesh and ParaDex even though checking filesystem readiness needs
    # neither. This mirrors get_obj_root()'s v8 convention.
    return (Path.home() / "shared_data/object_processing" if version == "v8"
            else Path.home() / "shared_data/AutoDex/object/paradex")


def _default_candidate_root(hand: str, version: str) -> Path:
    return Path.home() / "shared_data/AutoDex/candidates" / hand / version


@dataclass(frozen=True)
class ObjectReadiness:
    name: str
    mesh: str
    foundpose_repre: str
    anchor_bank: str
    candidate_count: int
    runnable_candidate_count: int
    successful_candidate_count: int
    ready: bool
    missing: List[str]


def _successful_candidate_count(paths: Iterable[Path], arm: Optional[str]) -> int:
    """Count executable, arm-compatible candidate records marked successful.

    Candidate pools are shared by hand, while results distinguish FR3 and
    xArm.  A successful xArm grasp must not let an FR3 demo pass preflight.
    Historical records lacking ``arm`` were xArm runs, matching AutoDex's
    existing coverage reader.
    """
    count = 0
    for wrist_path in paths:
        if not (wrist_path.parent / "pregrasp_pose.npy").is_file():
            continue
        result_path = wrist_path.parent / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text())
            if (result.get("success") is True
                    and (arm is None or str(result.get("arm", "xarm")) == arm)):
                count += 1
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return count


def check_object(
    item: CatalogObject,
    *,
    object_root: Path,
    assets_base: Path,
    candidate_root: Path,
    anchor_root: Path,
    require_gotrack: bool,
    arm: Optional[str] = None,
) -> ObjectReadiness:
    mesh = object_root / item.name / "raw_mesh" / f"{item.name}.obj"
    repre = assets_base / item.name / "object_repre" / "v1" / item.name / "1" / "repre.pth"
    anchor = anchor_root / f"{item.name}.npz"
    wrists = sorted((candidate_root / item.name).glob("**/wrist_se3.npy"))
    runnable_count = sum(
        (wrist.parent / "pregrasp_pose.npy").is_file() for wrist in wrists
    )
    success_count = _successful_candidate_count(wrists, arm)
    missing: List[str] = []
    if not mesh.is_file():
        missing.append("mesh")
    if not repre.is_file():
        missing.append("foundpose_repre")
    if not wrists:
        missing.append("grasp_candidates")
    elif runnable_count == 0:
        missing.append("runnable_grasp_candidates")
    if success_count == 0:
        missing.append("successful_grasp")
    if require_gotrack and not anchor.is_file():
        missing.append("gotrack_anchor_bank")
    return ObjectReadiness(
        name=item.name, mesh=str(mesh), foundpose_repre=str(repre), anchor_bank=str(anchor),
        candidate_count=len(wrists), runnable_candidate_count=runnable_count,
        successful_candidate_count=success_count,
        ready=not missing, missing=missing,
    )


def build_report(
    catalogue: Iterable[CatalogObject],
    *,
    object_root: Path,
    assets_base: Path,
    candidate_root: Path,
    anchor_root: Path,
    require_gotrack: bool,
    arm: Optional[str] = None,
) -> List[ObjectReadiness]:
    return [check_object(item, object_root=object_root, assets_base=assets_base,
                         candidate_root=candidate_root, anchor_root=anchor_root,
                         require_gotrack=require_gotrack, arm=arm)
            for item in catalogue]


def require_ready(rows: Sequence[ObjectReadiness]) -> None:
    """Raise one concise error for all catalogue entries that cannot run."""
    incomplete = [row for row in rows if not row.ready]
    if not incomplete:
        return
    details = "; ".join(
        f"{row.name}: {', '.join(row.missing)}" for row in incomplete
    )
    raise RuntimeError(f"catalogue readiness check failed ({details})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True,
                        help="object_name or object_name=YOLO-E prompt")
    parser.add_argument("--hand", default="inspire")
    parser.add_argument("--arm", default="franka", choices=["franka", "xarm"],
                        help="require successful grasp records for this robot arm")
    parser.add_argument("--version", default="v8")
    parser.add_argument("--assets-base", default=str(DEFAULT_ASSETS_BASE))
    parser.add_argument("--anchor-root", default=str(DEFAULT_ANCHOR_ROOT))
    parser.add_argument("--object-root", default=None,
                        help="override object root; defaults to candidate-version asset root")
    parser.add_argument("--candidate-root", default=None,
                        help="override candidates/{hand}/{version} root")
    parser.add_argument("--no-gotrack", action="store_true",
                        help="allow FoundPose-only fallback without anchor banks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    object_root = (Path(args.object_root).expanduser()
                   if args.object_root else _default_object_root(args.version))
    candidate_root = (Path(args.candidate_root).expanduser()
                      if args.candidate_root else _default_candidate_root(args.hand, args.version))
    rows = build_report(
        parse_catalog(args.objects), object_root=object_root,
        assets_base=Path(args.assets_base).expanduser(), candidate_root=candidate_root,
        anchor_root=Path(args.anchor_root).expanduser(), require_gotrack=not args.no_gotrack,
        arm=args.arm,
    )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(f"{'object':<24} {'cand':>5} {'run':>5} {'success':>8}  status")
        for row in rows:
            status = "READY" if row.ready else "MISSING: " + ", ".join(row.missing)
            print(f"{row.name:<24} {row.candidate_count:>5} {row.runnable_candidate_count:>5} "
                  f"{row.successful_candidate_count:>8}  {status}")
    try:
        require_ready(rows)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
