#!/usr/bin/env python3
"""Check the v8/Inspire assets required before collecting P2 grasps.

This does not initialise cameras or move a robot.  It checks precisely the
NAS paths that ``src/execution/run_auto.py`` resolves for P2:

    python src/demo/p2/preflight.py
    python src/execution/run_auto.py --obj apple --arm franka --hand inspire \\
        --grasp_version v8 --auto

The collection runner is intentionally not given basket targets: it collects
and labels grasps on the Charuco board.  The P2 basket routing configuration
in :mod:`src.demo.p2.protocol` is for the subsequent inference demo.  P2 is
currently collecting into ``v8``; candidate lookup is keyed by object, hand,
and grasp version only, never by arm type.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.p2.protocol import (  # noqa: E402
    P2_GRASP_VERSION,
    P2_HAND,
    P2_OBJECTS,
    P2_FOUNDPOSE_ROOT,
    P2_PROJECT_ROOT,
    P2Object,
    candidate_lookup_path,
    coverage_json_path,
    foundpose_repre_path,
)


DEFAULT_OBJECT_ROOT = Path.home() / "shared_data/object_processing"
DEFAULT_FOUNDPOSE_ROOT = P2_FOUNDPOSE_ROOT
DEFAULT_CANDIDATE_ROOT = P2_PROJECT_ROOT / "candidates"
DEFAULT_PROJECT_ROOT = P2_PROJECT_ROOT


@dataclass
class P2Readiness:
    object: str
    semantic_class: str
    basket: str
    tabletop_stems: list[str] = field(default_factory=list)
    coverage_stems: list[str] = field(default_factory=list)
    candidate_count: int = 0
    candidate_source: str = ""
    ready: bool = False
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _existing_tabletop_stems(root: Path, obj: str) -> list[str]:
    table_dir = root / obj / "processed_data/info/tabletop"
    try:
        return sorted(path.stem for path in table_dir.glob("*.npy"))
    except OSError:
        return []


def _archive_for(candidate_dir: Path) -> Path | None:
    for suffix in (".tar.gz", ".tgz", ".zip"):
        archive = candidate_dir.with_suffix(suffix)
        if _is_file(archive):
            return archive
    return None


def _coverage_document(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _record_path(candidate_dir: Path, grasp: dict) -> Path | None:
    try:
        return (candidate_dir / str(grasp["type"]) / str(grasp["sid"])
                / str(grasp["gid"]) / "wrist_se3.npy")
    except KeyError:
        return None


def _candidate_file_set(candidate_dir: Path) -> set[str]:
    """List a NAS candidate tree once for strict validation.

    Calling ``Path.is_file`` three times for every one of several thousand
    candidates is surprisingly slow on the NAS.  ``rg --files`` walks it once
    and gives us the exact same existence evidence.  Keep a pathlib fallback
    so the preflight remains usable on a minimal host.
    """
    if shutil.which("rg"):
        try:
            proc = subprocess.run(
                ["rg", "--files", str(candidate_dir)], text=True,
                capture_output=True, check=True,
            )
            return {
                str(Path(line).relative_to(candidate_dir))
                for line in proc.stdout.splitlines() if line
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        return {
            str(path.relative_to(candidate_dir))
            for path in candidate_dir.rglob("*") if path.is_file()
        }
    except OSError:
        return set()


def check_object(
    item: P2Object,
    *,
    object_root: Path,
    foundpose_root: Path,
    candidate_root: Path,
    project_root: Path,
    hand: str,
    version: str,
    verify_candidate_links: bool = False,
) -> P2Readiness:
    row = P2Readiness(object=item.name, semantic_class=item.semantic_class,
                      basket=item.basket)
    raw_mesh = object_root / item.name / "raw_mesh" / f"{item.name}.obj"
    planner_mesh = object_root / item.name / "processed_data/mesh/simplified.obj"
    repre = foundpose_repre_path(item.name, foundpose_root)
    for label, path in (("raw_mesh", raw_mesh), ("planner_mesh", planner_mesh),
                        ("foundpose_repre", repre)):
        if not _is_file(path):
            row.missing.append(label)

    row.tabletop_stems = _existing_tabletop_stems(object_root, item.name)
    if not row.tabletop_stems:
        row.missing.append("tabletop_poses")

    # Grasp lookup intentionally excludes arm.  The selected hand grasp is
    # subsequently screened by the arm-specific planner at execution time.
    candidate_dir = candidate_lookup_path(item.name, hand, version, candidate_root)
    archive = _archive_for(candidate_dir)
    direct_candidates = candidate_dir.is_dir()
    if direct_candidates:
        row.candidate_source = str(candidate_dir)
    elif archive is not None:
        # Read-only inference can extract a NAS archive to /tmp on demand, but
        # collection writes candidate ``result.json`` records back beside the
        # candidate itself.  Require the normal expanded NAS hierarchy here.
        row.candidate_source = str(archive)
        row.missing.append("expanded_grasp_candidates_for_collection")
    else:
        row.missing.append("grasp_candidates")

    coverage_file = coverage_json_path(item.name, version, project_root)
    coverage = _coverage_document(coverage_file) if _is_file(coverage_file) else None
    if coverage is None:
        row.missing.append("coverage_json")
    else:
        if coverage.get("object") != item.name:
            row.missing.append("coverage_object_mismatch")
        scenes = coverage.get("scenes")
        grasps = coverage.get("grasps")
        if not isinstance(scenes, list) or not isinstance(grasps, list):
            row.missing.append("coverage_schema")
        else:
            row.coverage_stems = sorted({str(scene.get("pose_idx", ""))
                                         for scene in scenes if scene.get("pose_idx") is not None})
            absent = sorted(set(row.tabletop_stems) - set(row.coverage_stems))
            stale = sorted(set(row.coverage_stems) - set(row.tabletop_stems))
            if absent:
                row.missing.append("coverage_for_tabletop=" + ",".join(absent))
            if stale:
                row.warnings.append("stale_coverage_tabletop=" + ",".join(stale))
            if not grasps:
                row.missing.append("coverage_grasps")
            else:
                # ``run_auto`` selects candidates by these coverage keys, so
                # the coverage grasp count is the relevant operational count
                # (rather than unrelated files left under the candidate tree).
                row.candidate_count = len(grasps)
                if direct_candidates:
                    paths = [_record_path(candidate_dir, grasp) for grasp in grasps]
                    # NFS metadata for thousands of candidates can take longer
                    # than a robot's operator-preflight window.  The default
                    # checks representative coverage keys; the strict option
                    # verifies every selectable wrist + both hand poses.
                    if not verify_candidate_links and len(paths) > 3:
                        paths = [paths[0], paths[len(paths) // 2], paths[-1]]
                    if verify_candidate_links:
                        files = _candidate_file_set(candidate_dir)
                        missing_records = sum(
                            wrist is None or any(
                                str(path.relative_to(candidate_dir)) not in files
                                for path in (
                                    wrist,
                                    wrist.parent / "pregrasp_pose.npy",
                                    wrist.parent / "grasp_pose.npy",
                                )
                            )
                            for wrist in paths
                        )
                    else:
                        missing_records = sum(
                            wrist is None or not _is_file(wrist)
                            for wrist in paths
                        )
                    if missing_records:
                        row.missing.append(f"coverage_candidate_links({missing_records})")

    # A material-less OBJ remains runnable, but it is not an acceptable claim
    # of textured perception.  Expose this separately rather than blocking the
    # collection command, which only needs a mesh-frame-consistent repre.pth.
    if item.name == "pringles":
        mtl_name = next((line.split(maxsplit=1)[1].strip() for line in raw_mesh.read_text(errors="ignore").splitlines()
                         if line.startswith("mtllib ") and len(line.split(maxsplit=1)) == 2), None) if _is_file(raw_mesh) else None
        if not mtl_name or not _is_file(raw_mesh.parent / mtl_name):
            row.warnings.append("raw_mesh_has_uv_but_no_resolved_texture_material")

    row.ready = not row.missing
    return row


def build_report(
    items: Iterable[P2Object] = P2_OBJECTS,
    *,
    object_root: Path = DEFAULT_OBJECT_ROOT,
    foundpose_root: Path = DEFAULT_FOUNDPOSE_ROOT,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    hand: str = P2_HAND,
    version: str = P2_GRASP_VERSION,
    verify_candidate_links: bool = False,
) -> list[P2Readiness]:
    return [check_object(item, object_root=object_root,
                         foundpose_root=foundpose_root,
                         candidate_root=candidate_root,
                         project_root=project_root, hand=hand,
                         version=version,
                         verify_candidate_links=verify_candidate_links)
            for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--foundpose-root", type=Path, default=DEFAULT_FOUNDPOSE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="AutoDex NAS root containing experiment/, candidates/, and foundpose_assets/")
    parser.add_argument("--hand", default=P2_HAND)
    parser.add_argument("--version", default=P2_GRASP_VERSION)
    parser.add_argument("--verify-candidate-links", action="store_true",
                        help="check every coverage-selected candidate record; slower on NAS")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = build_report(object_root=args.object_root.expanduser(),
                        foundpose_root=args.foundpose_root.expanduser(),
                        candidate_root=args.candidate_root.expanduser(),
                        project_root=args.project_root.expanduser(),
                        hand=args.hand, version=args.version,
                        verify_candidate_links=args.verify_candidate_links)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(f"{'object':<10} {'class':<9} {'basket':<6} {'cand':>5}  status")
        for row in rows:
            status = "READY" if row.ready else "MISSING: " + ", ".join(row.missing)
            print(f"{row.object:<10} {row.semantic_class:<9} {row.basket:<6} "
                  f"{row.candidate_count:>5}  {status}")
            for warning in row.warnings:
                print(f"  warning: {warning}")
    incomplete = [row.object for row in rows if not row.ready]
    if incomplete:
        raise SystemExit("P2 assets incomplete: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
