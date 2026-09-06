import os
import random
import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import trimesh

home_path = os.path.expanduser("~")
code_path = os.path.join(home_path, "RSS_2026")
shared_dir = os.path.join(home_path, "shared_data")
project_dir = os.path.join(shared_dir, "AutoDex")
bodex_path = os.path.join(code_path, "BODex_outputs")
repo_dir = os.path.join(home_path, "AutoDex")
candidate_path = os.path.join(project_dir, "candidates", "allegro")  # default, use get_candidate_path() for other hands

robot_configs_path = os.path.join(project_dir, "content", "configs", "robot")
obj_path = os.path.join(project_dir, "object", "paradex")
urdf_path = os.path.join(project_dir, "content", "assets", "robot", "allegro_description")

# v8 scenes + grasps were generated against the object_processing asset tree,
# whose tabletop set and simplified mesh differ from the older paradex tree
# (e.g. attached_container: paradex {000,001,006,009,016} vs op
# {000,001,002,007,010}). Mixing the two mislabels pose_idx, so every
# mesh/tabletop lookup on a v8 pool must resolve against object_processing.
object_processing_path = os.path.join(shared_dir, "object_processing")

# Candidate-pool versions whose meshes/tabletops live in object_processing.
OP_VERSIONS = {"v8"}


def get_obj_root(version: str = None) -> str:
    """Asset root (mesh + processed_data/info/tabletop) for a candidate pool.

    ``v8`` resolves to ``object_processing``; everything else (v7,
    selected_100, table_only, reset, ...) keeps the legacy ``obj_path``.
    Pass ``None`` to get the legacy root.
    """
    return object_processing_path if version in OP_VERSIONS else obj_path

# Scenes are hand-specific: the obstacle gap (wall/shelf/box clearance) is
# adapted per hand, so allegro and inspire scenes for the same object differ.
# They live under the AutoDex NAS, keyed by hand first so different hands never
# overwrite each other: {project_dir}/scene/{hand}/{obj}/{scene_type}/{id}.json
scene_root = os.path.join(project_dir, "scene")


def get_candidate_path(hand: str = "allegro") -> str:
    return os.path.join(project_dir, "candidates", hand)


RESET_RELEASE_HEIGHTS_CM = (0, 4, 8, 12)


def get_reset_candidate_root(hand: str, h_cm: int, *, version: str = "v8") -> str:
    """Candidate root for one reorientation release-height stage.

    ``reset_0``, ``reset_4``, ``reset_8`` and ``reset_12`` name the release
    height in centimetres above the floor — they are not dataset versions.
    v8 identifies the object mesh/tabletop asset contract and is enforced
    independently from this directory naming.
    """
    if version != "v8":
        raise ValueError(
            f"reset/reorient supports only the v8 asset contract, got {version!r}")
    if h_cm not in RESET_RELEASE_HEIGHTS_CM:
        raise ValueError(
            f"unsupported reset release height {h_cm!r}; expected one of "
            f"{RESET_RELEASE_HEIGHTS_CM}")
    return os.path.join(get_candidate_path(hand), f"reset_{h_cm}")


def iter_reset_candidate_roots(hand: str, *, version: str = "v8"):
    """Yield v8 reset roots from lowest to highest release height."""
    for h_cm in RESET_RELEASE_HEIGHTS_CM:
        yield h_cm, get_reset_candidate_root(hand, h_cm, version=version)


def _safe_archive_destination(root: Path, member_name: str) -> Path:
    """Return a safe extraction destination for one archive member.

    Candidate archives are NAS supplied data, so do not delegate path handling
    to ``extractall``: a malformed member such as ``../../...`` must never be
    able to write outside the local cache.
    """
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe candidate archive member: {member_name!r}")
    return root.joinpath(*relative.parts)


def _extract_candidate_archive(archive_path: Path, obj_name: str) -> str:
    """Extract a read-only NAS candidate archive to a versioned local cache."""
    stat = archive_path.stat()
    identity = (f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
    cache_key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    cache_base = Path(os.environ.get(
        "AUTODEX_CANDIDATE_CACHE", "/tmp/autodex_candidate_archives"))
    cache_dir = cache_base / cache_key
    cached_obj = cache_dir / obj_name
    if cached_obj.is_dir():
        return str(cached_obj)

    cache_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{cache_key}.", dir=cache_base))
    try:
        suffixes = archive_path.suffixes
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    destination = _safe_archive_destination(tmp_dir, member.filename)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, open(destination, "wb") as out:
                        shutil.copyfileobj(source, out)
        elif suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.lower() == ".tgz":
            with tarfile.open(archive_path, "r:*") as archive:
                for member in archive.getmembers():
                    destination = _safe_archive_destination(tmp_dir, member.name)
                    # Candidate trees never need links or device nodes. Refuse
                    # them rather than creating an unexpected local reference.
                    if member.issym() or member.islnk() or member.isdev():
                        raise ValueError(
                            f"unsupported candidate archive member: {member.name!r}")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(
                            f"cannot read candidate archive member: {member.name!r}")
                    with source, open(destination, "wb") as out:
                        shutil.copyfileobj(source, out)
        else:
            raise ValueError(f"unsupported candidate archive format: {archive_path}")

        extracted_obj = tmp_dir / obj_name
        if not extracted_obj.is_dir():
            raise ValueError(
                f"candidate archive {archive_path} does not contain {obj_name!r}/")
        try:
            os.rename(tmp_dir, cache_dir)
        except FileExistsError:
            # A concurrent demo populated this immutable cache first.
            shutil.rmtree(tmp_dir)
        if not cached_obj.is_dir():
            raise RuntimeError(f"candidate archive cache incomplete: {archive_path}")
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    print(f"[candidates] using NAS archive cache: {archive_path.name}")
    return str(cached_obj)


def resolve_candidate_object_path(root: str | Path, version: str,
                                  obj_name: str) -> str | None:
    """Return an extracted candidate object directory, accepting NAS archives.

    ``root`` is the hand-specific candidate root (for example
    ``.../candidates/inspire``); ``version`` is normally ``v8``. This public
    resolver also lets success-library loaders recover the exact original
    candidate from an archive rather than falling back to approximated data.
    """
    candidate_obj_path = Path(root) / version / obj_name
    if candidate_obj_path.is_dir():
        return str(candidate_obj_path)
    for archive_path in (
        candidate_obj_path.with_suffix(".tar.gz"),
        candidate_obj_path.with_suffix(".tgz"),
        candidate_obj_path.with_suffix(".zip"),
    ):
        if archive_path.is_file():
            return _extract_candidate_archive(archive_path, obj_name)
    return None


def get_scene_dir(hand: str, obj_name: str, scene_type: str = None) -> str:
    """Directory holding an object's scene JSONs for a given hand.

    ``{project_dir}/scene/{hand}/{obj}[/{scene_type}]``. Pass ``scene_type=None``
    to get the per-object root (whose subdirs are the scene types).
    """
    base = os.path.join(scene_root, hand, obj_name)
    return base if scene_type is None else os.path.join(base, scene_type)


def get_object_mesh(obj_name):
    mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_candidate(obj_name, obj_pose, version, shuffle=True, skip_done=True,
                    success_only=False, hand="allegro", scene_id=None,
                    scene_type_filter=None,
                    skip_scenes_with_success=False,
                    tabletop_pose_stem=None,
                    candidate_order=None,
                    candidates_root=None):
    """Load all grasp candidates under ``{candidates}/{hand}/{version}/{obj}``.

    If NAS stores the object pool as ``{obj}.tar.gz``, ``{obj}.tgz`` or
    ``{obj}.zip`` instead of an expanded directory, it is extracted once to
    ``$AUTODEX_CANDIDATE_CACHE`` (``/tmp/autodex_candidate_archives`` by
    default) and then loaded identically. The read-only NAS is never modified.

    Supports both layouts (auto-detected by walking until ``wrist_se3.npy`` is found):
        nested: ``{obj}/{scene_type}/{scene_id}/{grasp_idx}/wrist_se3.npy``
        flat:   ``{obj}/{scene_id}/{grasp_idx}/wrist_se3.npy``

    In the flat case the returned scene_info has ``scene_type=""``.

    If ``scene_id`` is given, only grasps whose dir name matches are kept.

    If ``scene_type_filter`` is given, only grasps under that scene_type subdir
    are kept (e.g. ``"wall"`` for v7 wall scenes). Use ``""`` to keep only flat
    layout candidates. ``None`` keeps everything.
    """
    wrist_se3_list = []
    pregrasp_pose_list = []
    grasp_pose_list = []
    scene_info = []

    root = candidates_root or get_candidate_path(hand)
    candidate_obj_path = resolve_candidate_object_path(root, version, obj_name)
    if candidate_obj_path is None:
        return np.empty((0, 4, 4)), np.empty((0, 0)), np.empty((0, 0)), []

    # Walk to find every grasp dir (one containing wrist_se3.npy).
    grasp_dirs = []
    for dirpath, dirnames, filenames in os.walk(candidate_obj_path):
        if "wrist_se3.npy" in filenames:
            grasp_dirs.append(dirpath)
            dirnames[:] = []  # don't descend further

    if candidate_order is not None:
        # Filter+sort by explicit priority list. Drop grasp dirs not in the
        # list (caller's whitelist).
        rank = {(str(t), str(s), str(g)): i
                for i, (t, s, g) in enumerate(candidate_order)}
        def _key(d):
            rel = os.path.relpath(d, candidate_obj_path)
            parts = rel.split(os.sep)
            if len(parts) == 3:
                tup = (parts[0], parts[1], parts[2])
            elif len(parts) == 2:
                tup = ("", parts[0], parts[1])
            else:
                return None
            return rank.get(tup)
        grasp_dirs = [d for d in grasp_dirs if _key(d) is not None]
        grasp_dirs.sort(key=_key)
    elif shuffle:
        random.shuffle(grasp_dirs)
    else:
        grasp_dirs.sort()

    # Pre-compute scenes (scene_type, scene_id_dir) that have any successful
    # grasp, so we can drop ALL grasps in those scenes (user policy: once a
    # scene has succeeded, don't re-attempt it).
    done_scenes = set()
    if skip_scenes_with_success:
        import json as _json
        for base in grasp_dirs:
            rel = os.path.relpath(base, candidate_obj_path)
            parts = rel.split(os.sep)
            if len(parts) == 3:
                st, sid, _ = parts
            elif len(parts) == 2:
                st = ""; sid = parts[0]
            else:
                continue
            rp = os.path.join(base, "result.json")
            if os.path.exists(rp):
                try:
                    with open(rp) as f:
                        if _json.load(f).get("success", False):
                            done_scenes.add((st, sid))
                except Exception:
                    pass

    for base in grasp_dirs:
        rel = os.path.relpath(base, candidate_obj_path)
        parts = rel.split(os.sep)
        if len(parts) == 3:
            scene_type, scene_id_dir, grasp_idx = parts
        elif len(parts) == 2:
            scene_type = ""
            scene_id_dir, grasp_idx = parts
        else:
            # Unexpected depth — skip.
            continue

        if scene_id is not None and scene_id_dir != scene_id:
            continue
        if scene_type_filter is not None and scene_type != scene_type_filter:
            continue
        if (scene_type, scene_id_dir) in done_scenes:
            continue
        if tabletop_pose_stem is not None and scene_type:
            scene_json = os.path.join(
                get_scene_dir(hand, obj_name, scene_type), f"{scene_id_dir}.json"
            )
            if not os.path.exists(scene_json):
                continue
            try:
                import json as _json
                with open(scene_json) as _f:
                    meta = _json.load(_f).get("meta", {})
                if str(meta.get("pose_idx", "")) != tabletop_pose_stem:
                    continue
            except Exception:
                continue

        result_path = os.path.join(base, "result.json")
        has_result = os.path.exists(result_path)
        if success_only:
            if not has_result:
                continue
            import json
            with open(result_path) as f:
                if not json.load(f).get("success", False):
                    continue
        elif skip_done and has_result:
            # Only skip if PRIOR result was a success — failed attempts
            # should remain available for retry (charuco/place fails can
            # be transient or fixable by re-running, while a success
            # genuinely means the scene is covered and no point repeating).
            try:
                import json as _json
                with open(result_path) as _f:
                    if _json.load(_f).get("success", False):
                        continue
            except Exception:
                continue

        pregrasp = np.load(os.path.join(base, "pregrasp_pose.npy"))
        pregrasp_pose_list.append(pregrasp)
        grasp_file = os.path.join(base, "grasp_pose.npy")
        grasp_pose_list.append(np.load(grasp_file) if os.path.exists(grasp_file) else pregrasp)
        wrist_se3_obj = np.load(os.path.join(base, "wrist_se3.npy"))
        wrist_se3_list.append(obj_pose @ wrist_se3_obj)
        scene_info.append((scene_type, scene_id_dir, grasp_idx))

    wrist_se3 = np.array(wrist_se3_list)
    grasp_pose = np.array(grasp_pose_list)
    pregrasp_pose = np.array(pregrasp_pose_list)

    return wrist_se3, pregrasp_pose, grasp_pose, scene_info


def load_openpose_for_candidates(obj_name, scene_info, hand, version, pose_stem):
    """Load ``openpose_{pose_stem}.npy`` for each candidate in ``scene_info``.

    ``pose_stem`` is the tabletop pose filename stem (e.g. ``"002"`` for the
    file ``002.npy`` under ``{obj}/processed_data/info/tabletop/``). For each
    grasp candidate, looks for the matching openpose file inside that
    candidate's directory and returns the (6,) finger configuration; missing
    files yield ``None``.

    The candidate's own scene (``scene_id_dir``) is GUARANTEED to have an
    openpose file matching the scene's start tabletop pose, so the typical
    usage is::

        scene_info = ik_res["scene_info"]
        pose_stem  = tb_before["filename"].replace(".npy", "")
        openpose   = load_openpose_for_candidates(obj, scene_info, hand,
                                                   version, pose_stem)

    Returns: list[Optional[np.ndarray (6,)]] of length len(scene_info).
    """
    cand_root = os.path.join(get_candidate_path(hand), version, obj_name)
    out = []
    for entry in scene_info:
        scene_type, scene_id_dir, grasp_idx = entry
        if scene_type:
            grasp_dir = os.path.join(cand_root, scene_type,
                                      scene_id_dir, str(grasp_idx))
        else:
            grasp_dir = os.path.join(cand_root, scene_id_dir, str(grasp_idx))
        fpath = os.path.join(grasp_dir, f"openpose_{pose_stem}.npy")
        out.append(np.load(fpath) if os.path.exists(fpath) else None)
    return out
