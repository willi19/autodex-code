import os
import random
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

# Scenes are hand-specific: the obstacle gap (wall/shelf/box clearance) is
# adapted per hand, so allegro and inspire scenes for the same object differ.
# They live under the AutoDex NAS, keyed by hand first so different hands never
# overwrite each other: {project_dir}/scene/{hand}/{obj}/{scene_type}/{id}.json
scene_root = os.path.join(project_dir, "scene")


def get_candidate_path(hand: str = "allegro") -> str:
    return os.path.join(project_dir, "candidates", hand)


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
                    candidate_order=None):
    """Load all grasp candidates under ``{candidates}/{hand}/{version}/{obj}``.

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

    candidate_obj_path = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(candidate_obj_path):
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
