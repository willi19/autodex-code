"""Coverage viewer over EXECUTED grasps (from autodex_dataset), grouped by
(object, tabletop_pose), with obstacle collision.

Each capture trial (autodex_dataset/selected_100/{obj}/{ts}) is an executed
grasp. We recover it with the CURRENT urdf:
    joint qpos (raw/arm/position.npy) --FK--> link6 @ LINK6_TO_WRIST = wrist(robot)
    object(robot) = inv(C2R) @ pose_world
    grasp(object) = inv(object_robot) @ wrist(robot)
    fingers = raw/hand/position.npy at the grasp frame
The grasp frame is the last frame near min wrist-z (grasped, just before lift).

Pick object + tabletop pose; every v8 scene at that pose is tiled. The selected
EXECUTED grasp's hand is placed in each scene; object+hand are GREEN if that
grasp clears the obstacles, RED if it collides.

    conda activate mingi
    python src/visualization/coverage_grid_viewer.py
"""
import os
import sys
import json
import glob
import numpy as np
import trimesh
import yourdfpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_process"))
import scene_grid_viewer as sg  # noqa: E402
from paradex.visualization.visualizer.viser import ViserViewer  # noqa: E402
from autodex.utils.conversion import cart2se3  # noqa: E402
from autodex.utils.robot_config import ALLEGRO_LINK6_TO_WRIST as L6W  # noqa: E402

# scenes are generated fresh from the CURRENT tabletop (not read from possibly-
# stale on-disk JSONs whose pose_idx labels don't match the re-processed tabletop)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src", "grasp_generation"))
sys.path.insert(0, os.path.join(_REPO, "src", "scene_generation"))
sys.path.insert(0, os.path.join(_REPO, "src", "experiment", "reset"))
from tabletop_pose import _z_aligned_geodesic_deg as _zalign  # noqa: E402
from autodex.utils.path import get_scene_dir  # noqa: E402
HAND = "allegro"   # all executed grasps in the dataset are allegro

DS_ROOTS = [os.path.expanduser("~/shared_data/autodex_dataset/selected_100"),
            os.path.expanduser("~/shared_data/autodex_dataset/corl_selected_100")]
OBJROOT = os.path.expanduser("~/shared_data/object_processing")
SKIP_OBJS = set()   # keep apple (data intact); its high rot_err is just left as-is
ARM_URDF = os.path.expanduser("~/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf")
HAND_URDF = os.path.expanduser("~/shared_data/AutoDex/content/assets/robot/"
                               "allegro_description/allegro_hand_description_right.urdf")
THRESH_DEG = 30.0
GREEN = [70, 200, 90]     # the SELECTED executed grasp fits this scene
YELLOW = [235, 205, 55]   # some executed grasp (any at this pose) fits
RED = [214, 74, 92]       # no executed grasp fits
TABLE = [235, 235, 240]
OBST = [119, 136, 153]
SCENE_TYPES = ["wall", "shelf", "box"]
UNC_LABEL = "◇ UNCOVERABLE"   # captured resting pose has no matching tabletop pose
GREY = [120, 130, 140]

_arm = yourdfpy.URDF.load(ARM_URDF, load_meshes=False, build_collision_scene_graph=False)
_aj = _arm.actuated_joint_names[:6]
_hand = yourdfpy.URDF.load(HAND_URDF, load_meshes=True, build_collision_scene_graph=False)
_hj = _hand.actuated_joint_names


def paint(mesh, rgb, a=255):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(
        m, vertex_colors=np.tile(np.array(rgb + [a], np.uint8), (len(m.vertices), 1)))
    return m


def grav(T):
    return T[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def _fk_wrist(q6):
    _arm.update_cfg({_aj[i]: float(q6[i]) for i in range(6)})
    return _arm.get_transform("link6", "world") @ L6W


def _hand_mesh(wrist_obj, finger):
    _hand.update_cfg({_hj[i]: float(finger[i]) for i in range(min(len(_hj), len(finger)))})
    return _hand.scene.to_geometry().copy().apply_transform(wrist_obj)


def tabletop(obj):
    fs = sorted(glob.glob(f"{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy"))
    return {os.path.basename(f)[:-4]: grav(np.load(f)) for f in fs}


_EXEC_CACHE = {}
_SCENE_CACHE = {}


def extract_executed(obj):
    """Load the precomputed executed grasps ({trial}/executed_grasp/), one per trial.
    Built by build_exec_grasps.py + refine_exec.py:
      wrist_se3.npy = FK executed wrist (object frame), grasp_pose.npy = finger,
      meta.json has pose_id / scene_info / grasp_time / finger_L2."""
    if obj in _EXEC_CACHE:
        return _EXEC_CACHE[obj]
    out = []
    mps = []
    for DS in DS_ROOTS:
        mps += glob.glob(f"{DS}/{obj}/*/executed_grasp/meta.json")
    for mp in sorted(mps):
        d = os.path.dirname(mp)
        trial = os.path.dirname(d)
        try:
            m = json.load(open(mp))
            wrist = np.load(f"{d}/wrist_se3.npy")
            finger = np.load(f"{d}/grasp_pose.npy")
        except Exception:
            continue
        objT = None   # captured object pose in robot frame (inv(C2R) @ pose_world)
        try:
            objT = np.linalg.inv(np.load(f"{trial}/C2R.npy")) @ np.load(f"{trial}/pose_world.npy")
        except Exception:
            pass
        pid = m.get("pose_id")
        out.append({"mesh": _hand_mesh(wrist, finger),
                    "pose_id": (str(pid) if pid is not None else None),
                    "coverable": bool(m.get("coverable")), "objT": objT,
                    "rot_err": m.get("tabletop_rot_err"),
                    "ts": m.get("ts", ""), "scene_info": m.get("scene_info")})
    return out


def scenes_by_pose(obj):
    """Build the v8 deployment scenes with PROPER obstacle geometry and group them
    by tabletop pose_idx.

    Rather than the raw on-disk cuboids (box = just a raised floor, no walls), we
    rebuild each scene with the sg builders so obstacles render nicely:
      box   -> 4 side walls (box_front/back/left/right)
      shelf -> back / side / up panels (per its up/side/back flags)
      wall  -> single wall panel
    Geometry params come from each on-disk scene json's ``meta.param``
    (gap / z_rotation_deg / up-side-back / height_offset) + the canonical tabletop
    pose ({obj}/processed_data/info/tabletop/{pose_idx}.npy). No adaptive summary
    needed. Returns {pose_idx: [(scene_type, scene_dict)]}; reorient_*/poseless
    scenes skipped."""
    if obj in _SCENE_CACHE:
        return _SCENE_CACHE[obj]
    tt = {os.path.basename(f)[:-4]: np.load(f)
          for f in sorted(glob.glob(f"{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy"))}
    try:
        obb = json.load(open(f"{OBJROOT}/{obj}/processed_data/info/simplified.json"))
    except Exception:
        obb = None
    idx = {}
    for st in SCENE_TYPES:
        for f in sorted(glob.glob(f"{OBJROOT}/{obj}/scene/{st}/*.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            pose = str(d.get("meta", {}).get("pose_idx"))
            if pose in ("None", "") or pose not in tt:
                continue
            p = d.get("meta", {}).get("param", {})
            ttp = tt[pose]
            try:
                if st == "wall":
                    scene = sg.get_wall_scene(obj, ttp, obb,
                                              p.get("z_rotation_deg", 0.0), p.get("gap", 0.0))
                elif st == "shelf":
                    scene = sg.get_shelf_scene(obj, ttp, obb,
                                               p.get("z_rotation_deg", 0.0), p.get("gap", 0.0),
                                               up=p.get("up", True), side=p.get("side", True),
                                               back=p.get("back", True))
                elif st == "box":
                    scene = sg.get_box_scene(obj, ttp, p.get("height_offset", 0.0))
                else:
                    continue
            except Exception:
                continue
            if scene is None:
                continue
            idx.setdefault(pose, []).append((st, scene))
    _SCENE_CACHE[obj] = idx
    return idx


def _boxes(cuboids):
    # include table: after initial-frame reclassification a coverable grasp is
    # always above the table (world-z match), so the table never false-flags a
    # valid grasp; any table hit means a genuinely bad placement.
    b = []
    for name, c in cuboids.items():
        Tb = cart2se3(np.array(c["pose"], float))
        b.append((np.linalg.inv(Tb), np.array(c["dims"], float) / 2))
    return b


def collides(Vo, obj_pose, boxes):
    Vw = (obj_pose[:3, :3] @ Vo.T).T + obj_pose[:3, 3]
    for Tinv, half in boxes:
        Vl = (Tinv[:3, :3] @ Vw.T).T + Tinv[:3, 3]
        if np.any(np.all(np.abs(Vl) <= half, axis=1)):
            return True
    return False


def objects():
    # can generate scenes on-the-fly iff tabletop + mesh exist (on-disk scene
    # JSONs may be absent, e.g. soaptray/lemon_squeezer had theirs deleted)
    return sorted(o for o in os.listdir(OBJROOT)
                  if glob.glob(f"{OBJROOT}/{o}/processed_data/info/tabletop/*.npy")
                  and os.path.exists(f"{OBJROOT}/{o}/processed_data/info/simplified.json")
                  and o not in SKIP_OBJS
                  and any(glob.glob(f"{DS}/{o}/*/executed_grasp/meta.json") for DS in DS_ROOTS))


def main():
    vis = ViserViewer()
    sg.vis = vis
    objs = objects()
    st = {"idx": {}, "exec": [], "byp": {}, "unc": []}

    with vis.server.gui.add_folder("Executed-grasp coverage"):
        obj_dd = vis.server.gui.add_dropdown("Object", options=tuple(objs), initial_value=objs[0])
        pose_dd = vis.server.gui.add_dropdown("Tabletop Pose", options=("",), initial_value="")
        gr_sl = vis.server.gui.add_slider("Exec grasp #", min=0, max=1, step=1, initial_value=0)
        stat = vis.server.gui.add_text("status", initial_value="")

    def load_object(obj):
        stat.value = f"{obj}: recovering executed grasps (FK)..."
        st["idx"] = scenes_by_pose(obj)
        st["exec"] = extract_executed(obj)
        st["byp"] = {}; st["unc"] = []; st["nopose"] = 0
        for g in st["exec"]:
            if g["pose_id"] is not None:   # coverable is DERIVED from pose_id (the
                st["byp"].setdefault(g["pose_id"], []).append(g)  # coverable flag gets
            elif g["objT"] is not None:    # stripped by harmonize's 17-key schema)
                st["unc"].append(g)   # no matching tabletop pose
            else:
                st["nopose"] += 1   # missing C2R/pose_world -> unclassifiable, NOT uncoverable
        # idx already keyed by CURRENT tabletop pose (scenes generated fresh)
        poses = sorted(st["idx"].keys())
        if st["unc"]:
            poses = poses + [UNC_LABEL]
        pose_dd.options = tuple(poses) if poses else ("",)
        if poses:
            pose_dd.value = poses[0]
        cap = sorted(st["byp"].keys())
        stat.value = (f"{obj}: {len(st['exec'])} executed grasps · coverable at {cap} · "
                      f"{len(st['unc'])} uncoverable"
                      + (f" · {st['nopose']} no-pose (missing C2R/pose_world)" if st['nopose'] else ""))

    def load_grid():
        sg.clear_all()
        obj, pose = obj_dd.value, pose_dd.value
        if pose == UNC_LABEL:
            ung = st["unc"]
            obb = json.load(open(f"{OBJROOT}/{obj}/processed_data/info/simplified.json"))
            raw = sg.load_mesh(f"{OBJROOT}/{obj}/processed_data/mesh/simplified.obj")
            spacing = float(np.linalg.norm(np.array(obb["obb"]))) * 3.0 + 0.1
            ttfs = sorted(glob.glob(f"{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy"))
            items = [("tt", os.path.basename(f)[:-4], np.load(f)) for f in ttfs] \
                + [("ug", i, g) for i, g in enumerate(ung)]
            cols = int(np.ceil(np.sqrt(max(1, len(items)))))
            for k, (kind, key, val) in enumerate(items):
                off = np.array([((k % cols) - cols // 2) * spacing,
                                ((k // cols) - cols // 2) * spacing, 0.0])
                if kind == "tt":                    # grey reference: the tabletop poses we DO have
                    G = np.eye(4); G[:3, :3] = val[:3, :3]; G[:3, 3] = [off[0], off[1], val[2, 3]]
                    vis.add_object(f"ref_{key}", paint(raw, GREY, 130), obj_T=G)
                else:                               # captured resting pose (no match)
                    T = val["objT"].copy(); T[:3, 3] = [off[0], off[1], val["objT"][2, 3]]
                    vis.add_object(f"ug{key}_o", paint(raw, YELLOW, 70), obj_T=T)   # object see-through
                    vis.add_object(f"ug{key}_h", paint(val["mesh"], [230, 120, 30], 255), obj_T=T)  # solid hand
            stat.value = (f"{obj} UNCOVERABLE: {len(ung)} grasps (yellow, at captured rest) vs "
                          f"{len(ttfs)} tabletop poses (grey) — captured rest not among them")
            return
        scenes = st["idx"].get(pose, [])
        grasps = st["byp"].get(pose, [])
        gr_sl.max = max(1, len(grasps) - 1)
        if not scenes:
            stat.value = f"{obj} pose {pose}: no scenes"
            return
        _v = gr_sl.value
        gi = min(int(_v) if _v == _v else 0, len(grasps) - 1) if grasps else -1
        obb = json.load(open(f"{OBJROOT}/{obj}/processed_data/info/simplified.json"))
        raw = sg.load_mesh(f"{OBJROOT}/{obj}/processed_data/mesh/simplified.obj")
        spacing = float(np.linalg.norm(np.array(obb["obb"]))) * 3.0 + 0.1
        cols = int(np.ceil(np.sqrt(len(scenes))))
        nfit = nany = 0
        for i, (stype, scene) in enumerate(scenes):
            if scene is None:
                continue
            op = cart2se3(np.array(scene["mesh"]["target"]["pose"], float))
            boxes = _boxes(scene.get("cuboid", {}))
            sel_fit = gi >= 0 and not collides(grasps[gi]["mesh"].vertices, op, boxes)
            any_fit = any(not collides(g["mesh"].vertices, op, boxes) for g in grasps)
            color = GREEN if sel_fit else (YELLOW if any_fit else RED)   # green=selected, yellow=any, red=none
            nfit += int(sel_fit); nany += int(any_fit)
            row, col = i // cols, i % cols
            off = np.array([(col - cols // 2) * spacing, (row - cols // 2) * spacing, 0.0])
            sid = f"{stype}{i}"
            for mn, mi in scene.get("mesh", {}).items():
                mesh = raw if mn == "target" else sg.load_mesh(mi["file_path"])
                pm = sg.parse_pose(mi["pose"]); pm[:3, 3] += off
                if mn == "target":
                    mesh = paint(mesh, color)
                vis.add_object(f"s{sid}_{mn}", mesh, obj_T=pm)
            for cn, ci in scene.get("cuboid", {}).items():
                box = paint(trimesh.creation.box(extents=ci["dims"]),
                            TABLE if cn == "table" else OBST, 150)
                pc = sg.parse_pose(ci["pose"]); pc[:3, 3] += off
                vis.add_object(f"s{sid}_{cn}", box, obj_T=pc)
            if gi >= 0:
                hp = op.copy(); hp[:3, 3] += off
                vis.add_object(f"s{sid}_hand", paint(grasps[gi]["mesh"], GREEN if sel_fit else RED, 235),
                               obj_T=hp)
        if gi < 0:
            stat.value = f"{obj} pose {pose}: NO executed grasp captured here ({len(scenes)} scenes)"
        else:
            stat.value = (f"{obj} pose {pose}: exec grasp {gi+1}/{len(grasps)} ({grasps[gi]['ts']}) "
                          f"GREEN {nfit}/{len(scenes)} | YELLOW(any) {nany}/{len(scenes)}")
        print(stat.value)

    obj_dd.on_update(lambda _: (load_object(obj_dd.value), load_grid()))
    pose_dd.on_update(lambda _: load_grid())
    gr_sl.on_update(lambda _: load_grid())

    load_object(obj_dd.value)
    load_grid()
    print("[coverage_grid_viewer] ready")
    vis.start_viewer()


if __name__ == "__main__":
    main()
