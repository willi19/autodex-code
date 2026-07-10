"""Validate capture -> v8-scene coverage in viser.

Coverage rule (tabletop-pose level): a scene at pose_idx p is COVERED iff some
real capture of that object was in tabletop pose p (gravity-in-object angle
<= THRESH between the capture's robot-frame pose inv(C2R)@pose_world and the
canonical tabletop pose).

GUI:
  object      dropdown
  scene_type  dropdown (wall / shelf / box)
  scene       slider   (browse scenes of that type)

Renders the selected scene: object mesh at its target pose (GREEN if the
scene's pose_idx is covered by a capture, RED otherwise) + obstacle cuboids
(grey). Text shows pose_idx, covered?, and object-level covered-scene count.

    conda activate mingi
    python src/visualization/coverage_viewer.py --port 8091
"""
import os
import json
import time
import glob
import argparse
import numpy as np
import trimesh
import viser

from autodex.utils.conversion import cart2se3

DS = "/home/mingi/shared_data/autodex_dataset/selected_100"
OBJROOT = "/home/mingi/shared_data/object_processing"
THRESH_DEG = 30.0
SCENE_TYPES = ["wall", "shelf", "box"]


def grav_obj(T):
    return T[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def load_mesh(obj):
    return trimesh.load(f"{OBJROOT}/{obj}/processed_data/mesh/simplified.obj", process=False)


def tabletop_poses(obj):
    fs = sorted(glob.glob(f"{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy"))
    return {os.path.basename(f)[:-4]: np.load(f) for f in fs}


def covered_poses(obj, ttv):
    """Set of tabletop-pose ids matched by at least one capture."""
    cov = set()
    for pw in sorted(glob.glob(f"{DS}/{obj}/*/pose_world.npy")):
        d = os.path.dirname(pw)
        c2r = os.path.join(d, "C2R.npy")
        if not os.path.exists(c2r):
            continue
        try:
            P = np.linalg.inv(np.load(c2r)) @ np.load(pw)
        except Exception:
            continue
        v = grav_obj(P)
        k, ang = None, 999.0
        for kk, vt in ttv.items():
            a = np.degrees(np.arccos(np.clip(v @ vt, -1, 1)))
            if a < ang:
                k, ang = kk, a
        if ang <= THRESH_DEG:
            cov.add(k)
    return cov


def scene_files(obj, st):
    return sorted(glob.glob(f"{OBJROOT}/{obj}/scene/{st}/*.json"))


def colored(mesh, rgb, alpha=255):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(
        m, vertex_colors=np.tile(np.array(rgb + [alpha], np.uint8), (len(m.vertices), 1)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    objs = sorted(o for o in os.listdir(DS)
                  if os.path.isdir(f"{DS}/{o}")
                  and glob.glob(f"{OBJROOT}/{o}/processed_data/info/tabletop/*.npy"))

    server = viser.ViserServer(port=args.port)
    dd_obj = server.gui.add_dropdown("object", options=objs, initial_value=objs[0])
    dd_st = server.gui.add_dropdown("scene_type", options=SCENE_TYPES, initial_value="wall")
    sl = server.gui.add_slider("scene", min=0, max=0, step=1, initial_value=0)
    info = server.gui.add_text("scene", initial_value="")
    summ = server.gui.add_text("object coverage", initial_value="")

    state = {"files": [], "cov": set(), "mesh": None}

    def load_obj(obj):
        tts = tabletop_poses(obj)
        ttv = {k: grav_obj(T) for k, T in tts.items()}
        state["cov"] = covered_poses(obj, ttv)
        state["mesh"] = load_mesh(obj)
        state["ntt"] = len(tts)

    def refresh_files():
        obj, st = dd_obj.value, dd_st.value
        state["files"] = scene_files(obj, st)
        sl.max = max(0, len(state["files"]) - 1)
        if sl.value > sl.max:
            sl.value = 0
        # object-level coverage across all scene types
        tot = cov = 0
        for s in SCENE_TYPES:
            for f in scene_files(obj, s):
                tot += 1
                pidx = json.load(open(f)).get("meta", {}).get("pose_idx")
                if pidx in state["cov"]:
                    cov += 1
        summ.value = (f"{obj}: covered scenes {cov}/{tot}  "
                      f"covered tabletop-poses {sorted(state['cov'])} / {state['ntt']}")

    def render():
        server.scene.reset()
        files = state["files"]
        if not files:
            info.value = "(no scenes)"
            return
        f = files[int(sl.value)]
        d = json.load(open(f))
        scene = d["scene"]
        pidx = d.get("meta", {}).get("pose_idx")
        is_cov = pidx in state["cov"]

        # object mesh at target pose
        tgt = scene["mesh"]["target"]
        T = cart2se3(np.array(tgt["pose"], float))
        rgb = [70, 200, 90] if is_cov else [220, 70, 70]
        m = state["mesh"].copy()
        m.apply_transform(T)
        server.scene.add_mesh_trimesh("/obj", colored(m, rgb))

        # obstacle cuboids
        for name, cub in scene.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=np.array(cub["dims"], float))
            box.apply_transform(cart2se3(np.array(cub["pose"], float)))
            server.scene.add_mesh_trimesh(f"/cub/{name}", colored(box, [119, 136, 153], 110))

        info.value = (f"{os.path.basename(f)}  pose_idx={pidx}  "
                      f"{'COVERED' if is_cov else 'not covered'}  "
                      f"gap={d.get('meta',{}).get('param',{}).get('gap')}")

    @dd_obj.on_update
    def _(_):
        load_obj(dd_obj.value)
        refresh_files()
        render()

    @dd_st.on_update
    def _(_):
        refresh_files()
        render()

    @sl.on_update
    def _(_):
        render()

    load_obj(dd_obj.value)
    refresh_files()
    render()
    print(f"[coverage_viewer] serving on port {args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
