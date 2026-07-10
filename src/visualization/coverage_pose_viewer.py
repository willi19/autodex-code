"""(object x tabletop-pose) coverage in viser.

Pick an object; it lays out the object mesh once per canonical tabletop pose,
colored GREEN if a real capture landed in that pose (gravity-in-object angle
<= THRESH, robot frame inv(C2R)@pose_world) or RED if not. Label shows the
pose id and how many captures fell in it.

Granularity is (object, tabletop_pose) on purpose: a captured grasp only
transfers within the SAME resting pose, so that is the unit of coverage.

    conda activate mingi
    python src/visualization/coverage_pose_viewer.py --port 8091
"""
import os
import time
import glob
import argparse
import numpy as np
import trimesh
import viser

DS = "/home/mingi/shared_data/autodex_dataset/selected_100"
OBJROOT = "/home/mingi/shared_data/object_processing"
THRESH_DEG = 30.0


def grav(T):
    return T[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def tabletop(obj):
    fs = sorted(glob.glob(f"{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy"))
    return {os.path.basename(f)[:-4]: np.load(f) for f in fs}


def capture_counts(obj, ttv):
    """captures landed per tabletop-pose id."""
    n = {k: 0 for k in ttv}
    for pw in sorted(glob.glob(f"{DS}/{obj}/*/pose_world.npy")):
        d = os.path.dirname(pw)
        if not os.path.exists(f"{d}/C2R.npy"):
            continue
        try:
            P = np.linalg.inv(np.load(f"{d}/C2R.npy")) @ np.load(pw)
        except Exception:
            continue
        v = grav(P)
        k, a = min(((kk, np.degrees(np.arccos(np.clip(v @ vt, -1, 1))))
                    for kk, vt in ttv.items()), key=lambda x: x[1])
        if a <= THRESH_DEG:
            n[k] += 1
    return n


def colored(mesh, rgb):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(
        m, vertex_colors=np.tile(np.array(rgb + [255], np.uint8), (len(m.vertices), 1)))
    return m


def placed(mesh, R, x, y):
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = [x, y, 0.0]
    m = mesh.copy()
    m.apply_transform(G)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    objs = sorted(o for o in os.listdir(DS)
                  if os.path.isdir(f"{DS}/{o}")
                  and glob.glob(f"{OBJROOT}/{o}/processed_data/info/tabletop/*.npy"))

    server = viser.ViserServer(port=args.port)
    dd = server.gui.add_dropdown("object", options=objs, initial_value=objs[0])
    info = server.gui.add_text("coverage", initial_value="")

    def render(obj):
        server.scene.reset()
        mesh = trimesh.load(f"{OBJROOT}/{obj}/processed_data/mesh/simplified.obj", process=False)
        tts = tabletop(obj)
        ttv = {k: grav(T) for k, T in tts.items()}
        ncap = capture_counts(obj, ttv)
        pitch = float(np.linalg.norm(mesh.extents)) * 1.6

        ncov = 0
        for i, (k, T) in enumerate(tts.items()):
            cov = ncap[k] > 0
            ncov += int(cov)
            rgb = [70, 200, 90] if cov else [214, 74, 92]
            server.scene.add_mesh_trimesh(f"/p/{k}", placed(colored(mesh, rgb), T[:3, :3], i * pitch, 0))
            server.scene.add_label(f"/l/{k}", f"pose {k}  ·  {ncap[k]} cap",
                                   position=(i * pitch, -pitch * 0.55, 0))

        info.value = f"{obj}: covered {ncov}/{len(tts)} poses  ({[k for k in tts if ncap[k]>0]})"
        print(info.value)

    @dd.on_update
    def _(_):
        render(dd.value)

    render(dd.value)
    print(f"[coverage_pose_viewer] serving on port {args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
