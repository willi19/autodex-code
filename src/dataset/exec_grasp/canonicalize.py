"""#1: canonicalize symmetry-variant uncoverable grasps.
For each uncoverable capture P, if S=P_rot^T @ T_rot is a mesh symmetry (dev<THR)
for some tabletop pose T, re-express the grasp in the canonical frame:
  wrist6d_new = inv(S) @ wrist6d_old   (physical wrist unchanged -> no penetration)
and set pose_id=T. Verify hand doesn't hit the table before committing.
Backs up original wrist to wrist_se3_orig.npy."""
import numpy as np, glob, json, os, sys, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, 'src/visualization')
import coverage_grid_viewer as cov
OP = "/home/mingi/shared_data/object_processing"
ROOTS = ['/home/mingi/shared_data/autodex_dataset/selected_100',
         '/home/mingi/shared_data/autodex_dataset/corl_selected_100']
THR = 0.08
_mc = {}
def meshinfo(obj):
    if obj not in _mc:
        m = trimesh.load(f"{OP}/{obj}/processed_data/mesh/simplified.obj", process=False)
        V = np.asarray(m.vertices)
        _mc[obj] = (V, cKDTree(V), float(np.linalg.norm(V.max(0) - V.min(0))))
    return _mc[obj]
def dev(obj, S):
    V, tree, scale = meshinfo(obj)
    d, _ = tree.query(V @ S.T); return d.max() / scale
_tt = {}
def tabletops(obj):
    if obj not in _tt:
        _tt[obj] = {os.path.basename(f)[:-4]: np.load(f)
                    for f in sorted(glob.glob(f"{OP}/{obj}/processed_data/info/tabletop/*.npy"))}
    return _tt[obj]

def run(objs=None):
    total = ncanon = npen = 0
    for R in ROOTS:
        for mp in glob.glob(f"{R}/*/*/executed_grasp/meta.json"):
            m = json.load(open(mp))
            if m.get("pose_id") is not None:
                continue
            obj = m["obj"]
            if objs and obj not in objs:
                continue
            total += 1
            d0 = os.path.dirname(mp); trial = os.path.dirname(d0)
            try:
                P = np.linalg.inv(np.load(f"{trial}/C2R.npy")) @ np.load(f"{trial}/pose_world.npy")
                wrist = np.load(f"{d0}/wrist_se3.npy"); finger = np.load(f"{d0}/grasp_pose.npy")
            except Exception:
                continue
            tt = tabletops(obj)
            best = None
            for stem, T in tt.items():
                S = P[:3, :3].T @ T[:3, :3]
                dd = dev(obj, S)
                if dd < THR and (best is None or dd < best[2]):
                    best = (stem, T, dd, S)
            if best is None:
                continue
            stem, T, dd, S = best
            Sse3 = np.eye(4); Sse3[:3, :3] = S
            new_wrist = np.linalg.inv(Sse3) @ wrist
            hand = cov._hand_mesh(new_wrist, finger)
            Vw = (T[:3, :3] @ hand.vertices.T).T + T[:3, 3]
            if Vw[:, 2].min() < -0.01:
                npen += 1; continue          # would penetrate -> leave uncoverable
            if not os.path.exists(f"{d0}/wrist_se3_orig.npy"):
                np.save(f"{d0}/wrist_se3_orig.npy", wrist)
            np.save(f"{d0}/wrist_se3.npy", new_wrist)
            m["pose_id"] = stem; m["coverable"] = True; m["canonicalized"] = True
            m["canon_sym_dev"] = round(float(dd), 3)
            json.dump(m, open(mp, "w"), indent=1)
            ncanon += 1
    print(f"uncoverable {total} -> canonicalized {ncanon} | skipped-penetrate {npen}")
