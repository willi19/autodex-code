"""#1 v2: canonicalize via SYMMETRY GROUP enumeration + z-aligned matching.
Detect the object's discrete symmetry group G (subset of {I,x180,y180,z180,+90s}
that leave the mesh invariant, dev<THR). For an uncoverable capture P, if P@G
z-aligns (<=ANG) to a tabletop pose T for some G, canonicalize:
  wrist6d_new = inv(G) @ wrist6d_old   (physical wrist unchanged -> no penetration)
  pose_id = T   (azimuth handled by scene z_rots)."""
import numpy as np, glob, json, os, sys, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rt
sys.path.insert(0, 'src/dataset/exec_grasp'); import reclassify_plain as rp  # _z_aligned via tabletop_pose
sys.path.insert(0, 'src/experiment/reset'); from tabletop_pose import _z_aligned_geodesic_deg
sys.path.insert(0, 'src/visualization'); import coverage_grid_viewer as cov
OP = "/home/mingi/shared_data/object_processing"
ROOTS = ['/home/mingi/shared_data/autodex_dataset/selected_100',
         '/home/mingi/shared_data/autodex_dataset/corl_selected_100']
DEV_THR = 0.18; ANG = 25.0
_mc = {}; _grp = {}; _tt = {}
def meshinfo(o):
    if o not in _mc:
        m = trimesh.load(f"{OP}/{o}/processed_data/mesh/simplified.obj", process=False)
        V = np.asarray(m.vertices); _mc[o] = (V, cKDTree(V), float(np.linalg.norm(V.max(0)-V.min(0))))
    return _mc[o]
def group(o):
    """Rotational-symmetry group from the object_processing detector's output
    (info/symmetry.json: per-axis unit vector + fold order, verified by RMS
    point-to-surface residual). Build the discrete rotations about each detected
    axis (k*2pi/fold; 'inf'=continuous -> sample 36). Returns {key: 3x3}."""
    if o not in _grp:
        g = {'I': np.eye(3)}
        sp = f"{OP}/{o}/processed_data/info/symmetry.json"
        if os.path.exists(sp):
            s = json.load(open(sp))
            for j, a in enumerate(s.get("axes", [])):
                ax = np.asarray(a["axis"], float); ax /= np.linalg.norm(ax)
                fold = a["fold"]
                n = 36 if fold == "inf" else int(fold)
                for k in range(1, n):
                    M = Rt.from_rotvec(ax * (2 * np.pi * k / n)).as_matrix()
                    if all(np.degrees(np.arccos(np.clip((np.trace(M.T @ U) - 1) / 2, -1, 1))) > 8
                           for U in g.values()):
                        g[f"a{j}_{k}"] = M
        _grp[o] = g
    return _grp[o]
def tts(o):
    if o not in _tt:
        _tt[o] = {os.path.basename(f)[:-4]: np.load(f) for f in sorted(glob.glob(f"{OP}/{o}/processed_data/info/tabletop/*.npy"))}
    return _tt[o]
def run(objs=None):
    tot=nc=npen=0
    for R in ROOTS:
        for mp in glob.glob(f"{R}/*/*/executed_grasp/meta.json"):
            m=json.load(open(mp))
            if m.get("pose_id") is not None: continue
            o=m["obj"]
            if objs and o not in objs: continue
            tot+=1
            d0=os.path.dirname(mp); trial=os.path.dirname(d0)
            try:
                P=np.linalg.inv(np.load(f"{trial}/C2R.npy"))@np.load(f"{trial}/pose_world.npy")
                _wo=f"{d0}/wrist_se3_orig.npy"
                wrist=np.load(_wo) if os.path.exists(_wo) else np.load(f"{d0}/wrist_se3.npy")
                finger=np.load(f"{d0}/grasp_pose.npy")
            except Exception: continue
            G=group(o); TT=tts(o); best=None
            for gk,Gm in G.items():
                PG=P[:3,:3]@Gm
                for stem,T in TT.items():
                    e=_z_aligned_geodesic_deg(PG,T[:3,:3])
                    if e<ANG and (best is None or e<best[3]): best=(gk,Gm,stem,e,T)
            if best is None: continue
            gk,Gm,stem,e,T=best
            Gse3=np.eye(4); Gse3[:3,:3]=Gm
            new_wrist=np.linalg.inv(Gse3)@wrist
            hand=cov._hand_mesh(new_wrist,finger)
            Vw=(T[:3,:3]@hand.vertices.T).T+T[:3,3]
            if Vw[:,2].min()<-0.01: npen+=1; continue
            if not os.path.exists(f"{d0}/wrist_se3_orig.npy"): np.save(f"{d0}/wrist_se3_orig.npy",wrist)
            np.save(f"{d0}/wrist_se3.npy",new_wrist)
            m["pose_id"]=stem; json.dump(m,open(mp,"w"),indent=1); nc+=1
    print(f"uncoverable {tot} -> canonicalized {nc} | skip-penetrate {npen}")
