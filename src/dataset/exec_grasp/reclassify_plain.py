"""Initial-frame pose_id: plain z-aligned geodesic (world-z only, NO symmetry
fold) against object_processing tabletop. A world-z match keeps the grasp above
the table (no penetration). Residual > threshold => uncoverable (pose not in set).

Threshold is PER-OBJECT, read from src/scene_generation/pose_match_threshold.json
(key "_default" is the fallback, default 25). Objects whose kept tabletop poses are
close but distinct use a lower threshold; objects that absorb removed poses use a
higher one."""
import os, sys, glob, json, numpy as np
sys.path.insert(0, 'src/experiment/reset')
from tabletop_pose import _z_aligned_geodesic_deg

OP = "/home/mingi/shared_data/object_processing"
_THR_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "scene_generation", "pose_match_threshold.json")
try:
    _c = json.load(open(_THR_CFG))
    DEFAULT_THRESH = float(_c.get("_default", 25.0))
    _THR = {k: float(v) for k, v in _c.items() if not k.startswith("_")}
except Exception:
    DEFAULT_THRESH = 25.0
    _THR = {}

_tt = {}
def _poses(obj):
    if obj not in _tt:
        _tt[obj] = [(os.path.basename(f)[:-4], np.load(f))
                    for f in sorted(glob.glob(f"{OP}/{obj}/processed_data/info/tabletop/*.npy"))]
    return _tt[obj]

def threshold(obj):
    return _THR.get(obj, DEFAULT_THRESH)

def classify(obj, R_est):
    ps = _poses(obj)
    if not ps:
        return None
    thr = threshold(obj)
    errs = [_z_aligned_geodesic_deg(R_est, T[:3, :3]) for _, T in ps]
    i = int(np.argmin(errs))
    return {"pose_id": ps[i][0], "rot_err_deg": float(errs[i]),
            "coverable": bool(errs[i] <= thr), "thresh": thr}
