"""Initial-frame pose_id: plain z-aligned geodesic (world-z only, NO symmetry
fold) against object_processing tabletop. A world-z match keeps the grasp above
the table (no penetration). Residual > THRESH => uncoverable (pose not in set)."""
import os, sys, glob, numpy as np
sys.path.insert(0, 'src/experiment/reset')
from tabletop_pose import _z_aligned_geodesic_deg
OP = "/home/mingi/shared_data/object_processing"
THRESH = 25.0
_tt = {}
def _poses(obj):
    if obj not in _tt:
        _tt[obj] = [(os.path.basename(f)[:-4], np.load(f))
                    for f in sorted(glob.glob(f"{OP}/{obj}/processed_data/info/tabletop/*.npy"))]
    return _tt[obj]
def classify(obj, R_est):
    ps = _poses(obj)
    if not ps: return None
    errs = [_z_aligned_geodesic_deg(R_est, T[:3,:3]) for _, T in ps]
    i = int(np.argmin(errs))
    return {"pose_id": ps[i][0], "rot_err_deg": float(errs[i]),
            "coverable": bool(errs[i] <= THRESH)}
