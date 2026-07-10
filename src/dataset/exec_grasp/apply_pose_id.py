"""Apply the canonical initial-frame pose_id to executed_grasp metas.

Step 6 of the corl->rss format pipeline, for the inspire dataset (or any root).
For each trial's executed_grasp/meta.json, recompute the initial-frame object
rotation R_est = (inv(C2R) @ pose_world)[:3,:3] and classify it against the
object_processing tabletop set with reclassify_plain (plain world-z geodesic,
>25 deg => uncoverable). Writes the same fields corl/RSS carry:

    pose_id (None if uncoverable), pose_id_prev, tabletop_before{idx,filename,
    rot_err_deg}, tabletop_rot_err, tabletop_src, coverable, pose_match.

Run AFTER harmonize_merge (harmonize drops non-canonical keys; these are added
last, exactly like corl):
    ~/miniconda3/envs/mingi/bin/python -m src.dataset.exec_grasp.apply_pose_id
    ~/miniconda3/envs/mingi/bin/python -m src.dataset.exec_grasp.apply_pose_id --root <dataset_root>
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, 'src/dataset/exec_grasp')
import reclassify_plain as rp

DEFAULT_ROOT = '/home/mingi/shared_data/autodex_dataset/selected_100_inspire'


def _tt_index(obj):
    """sorted tabletop stems for the object (same order reclassify_plain uses)."""
    return [stem for stem, _ in rp._poses(obj)]


def _grav(R):
    return np.asarray(R)[:3, :3].T @ np.array([0, 0, 1.])


def _basic_pose_id(obj, R_est):
    """corl_exec's pre-reclassify pose_id: nearest tabletop by gravity vector.
    Used for pose_id_prev so restores are independent of the current meta."""
    ps = rp._poses(obj)
    if not ps:
        return None
    v = _grav(R_est)
    return min(ps, key=lambda st: np.arccos(np.clip(v @ _grav(st[1]), -1, 1)))[0]


def process(mp):
    d = os.path.dirname(os.path.dirname(mp))          # trial dir
    obj = os.path.basename(os.path.dirname(d))
    if not (os.path.exists(f'{d}/C2R.npy') and os.path.exists(f'{d}/pose_world.npy')):
        return 'no_pose'
    C2R = np.load(f'{d}/C2R.npy'); pw = np.load(f'{d}/pose_world.npy')
    R_est = (np.linalg.inv(C2R) @ pw)[:3, :3]
    res = rp.classify(obj, R_est)
    if res is None:
        return 'no_tabletop'
    m = json.load(open(mp))
    stems = _tt_index(obj)
    idx = stems.index(res['pose_id']) if res['pose_id'] in stems else None
    m['pose_id_prev'] = _basic_pose_id(obj, R_est)
    m['pose_id'] = res['pose_id'] if res['coverable'] else None
    m['tabletop_before'] = {'idx': idx, 'filename': f"{res['pose_id']}.npy",
                            'rot_err_deg': round(res['rot_err_deg'], 2)}
    m['tabletop_rot_err'] = res['rot_err_deg']
    m['tabletop_src'] = 'object_processing'
    m['coverable'] = res['coverable']
    m['pose_match'] = 'initial_frame_zaligned'
    json.dump(m, open(mp, 'w'), indent=1)
    return 'coverable' if res['coverable'] else 'uncoverable'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    args = ap.parse_args()
    st = {}
    for mp in sorted(glob.glob(f'{args.root}/*/*/executed_grasp/meta.json')):
        r = process(mp); st[r] = st.get(r, 0) + 1
    print(st)


if __name__ == '__main__':
    main()
