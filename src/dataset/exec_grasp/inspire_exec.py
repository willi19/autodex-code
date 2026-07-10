"""inspire executed_grasp recovery for selected_100_inspire.

Mirror of corl_exec.py for the 6-DOF inspire hand. Differences vs allegro:
  - plan/traj[-1, 6:12] is the 6-DOF grasp finger (URDF order, radians)
  - raw/hand/action is inspire controller units (0-1000), so match the grasp
    frame in radians via convert_inspire_raw() (URDF order) -- NOT raw units
  - xarm_inspire URDF + INSPIRE_LINK6_TO_WRIST for the executed-wrist FK

Inputs raw/plan/object_tracking come from the source experiment; C2R + the
(recomputed) pose_world come from the dataset trial, and executed_grasp/ is
written into the dataset trial so it is consistent with the shipped pose.

    ~/miniconda3/envs/mingi/bin/python -m src.dataset.exec_grasp.inspire_exec            # all objects
    ~/miniconda3/envs/mingi/bin/python -m src.dataset.exec_grasp.inspire_exec attached_container
"""
import numpy as np, glob, os, json, sys, yourdfpy
from autodex.utils.robot_config import INSPIRE_LINK6_TO_WRIST as L6W
from autodex.utils.sync import convert_inspire_raw

SRC = '/home/mingi/shared_data/AutoDex/experiment/selected_100/inspire'
DST = '/home/mingi/shared_data/autodex_dataset/selected_100_inspire'
OBJROOT = '/home/mingi/shared_data/object_processing'
URDF = '/home/mingi/shared_data/AutoDex/content/assets/robot/inspire_description/xarm_inspire.urdf'
u = yourdfpy.URDF.load(URDF, load_meshes=False, build_collision_scene_graph=False)
aj = u.actuated_joint_names[:6]


def grav(T):
    return T[:3, :3].T @ np.array([0, 0, 1.])


def fkw(q6):
    u.update_cfg({aj[i]: float(q6[i]) for i in range(6)})
    return u.get_transform('link6', 'world') @ L6W


def obj_success(ot, grasp_time):
    """object lifted after grasp? z rise > 50mm + tracked."""
    wp = f'{ot}/world_pose_records.json'
    if not os.path.exists(wp):
        return None
    try:
        recs = json.load(open(wp))
    except Exception:
        return None
    zt = [(r['time_sec'], r['translation_world_mm'][2]) for r in recs
          if r.get('status') == 'ok' and r.get('translation_world_mm')]
    if len(zt) < 5:
        return None
    zt.sort()
    ts = np.array([t for t, _ in zt])
    zs = np.array([z for _, z in zt])
    at_grasp = zs[np.abs(ts - grasp_time).argmin()]
    after = zs[ts > grasp_time]
    if len(after) == 0:
        return None
    return bool(after.max() - at_grasp > 50.0)


def process(obj, ts):
    s = f'{SRC}/{obj}/{ts}'
    d = f'{DST}/{obj}/{ts}'
    if not os.path.exists(f'{s}/raw/hand/action.npy') or not os.path.exists(f'{s}/plan/traj.npy'):
        return 'no_data'
    if not os.path.exists(f'{d}/pose_world.npy') or not os.path.exists(f'{d}/C2R.npy'):
        return 'no_dataset_pose'   # recompute rejected / not built yet
    q = np.load(f'{s}/raw/arm/position.npy'); at = np.load(f'{s}/raw/arm/time.npy')
    ha = np.load(f'{s}/raw/hand/action.npy'); ht = np.load(f'{s}/raw/hand/time.npy')
    traj = np.load(f'{s}/plan/traj.npy')
    gfinger = traj[-1, 6:12]                       # 6-DOF grasp finger (URDF order, rad)
    har = convert_inspire_raw(ha)                  # (T,6) rad, URDF order
    C2R = np.load(f'{d}/C2R.npy'); pw = np.load(f'{d}/pose_world.npy')
    obj_r = np.linalg.inv(C2R) @ pw
    dfl = np.linalg.norm(har - gfinger, axis=1)
    ghf = int(dfl.argmin()); fL2 = float(dfl[ghf])
    gt = ht[ghf]; ga = int(np.abs(at - gt).argmin())
    wrist_obj = np.linalg.inv(obj_r) @ fkw(q[ga])
    finger = gfinger
    # execution_states: arm-based init/approach, hand-based grasp/squeeze, arm-z lift
    dj0 = np.abs(q[:, 0] - q[0, 0]); dj15 = np.abs(q[:, 1:6] - q[0, 1:6]).max(1)
    init = int(np.argmax(dj0 > 0.1)) if (dj0 > 0.1).any() else 0
    approach = int(np.argmax(dj15 > 0.1)) if (dj15 > 0.1).any() else init
    clo = har.sum(1); sq = int(ghf + clo[ghf:].argmax())   # most-closed after grasp
    idx = np.arange(0, len(q), 15)
    zs = np.array([fkw(q[t])[2, 3] for t in idx]); rise = idx[zs > zs.min() + 0.05]
    aft = rise[rise > ga]; lift = int(aft[0]) if len(aft) else int(idx[zs.argmax()])
    states = [('init', at[init]), ('approach', at[approach]), ('grasp', ht[ghf]),
              ('squeeze', ht[sq]), ('lift', at[lift])]
    states = sorted(states, key=lambda x: x[1]); t0 = states[0][1]
    # tabletop pose_id (basic gravity-vector nearest; reclassify_plain refines later)
    tt = sorted(glob.glob(f'{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy'))
    pose_id = None
    if tt:
        ttv = {os.path.basename(t)[:-4]: grav(np.load(t)) for t in tt}; v = grav(obj_r)
        pose_id = min(ttv.items(), key=lambda kv: np.arccos(np.clip(v @ kv[1], -1, 1)))[0]
    succ = obj_success(f'{s}/object_tracking/gotrack_output', float(gt))
    cand_idx = None
    if os.path.exists(f'{s}/plan/timing.json'):
        cand_idx = json.load(open(f'{s}/plan/timing.json')).get('candidate_idx')
    # grasp frame on the synced video timeline
    gfs = None
    vtp = f'{d}/raw/timestamps/timestamp.npy'
    if os.path.exists(vtp):
        vt = np.load(vtp)
        gfs = int(np.abs(vt - float(gt)).argmin())
    out = f'{d}/executed_grasp'; os.makedirs(out, exist_ok=True)
    np.save(f'{out}/wrist_se3.npy', wrist_obj)
    np.save(f'{out}/grasp_pose.npy', finger)
    meta = {'obj': obj, 'ts': ts, 'hand': 'inspire', 'note': 'inspire',
            'candidate_idx': cand_idx, 'finger_L2': fL2, 'grasp_frame': ghf,
            'grasp_frame_synced': gfs, 'grasp_time': float(gt), 'pose_id': pose_id,
            'wrist_source': 'fk_executed', 'success': succ,
            'execution_states': [{'state': s_, 'time': float(t), 't_rel_s': round(float(t - t0), 3)}
                                 for s_, t in states]}
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    return 'ok'


if __name__ == '__main__':
    objs = sys.argv[1:] or [o for o in sorted(os.listdir(SRC)) if os.path.isdir(f'{SRC}/{o}')]
    st = {}
    for obj in objs:
        od = f'{SRC}/{obj}'
        for ts in sorted(os.listdir(od)):
            if os.path.isdir(f'{od}/{ts}'):
                r = process(obj, ts); st[r] = st.get(r, 0) + 1
    print(st)
