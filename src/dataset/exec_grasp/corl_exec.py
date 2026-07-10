"""corl_selected_100 executed_grasp recovery (AutoDex/experiment/selected_100).
Uses plan/traj[-1] as the grasp config (no external candidate needed), FK wrist
from raw qpos, execution_states detection, and success from object_tracking 6d."""
import numpy as np, glob, os, json, sys, yourdfpy
from autodex.utils.robot_config import ALLEGRO_LINK6_TO_WRIST as L6W
EXP='/home/mingi/shared_data/AutoDex/experiment/selected_100'
OBJROOT='/home/mingi/shared_data/object_processing'
URDF='/home/mingi/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf'
u=yourdfpy.URDF.load(URDF,load_meshes=False,build_collision_scene_graph=False); aj=u.actuated_joint_names[:6]
def conv(hp): o=hp.copy(); o[...,:4]=hp[...,12:]; o[...,4:]=hp[...,:12]; return o
def grav(T): return T[:3,:3].T@np.array([0,0,1.])
def fkw(q6): u.update_cfg({aj[i]:float(q6[i]) for i in range(6)}); return u.get_transform('link6','world')@L6W

def obj_success(ot, grasp_time):
    """object lifted after grasp? z rise > 50mm + tracked."""
    wp=f'{ot}/world_pose_records.json'
    if not os.path.exists(wp): return None
    try: recs=json.load(open(wp))
    except: return None
    zt=[(r['time_sec'], r['translation_world_mm'][2]) for r in recs
        if r.get('status')=='ok' and r.get('translation_world_mm')]
    if len(zt)<5: return None
    zt.sort(); ts=np.array([t for t,_ in zt]); zs=np.array([z for _,z in zt])
    at_grasp=zs[np.abs(ts-grasp_time).argmin()]
    after=zs[ts>grasp_time]
    if len(after)==0: return None
    return bool(after.max()-at_grasp > 50.0)   # lifted >5cm

def process(hand, d):
    if not os.path.exists(f'{d}/raw/hand/action.npy') or not os.path.exists(f'{d}/plan/traj.npy'): return 'no_data'
    q=np.load(f'{d}/raw/arm/position.npy'); at=np.load(f'{d}/raw/arm/time.npy')
    ha=np.load(f'{d}/raw/hand/action.npy'); ht=np.load(f'{d}/raw/hand/time.npy')
    traj=np.load(f'{d}/plan/traj.npy'); gfinger=conv(traj[-1,6:22])
    C2R=np.load(f'{d}/C2R.npy'); pw=np.load(f'{d}/pose_world.npy'); obj_r=np.linalg.inv(C2R)@pw
    dfl=np.linalg.norm(ha-gfinger,axis=1); ghf=int(dfl.argmin()); fL2=float(dfl[ghf])
    gt=ht[ghf]; ga=int(np.abs(at-gt).argmin())
    wrist_obj=np.linalg.inv(obj_r)@fkw(q[ga])
    finger=traj[-1,6:22]  # grasp finger (cuRobo order, like RSS grasp_pose)
    # execution_states
    dj0=np.abs(q[:,0]-q[0,0]); dj15=np.abs(q[:,1:6]-q[0,1:6]).max(1)
    init=int(np.argmax(dj0>0.1)) if (dj0>0.1).any() else 0
    approach=int(np.argmax(dj15>0.1)) if (dj15>0.1).any() else init
    pp=f'{d}/plan/wrist_se3.npy'  # no separate pregrasp finger; approx pregrasp via hand open->close midpoint skipped
    clo=ha.sum(1); sq=int(ghf+clo[ghf:].argmax())
    idx=np.arange(0,len(q),15); zs=np.array([fkw(q[t])[2,3] for t in idx]); rise=idx[zs>zs.min()+0.05]
    aft=rise[rise>ga]; lift=int(aft[0]) if len(aft) else int(idx[zs.argmax()])
    states=[('init',at[init]),('approach',at[approach]),('grasp',ht[ghf]),('squeeze',ht[sq]),('lift',at[lift])]
    states=sorted(states,key=lambda x:x[1]); t0=states[0][1]
    # tabletop pose_id
    tt=sorted(glob.glob(f'{OBJROOT}/{os.path.basename(os.path.dirname(d))}/processed_data/info/tabletop/*.npy'))
    pose_id=None
    if tt:
        ttv={os.path.basename(t)[:-4]:grav(np.load(t)) for t in tt}; v=grav(obj_r)
        pose_id=min(ttv.items(),key=lambda kv:np.arccos(np.clip(v@kv[1],-1,1)))[0]
    succ=obj_success(f'{d}/object_tracking/gotrack_output', gt)
    out=f'{d}/executed_grasp'; os.makedirs(out,exist_ok=True)
    np.save(f'{out}/wrist_se3.npy', wrist_obj); np.save(f'{out}/grasp_pose.npy', finger)
    meta={'obj':os.path.basename(os.path.dirname(d)),'ts':os.path.basename(d),'hand':hand,
          'candidate_idx':json.load(open(f'{d}/plan/timing.json')).get('candidate_idx'),
          'finger_L2':fL2,'grasp_frame':ghf,'grasp_time':float(gt),'pose_id':pose_id,
          'wrist_source':'fk_executed','success':succ,
          'execution_states':[{'state':s,'time':float(t),'t_rel_s':round(float(t-t0),3)} for s,t in states]}
    json.dump(meta, open(f'{out}/meta.json','w'), indent=1)
    return 'ok'

if __name__=='__main__':
    hands=sys.argv[1:] or ['allegro']
    for hand in hands:
        st={}
        for d in sorted(glob.glob(f'{EXP}/{hand}/*/*/')):
            r=process(hand, d.rstrip('/')); st[r]=st.get(r,0)+1
        print(hand, st)
