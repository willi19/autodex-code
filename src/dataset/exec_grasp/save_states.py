import numpy as np, glob, os, json, yourdfpy
from autodex.utils.robot_config import ALLEGRO_LINK6_TO_WRIST as L6W
URDF='/home/mingi/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf'
u=yourdfpy.URDF.load(URDF,load_meshes=False,build_collision_scene_graph=False); aj=u.actuated_joint_names[:6]
def conv(hp): o=hp.copy(); o[...,:4]=hp[...,12:]; o[...,4:]=hp[...,:12]; return o
def fkwz(q6):
    u.update_cfg({aj[i]:float(q6[i]) for i in range(6)}); return (u.get_transform('link6','world')@L6W)[2,3]

def detect(d, cand_path):
    q=np.load(f'{d}/raw/arm/position.npy'); at=np.load(f'{d}/raw/arm/time.npy')
    ha=np.load(f'{d}/raw/hand/action.npy'); ht=np.load(f'{d}/raw/hand/time.npy')
    gp=conv(np.load(cand_path.replace('wrist_se3','grasp_pose')))
    pp_f=cand_path.replace('wrist_se3','pregrasp_pose')
    pg=conv(np.load(pp_f)) if os.path.exists(pp_f) else None
    st={}
    # arm-based: init(joint0), approach(joints1-5), lift(wrist z rise)
    dj0=np.abs(q[:,0]-q[0,0]); dj15=np.abs(q[:,1:6]-q[0,1:6]).max(1)
    st['init']=int(np.argmax(dj0>0.1)) if (dj0>0.1).any() else 0
    st['approach']=int(np.argmax(dj15>0.1)) if (dj15>0.1).any() else st['init']
    # hand-based: pregrasp, grasp
    def hframe(target): return int(np.linalg.norm(ha-target,axis=1).argmin())
    hg=hframe(gp)  # grasp
    st['grasp_hf']=hg
    if pg is not None: st['pregrasp_hf']=hframe(pg)
    # squeeze: after grasp, max total closure
    clo=ha.sum(1)  # more positive = more closed (approx)
    st['squeeze_hf']=int(hg+clo[hg:].argmax())
    # lift: wrist z rise. FK z sparsely
    idx=np.arange(0,len(q),15); zs=np.array([fkwz(q[t]) for t in idx])
    zmin=zs.min(); rise=idx[zs>zmin+0.05]
    # first rise AFTER grasp time
    gt=ht[hg]; ga=int(np.abs(at-gt).argmin())
    after=rise[rise>ga]; st['lift']=int(after[0]) if len(after) else int(idx[zs.argmax()])
    # convert to times
    out=[('init',at[st['init']]),('approach',at[st['approach']])]
    if 'pregrasp_hf' in st: out.append(('pregrasp',ht[st['pregrasp_hf']]))
    out.append(('grasp',ht[hg])); out.append(('squeeze',ht[st['squeeze_hf']])); out.append(('lift',at[st['lift']]))
    return sorted(out,key=lambda x:x[1]), st


import sys
DS='/home/mingi/shared_data/autodex_dataset/selected_100'
n_ok=n_err=0
for mp in glob.glob(f'{DS}/*/*/executed_grasp/meta.json'):
    d=os.path.dirname(os.path.dirname(mp))
    try:
        m=json.load(open(mp))
        states,_=detect(d, m['cand_path'])
        t0=states[0][1]
        m['execution_states']=[{'state':s,'time':float(t),'t_rel_s':round(float(t-t0),3)} for s,t in states]
        json.dump(m, open(mp,'w'), indent=1); n_ok+=1
    except Exception as e:
        n_err+=1
print(f'execution_states saved: {n_ok}, errors: {n_err}')
