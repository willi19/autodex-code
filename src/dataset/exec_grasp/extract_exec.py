import numpy as np, glob, os, yourdfpy
from autodex.utils.robot_config import ALLEGRO_LINK6_TO_WRIST as L6W

URDF='/home/mingi/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf'
OBJROOT='/home/mingi/shared_data/object_processing'
DS='/home/mingi/shared_data/autodex_dataset/selected_100'
CANDS='/home/mingi/shared_data/AutoDex/candidates/allegro/selected_100'
obj='apple'

u=yourdfpy.URDF.load(URDF, load_meshes=False, build_collision_scene_graph=False)
aj=u.actuated_joint_names[:6]
def fk_wrist(q6):
    u.update_cfg({aj[i]: float(q6[i]) for i in range(6)})
    return u.get_transform('link6','world') @ L6W

def grav(T): return T[:3,:3].T@np.array([0,0,1.])
tt=sorted(glob.glob(f'{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy'))
ttv={os.path.basename(t)[:-4]:grav(np.load(t)) for t in tt}
cand=np.array([np.load(w) for w in glob.glob(f'{CANDS}/{obj}/*/*/*/wrist_se3.npy')])

for d in sorted(glob.glob(f'{DS}/{obj}/*/')):
    q=np.load(f'{d}/raw/arm/position.npy')
    C2R=np.load(f'{d}/C2R.npy'); pw=np.load(f'{d}/pose_world.npy')
    obj_robot=np.linalg.inv(C2R)@pw
    # wrist z over traj
    zs=np.array([fk_wrist(q[t])[2,3] for t in range(0,len(q),20)])
    idx=np.arange(0,len(q),20)
    # grasp frame = last frame near min-z before final lift
    zmin=zs.min(); thr=zmin+0.03
    near=idx[zs<=thr]; gf=near[-1] if len(near) else idx[zs.argmin()]
    wr=fk_wrist(q[gf]); g_obj=np.linalg.inv(obj_robot)@wr
    # tabletop pose match
    v=grav(obj_robot); k,a=min(((kk,np.degrees(np.arccos(np.clip(v@vt,-1,1)))) for kk,vt in ttv.items()),key=lambda x:x[1])
    # candidate match
    cd=np.linalg.norm(cand[:,:3,3]-g_obj[:3,3],axis=1); j=cd.argmin()
    print(f'{os.path.basename(d.rstrip("/"))}: grasp_frame={gf} pose={k}({a:.0f}d) grasp_obj={np.round(g_obj[:3,3],3)} nearest_cand#{j} d={cd[j]*1000:.0f}mm')
