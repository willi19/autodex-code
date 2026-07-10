import numpy as np, glob, os, json, yourdfpy
from autodex.utils.robot_config import ALLEGRO_LINK6_TO_WRIST as L6W
DS='/home/mingi/shared_data/autodex_dataset/selected_100'
URDF='/home/mingi/shared_data/AutoDex/content/assets/robot/allegro_description/xarm_allegro.urdf'
u=yourdfpy.URDF.load(URDF,load_meshes=False,build_collision_scene_graph=False)
aj=u.actuated_joint_names[:6]
def fkw(q6):
    u.update_cfg({aj[i]:float(q6[i]) for i in range(6)}); return u.get_transform('link6','world')@L6W

metas=glob.glob(f'{DS}/*/*/executed_grasp/meta.json')
n=0; nfk=0
for mp in metas:
    m=json.load(open(mp)); d=os.path.dirname(os.path.dirname(mp))
    # scene_info from cand_path: .../{obj}/{type}/{sid}/{gid}/wrist_se3.npy
    parts=m['cand_path'].split('/'); m['scene_info']=[parts[-4],parts[-3],parts[-2]]
    # FK wrist at grasp time
    try:
        at=np.load(f'{d}/raw/arm/time.npy'); q=np.load(f'{d}/raw/arm/position.npy')
        gt=m.get('grasp_time')
        fi=int(np.abs(at-gt).argmin()) if gt is not None else int(m['match_frame']*len(q)//1)
        C2R=np.load(f'{d}/C2R.npy'); pw=np.load(f'{d}/pose_world.npy'); obj_r=np.linalg.inv(C2R)@pw
        w_obj=np.linalg.inv(obj_r)@fkw(q[min(fi,len(q)-1)])
        np.save(f'{os.path.dirname(mp)}/wrist_se3.npy', w_obj)   # overwrite: FK executed wrist (object frame)
        m['wrist_source']='fk_executed'; m['arm_frame']=fi; nfk+=1
    except Exception as e:
        m['wrist_source']='candidate(fk_failed)'; m['fk_err']=str(e)
    json.dump(m, open(mp,'w'), indent=1); n+=1
print(f'refined {n} metas, FK wrist for {nfk}')
