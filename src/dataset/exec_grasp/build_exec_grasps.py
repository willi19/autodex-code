import numpy as np, glob, os, json, sys
DS='/home/mingi/shared_data/autodex_dataset/selected_100'
RSSB='/home/mingi/shared_data/RSS2026_Mingi/candidates/selected_100_exp'
OBJROOT='/home/mingi/shared_data/object_processing'
MATCH_THRESH=0.05

def conv(hp):  # _convert_allegro: thumb(last4)->front  (candidate grasp_pose -> hand action order)
    o=hp.copy(); o[...,:4]=hp[...,12:]; o[...,4:]=hp[...,:12]; return o
def grav(T): return T[:3,:3].T@np.array([0,0,1.])

def process(obj):
    cpaths=sorted(glob.glob(f'{RSSB}/{obj}/*/*/*/wrist_se3.npy'))
    if not cpaths: return None
    cw=np.array([np.load(p) for p in cpaths])
    cf=np.array([np.load(p.replace('wrist_se3','grasp_pose')) for p in cpaths])
    cf_act=conv(cf)  # in hand-action order for matching
    tt=sorted(glob.glob(f'{OBJROOT}/{obj}/processed_data/info/tabletop/*.npy'))
    ttv={os.path.basename(t)[:-4]:grav(np.load(t)) for t in tt}
    n_ok=n_miss=0
    for d in sorted(glob.glob(f'{DS}/{obj}/*/')):
        haf=f'{d}/raw/hand/action.npy'
        if not os.path.exists(haf): continue
        ha=np.load(haf)
        if ha.shape[1]!=16: continue  # allegro only
        best=(9,-1,-1)
        for t in range(0,len(ha),3):
            dd=np.linalg.norm(cf_act-ha[t],axis=1); j=dd.argmin()
            if dd[j]<best[0]: best=(dd[j],t,j)
        L2,frame,ci=best
        if L2>MATCH_THRESH: n_miss+=1; continue
        # pose id
        pose_id='?'
        try:
            C2R=np.load(f'{d}/C2R.npy'); pw=np.load(f'{d}/pose_world.npy'); obj_r=np.linalg.inv(C2R)@pw
            v=grav(obj_r); pose_id=min(ttv.items(),key=lambda kv:np.arccos(np.clip(v@kv[1],-1,1)))[0]
        except Exception: pass
        # grasp timing: raw hand-action frame -> timestamp -> synced(video) frame
        gtime=None; synced=None
        try:
            ht=np.load(f'{d}/raw/hand/time.npy'); gtime=float(ht[min(frame,len(ht)-1)])
            sts=np.load(f'{d}/raw/timestamps/timestamp.npy'); synced=int(np.abs(sts-gtime).argmin())
        except Exception: pass
        out=os.path.join(d,'executed_grasp'); os.makedirs(out,exist_ok=True)
        np.save(f'{out}/wrist_se3.npy', cw[ci])          # object frame 4x4
        np.save(f'{out}/grasp_pose.npy', cf[ci])         # finger (cuRobo order)
        json.dump({'obj':obj,'ts':os.path.basename(d.rstrip('/')),'cand_path':cpaths[ci],
                   'finger_L2':float(L2),'match_frame':int(frame),'grasp_time':gtime,
                   'grasp_frame_synced':synced,'pose_id':pose_id},
                  open(f'{out}/meta.json','w'), indent=1)
        n_ok+=1
    return n_ok,n_miss

if __name__=='__main__':
    import os as _os
    objs=sys.argv[1:] or sorted(d for d in _os.listdir(DS) if _os.path.isdir(f'{DS}/{d}'))
    for o in objs:
        r=process(o)
        print(o, '->', ('no RSS cand' if r is None else f'saved {r[0]}, miss {r[1]}'))
