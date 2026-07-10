"""Unify executed_grasp meta schema across RSS + corl and tag note=version."""
import json, glob
CANON=['obj','ts','hand','note','pose_id','tabletop_before','finger_L2','grasp_frame',
       'grasp_time','wrist_source','execution_states','candidate_idx','cand_path',
       'scene_info','success','grasp_frame_synced','arm_frame']
def harmonize(pat, note):
    n=0
    for mp in glob.glob(pat):
        m=json.load(open(mp))
        m['note']=note
        m.setdefault('hand','allegro')
        if 'grasp_frame' not in m and 'match_frame' in m: m['grasp_frame']=m['match_frame']
        for k in CANON: m.setdefault(k, None)
        extra=[k for k in m if k not in CANON]  # keep post-canonical keys (reclassify fields)
        m={k:m.get(k) for k in CANON+extra}     # reorder to canonical, non-destructive
        json.dump(m, open(mp,'w'), indent=1); n+=1
    return n
r=harmonize('/home/mingi/shared_data/autodex_dataset/selected_100/*/*/executed_grasp/meta.json','RSS')
c=harmonize('/home/mingi/shared_data/autodex_dataset/corl_selected_100/*/*/executed_grasp/meta.json','corl')
i=harmonize('/home/mingi/shared_data/autodex_dataset/selected_100_inspire/*/*/executed_grasp/meta.json','inspire')
print(f'harmonized: RSS={r}, corl={c}, inspire={i}')
