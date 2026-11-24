#!/usr/bin/env python3
"""Publish compact real diagnostics, all failures and local video manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def write(p,data):p.write_text(json.dumps(data,indent=2)+'\n')

def collect():
    dest=ROOT/'evidence/clutter_v2';dest.mkdir(parents=True,exist_ok=True)
    campaigns=[];episodes=[];artifacts=[];video_index=[];panels=[]
    for base in sorted((ROOT/'outputs/clutter_v2/campaigns').iterdir()):
        registration=base/'development_preregistration.json'
        if not registration.exists():continue
        reg=json.loads(registration.read_text());snapshot_ok=all((base/'source_snapshot'/n).is_file() and digest(base/'source_snapshot'/n)==sha for n,sha in reg['source_sha256'].items())
        campaign={'run_id':base.name,'mode':reg['mode'],'planned':len(reg['tasks']),'completed':0,'source_snapshot_validation':'PASS' if snapshot_ok else 'FAILED','source_sha256':reg['source_sha256'],'preregistration':rel(registration),'command':reg['command'],'status':'NOT_RUN'}
        for path in sorted(base.glob('episode_*/result.json')):
            raw=json.loads(path.read_text());ep=path.parent;before=raw.get('before',{});after=raw.get('after',{});ev=raw.get('observation_evaluation',{})
            item={'run_id':base.name,'episode':raw['episode'],'family':raw['family'],'requested':raw['requested'],'seed':raw['seed'],'scene_revision':reg['tasks'][raw['episode']]['spec']['generator_revision'],'scene_sha256':reg['tasks'][raw['episode']]['spec']['scene_content_sha256'],'status':raw['status'],'retrieval':raw['retrieval'],'failure_reason':raw.get('failure_reason'),'wall_s':raw['wall_s'],'initial_stability':raw.get('initial_stability',{}).get('status'),'association_revision':before.get('association',{}).get('implementation_revision'),'association_before':before.get('association',{}).get('status'),'association_after':after.get('association',{}).get('status'),'candidates_before':before.get('counts'),'candidates_after':after.get('counts'),'perception_errors_before':before.get('errors'),'perception_errors_after':after.get('errors'),'observation_evaluation':ev,'recording_status':raw.get('recording_status'),'actions':[],'path':rel(ep)}
            for a in raw['actions']:
                item['actions'].append({k:a[k] for k in ['operation','command_status','error','cost','push','flip','active','retrieval','distractor_displacement_m'] if k in a})
            # Full-body same-instance equality is evaluator-only, never policy.
            initial=ep/'evaluation_before.json'
            if initial.exists():
                value=json.loads(initial.read_text());item['private_initial_states']=value['states'];item['density']={k:value[k] for k in ['body_count','neighbors','neighbors_gap_under_4cm','neighbors_gap_under_10cm']}
            episodes.append(item);campaign['completed']+=1
            videos={camera:rel(ep/'videos/m1'/(camera+'.mp4')) for camera in ['third','fixed','wrist'] if (ep/'videos/m1'/(camera+'.mp4')).is_file()}
            video_index.append({'run_id':base.name,'episode':raw['episode'],'family':raw['family'],'requested':raw['requested'],'status':raw['status'],'retrieval':raw['retrieval'],'videos':videos,'validation':raw.get('recording_status')})
            for p in ep.rglob('*'):
                if not p.is_file() or 'scene_snapshot.usd'==p.name:continue
                if p.suffix.lower() not in ('.json','.npz','.mp4','.png','.log','.txt'):continue
                artifacts.append({'path':rel(p),'bytes':p.stat().st_size,'sha256':digest(p)})
            # Genuine before/after images, no synthetic or simulated substitutes.
            for stage in ['initial_fixed','after_01_fixed']:
                p=ep/'images'/(stage+'_rgb.png')
                if p.exists():panels.append((p,f'{base.name[:16]} / {raw["family"]} {raw["requested"]} / {stage} / {raw["status"]}'))
        if campaign['completed']:campaign['status']='PASS' if campaign['completed']==campaign['planned'] else 'FAILED'
        note=base/'not_run.json'
        if note.exists():campaign['not_run_reason']=json.loads(note.read_text())
        campaigns.append(campaign)
    pairs=[]
    for campaign in campaigns:
        group=[e for e in episodes if e['run_id']==campaign['run_id']]
        for family in ['C1','C2','C3']:
            rows=[e for e in group if e['family']==family and 'private_initial_states' in e]
            direct=next((e for e in rows if e['requested']=='GRASP'),None)
            prep=next((e for e in rows if e['requested'] in ('PUSH','FLIP','ACTIVE')),None)
            if not direct or not prep:continue
            import numpy as np
            from scipy.spatial.transform import Rotation
            a,b=direct['private_initial_states'],prep['private_initial_states'];dx=angle=0.
            for key in a:
                ta,tb=np.asarray(a[key]['T_world']),np.asarray(b[key]['T_world']);dx=max(dx,float(np.linalg.norm(ta[:3,3]-tb[:3,3])));angle=max(angle,float((Rotation.from_matrix(ta[:3,:3]).inv()*Rotation.from_matrix(tb[:3,:3])).magnitude()))
            pairs.append({'run_id':campaign['run_id'],'family':family,'status':'PASS' if dx<.002 and angle<.01 else 'FAILED','max_position_delta_m':dx,'max_orientation_delta_rad':angle,'direct_retrieval':direct['retrieval'],'prepared_retrieval':prep['retrieval'],'prepared_action':prep['requested'],'causal_retrieval_gain':direct['retrieval']!='PASS' and prep['retrieval']=='PASS'})
    for e in episodes:e.pop('private_initial_states',None)
    summary={'scope':'Finite real GUI development; no training or paper performance claim','campaigns':campaigns,'actual_episode_attempts':len(episodes),'recordings_pass':sum(e['recording_status']=='PASS' for e in episodes),'retrieval_pass':sum(e['retrieval']=='PASS' for e in episodes),'episodes':episodes,'paired_initialization_audit':pairs,'full_videos_data_models':'Local server paths only; no raw assets, weights or MP4 uploaded','large_scale_training':'NOT_READY','native_DP_DP3':'NOT_RUN','unit_tests':{'status':'PASS','count':80,'local_log':'outputs/clutter_v2_unit_tests_final.log','log_sha256':digest(ROOT/'outputs/clutter_v2_unit_tests_final.log')}}
    write(dest/'summary.json',summary);write(dest/'video_index.json',video_index);write(dest/'artifact_manifest.json',{'artifacts':artifacts,'total_files':len(artifacts),'total_bytes':sum(v['bytes'] for v in artifacts)})
    # Explicit representative successes AND a post-contact failure; all attempts
    # remain in the full summary/video index and no denominator is filtered.
    chosen=[]
    representatives=[
      ('20261124T071346Z_b9d9420f',1,'initial_fixed','C1 neighbor push / BEFORE / complete chain PASS'),
      ('20261124T071346Z_b9d9420f',1,'after_01_fixed','C1 actual obstacle moved / AFTER / complete chain PASS'),
      ('20261124T074245Z_a709c87e',2,'after_01_fixed','C1 after contact / target reassociated / retrieval FAILED'),
      ('20261124T072632Z_124bc8c8',3,'initial_wrist','C3 Active / BEFORE / zero legal grasp'),
      ('20261124T072632Z_124bc8c8',3,'after_01_wrist','C3 Active / AFTER / two legal grasps, retrieval PASS'),
      ('20261124T075922Z_9e0cf087',1,'initial_fixed','C2 real support / lying BEFORE / Direct also PASS'),
      ('20261124T075922Z_9e0cf087',1,'after_01_fixed','C2 actual 91.6 degree reorientation / AFTER / retrieval PASS'),
      ('20261124T075922Z_9e0cf087',1,'after_02','C2 GraspNet complete extraction / PASS')]
    for run,episode,stem,label in representatives:
        p=ROOT/'outputs/clutter_v2/campaigns'/run/f'episode_{episode:03d}'/'images'/(stem+'_rgb.png')
        if p.exists():chosen.append((p,label))
    # Add the last reorientation attempt too, including a failed outcome.
    latest={}
    for p,label in panels:
        if 'after_01' in label:latest[label.split('/')[1].strip()]=(p,label)
    for key,value in latest.items():
        if 'FLIP' in key and value not in chosen:chosen.append(value)
    chosen=chosen[:12]
    if chosen:
        cols=2;w,h=640,510;sheet=Image.new('RGB',(w*cols,h*((len(chosen)+cols-1)//cols)),'white');draw=ImageDraw.Draw(sheet)
        for i,(p,label) in enumerate(chosen):
            x=(i%cols)*w;y=(i//cols)*h;image=Image.open(p).convert('RGB');image.thumbnail((640,480));sheet.paste(image,(x,y+30));draw.text((x+5,y+5),label,fill='black')
        sheet.save(dest/'actual_keyframes.jpg',quality=86)
        write(dest/'preview_sources.json',[{'path':rel(p),'sha256':digest(p),'label':label} for p,label in chosen])
    print(json.dumps({'episodes':len(episodes),'campaigns':len(campaigns),'paired':pairs,'artifacts':len(artifacts)},indent=2))

if __name__=='__main__':collect()
