#!/usr/bin/env python3
"""Audit retained real RGBD/trajectory/source evidence; never run physics."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.evaluation.m1_metrics import pose_stability
from scripts.prepare_baseline_protocol import verify_protocol_sources

def main():
    checks=[]
    for base in sorted((ROOT/'outputs/clutter_v2/campaigns').iterdir()):
        reg=base/'development_preregistration.json'
        if not reg.exists():continue
        registration=json.loads(reg.read_text())
        for file in sorted(base.glob('episode_*/result.json')):
            result=json.loads(file.read_text());directory=file.parent;errors=[];count=0
            for name,sha in registration['source_sha256'].items():
                p=base/'source_snapshot'/name
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:errors.append('Source snapshot: '+name)
            for p in sorted(directory.glob('observation_*/rgbd.npz')):
                with np.load(p,allow_pickle=False) as d:
                    for camera in ['fixed','wrist']:
                        rgb=d[camera+'_rgb'];depth=d[camera+'_depth_m'];mask=d[camera+'_mask'];t=d[camera+'_T_world_optical'];k=d[camera+'_K']
                        if rgb.shape!=(480,640,3) or depth.shape!=(480,640) or mask.shape!=depth.shape or mask.dtype!=bool:errors.append('RGBD shape/mask '+str(p.relative_to(ROOT)))
                        if not np.isfinite(t).all() or not np.isfinite(k).all() or abs(np.linalg.det(t[:3,:3])-1)>1e-5:errors.append('Camera calibration '+str(p.relative_to(ROOT)))
                    if any('oracle' in key or 'asset_id' in key or 'T_world_object' in key for key in d.files):errors.append('GT in sensor record')
                count+=1
            telemetry=directory/'telemetry.json';stability={'status':'NOT_RUN'};bilateral=roll_frames=0
            if telemetry.exists():
                rows=json.loads(telemetry.read_text());times=[r['time_s'] for r in rows]
                if not all(b>a for a,b in zip(times,times[1:])):errors.append('Non-monotonic actual telemetry')
                time=result.get('before',{}).get('association',{}).get('time_s')
                if time is not None:
                    initial=[r for r in rows if r['time_s']<=time+1e-4]
                    if initial:stability=pose_stability(initial,list(initial[-1]['m1_objects']))
                for row in rows:
                    if not row['phase'].startswith('flip_'):continue
                    both=all(any('/World/Object' in pair and any('/ral_'+side+'_finger' in v for v in pair) for pair in row['contacts']) for side in ['left','right'])
                    bilateral+=int(both)
                    if row['phase']=='flip_reorient':roll_frames+=int(both)
            else:errors.append('Missing actual telemetry')
            for required in ['commands.json','contacts.json','association_history.json']:
                if not (directory/required).exists():errors.append('Missing '+required)
            if result.get('recording_status')!='PASS':errors.append('Recording validation failed')
            checks.append({'run_id':base.name,'episode':result['episode'],'family':result['family'],'requested':result['requested'],'status':'FAILED' if errors else 'PASS','errors':errors,'real_rgbd_records':count,'post_view_initial_stability':stability,'flip_bilateral_target_contact_frames':bilateral,'roll_bilateral_target_contact_frames':roll_frames})
    result={'scope':'Source/data/recording integrity and private measured initial stability; not a success-rate substitute','status':'PASS' if checks and all(r['status']=='PASS' for r in checks) else 'FAILED','episodes':len(checks),'frozen_E1_v3_sources':verify_protocol_sources(json.loads((ROOT/'configs/m1/e1_protocol_v3.json').read_text())),'checks':checks}
    dest=ROOT/'evidence/clutter_v2';dest.mkdir(parents=True,exist_ok=True);(dest/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'episodes':len(checks),'stability_failures':[(r['run_id'],r['episode']) for r in checks if r['post_view_initial_stability']['status']!='PASS'],'errors':[r for r in checks if r['errors']],'E1':result['frozen_E1_v3_sources']},indent=2))

if __name__=='__main__':main()
