#!/usr/bin/env python3
"""Conservative summary: every required gate needs an actual PASS measurement."""
import datetime as dt
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environment_check'


def load(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else {'status':'NOT_RUN'}


def main():
    runs=[]
    for p in sorted((OUT/'runs').glob('*/result.json')):
        runs.append(json.loads(p.read_text()))
    latest={}
    for r in runs:
        if r.get('purpose')=='interactive_preview' or '--interactive' in r.get('command',[]):continue
        latest[(r['test'],r['mode'],r.get('scene'),r.get('asset_id'))]=r
    def status(test,mode=None,scene=None):
        selected=[r for r in latest.values() if r['test']==test and (mode is None or r['mode']==mode) and (scene is None or r.get('scene')==scene)]
        if not selected:return 'NOT_RUN'
        r=selected[-1]
        return 'NOT_RUN' if r['status']=='PASS' and not r.get('checks',{}).get('simulation_clock',False) else r['status']
    object_passes={o for r in latest.values() if r['test']=='objects' and r['status']=='PASS' and r.get('checks',{}).get('simulation_clock',False) for o in r.get('objects',[])}
    gates={
        'headless_physics':status('physics','headless'), 'gui_physics':status('physics','gui'),
        'franka_control_and_gripper':status('robot'), 'fixed_and_wrist_rgbd':status('camera'),
        'twelve_independent_ycb_drops':'PASS' if len(object_passes)>=12 else 'NOT_RUN',
        's1_headless':status('clutter','headless','s1'), 's2_headless':status('clutter','headless','s2'),
        's1_gui':status('clutter','gui','s1'), 's2_gui':status('clutter','gui','s2'),
        'local_usd_dependencies':load('offline_asset_check.json')['status'],
        'storage_audit':load('storage_audit.json')['status'],
        'gpu_compute_and_rendering':load('gpu_check.json').get('rtx_rendering','NOT_RUN'),
        'installation':load('environment_versions.json').get('installation_status','NOT_RUN'),
        'github_sync':load('github_sync.json').get('status','NOT_RUN')}
    cuda=load('cuda_driver_check.json')
    rendered_physics=any(r.get('status')=='PASS' and r.get('test')=='physics' and r.get('camera_rgbd_valid') for r in runs)
    if cuda.get('cuInit_returncode')==0 and rendered_physics:
        gates['gpu_compute_and_rendering']='PASS'
    if cuda.get('cuInit_returncode',0) != 0:
        gates['gpu_compute_and_rendering']='BLOCKED'
        for k in ['headless_physics','gui_physics','franka_control_and_gripper','fixed_and_wrist_rgbd','twelve_independent_ycb_drops',
                  's1_headless','s2_headless','s1_gui','s2_gui','gpu_compute_and_rendering']:
            if gates[k]!='PASS': gates[k]='BLOCKED'
    ready=all(v=='PASS' for v in gates.values())
    registry=json.loads((ROOT/'assets/manifests/object_registry.json').read_text())
    result={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'PASS' if ready else 'BLOCKED',
            'APFG_M0_READY':ready,'gates':gates,'registered_ycb_count':len(registry['objects']),
            'physically_verified_ycb_count':len(object_passes),'runs':runs,
            'limitation':'Static conversion and dependency PASS do not establish stable contact or render correctness.'}
    (OUT/'m0_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','APFG_M0_READY','registered_ycb_count','physically_verified_ycb_count','gates']},indent=2))
    return 0 if ready else 2


if __name__=='__main__': raise SystemExit(main())
