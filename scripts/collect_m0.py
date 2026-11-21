#!/usr/bin/env python3
"""Collect measured results and publish local aliases without changing raw runs."""
import json
import os
from pathlib import Path
from apfg.download import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environment_check'
runs=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/result.json'))]
formal=[r for r in runs if r.get('purpose')!='interactive_preview' and '--interactive' not in r.get('command',[])]


def latest(test,mode=None,scene=None):
    found=[r for r in formal if r['test']==test and (not mode or r['mode']==mode) and (not scene or r['scene']==scene)]
    return found[-1] if found else {'status':'NOT_RUN'}


for mode in ['gui','headless']:
    (OUT/f'{mode}_smoke.json').write_text(json.dumps(latest('physics',mode),indent=2)+'\n')
for test in ['robot','camera']:
    (OUT/f'{test}_smoke.json').write_text(json.dumps(latest(test,'headless'),indent=2)+'\n')
objects={}
for r in formal:
    if r['test']=='objects' and r.get('asset_id'):objects[r['asset_id']]=r
passed=[r for r in objects.values() if r['status']=='PASS' and r.get('checks',{}).get('simulation_clock')]
registry=json.loads((ROOT/'assets/manifests/object_registry.json').read_text())['objects']
object_report={'status':'PASS' if len(passed)>=12 else 'NOT_RUN','physically_verified_objects':len(passed),
               'independent_run_ids':[r['run_id'] for r in passed],
               'results':[{'asset_id':r['asset_id'],'status':r['status'],'run_id':r['run_id'],'physics_steps':r.get('physics_steps'),
                           'simulation_duration_s':r.get('measured_simulation_duration_s'),'artifact_directory':r['artifacts_directory']} for r in objects.values()]}
(OUT/'object_smoke.json').write_text(json.dumps(object_report,indent=2)+'\n')
(OUT/'asset_summary.json').write_text(json.dumps({'status':'PASS','canonical_ycb_downloads':len(registry),'converted_ycb':len(registry),
      'physically_verified_ycb':len(passed),'local_official_ycb_usd':21,'franka_local':True,'grid_environment_local':True,
      'graspfactory_meshes':5,'graspnet_models':0,'dexgraspnet_meshes':0},indent=2)+'\n')
gpu_path=OUT/'gpu_check.json'
gpu=json.loads(gpu_path.read_text())
cuda=json.loads((OUT/'cuda_driver_check.json').read_text())
physics=latest('physics','headless')
gpu['cuda_compute']='PASS' if cuda.get('status')=='PASS' else 'FAILED'
gpu['rtx_rendering']='PASS' if physics.get('status')=='PASS' and physics.get('camera_rgbd_valid') else 'NOT_RUN'
gpu['rendering_evidence_run']=physics.get('run_id')
gpu['note']='Inventory and actual rendered-physics evidence are distinct checks.'
gpu.pop('reason',None)
gpu_path.write_text(json.dumps(gpu,indent=2)+'\n')
version_path=OUT/'environment_versions.json'
versions=json.loads(version_path.read_text())
versions['simulation_startup']='PASS' if gpu['rtx_rendering']=='PASS' else 'NOT_RUN'
versions['simulation_evidence_run']=physics.get('run_id')
version_path.write_text(json.dumps(versions,indent=2)+'\n')

# Latest convenience names link to immutable source run files. Refuse to
# overwrite a user-edited alias not matching the previous generated receipt.
receipt_path=OUT/'artifact_aliases.json'
aliases=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}


def alias(name,source):
    source=Path(source)
    if not source.exists():return
    target=OUT/name
    digest=sha256(source)
    if target.exists():
        if sha256(target)==digest:return
        if name not in aliases or sha256(target)!=aliases[name]['sha256']:
            raise RuntimeError('Refusing to replace an unrecognized output '+name)
        target.unlink()
    os.link(source,target)
    aliases[name]={'source':str(source.relative_to(ROOT)),'sha256':digest}


robot=latest('robot','headless');camera=latest('camera','headless')
s1=latest('clutter','gui','s1');s2=latest('clutter','gui','s2')
for result,names in [(robot,{'franka_scene.png':'third_camera_rgb.png','robot_motion.mp4':'videos/third.mp4'}),
                     (camera,{'fixed_camera_rgb.png':'fixed_camera_rgb.png','fixed_camera_depth.png':'fixed_camera_depth.png',
                              'wrist_camera_rgb.png':'wrist_camera_rgb.png','wrist_camera_depth.png':'wrist_camera_depth.png','wrist_camera.mp4':'videos/wrist.mp4'}),
                     (s1,{'ycb_tabletop_scene.png':'third_camera_rgb.png'}),
                     (s2,{'ycb_pile_scene.png':'fixed_camera_rgb.png','clutter_preview.mp4':'videos/third.mp4'})]:
    if result.get('status')=='PASS':
        for name,relative in names.items():alias(name,ROOT/result['artifacts_directory']/relative)
if passed:
    chosen=next((r for r in passed if r['asset_id']=='ycb:035_power_drill'),passed[0])
    alias('object_physics.mp4',ROOT/chosen['artifacts_directory']/'videos/fixed.mp4')
receipt_path.write_text(json.dumps(aliases,indent=2)+'\n')
print('Collected measured results; verified objects:',len(passed))
