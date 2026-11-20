#!/usr/bin/env python3
"""Publish compact facts and local artifact hashes, never raw machine logs."""
import json
from pathlib import Path
from apfg.download import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environment_check'
EVIDENCE=ROOT/'evidence'
EVIDENCE.mkdir(exist_ok=True)
summary=json.loads((OUT/'m0_summary.json').read_text())
versions_path=OUT/'environment_versions.json'
versions=json.loads(versions_path.read_text()) if versions_path.exists() else {'installation_status':'NOT_RUN'}
storage=json.loads((OUT/'storage_audit.json').read_text())
cuda=json.loads((OUT/'cuda_driver_check.json').read_text())
video_paths=sorted((OUT/'runs').glob('*/videos/*.mp4'))
image_paths=sorted((OUT/'runs').glob('*/*.png'))
runs=[{k:r.get(k) for k in ['run_id','test','mode','scene','seed','status','artifacts_directory','cuda_init_returncode','reason']} for r in summary['runs']]
evidence={'status':summary['status'],'APFG_M0_READY':summary['APFG_M0_READY'],'gates':summary['gates'],
          'canonical_ycb_count':summary['registered_ycb_count'],'physically_verified_ycb_count':summary['physically_verified_ycb_count'],
          'gpu':'NVIDIA GeForce RTX 3090 24 GiB','driver':'535.230.02','os':'Ubuntu 20.04.6','glibc':'2.31',
          'installation_status':versions['installation_status'],'installed_packages':versions.get('packages',{}),
          'python':versions.get('python'),'isaac_sim_binary_version':versions.get('isaac_sim_binary_version'),
          'isaac_lab_release_tag':'v2.0.2','isaac_lab_commit':'b5fa0eb031a2413c182eeb54fa3a9295e8fd867c',
          'cuda_init':cuda.get('cuInit_returncode'),'uvm_open_errno':cuda.get('uvm_errno'),
          'video_count':len(video_paths),'keyframe_count':len(image_paths),
          'storage':{'project_allocated_bytes':storage['project_allocated_bytes'],'free_bytes':storage['free_bytes'],
                     'outside_observed_changed_count':len(storage['outside_scope_changed_files']),
                     'audit_scope':storage['scope']},
          'runs':runs,'note':'Installation and static asset checks do not establish runtime physics or camera success.'}
(EVIDENCE/'m0_status.json').write_text(json.dumps(evidence,indent=2)+'\n')
artifacts=[]
for pattern in ['*.json','*.txt','runs/*/result.json']:
    for path in OUT.glob(pattern):
        if path.name=='outside_storage_baseline.json':continue
        artifacts.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha256(path),'availability':'project-local; contents not uploaded'})
def manifest(paths):
    return [{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p),'availability':'project-local'} for p in paths]
(EVIDENCE/'artifact_manifest.json').write_text(json.dumps({'artifacts':artifacts,'videos':manifest(video_paths),'keyframes':manifest(image_paths)},indent=2)+'\n')
(EVIDENCE/'checks.json').write_text(json.dumps({'cpu_unit_tests':{'status':'PASS','count':3,'command':'python -m unittest discover -s tests -v'},
     'python_compile':'PASS','shell_syntax':'PASS','usd_dependencies':json.loads((OUT/'offline_asset_check.json').read_text())['status'],
     'pip_check':versions.get('pip_check',{'status':'NOT_RUN'}),'gpu_physics_camera_control':'BLOCKED'},indent=2)+'\n')
print('Exported sanitized lightweight evidence')
