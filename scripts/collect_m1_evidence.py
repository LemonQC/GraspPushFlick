#!/usr/bin/env python3
"""Publish compact metadata/real sensor previews, retaining every M1 pilot."""
import hashlib
import json
from pathlib import Path
import math
import re

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative(value):
    if isinstance(value,dict):return {k:relative(v) for k,v in value.items()}
    if isinstance(value,list):return [relative(v) for v in value]
    if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
    return value


def main():
    from PIL import Image,ImageDraw
    out=ROOT/'evidence/m1';out.mkdir(parents=True,exist_ok=True)
    unit_log=ROOT/'outputs/environment_check/m1_unit_tests.log'
    unit_text=unit_log.read_text();unit_count=re.search(r'Ran (\d+) tests?',unit_text)
    unit_passed=unit_text.rstrip().endswith('OK') and unit_count is not None
    runs=[];artifacts=[]
    for directory in sorted((ROOT/'outputs/m1/runs').iterdir()):
        if not directory.is_dir():continue
        config=json.loads((directory/'run_config.json').read_text())
        scene=json.loads((directory/'scene.json').read_text())
        file=directory/'result.json'
        if not file.exists():raise RuntimeError(f'Unfinished pilot must be accounted for: {directory.name}')
        result=json.loads(file.read_text());evaluation=result.get('action_evaluation',[])
        camera_motion=None
        initial_camera=directory/'images/initial_wrist_calibration.json'
        final_camera=directory/'images/final_wrist_calibration.json'
        if initial_camera.exists() and final_camera.exists():
            first=json.loads(initial_camera.read_text());last=json.loads(final_camera.read_text())
            camera_motion={'translation_m':math.dist(first['position_m'],last['position_m']),
                'source':'actual rendered wrist calibration poses; not object GT or information-gain score'}
        compact=[]
        for e in evaluation:
            compact.append({k:v for k,v in e.items() if k not in ['before','after']})
        timing=[]
        for p in directory.glob('videos/*/timing.json'):
            samples=json.loads(p.read_text())['sampling']
            if len(samples)>1:
                sim=samples[-1]['simulation_time_s']-samples[0]['simulation_time_s']
                wall=samples[-1]['wall_monotonic_s']-samples[0]['wall_monotonic_s']
                timing.append({'path':str(p.relative_to(ROOT)),'frames':len(samples),
                    'actual_simulation_capture_hz':(len(samples)-1)/sim,'wall_capture_hz':(len(samples)-1)/wall})
        runs.append(relative({'run_id':directory.name,'family':scene['family'],'seed':scene['seed'],
            'scene_generator_revision':scene.get('generator_revision',1),'scene_sha256':scene['scene_content_sha256'],
            'action':config['args']['action'],'mode':result['mode'],'information_mode':result['information_mode'],
            'status':result['status'],'reason':result.get('reason'),'command':['bash','scripts/run_m1.sh']+config['command'][1:],
            'scope':'Development smoke only; not a formal E1 episode or necessity label',
            'scene_stability':result.get('scene_stability'),'actual_settling_duration_s':result.get('actual_settling_duration_s'),
            'evaluation':compact,'initial_visible_hint_pixels':result.get('initial_visible_hint_pixels'),
            'final_visible_hint_pixels':result.get('final_visible_hint_pixels'),
            'costs':result.get('costs'),'performance':result.get('performance'),'videos':result.get('videos'),
            'capture_timing':timing,'local_directory':str(directory.relative_to(ROOT))}))
        runs[-1]['observed_wrist_camera_motion']=camera_motion
        # All local run files are hashed, including failed runs, RGB-D, raw
        # predictions, contact logs and full videos. No binary is uploaded here.
        for p in sorted(directory.rglob('*')):
            if p.is_file():artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
    by_id={r['run_id']:r for r in runs}
    final={'Direct':'20261122T095317Z_45c90433','C1':'20261122T094514Z_639cae5e',
           'C2':'20261122T094547Z_5578e311','C3':'20261122T095447Z_aa41cca7','C4':'20261122T095446Z_81d1e0be'}
    primitives={'Active':'20261122T095447Z_aa41cca7','Push':'20261122T095141Z_33fd0fc3',
                'Flip_ORACLE_GUI':'20261122T095316Z_b417c953','Grasp_RGBD_full_retrieval':'20261122T095317Z_45c90433'}
    summary={'scope':'M1 implementation and small development pilots; no paper success-rate claim',
        'stack':{'gpu':'RTX 3090','driver':'535.230.02','isaac_sim':'4.5.0 binary','isaac_lab':'v2.0.2',
                 'python':'3.10.22','torch':'2.5.1+cu118','physics':'CPU PhysX MBP TGS 480Hz','rendering':'RTX RGB-D 640x480, recorded 30fps'},
        'scene_stability_gates':{k:{'status':by_id[v]['scene_stability']['status'],'run_id':v} for k,v in final.items()},
        'primitive_gates':{k:{'status':by_id[v]['status'],'run_id':v} for k,v in primitives.items()},
        'GUI':{'status':'PASS','run_ids':['20261122T094653Z_20726c4f','20261122T095316Z_b417c953']},
        'Headless':{'status':'PASS','run_ids':list(primitives.values())[:2]+['20261122T095317Z_45c90433']},
        'E1':{'status':'NOT_RUN','protocol':'configs/m1/e1_protocol_v2.json','cases':16,'planned_episodes':64},
        'E2':'NOT_RUN','E3':'NOT_RUN','training':'NOT_RUN','mechanism_qualification':'NOT_RUN',
        'CPU_validation':{'status':'PASS' if unit_passed else 'FAILED','unit_tests':int(unit_count.group(1)) if unit_count else None,'command':'source scripts/project_env.sh; "$APFG_PYTHON" -m unittest discover -s tests -v',
            'local_log':'outputs/environment_check/m1_unit_tests.log','log_sha256':digest(unit_log),
            'additional_checks':['Python compileall apfg scripts','bash -n scripts/run_m1.sh']},
        'limitations':['Scene families are unverified mechanism hypotheses; C4 is inspection only.',
            'Active passed motion/no-object-disturbance; information or retrieval benefit is NOT_RUN. Visible cue pixels decreased in the tested move.',
            'Flip primitive used explicit ORACLE_DIAGNOSTIC contact parameters; C2 Flip-to-Grasp is NOT_RUN.',
            'Push displaced and toppled the target; it did not preserve upright orientation.',
            'RGB cue/ROI is explicit task input, not general target recognition. Connected measured surfaces may merge touching objects.',
            'Bounded source-convex-hull/multiview measured-cloud checks do not certify unobserved geometry.',
            'PGIA/camera source boxes/extrinsics and YCB mass/friction/inertia remain approximate or estimated.',
            'C4 cracker rocking failures retained; final C4-only front distractor is real YCB wood block, unsuitable as an 80mm-jaw target.'],
        'all_pilots':runs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sources=[]
    panels=[('C1','20261122T094514Z_639cae5e','initial_scene'),('C2','20261122T094547Z_5578e311','initial_scene'),
            ('C3','20261122T095447Z_aa41cca7','initial_scene'),('C4','20261122T095446Z_81d1e0be','initial_scene'),
            ('Flip before ORACLE','20261122T095316Z_b417c953','initial_scene'),('Flip after ORACLE','20261122T095316Z_b417c953','final_scene'),
            ('GraspNet candidates RGBD','20261122T095317Z_45c90433','grasp_candidates'),('Complete extraction RGBD','20261122T095317Z_45c90433','final_scene')]
    sheet=Image.new('RGB',(960,4*388),(240,240,240));draw=ImageDraw.Draw(sheet)
    for i,(label,rid,name) in enumerate(panels):
        source=ROOT/'outputs/m1/runs'/rid/'images'/(name+'_rgb.png')
        frame=Image.open(source).convert('RGB');frame.thumbnail((480,360));x=(i%2)*480;y=(i//2)*388
        sheet.paste(frame,(x,y+28));draw.text((x+6,y+6),label,fill=(0,0,0))
        sources.append({'panel':label,'source':str(source.relative_to(ROOT)),'source_sha256':digest(source)})
    sheet.save(out/'scene_and_actions.jpg',quality=85)
    (out/'preview_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    for p in [ROOT/'com2.md',ROOT/'docs/M1_M6_SCENE_TRAINING_BLUEPRINT_2026-11-22.md',ROOT/'configs/m1/e1_protocol_v1.json',ROOT/'configs/m1/e1_protocol_v2.json']:
        artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
    (out/'artifact_manifest.json').write_text(json.dumps({'storage':'Full artifacts are server-local only, not GitHub downloads','files':artifacts},indent=2)+'\n')
    source_names=['AGENTS.md','README.md','apfg/primitives/sr5_control.py','apfg/simulation/sr5_hardware.py',
        'apfg/evaluation/m1_metrics.py','apfg/perception/m1_observation.py','apfg/primitives/m1_actions.py',
        'apfg/primitives/swept.py','apfg/scenes/m1.py','apfg/simulation/m1_env.py','scripts/run_m1.py',
        'scripts/run_m1.sh','scripts/prepare_m1.py','scripts/collect_m1_evidence.py','tests/test_m1.py',
        'docs/M1_REPORT_ZH.md','docs/E1_PROTOCOL_ZH.md','docs/REPRODUCIBILITY.md']
    source_names+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'configs/m1').glob('*.json'))
    (out/'source_manifest.json').write_text(json.dumps({'scope':'Final M1 source/config/document bytes; early transient development source was not separately committed',
        'files':[{'path':name,'sha256':digest(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in source_names]},indent=2)+'\n')
    print(json.dumps({'pilots_retained':len(runs),'failed_run_ids':[r['run_id'] for r in runs if r['status']=='FAILED'],
                      'manifest_files':len(artifacts),'evidence':str(out.relative_to(ROOT))}))


if __name__=='__main__':main()
