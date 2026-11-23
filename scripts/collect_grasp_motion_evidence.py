#!/usr/bin/env python3
"""Measured before/after motion evidence; never manufactures rollout frames."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative(data):
    if isinstance(data,dict):return {k:relative(v) for k,v in data.items()}
    if isinstance(data,list):return [relative(v) for v in data]
    if isinstance(data,str):return data.replace(str(ROOT)+'/', '')
    return data


def measure(directory,phase):
    directory=ROOT/directory
    result=json.loads((directory/'result.json').read_text())
    rows=json.loads((directory/'telemetry.json').read_text())
    indices=[i for i,r in enumerate(rows) if r['phase']==phase]
    before=max(0,indices[0]-1);motion=rows[before:indices[-1]+1]
    rotations=Rotation.from_matrix(np.asarray([r['frames_world']['ral_tcp'] for r in motion])[:,:3,:3])
    joints=np.asarray([r['q'] for r in motion])
    time=np.asarray([r['time_s'] for r in motion]);time-=time[0]
    increments=np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude())
    cumulative=np.r_[0,np.cumsum(increments)]
    selected=result.get('selected_candidate')
    if selected is None:selected=result['candidates'][result['selected_rank']]
    physical=result.get('physical_evaluation')
    if physical is None:physical=next(e['retrieval'] for e in result['action_evaluation'] if e['operation']=='GRASP')
    compact={'run_id':directory.name,'directory':str(directory.relative_to(ROOT)),'status':result['status'],
        'command':relative(json.loads((directory/'run_config.json').read_text())['command']),
        'phase':phase,'pregrasp_duration_s':float(time[-1]),
        'tcp_net_rotation_deg':float(np.degrees((rotations[0].inv()*rotations[-1]).magnitude())),
        'tcp_cumulative_rotation_deg':float(cumulative[-1]),
        'joint6_net_displacement_deg':float(np.degrees(joints[-1,5]-joints[0,5])),
        'joint6_absolute_travel_deg':float(np.degrees(abs(np.diff(joints[:,5]))).sum()),
        'selected_prediction_index':selected['index'],'selected_raw_candidate':selected,
        'execution_plan':result.get('grasp_plan'),'physical_result':physical,'videos':relative(result.get('videos'))}
    return compact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m1-run',type=Path,required=True);p.add_argument('--gui-run',type=Path,required=True);args=p.parse_args()
    pairs=[('M1 complete retrieval',Path('outputs/m1/runs/20261122T095317Z_45c90433'),args.m1_run,'grasp_hover'),
           ('Standalone GUI grasp',Path('outputs/sr5_graspnet/20261122T084930Z_e8d15d13'),args.gui_run,'predicted_hover')]
    out=ROOT/'evidence/grasp_motion';out.mkdir(parents=True,exist_ok=True)
    comparisons=[];files=[]
    for name,baseline,current,phase in pairs:
        old=measure(baseline,phase);new=measure(current,phase)
        same=old['selected_raw_candidate']['raw_row']==new['selected_raw_candidate']['raw_row']
        comparisons.append({'case':name,'same_original_prediction_values':same,'before':old,'after':new})
        for directory in [baseline,current]:
            for file in sorted((ROOT/directory).rglob('*')):
                if file.is_file():files.append({'path':str(file.relative_to(ROOT)),'bytes':file.stat().st_size,'sha256':sha256(file)})
    from PIL import Image
    source=ROOT/args.gui_run/'images/execution_final_rgb.png'
    Image.open(source).convert('RGB').save(out/'gui_grasp.jpg',quality=90)
    unit=ROOT/'outputs/grasp_motion_check/unit_tests.log'
    unit_text=unit.read_text();count=re.search(r'Ran (\d+) tests?',unit_text)
    robot=ROOT/'assets/imported_sr5/isaac_sr5_export/robot/assembled_portable.urdf'
    export=json.loads((robot.parents[1]/'export_report.json').read_text())
    assert sha256(robot)==export['robot_urdf_sha256'],'Source robot changed'
    source_names=['apfg/robots/sr5.py','apfg/primitives/sr5_control.py','apfg/primitives/grasp_motion.py',
        'apfg/simulation/m1_env.py','apfg/visualization/grasp_candidates.py','scripts/run_m1.py',
        'scripts/sr5_graspnet_demo.py','scripts/collect_grasp_motion_evidence.py','tests/test_grasp_motion.py']
    playback=[]
    for name in json.loads((ROOT/'outputs/grasp_motion_check/playback_copies.json').read_text()):
        file=ROOT/name
        playback.append({'path':name,'sha256':sha256(file),'validation':json.loads(file.with_name('validation.json').read_text())})
    summary={'scope':'Controller correction and physical regression smokes; no E1/E2/E3 comparison or training',
        'controller_revision':'parallel_jaw_execution_v1','source_robot_changed':False,
        'source_robot_sha256':sha256(robot),
        'software_stack':json.loads((ROOT/'evidence/m1/summary.json').read_text())['stack'],
        'source_file_sha256':{name:sha256(ROOT/name) for name in source_names},
        'changes':['Nearest limit-valid 2pi IK representation; no changes to requested physical pose.',
            'Separate raw GraspNet prediction and checked parallel-jaw execution pose; approach/position preserved.',
            'Both full source tool assemblies checked independently; camera/plate never assumed symmetric.',
            'Executed joint goals are the same goals used by the bounded path check; no per-frame collision preflight.'],
        'unit_tests':{'status':'PASS' if count is not None and unit_text.rstrip().endswith('OK') else 'FAILED',
                      'count':int(count.group(1)) if count else None,'log_sha256':sha256(unit)},
        'comparisons':comparisons,'preview_source':{'path':str(source.relative_to(ROOT)),'sha256':sha256(source)},
        'limitations':['Necessary 6-DoF orientation changes remain; zero rotation is not guaranteed.',
            'A shorter finger-exchange variant can be rejected by geometry; no unsafe forced symmetry.',
            'Frozen E1 v1/v2 files and historical outcomes unchanged; preregister the controller revision before formal comparisons.'],
        'playback_copies':playback}
    (out/'summary.json').write_text(json.dumps(relative(summary),indent=2)+'\n')
    (out/'artifact_manifest.json').write_text(json.dumps({'storage':'Full videos and original data remain project-local','files':files},indent=2)+'\n')
    print(json.dumps({'comparisons':[{ 'case':c['case'],'same_raw_prediction':c['same_original_prediction_values'],
        'before_rotation_deg':c['before']['tcp_net_rotation_deg'],'after_rotation_deg':c['after']['tcp_net_rotation_deg'],
        'actual_after_status':c['after']['status']} for c in comparisons]}))


if __name__=='__main__':main()
