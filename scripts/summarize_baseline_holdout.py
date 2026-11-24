#!/usr/bin/env python3
"""Audit all six physical methods' initial conditions and actual inference times."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'evidence/baselines/heldout_audit.json');args=p.parse_args()
    records=[json.loads(line) for line in (args.campaign/'episodes.jsonl').read_text().splitlines()]
    registration=json.loads((args.campaign/'heldout_preregistration.json').read_text())
    actual=hashlib.sha256((ROOT/'scripts/run_baseline_holdout.py').read_bytes()).hexdigest()
    if actual!=registration['evaluation_runner_sha256']:raise RuntimeError('Evaluation runner changed after preregistration')
    methods=['grasp_only','heuristic','random_valid','dqn','bc','dp'];scenes=defaultdict(list)
    for record in records:scenes[record['scene_id']].append(record)
    audits=[]
    for scene,group in scenes.items():
        errors=[];distance=angle=calibration=observation_difference=candidate_difference=0.
        masks_equal=pools_equal=ids_equal=True
        if sorted(r['method'] for r in group)!=sorted(methods):errors.append('Six methods not present exactly once')
        for a,b in combinations(group,2):
            if set(a['initial_state'])!=set(b['initial_state']):errors.append('Body identities differ');continue
            for name in a['initial_state']:
                at=np.asarray(a['initial_state'][name]['T_world']);bt=np.asarray(b['initial_state'][name]['T_world'])
                distance=max(distance,float(np.linalg.norm(at[:3,3]-bt[:3,3])))
                angle=max(angle,float((Rotation.from_matrix(at[:3,:3]).inv()*Rotation.from_matrix(bt[:3,:3])).magnitude()))
            with np.load(args.campaign/a['episode_id']/'decision_00/observation.npz',allow_pickle=False) as av,np.load(args.campaign/b['episode_id']/'decision_00/observation.npz',allow_pickle=False) as bv:
                masks_equal=masks_equal and np.array_equal(av['mask'],bv['mask'])
                pools_equal=pools_equal and np.array_equal(av['candidates'],bv['candidates'])
                observation_difference=max(observation_difference,float(abs(av['obs']-bv['obs']).max()))
                candidate_difference=max(candidate_difference,float(abs(av['candidates']-bv['candidates']).max()))
                for camera in ['fixed','wrist']:
                    for key in ['K','T_world_optical']:calibration=max(calibration,float(np.abs(av[camera+'_'+key]-bv[camera+'_'+key]).max()))
            ap=json.loads((args.campaign/a['episode_id']/'decision_00/candidates.json').read_text())['candidates']
            bp=json.loads((args.campaign/b['episode_id']/'decision_00/candidates.json').read_text())['candidates']
            ids_equal=ids_equal and [e['candidate_id'] for e in ap]==[e['candidate_id'] for e in bp]
        if distance>.002 or angle>.01:errors.append('Initial pose tolerance exceeded')
        if any(r['scene_stability']['status']!='PASS' for r in group):errors.append('An initial stability gate failed')
        if not masks_equal or not pools_equal or not ids_equal:errors.append('Initial candidate pool/mask differs between methods')
        audits.append({'scene_id':scene,'status':'FAILED' if errors else 'PASS','errors':errors,
            'methods':[r['method'] for r in group],'max_pairwise_position_m':distance,'max_pairwise_orientation_rad':angle,
            'max_camera_K_T_element_difference':calibration,'unique_actual_initial_input_hashes':len({r['initial_observation_sha256'] for r in group}),
            'exact_same_initial_masks':masks_equal,'exact_same_initial_candidate_arrays':pools_equal,'exact_same_initial_candidate_ids':ids_equal,
            'max_initial_observation_feature_difference':observation_difference,'max_initial_candidate_feature_difference':candidate_difference,
            'note':'Physics/parameters audited; any real rendering hash variation is retained, not replaced with cached synthetic frames'})
    inference={}
    for method in methods:
        rows=[r for r in records if r['method']==method];times=[d['elapsed_s'] for r in rows for d in r['policy_inference']]
        inference[method]={'episodes':len(rows),'successes':sum(r['success'] for r in rows),'decisions':len(times),
            'mean_actual_act_wall_s':float(np.mean(times)) if times else None,'max_actual_act_wall_s':max(times) if times else None,
            'actual_actions':[[s['type'] for s in r['steps']] for r in rows]}
    result={'status':'PASS' if len(audits)==2 and all(a['status']=='PASS' for a in audits) else 'FAILED',
        'campaign':str(args.campaign.resolve().relative_to(ROOT)),'registration':str((args.campaign/'heldout_preregistration.json').resolve().relative_to(ROOT)),
        'evaluation_runner_sha256_verified':actual,'same_six_methods_and_two_cases':len(audits)==2 and all(sorted(a['methods'])==sorted(methods) for a in audits),'initialization':audits,'actual_policy_inference':inference,
        'scope':'Private evaluator pose comparisons and real act() timing; no oracle state fed to models; no statistical method ranking'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
