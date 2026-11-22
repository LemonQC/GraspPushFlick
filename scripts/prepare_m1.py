#!/usr/bin/env python3
"""Generate auditable M1 metadata and freeze a NOT_RUN E1 protocol."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.scenes.m1 import generate,registry,suitability,TARGET_POOL,FAMILIES


def main():
    out=ROOT/'configs/m1';out.mkdir(parents=True,exist_ok=True)
    samples={name:generate(name,2026112201) for name in FAMILIES}
    (out/'scene_examples.json').write_text(json.dumps(samples,indent=2)+'\n')
    r=registry();screen={n:suitability(r['ycb:'+n]) for n in TARGET_POOL+['036_wood_block']}
    (out/'target_suitability.json').write_text(json.dumps(screen,indent=2)+'\n')
    protocol={'protocol_version':'E1-development-preregister-v2','status':'NOT_RUN','scope':'Future E1 development gate, not formal paper evaluation',
        'supersedes':'e1_protocol_v1.json (kept immutable, all formal episodes NOT_RUN)',
        'pilot_revision_reason':'Tighter explicit yellow-vs-red cue, bounded settling extension, dual-RGBD observed obstacle fusion and source-housing precontact clearance. Numeric physical outcome thresholds unchanged.',
        'families':['Direct','C1','C2','C3'],'seeds':[2026112201,2026112202],
        'target_assets':['ycb:004_sugar_box','ycb:005_tomato_soup_can'],
        'arms':['DIRECT_GRASP','PUSH_GRASP','FLIP_GRASP','VIEW_GRASP'],
        'information_mode':'RGBD_POLICY','max_preparation_actions':1,'max_grasp_attempts':1,'max_candidates_checked':8,
        'thresholds':{'settling_window_s':1.,'settling_position_m':.002,'settling_rotation_rad':.01,
            'push_displacement_m':.03,'flip_non_yaw_deg':35.,'flip_horizontal_rotvec_deg':25.,
            'view_max_object_motion_m':.002,'target_entire_aabb_min_y_m':.30,'minimum_lift_m':.10,'bilateral_hold_s':1.},
        'candidate_protocol':'Fixed measured workspace, explicit visible target cue, existing GraspNet checkpoint and same source full-arm kinematic checks. No family->skill labels.',
        'operation_costs':'All physical arm motion and network wall-time logged. Any mandatory re-observation motion must be explicit and charged.',
        'success':'Entire designated object outside task region, real bilateral grasp, sustained lift/hold, no catastrophic distractor drop. A 6cm/15cm lift alone is not task success.',
        'labels':'Layout family is only a generation hypothesis. Direct-success cases cannot be called push/flip-necessary. Audit and retain non-qualified scenes.',
        'failure_retention':True,'statistics':'No success-rate or novelty claim from the separate pilot subset; E2/E3 NOT_RUN',
        'case_list':[]}
    for family in protocol['families']:
        for seed in protocol['seeds']:
            for target in protocol['target_assets']:
                case=generate(family,seed,target)
                protocol['case_list'].append({'case_id':f'{family}_{seed}_{target.split(":")[1]}','family':family,'seed':seed,'target_asset':target,'scene_sha256':case['scene_content_sha256'],'status':'NOT_RUN'})
    canonical=json.dumps(protocol,sort_keys=True).encode();protocol['frozen_protocol_sha256']=hashlib.sha256(canonical).hexdigest()
    file=out/'e1_protocol_v2.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise RuntimeError('Existing frozen protocol differs; create a new explicit version instead of silently changing thresholds')
    file.write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'development_smoke_budget.json').write_text(json.dumps({'scope':'Pilot implementation checks, not E1 arm comparison',
        'seed':2026112201,'scene_audits':list(FAMILIES),'action_smokes':[
            {'family':'C3','action':'active','information_mode':'RGBD_POLICY'},
            {'family':'Direct','action':'push','information_mode':'RGBD_POLICY'},
            {'family':'Direct','action':'flip','information_mode':'ORACLE_DIAGNOSTIC'},
            {'family':'Direct','action':'grasp','information_mode':'RGBD_POLICY'}],
        'gui_cases':[{'family':'C1','action':'observe'},{'family':'Direct','action':'flip'}],
        'E2':'NOT_RUN','E3':'NOT_RUN','training':'NOT_RUN'},indent=2)+'\n')
    print(json.dumps({'scene_examples':len(samples),'frozen_E1_cases':len(protocol['case_list']),'E1_episodes_NOT_RUN':len(protocol['case_list'])*4,'protocol_sha256':protocol['frozen_protocol_sha256']}))


if __name__=='__main__':main()
