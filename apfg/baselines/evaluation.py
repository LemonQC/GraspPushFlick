"""Private physical evaluation and E1 pairing audits; not policy features."""
import hashlib
import json
from itertools import combinations
import numpy as np
from scipy.spatial.transform import Rotation

ARMS=('DIRECT_GRASP','PUSH_GRASP','FLIP_GRASP','VIEW_GRASP')


def initial_state_hash(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _state(record):
    if 'initial_state' in record:return record['initial_state'] or {}
    if 'settled_initial' in record:return record['settled_initial'] or {}
    if record and all(isinstance(v,dict) and 'T_world' in v for v in record.values()):return record
    return {}


def audit_pair(initializations,position_tolerance_m=.002,orientation_tolerance_rad=.01):
    """Compare every physical object's settled pose, including distractors.

    Entries may be direct {body: {T_world,asset_id}} mappings or results with
    initial_state and optional scene_stability fields. Image hashes are
    reported separately; rendering roundoff does not alter physical tolerance.
    """
    reasons=[];translation=angle=0.;per_body={}
    if len(initializations)!=4:reasons.append('Expected all four preregistered arms')
    if not initializations:return {'status':'FAILED','pair_status':'INVALID_PAIR','reasons':['No initializations']}
    first=_state(initializations[0]);hashes=[];states=[]
    for i,record in enumerate(initializations):
        state=_state(record);hashes.append(initial_state_hash(state))
        states.append(state)
        if not state:reasons.append(f'arm {i}: no valid physical initial state')
        stable=record.get('scene_stability',record.get('initial_stability',{}))
        if stable and stable.get('status')!='PASS':reasons.append(f'arm {i}: initial stability failed')
        if set(state)!=set(first):reasons.append(f'arm {i}: body set mismatch');continue
        for name,entry in state.items():
            if entry.get('asset_id')!=first[name].get('asset_id'):reasons.append(f'arm {i}: {name} asset mismatch')
            a=np.asarray(first[name]['T_world'],float);b=np.asarray(entry['T_world'],float)
            if a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite(np.r_[a.ravel(),b.ravel()]).all():
                reasons.append(f'arm {i}: {name} invalid finite transform');continue
            distance=float(np.linalg.norm(a[:3,3]-b[:3,3]))
            rotation=float((Rotation.from_matrix(a[:3,:3]).inv()*Rotation.from_matrix(b[:3,:3])).magnitude())
            translation=max(translation,distance);angle=max(angle,rotation)
            previous=per_body.setdefault(name,{'max_position_difference_m':0.,'max_orientation_difference_rad':0.})
            previous['max_position_difference_m']=max(previous['max_position_difference_m'],distance)
            previous['max_orientation_difference_rad']=max(previous['max_orientation_difference_rad'],rotation)
    # Compare all arm pairs, not only deviations from an arbitrary reference.
    # Opposite sub-tolerance deviations can otherwise exceed the pair bound.
    for astate,bstate in combinations(states,2):
        for name in set(astate)&set(bstate):
            a=np.asarray(astate[name]['T_world'],float);b=np.asarray(bstate[name]['T_world'],float)
            if a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite(np.r_[a.ravel(),b.ravel()]).all():continue
            distance=float(np.linalg.norm(a[:3,3]-b[:3,3]));rotation=float((Rotation.from_matrix(a[:3,:3]).inv()*Rotation.from_matrix(b[:3,:3])).magnitude())
            translation=max(translation,distance);angle=max(angle,rotation)
            previous=per_body.setdefault(name,{'max_position_difference_m':0.,'max_orientation_difference_rad':0.})
            previous['max_position_difference_m']=max(previous['max_position_difference_m'],distance)
            previous['max_orientation_difference_rad']=max(previous['max_orientation_difference_rad'],rotation)
    scene_hashes=[r.get('scene_sha256',r.get('scene_content_sha256')) for r in initializations]
    provided=[h for h in scene_hashes if h is not None]
    if len(set(provided))>1:reasons.append('Scene content hash mismatch')
    if translation>position_tolerance_m:reasons.append('Settled pose translation exceeds preregistered pair tolerance')
    if angle>orientation_tolerance_rad:reasons.append('Settled pose orientation exceeds preregistered pair tolerance')
    return {'status':'FAILED' if reasons else 'PASS','pair_status':'INVALID_PAIR' if reasons else 'VALID_PAIR',
            'reasons':reasons,'initial_state_sha256':hashes,'max_position_difference_m':translation,
            'max_orientation_difference_rad':angle,'per_body':per_body,
            'scene_content_sha256':scene_hashes,
            'initial_observation_sha256':[r.get('initial_observation_sha256') for r in initializations],
            'position_tolerance_m':position_tolerance_m,'orientation_tolerance_rad':orientation_tolerance_rad}


def summarize_e1(records,case_order):
    """Retain invalid/missing arms and reject a no-benefit scientific GO."""
    groups=[];invalid=[];benefit=[];matrix=[]
    for case_id in case_order:
        group=[r for r in records if r['case_id']==case_id]
        by_arm={r['arm']:r for r in group}
        duplicates=len(group)!=len(by_arm)
        ordered=[by_arm[a] for a in ARMS if a in by_arm]
        pair=audit_pair(ordered)
        if duplicates:pair.update(status='FAILED',pair_status='INVALID_PAIR');pair['reasons'].append('Duplicate arm records')
        if tuple(a for a in ARMS if a in by_arm)!=ARMS:pair.update(status='FAILED',pair_status='INVALID_PAIR')
        outcome={a:{'status':by_arm[a].get('status','FAILED'),'success':bool(by_arm[a].get('success',False)),
                    'failure_reason':by_arm[a].get('failure_reason')} if a in by_arm else {'status':'NOT_RUN','success':False,'failure_reason':'Missing arm'} for a in ARMS}
        preparation_benefit=pair['status']=='PASS' and not outcome['DIRECT_GRASP']['success'] and any(outcome[a]['success'] for a in ARMS[1:])
        if preparation_benefit:benefit.append(case_id)
        if pair['status']!='PASS':invalid.append(case_id)
        groups.append({'case_id':case_id,'pair_audit':pair,'outcomes':outcome,'observed_preparation_retrieval_benefit':preparation_benefit})
        matrix.append([outcome[a]['success'] if pair['status']=='PASS' and outcome[a]['status']!='NOT_RUN' else None for a in ARMS])
    reasons=[]
    if invalid:reasons.append('INVALID_PAIR: physical initialization/stability audit failed or arms missing')
    if not benefit:reasons.append('No valid same-instance preparation-to-retrieval benefit in this fixed subset')
    control_failures=[r for r in records if any(t in str(r.get('failure_reason','')).upper() for t in ['TRACKING','TCP_FINAL','GUI_CLOSED','CUDA','CRASH'])]
    if control_failures:reasons.append('Control/runtime failures remain; do not interpret them as a scientific need for learning')
    return {'status':'PASS' if not invalid and len(records)==16 else 'FAILED','executed':len(records),'planned':16,
            'arms':list(ARMS),'case_order':list(case_order),'outcome_matrix':matrix,'cases':groups,
            'go_for_remaining_48':not reasons,'NO_GO_REASON':reasons,'benefit_case_ids':benefit,
            'scope':'Fixed 16-run E1 qualification subset; not a formal paper success-rate estimate'}
