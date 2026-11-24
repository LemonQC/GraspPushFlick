#!/usr/bin/env python3
"""Freeze E1 v3 and disjoint baseline development partitions before execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,SCHEMA_VERSION

CONTROLLER_COMMIT='d51265bc76aab1d1cee45a16995a510d04f61a1b'
TRAINING_ASSETS_CONFIRMED=True
SOURCE_FILES=(
    'apfg/baselines/schema.py','apfg/baselines/common_env.py',
    'apfg/baselines/candidate_generator.py','apfg/baselines/heuristic.py',
    'apfg/baselines/reward.py','apfg/baselines/evaluation.py','apfg/baselines/gpu_lock.py',
    'apfg/simulation/m1_env.py','apfg/simulation/sr5_hardware.py',
    'apfg/primitives/grasp_motion.py','apfg/primitives/sr5_control.py',
    'apfg/primitives/m1_actions.py','apfg/primitives/preparation_motion.py','apfg/primitives/swept.py',
    'apfg/perception/m1_observation.py',
    'apfg/perception/graspnet_candidates.py','apfg/robots/sr5.py','apfg/scenes/m1.py',
    'scripts/graspnet_worker.py','scripts/run_baseline_experiments.py',
    'configs/baselines/common.yaml','configs/baselines/heuristic.yaml',
    'assets/manifests/object_registry.json','assets/manifests/graspnet_model.json',
)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical_hash(protocol):
    data={k:v for k,v in protocol.items() if k!='frozen_protocol_sha256'}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def build_protocol(allow_missing=False):
    v2=json.loads((ROOT/'configs/m1/e1_protocol_v2.json').read_text())
    protocol=copy.deepcopy(v2);protocol.pop('frozen_protocol_sha256')
    protocol.update(protocol_version='E1-development-preregister-v3',status='NOT_RUN',
        supersedes='e1_protocol_v2.json; earlier frozen files remain unchanged',
        pilot_revision_reason='New parallel_jaw_execution_v1 controller and common observable candidate pool; old control trajectories are not pooled into this E1 comparison.',
        controller_revision='parallel_jaw_execution_v1',controller_source_commit=CONTROLLER_COMMIT,
        schema_version=SCHEMA_VERSION,observation_dimension=OBS_DIM,candidate_dimension=CAND_DIM,
        maximum_legal_action_pool=MAX_CANDIDATES,maximum_raw_grasp_candidates=8,
        maximum_parallel_jaw_variants_per_raw_grasp=2,
        path_check_samples_per_actual_joint_segment=7,
        preparation_contact_height='Measured target quantiles with a source closed-finger/table-clearance lower bound; exact implementation must be frozen in preparation_motion.py and candidate_generator.py. No object GT.',
        pairing_tolerances={'all_body_position_m':.002,'all_body_orientation_rad':.01,
            'all_initial_stability_required':True,'scene_content_hash_equal':True,
            'initial_rgbd_input_hashes_recorded':True},
        candidate_protocol='All Track A methods receive the same sensor-derived candidate tensors and legal mask. No family/seed/object GT in model features. Fixed E1 arms constrain first type only; choose parameters within the same shared pool.',
        cameras={'views':['fixed','wrist'],'width':640,'height':480,'physics_hz':480,
            'sensor_frame_refresh_each_decision':True,'same_initial_view':True},
        experiment_gate='Execute all 16 frozen qualification arms, retain invalid arms. Continue remaining48 only with valid physical pairs and observable preparation-to-complete-retrieval benefit; otherwise NO_GO_REASON.',
        qualification_case_order=[],qualification_execution_order=[])
    for family in protocol['families']:
        case=next(c for c in protocol['case_list'] if c['family']==family and c['seed']==2026112201 and c['target_asset']=='ycb:004_sugar_box')
        protocol['qualification_case_order'].append(case['case_id'])
        for arm in protocol['arms']:
            protocol['qualification_execution_order'].append(dict(case,arm=arm,execution_id=case['case_id']+'__'+arm,status='NOT_RUN',split='qualification'))
    protocol['qualification_episodes']=16
    protocol['remaining_episodes']=48
    missing=[name for name in SOURCE_FILES if not (ROOT/name).is_file()]
    if missing and not allow_missing:raise RuntimeError('Cannot freeze before shared implementation exists: '+', '.join(missing))
    protocol['implementation_source_sha256']={name:digest(ROOT/name) for name in SOURCE_FILES if (ROOT/name).is_file()}
    protocol['missing_implementation_files']=missing
    config=ROOT/'configs/baselines/common.yaml'
    if config.exists():protocol['common_configuration_sha256']=digest(config)
    model=ROOT/'assets/models/graspnet/checkpoint-rs.tar'
    if model.exists():protocol['graspnet_checkpoint_sha256']=digest(model)
    protocol['baseline_partitions']={
        'training':{'object_ids':['ycb:006_mustard_bottle'],
            'asset_probe_status':'PASS_LIMITED_SINGLE_ASSET' if TRAINING_ASSETS_CONFIRMED else 'PENDING_REPLACEMENT_PROBE',
            'asset_probe_campaign':'outputs/baselines/campaigns/20261123T103216Z_cf3c2efa',
            'asset_probe_result':'2/2 actual full retrieval PASS in one reused Isaac process',
            'retained_failed_asset_probes':[
                {'object_id':'ycb:008_pudding_box','campaign':'outputs/baselines/campaigns/20261123T102827Z_fc7f02b9','episodes':2,'reason':'NO_LEGAL_GRASP'},
                {'object_id':'ycb:003_cracker_box','campaign':'outputs/baselines/campaigns/20261123T103421Z_40c6d1ae','episodes':2,'reason':'INITIAL_UNSTABLE'}],
            'seeds':list(range(2026112320,2026112336)),
            'demonstration_seeds':list(range(2026112320,2026112332)),
            'dqn_online_seeds':list(range(2026112332,2026112336)),
            'episode_budget':{'demonstrations':12,'dqn_online':4,'total':16},
            'scope':'Small chain smoke, single train asset; diversity inadequate for next-scale training. Actual manifest determines successful demonstrations; never fabricate success counts.'},
        'heldout':{'cases':[
            {'scene_id':'holdout_Direct_2026112309_004_sugar_box','family':'Direct','seed':2026112309,'target_asset':'ycb:004_sugar_box'},
            {'scene_id':'holdout_Direct_2026112310_005_tomato_soup_can','family':'Direct','seed':2026112310,'target_asset':'ycb:005_tomato_soup_can'}],
            'same_cases_for':['DQN','BC','DIFFUSION_HIGH_LEVEL'],'training_label_access':False},
        'qualification':{'training_label_access':False,'case_ids':protocol['qualification_case_order']},
        'split_invariants':['No train/holdout object or seed overlap.','BC and high-level DP use identical successful transition subset hashes.',
                            'Prior M1/grasp regression episodes are not relabeled as new E1 or new baseline training samples.']}
    protocol['frozen_protocol_sha256']=canonical_hash(protocol)
    return protocol


def freeze_protocol(path,protocol):
    path=Path(path)
    if protocol.get('missing_implementation_files'):raise ValueError('Cannot freeze incomplete implementation hashes')
    if protocol.get('baseline_partitions',{}).get('training',{}).get('asset_probe_status','PASS') not in ('PASS','PASS_LIMITED_SINGLE_ASSET'):
        raise ValueError('Cannot freeze training assets before the real source-geometry usability probe passes')
    if path.exists():
        existing=json.loads(path.read_text())
        if existing!=protocol:raise RuntimeError('Frozen protocol differs: preserve it and explicitly register a new version')
        return False
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(protocol,indent=2)+'\n')
    return True


def verify_protocol_sources(protocol):
    """Refuse using a stale freeze after changing shared environment/control."""
    errors=[]
    if canonical_hash(protocol)!=protocol.get('frozen_protocol_sha256'):errors.append('Frozen content hash mismatch')
    for name,expected in protocol.get('implementation_source_sha256',{}).items():
        file=ROOT/name
        if not file.is_file() or digest(file)!=expected:errors.append('Implementation changed: '+name)
    model=ROOT/'assets/models/graspnet/checkpoint-rs.tar'
    if protocol.get('graspnet_checkpoint_sha256') and (not model.exists() or digest(model)!=protocol['graspnet_checkpoint_sha256']):
        errors.append('GraspNet checkpoint changed')
    return {'status':'FAILED' if errors else 'PASS','errors':errors}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',action='store_true',help='Create immutable v3 only after common implementation is ready')
    p.add_argument('--output',type=Path,default=ROOT/'configs/m1/e1_protocol_v3.json')
    args=p.parse_args();protocol=build_protocol(allow_missing=not args.freeze)
    if args.freeze:freeze_protocol(args.output,protocol)
    else:
        preview=ROOT/'outputs/baselines/protocol_preview.json';preview.parent.mkdir(parents=True,exist_ok=True)
        preview.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'status':'NOT_RUN','frozen':args.freeze,'qualification_episodes':16,
        'full_episodes':64,'protocol_sha256':protocol['frozen_protocol_sha256'],
        'missing_implementation_files':protocol['missing_implementation_files']}))


if __name__=='__main__':main()
