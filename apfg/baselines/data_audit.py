"""Structural/provenance audit of real baseline datasets, including zero-step failures.

This module does not label policy inputs, create demonstrations, start Isaac,
or infer that a structurally valid dataset establishes scientific readiness.
"""
from collections import Counter,defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,TYPES,SCHEMA_VERSION

ROOT=Path(__file__).resolve().parents[2]
NUMERIC_KEYS={'obs','candidates','mask','action','reward','next_obs','next_candidates','next_mask','done'}
FORBIDDEN_POLICY_FIELDS={'oracle_state','ground_truth','gt','object_pose','object_T_world','m1_objects',
                         'family','seed','asset_id','object_id','success','retrieval','instance_segmentation'}
SIGNATURE_KEYS=('episode_id','scene_id','object_id','seed','method','split','wall_s_including_reset')


def project_path(path):
    path=Path(path)
    path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to(ROOT)
    return path


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _snapshot(path):
    path=project_path(path)
    if not path.exists():return [],{'path':str(path.relative_to(ROOT)),'exists':False,'rows':0,'sha256':None},[],False
    content=path.read_bytes();text=content.decode();lines=text.splitlines();rows=[];errors=[];pending=False
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('row is not an object')
            rows.append(row)
        except (json.JSONDecodeError,ValueError) as exc:
            if index==len(lines)-1 and not text.endswith('\n'):pending=True
            else:errors.append({'code':'INVALID_JSONL','path':str(path.relative_to(ROOT)),'line':index+1,'detail':str(exc)})
    return rows,{'path':str(path.relative_to(ROOT)),'exists':True,'rows':len(rows),'bytes':len(content),
                 'sha256':hashlib.sha256(content).hexdigest()},errors,pending


def _signature(row):return tuple(json.dumps(row.get(k),sort_keys=True) for k in SIGNATURE_KEYS)


def _error(errors,code,path=None,detail=None):
    item={'code':code}
    if path is not None:item['path']=str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path)
    if detail is not None:item['detail']=str(detail).replace(str(ROOT)+'/', '')[:500]
    errors.append(item)


def _array_shape(array,shape,name):
    if array.shape!=shape:raise ValueError(f'{name} shape {array.shape} != {shape}')


def _transform(value,name):
    _array_shape(value,(4,4),name)
    if not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1],atol=1e-5):raise ValueError('Invalid homogeneous transform: '+name)
    r=value[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(r),1,atol=1e-3):raise ValueError('Invalid optical rotation: '+name)


def _validate_transition(row):
    path=project_path(row['transition_path'])
    if file_sha256(path)!=row['data_sha256']:raise ValueError('Transition content hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=NUMERIC_KEYS:raise ValueError('Unexpected policy transition fields: '+str(sorted(set(archive.files)^NUMERIC_KEYS)))
        data={name:archive[name].copy() for name in archive.files}
    for name,shape in [('obs',(OBS_DIM,)),('next_obs',(OBS_DIM,)),
                       ('candidates',(MAX_CANDIDATES,CAND_DIM)),('next_candidates',(MAX_CANDIDATES,CAND_DIM))]:
        _array_shape(data[name],shape,name)
        if data[name].dtype!=np.float32 or not np.isfinite(data[name]).all():raise ValueError('Expected finite float32 '+name)
    for name in ['mask','next_mask']:
        _array_shape(data[name],(MAX_CANDIDATES,),name)
        if data[name].dtype!=np.bool_:raise ValueError('Expected boolean '+name)
    for name in ['action','reward','done']:_array_shape(data[name],(),name)
    if data['action'].dtype.kind not in 'iu':raise ValueError('Noninteger action')
    if data['done'].dtype!=np.bool_:raise ValueError('Nonboolean terminal flag')
    if data['reward'].dtype.kind not in 'fiu' or not np.isfinite(data['reward']):raise ValueError('Invalid numeric reward')
    action=int(data['action'])
    if not 0<=action<MAX_CANDIDATES or not data['mask'][action]:raise ValueError('Selected action violates legal mask')
    if row['action_type'] not in TYPES:raise ValueError('Unknown action semantic label')
    onehot=np.zeros(len(TYPES));onehot[TYPES.index(row['action_type'])]=1
    if not np.allclose(data['candidates'][action,:len(TYPES)],onehot,atol=1e-6):raise ValueError('Candidate feature/manifest action semantic mismatch')
    if not bool(data['done']) and not data['next_mask'].any():raise ValueError('Nonterminal transition without a legal next action')
    return data


def _validate_observation(path,transition,pool_path):
    path=project_path(path)
    with np.load(path,allow_pickle=False) as archive:
        forbidden=set(archive.files)&FORBIDDEN_POLICY_FIELDS
        if forbidden:raise ValueError('Forbidden hidden policy field: '+str(sorted(forbidden)))
        needed={'obs','candidates','mask','target_points','scene_points'}
        needed|={name+'_'+key for name in ['fixed','wrist'] for key in ['rgb','depth_m','K','T_world_optical']}
        if set(archive.files)!=needed:raise ValueError('Observation fields differ from common recorded sensor schema')
        data={name:archive[name].copy() for name in archive.files}
    quality={}
    for name in ['fixed','wrist']:
        rgb=data[name+'_rgb'];depth=data[name+'_depth_m'];k=data[name+'_K'];transform=data[name+'_T_world_optical']
        _array_shape(rgb,(480,640,3),name+'_rgb');_array_shape(depth,(480,640),name+'_depth_m');_array_shape(k,(3,3),name+'_K')
        if rgb.dtype!=np.uint8:raise ValueError('RGB must be uint8: '+name)
        if depth.dtype.kind!='f':raise ValueError('Depth must be floating optical-axis meters: '+name)
        finite=np.isfinite(depth)
        if not finite.any() or np.any(depth[finite]<0):raise ValueError('Missing/negative measured depth: '+name)
        if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],atol=1e-5):raise ValueError('Invalid camera intrinsics: '+name)
        _transform(transform,name+'_T_world_optical')
        pixels=rgb[::16,::16]
        quality[name]={'rgb_shape':list(rgb.shape),'depth_shape':list(depth.shape),'rgb_dtype':str(rgb.dtype),
            'depth_dtype':str(depth.dtype),'positive_finite_depth_fraction':float(np.mean(finite&(depth>0))),
            'mean_rgb':float(pixels.mean()),'rgb_standard_deviation':float(pixels.std())}
    for name in ['target_points','scene_points']:
        cloud=data[name]
        if cloud.ndim!=2 or cloud.shape[1]!=3 or cloud.dtype.kind!='f' or not np.isfinite(cloud).all():raise ValueError('Invalid measured point cloud: '+name)
    for name in ['obs','candidates','mask']:
        if not np.array_equal(data[name],transition[name]):raise ValueError('Transition does not match decision observation '+name)
    target=data['target_points']
    if len(target):
        if not np.allclose(data['obs'][11:14],np.median(target,axis=0),atol=2e-6):raise ValueError('Target feature median differs from recorded visible points')
        if not np.allclose(data['obs'][14:17],np.ptp(target,axis=0),atol=2e-6):raise ValueError('Target feature extent differs from recorded visible points')
    pool=json.loads(project_path(pool_path).read_text())
    if pool.get('schema')!=SCHEMA_VERSION or not isinstance(pool.get('input_hash'),str) or len(pool['input_hash'])!=64:raise ValueError('Candidate schema/input hash absent')
    entries=pool['candidates']
    if len(entries)>MAX_CANDIDATES:raise ValueError('Candidate pool exceeds frozen maximum')
    ids=[]
    for i,entry in enumerate(entries):
        if set(entry)&FORBIDDEN_POLICY_FIELDS:raise ValueError('Forbidden candidate metadata field')
        if entry['type'] not in TYPES:raise ValueError('Unknown candidate type')
        expected=np.zeros(len(TYPES));expected[TYPES.index(entry['type'])]=1
        vector=np.asarray(entry['features'],dtype=np.float32)
        if vector.shape!=(CAND_DIM,) or not np.isfinite(vector).all():raise ValueError('Invalid candidate descriptor')
        if not np.allclose(vector[:len(TYPES)],expected,atol=1e-6):raise ValueError('Candidate one-hot/type mismatch')
        if not np.array_equal(vector,data['candidates'][i]) or bool(entry['valid'])!=bool(data['mask'][i]):raise ValueError('Candidate pool/tensor mismatch')
        if entry.get('input_hash')!=pool['input_hash']:raise ValueError('Inconsistent candidate observation hash')
        ids.append(entry['candidate_id'])
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate IDs within one decision')
    if data['mask'][len(entries):].any() or np.any(data['candidates'][len(entries):]):raise ValueError('Padding slots must be zero/illegal')
    return quality,len(target),len(data['scene_points'])


def _trajectory_summary(path):
    path=project_path(path);rows=json.loads(path.read_text())
    if not isinstance(rows,list) or not rows:raise ValueError('Missing actual trajectory samples')
    for row in [rows[0],rows[len(rows)//2],rows[-1]]:
        q=np.asarray(row['q'])
        if q.shape!=(8,) or not np.isfinite(q).all():raise ValueError('Invalid eight-joint trajectory sample')
        _transform(np.asarray(row['frames_world']['ral_tcp']),'native TCP')
        _transform(np.asarray(row['T_world_optical']),'native wrist optical')
    times=np.asarray([r['time_s'] for r in rows],float)
    if not np.isfinite(times).all() or len(times)>1 and np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic native trajectory timestamps')
    return {'path':str(path.relative_to(ROOT)),'sha256':file_sha256(path),'samples':len(rows),
        'duration_s':float(times[-1]-times[0]),'actual_sample_hz':float((len(rows)-1)/(times[-1]-times[0])) if len(rows)>1 else None,
        'joint_dimensions':8,'includes_tcp_and_camera_extrinsics':True,
        'RGBD_cadence':'RGB-D at macro decision boundaries; no claim of dense synchronized native-DP depth trajectories'}


def audit_dataset(manifest,episodes_manifest=None,episode_roots=None):
    manifest=project_path(manifest);episodes_manifest=project_path(episodes_manifest or manifest.with_name('episodes_manifest.jsonl'))
    rows,manifest_snapshot,errors,pending=_snapshot(manifest)
    episodes,episode_snapshot,episode_errors,episode_pending=_snapshot(episodes_manifest)
    errors+=episode_errors;pending=pending or episode_pending;warnings=[]
    report={'schema_version':SCHEMA_VERSION,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'manifest_snapshot':manifest_snapshot,'episodes_snapshot':episode_snapshot,
        'scope':'Data/schema/provenance audit, not physical policy success or a scientific readiness claim',
        'policy_encoder':'32 common sensor-geometry/proprio features; not an RGB-D visual encoder'}
    if not rows and not episodes and not errors:
        return dict(report,status='BLOCKED' if pending else 'NOT_RUN',data_readiness='DATA_PENDING',errors=[],warnings=[],
                    counts={'episodes':0,'transitions':0,'zero_transition_episodes':0,'successful_train_episodes':0})
    if rows and not episodes:_error(errors,'MISSING_EPISODE_LEDGER')
    paths=[];groups=defaultdict(list);checked_observations={};trajectory_cache={};transitions_checked=0
    required={'transition_path','episode_id','scene_id','split','success','method','demonstrator_source','object_id','seed','action_type','step_index','data_sha256','rgbd_before','trajectory'}
    for index,row in enumerate(rows):
        try:
            missing=required-set(row)
            if missing:raise ValueError('Missing transition metadata: '+str(sorted(missing)))
            if not isinstance(row['success'],bool) or not isinstance(row['seed'],int) or not isinstance(row['step_index'],int):raise ValueError('Invalid metadata scalar types')
            if row['split'] not in ['train','holdout','qualification']:raise ValueError('Unexpected data split')
            path=project_path(row['transition_path']);paths.append(str(path));groups[path.parent].append(row)
            data=_validate_transition(row)
            rgbd=project_path(row['rgbd_before']);quality,target_count,scene_count=_validate_observation(rgbd,data,rgbd.with_name('candidates.json'))
            checked_observations[str(rgbd.relative_to(ROOT))]={'sha256':file_sha256(rgbd),'cameras':quality,
                'visible_target_points':target_count,'measured_scene_points':scene_count}
            trajectory=project_path(row['trajectory'])
            if trajectory not in trajectory_cache:trajectory_cache[trajectory]=_trajectory_summary(trajectory)
            transitions_checked+=1
        except (KeyError,ValueError,OSError,TypeError) as exc:_error(errors,'INVALID_TRANSITION',row.get('transition_path'),f'row {index}: {exc}')
    if len(paths)!=len(set(paths)):_error(errors,'DUPLICATE_TRANSITION_PATH')
    train=[r for r in rows if r.get('split')=='train'];held=[r for r in rows if r.get('split')=='holdout']
    for key in ['episode_id','scene_id','object_id','seed']:
        overlap={str(r.get(key)) for r in train}&{str(r.get(key)) for r in held}
        if overlap:_error(errors,'TRAIN_HOLDOUT_OVERLAP',detail={key:sorted(overlap)})
    # Locate physical episode records, including episodes that yielded no NPZ.
    files={directory/'episode.json' for directory in groups}
    for directory in episode_roots or [ROOT/'outputs/baselines']:
        files.update(project_path(directory).rglob('episode.json'))
    sources=defaultdict(list)
    for file in sorted(files):
        try:
            value=json.loads(file.read_text());sources[_signature(value)].append((file,value))
        except (OSError,ValueError):pass
    episode_audits=[];seen=set();represented=set();by_split=defaultdict(Counter);lengths=Counter();failure_counts=Counter();demo_support=Counter()
    for index,entry in enumerate(episodes):
        signature=_signature(entry)
        if signature in seen:_error(errors,'DUPLICATE_EPISODE_LEDGER_ROW',detail=index)
        seen.add(signature);matches=sources.get(signature,[])
        if len(matches)!=1:
            _error(errors,'UNRESOLVED_EPISODE_SOURCE',detail={'ledger_row':index,'matching_sources':len(matches)})
            continue
        file,source=matches[0];directory=file.parent;represented.add(directory);step_rows=sorted(groups.get(directory,[]),key=lambda r:r.get('step_index',-1))
        expected=len(source.get('steps',[]));actual=len(step_rows)
        if expected!=actual:
            if not (directory.parent/'summary.json').exists():pending=True;warnings.append({'code':'EPISODE_LEDGER_APPEND_IN_PROGRESS','episode':str(directory.relative_to(ROOT)),'actual_steps':expected,'manifest_steps':actual})
            else:_error(errors,'EPISODE_TRANSITION_COUNT_MISMATCH',file,f'{actual} manifest steps != {expected} actual steps')
        if len({r['step_index'] for r in step_rows})!=actual:_error(errors,'DUPLICATE_EPISODE_STEP',file)
        if [r['step_index'] for r in step_rows]!=list(range(actual)):_error(errors,'NONCONTIGUOUS_EPISODE_STEPS',file)
        if not isinstance(entry.get('success'),bool):_error(errors,'INVALID_EPISODE_SUCCESS_LABEL',file)
        for row in step_rows:
            if any(row.get(k)!=entry.get(k) for k in ['scene_id','seed','object_id','method','split','success']):_error(errors,'EPISODE_TRANSITION_METADATA_MISMATCH',file)
        if entry.get('success'):
            physical=any(step.get('type')=='GRASP' and step.get('physical_evaluation',{}).get('retrieval',{}).get('status')=='PASS' for step in source.get('steps',[]))
            if not physical:_error(errors,'SUCCESS_LABEL_WITHOUT_PHYSICAL_FULL_RETRIEVAL',file)
        initial=directory/'decision_00/observation.npz'
        if entry.get('initial_observation_file_sha256') and (not initial.exists() or file_sha256(initial)!=entry['initial_observation_file_sha256']):_error(errors,'INITIAL_OBSERVATION_HASH_MISMATCH',initial)
        split=entry.get('split','unknown');stats=by_split[split];stats['episodes']+=1;stats['successes']+=int(entry.get('success',False));stats['failures']+=int(not entry.get('success',False));stats['transitions']+=actual;stats['zero_transition_episodes']+=int(actual==0)
        lengths[str(actual)]+=1
        if not entry.get('success'):failure_counts[str(entry.get('failure_reason') or 'UNSPECIFIED')]+=1
        if split=='train' and entry.get('success'):
            for row in step_rows:demo_support[row['action_type']]+=1
        episode_audits.append({'ledger_row':index,'directory':str(directory.relative_to(ROOT)),
            'episode_manifest_ids':sorted({r['episode_id'] for r in step_rows}),'scene_id':entry.get('scene_id'),
            'object_id':entry.get('object_id'),'seed':entry.get('seed'),'method':entry.get('method'),'split':split,
            'success':entry.get('success'),'status':entry.get('status'),'failure_reason':entry.get('failure_reason'),
            'transitions':actual,'expected_physical_steps':expected,'source_sha256':file_sha256(file)})
    for directory in set(groups)-represented:_error(errors,'TRANSITION_WITHOUT_EPISODE_LEDGER',directory)
    # Check object/seed partition isolation even for zero-transition episodes.
    training=[r for r in episodes if r.get('split')=='train'];holdout=[r for r in episodes if r.get('split')=='holdout']
    for key in ['scene_id','object_id','seed']:
        overlap={str(r.get(key)) for r in training}&{str(r.get(key)) for r in holdout}
        if overlap:_error(errors,'EPISODE_TRAIN_HOLDOUT_OVERLAP',detail={key:sorted(overlap)})
    successes=sum(bool(r.get('success')) for r in training)
    if not training:readiness='NOT_READY_NO_TRAIN_EPISODES'
    elif successes<12:readiness='NOT_READY_FEWER_THAN_12_SUCCESSFUL_TRAIN_EPISODES'
    else:readiness='INTERFACE_SMOKE_DATA_AVAILABLE_NOT_A_SCIENTIFIC_GO'
    report.update(status='FAILED' if errors else 'BLOCKED' if pending else 'PASS',data_readiness=readiness,
        snapshot_in_progress=pending,errors=errors,warnings=warnings,
        counts={'episodes':len(episodes),'matched_physical_episodes':len(episode_audits),'transitions':len(rows),
            'validated_transitions':transitions_checked,'zero_transition_episodes':sum(e['transitions']==0 for e in episode_audits),
            'successful_train_episodes':successes,'successful_train_demonstration_transitions':sum(demo_support.values())},
        by_split={key:dict(value) for key,value in by_split.items()},episode_length_histogram=dict(lengths),
        action_types=dict(Counter(r.get('action_type','unknown') for r in rows)),
        successful_train_action_support=dict(demo_support),demonstrator_sources=dict(Counter(r.get('demonstrator_source','unknown') for r in train if r.get('success'))),
        failure_reasons=dict(failure_counts),episode_audits=episode_audits,observations=checked_observations,
        trajectories=list(trajectory_cache.values()),
        observation_shapes={'features':[OBS_DIM],'candidates':[MAX_CANDIDATES,CAND_DIM],'RGB':[480,640,3],'depth_m':[480,640]},
        boundary_audit={'status':'PASS' if not any('field' in str(e).lower() or 'semantic' in str(e).lower() for e in errors) else 'FAILED',
            'scope':'Strict saved tensor keys and candidate semantic consistency; no claim that file inspection replaces poison-oracle/source review.'},
        next_round_large_scale_training_ready=False,
        native_DP_data_readiness='NOT_READY: no dense synchronized RGB-D action-chunk dataset asserted by these macro-decision files')
    return report
