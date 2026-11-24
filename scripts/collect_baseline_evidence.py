#!/usr/bin/env python3
"""Collect compact actual Baseline evidence without running policies or physics.

Incomplete campaigns remain explicitly pending. Tests/fixtures are never
counted as physical sampling, training demonstrations or heldout episodes.
"""
import argparse
from collections import Counter,defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative(value):
    if isinstance(value,dict):return {k:relative(v) for k,v in value.items()}
    if isinstance(value,list):return [relative(v) for v in value]
    if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
    return value


def read_json(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).is_file() else []


def local(path):
    result=(ROOT/Path(path)).resolve();result.relative_to(ROOT);return result


def artifact(path):
    path=local(path)
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)}


def failure_class(reason):
    reason=str(reason or '')
    for category in ['INITIAL_UNSTABLE','NO_LEGAL_GRASP','NO_LEGAL_PUSH','NO_LEGAL_FLIP','NO_LEGAL_ACTIVE',
        'NO_LEGAL_CANDIDATE','IK_PATH_OR_TRACKING','GRASP_MISS_OR_INCOMPLETE_RETRIEVAL','SEVERE_DISTRACTOR_DROP',
        'EPISODE_TIMEOUT','RUNTIME']:
        if category in reason:return category
    return 'OTHER_FAILURE' if reason else None


def scan_campaigns(campaign_root):
    campaigns=[];episodes=[];pending=[]
    for directory in sorted(campaign_root.iterdir() if campaign_root.exists() else []):
        if not directory.is_dir():continue
        summary=read_json(directory/'summary.json')
        if summary is None:
            pending.append({'directory':str(directory.relative_to(ROOT)),'completed_records':len(read_jsonl(directory/'episodes.jsonl')),
                            'status':'NOT_RUN','note':'Campaign not finalized; not counted as completed results'})
            continue
        records=read_jsonl(directory/'episodes.jsonl')
        config=read_json(directory/'run_config.json',{})
        campaign={'run_id':directory.name,'mode':summary.get('mode'),'status':summary.get('status'),
            'declared_episodes':summary.get('episodes'),'recorded_episodes':len(records),
            'directory':str(directory.relative_to(ROOT)),'command':relative(config.get('command')),
            'summary':relative(summary),'args':relative(config.get('args',{}))}
        if summary.get('episodes')!=len(records):campaign['record_count_status']='FAILED'
        else:campaign['record_count_status']='PASS'
        campaigns.append(campaign)
        for record in records:
            record=dict(record,_campaign=directory.name,_mode=summary.get('mode'),_directory=directory/record['episode_id'])
            episodes.append(record)
    return campaigns,episodes,pending


def compact_episode(record):
    directory=record['_directory'];files=[p for p in directory.rglob('*') if p.is_file()]
    steps=record.get('steps',[]);wall=float(record.get('wall_s_including_reset',record.get('episode_wall_s',0.)) or 0.)
    sim=float(record.get('simulation_time_s',0.) or 0.)
    return relative({'campaign':record['_campaign'],'mode':record['_mode'],'episode_id':record['episode_id'],
        'method':record.get('method'),'scene_id':record.get('scene_id'),'case_id':record.get('case_id'),'arm':record.get('arm'),
        'split':record.get('split'),'object_id':record.get('object_id'),'seed':record.get('seed'),
        'status':record.get('status'),'success':bool(record.get('success',False)),
        'failure_reason':record.get('failure_reason'),'failure_class':failure_class(record.get('failure_reason')),
        'recording_validation':record.get('recording_validation','NOT_RUN'),
        'transition_count':len(steps),'zero_action_episode':not steps,'action_types':[s['type'] for s in steps],
        'wall_s_including_reset':wall,'simulation_time_s':sim,
        'gpu_device_memory_sampled_peak_mib':record.get('gpu_device_memory_sampled_peak_mib'),
        'gpu_peak_scope':record.get('gpu_peak_scope','1Hz sampled TOTAL device memory, includes desktop; not exact allocator peak' if record.get('gpu_device_memory_sampled_peak_mib') is not None else 'No peak measured in this historical probe'),
        'actual_disk_bytes':sum(p.stat().st_size for p in files),'local_directory':str(directory.relative_to(ROOT)),
        'episode_file':str((directory/'episode.json').relative_to(ROOT))})


def physical_groups(episodes):
    groups=defaultdict(list)
    for episode in episodes:groups[(episode.get('split'),episode.get('method'))].append(episode)
    result=[]
    for (split,method),rows in sorted(groups.items()):
        wall=sum(e['wall_s_including_reset'] for e in rows);sim=sum(e['simulation_time_s'] for e in rows)
        transitions=sum(e['transition_count'] for e in rows)
        result.append({'split':split,'method':method,'attempted_episodes':len(rows),
            'physical_retrieval_successes':sum(e['success'] for e in rows),'denominator_includes_zero_action_failures':True,
            'zero_action_episodes':sum(e['zero_action_episode'] for e in rows),
            'episode_status_counts':dict(Counter(e['status'] for e in rows)),
            'failure_taxonomy':dict(Counter(e['failure_class'] for e in rows if e['failure_class'])),
            'scene_ids':sorted({e['scene_id'] for e in rows if e['scene_id']}),
            'object_ids':sorted({e['object_id'] for e in rows if e['object_id']}),
            'seeds':sorted({e['seed'] for e in rows if e['seed'] is not None}),
            'actual_transitions':transitions,'wall_s_including_reset':wall,'simulation_time_s':sim,
            'simulation_seconds_per_wall_second':sim/wall if wall else None,
            'actual_transitions_per_hour':transitions*3600/wall if wall else None,
            'actual_disk_bytes':sum(e['actual_disk_bytes'] for e in rows),
            'max_sampled_total_device_memory_mib':max((e['gpu_device_memory_sampled_peak_mib'] for e in rows if e['gpu_device_memory_sampled_peak_mib'] is not None),default=None),
            'memory_note':'1Hz sampled total device use, including desktop/GraspNet; not exact process allocator peak'})
    return result


def write_curve(rows,path):
    if not rows:return None
    numeric=[]
    for row in rows:
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in row.values()):raise ValueError('Nonfinite actual curve: '+str(path))
        numeric.append(row)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in numeric for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(numeric)
    return {'path':str(path.relative_to(ROOT)),'rows':len(rows),'sha256':digest(path),
            'first':rows[0],'last':rows[-1]}


def dqn_checkpoint_audit(path):
    """Actually load online/target/replay/optimizer and test a legal replay state."""
    try:
        import torch
        torch.set_num_threads(1)
        from apfg.baselines.dqn import CandidateDQN
        agent=CandidateDQN.load(path,device='cpu')
        if agent.updates<=0 or not len(agent.replay):raise ValueError('No actual replay/gradient updates persisted')
        sample=agent.replay.storage[0];choice=agent.act(sample,epsilon=0)
        if choice is None or not sample['mask'][choice]:raise ValueError('Restored policy violates the legal mask')
        return {'status':'PASS','updates':agent.updates,'environment_steps':agent.environment_steps,
            'replay_transitions':len(agent.replay),'target_syncs':agent.target_syncs,
            'restored_legal_action':choice,'training_metadata':relative(agent.training_metadata)}
    except Exception as exc:return {'status':'FAILED','reason':str(exc)}


def scan_training(campaigns,training_roots,out):
    results=[]
    for campaign in campaigns:
        if campaign['mode']!='online':continue
        directory=ROOT/campaign['directory'];curve=read_json(directory/'training_curve.json',[]);checkpoint=directory/'dqn_online.pt'
        audit=dqn_checkpoint_audit(checkpoint) if checkpoint.exists() else {'status':'FAILED','reason':'No online checkpoint'}
        training_ok=bool(curve) and audit['status']=='PASS' and all('loss' in r and math.isfinite(r['loss']) for r in curve)
        results.append({'method':'dqn','kind':'actual_online_physical_interaction','directory':campaign['directory'],
            'status':'PASS' if training_ok else 'FAILED','implementation':'FROM_SCRATCH candidate-conditioned Double DQN',
            'curve_rows':len(curve),'gradient_updates':audit.get('updates',0),'curve':write_curve(curve,out/'curves'/(directory.name+'_dqn_online.csv')),
            'checkpoint':artifact(checkpoint) if checkpoint.exists() else None,'checkpoint_validation':audit})
    seen=set()
    for root in training_roots:
        for file in sorted(root.rglob('result.json') if root.exists() else []):
            if file in seen:continue
            seen.add(file);record=read_json(file);directory=file.parent
            config=read_json(directory/'config.json',read_json(directory/'run_config.json',{}))
            method=record.get('method',config.get('method','dqn' if record.get('algorithm')=='candidate-conditioned Double DQN' or config.get('config',{}).get('algorithm')=='candidate_conditioned_double_dqn' else None))
            if method not in ['dqn','bc','dp']:
                results.append({'method':method or 'unknown_training','kind':'trainer_result','directory':str(directory.relative_to(ROOT)),
                    'status':'FAILED' if record.get('status')=='FAILED' else 'NOT_RUN','actual_result':relative(record),
                    'reason':'Trainer result retained; method/complete chain not established'})
                continue
            curve=read_json(directory/'curve.json')
            if curve is None and (directory/'curve.csv').exists():
                with (directory/'curve.csv').open(newline='') as stream:curve=list(csv.DictReader(stream))
                for row in curve:
                    for key,value in row.items():
                        try:row[key]=float(value) if value else None
                        except (ValueError,TypeError):pass
            curve=curve or []
            checkpoint=local(record['checkpoint']) if record.get('checkpoint') else None
            hash_ok=checkpoint is not None and checkpoint.is_file() and digest(checkpoint)==record.get('checkpoint_sha256')
            flags=record.get('updates',0)>0 and bool(curve) and hash_ok and record.get('status')=='PASS'
            if method in ['bc','dp']:
                flags=flags and record.get('checkpoint_prediction_max_abs_difference')==0 and record.get('dataset',{}).get('optimization_train',{}).get('transitions',0)>0
                validation={'status':'PASS' if flags else 'FAILED','prediction_max_abs_difference':record.get('checkpoint_prediction_max_abs_difference'),
                    'source':'Actual trainer forward/backward and checkpoint prediction roundtrip result; not inferred from checkpoint existence'}
            else:
                validation=dqn_checkpoint_audit(checkpoint) if hash_ok else {'status':'FAILED','reason':'Checkpoint hash missing/mismatch'}
                flags=flags and validation['status']=='PASS'
            results.append({'method':method,'kind':'real_data_training','directory':str(directory.relative_to(ROOT)),
                'status':'PASS' if flags else 'FAILED','implementation':record.get('implementation',
                    'FROM_SCRATCH_BC' if method=='bc' else 'ADAPTED_OFFICIAL_UNET_HIGH_LEVEL_DP_LOCAL_DDPM'),
                'curve_rows':len(curve),'gradient_updates':int(record.get('updates',0)),'curve':write_curve(curve,out/'curves'/(directory.name+'_'+method+'.csv')),
                'checkpoint':artifact(checkpoint) if checkpoint is not None and checkpoint.exists() else None,
                'checkpoint_validation':validation,'actual_result':relative(record)})
    return results


def dataset_audit(manifest,episodes,training):
    rows=read_jsonl(manifest);splits={}
    for split in sorted({r['split'] for r in rows}):
        chosen=[r for r in rows if r['split']==split];counts=Counter(r['episode_id'] for r in chosen)
        splits[split]={'transitions':len(chosen),'episodes_with_transitions':len(counts),
            'success_episodes_with_transitions':len({r['episode_id'] for r in chosen if r.get('success') is True}),
            'failure_episodes_with_transitions':len({r['episode_id'] for r in chosen if r.get('success') is False}),
            'object_ids':sorted({r['object_id'] for r in chosen}),'seeds':sorted({int(r['seed']) for r in chosen}),
            'scene_ids':sorted({r['scene_id'] for r in chosen}),'episode_ids':sorted(counts),
            'episode_length_histogram':dict(Counter(str(n) for n in counts.values())),
            'action_types':dict(Counter(r['action_type'] for r in chosen)),
            'sources':dict(Counter(r.get('demonstrator_source','unspecified') for r in chosen))}
    train=[r for r in rows if r['split']=='train'];hold=[r for r in rows if r['split']=='holdout']
    overlap={key:sorted({str(r[key]) for r in train}&{str(r[key]) for r in hold}) for key in ['object_id','seed','scene_id','episode_id']}
    before_rows=0;bad=[]
    for row in rows:
        file=local(row['transition_path']);before_rows+=1
        if not file.is_file() or digest(file)!=row.get('data_sha256'):bad.append(row['transition_path'])
    subsets={m:sorted({t.get('actual_result',{}).get('successful_demo_subset_sha256') for t in training if t['method']==m and t.get('actual_result',{}).get('successful_demo_subset_sha256')}) for m in ['bc','dp']}
    subset_same=bool(subsets['bc']) and subsets['bc']==subsets['dp']
    return {'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':digest(manifest) if manifest.exists() else None,
        'rows':len(rows),'partitions':splits,'transition_hash_audit':{'status':'FAILED' if bad else 'PASS','checked':before_rows,'bad_paths':bad},
        'train_holdout_overlap':overlap,'train_holdout_disjoint_status':'PASS' if not any(overlap.values()) and train and hold else 'NOT_RUN' if not hold else 'FAILED',
        'BC_DP_successful_demo_subset_sha256':subsets,'BC_DP_same_successful_subset_status':'PASS' if subset_same else 'NOT_RUN' if not all(subsets.values()) else 'FAILED',
        'zero_action_episode_note':'Manifest contains transitions only. All failed zero-action episodes remain in the physical episode ledger and denominators.',
        'actual_policy_tensor_shapes':{'obs':[32],'candidates':[24,24],'mask':[24]},
        'observation_scope':'Shared RGB-D/proprio/visible-cloud-derived features; no learned raw image encoder or native DP3 point-cloud policy'}


def holdout_audit(episodes):
    groups={m:[e for e in episodes if e.get('split')=='holdout' and e.get('method')==m] for m in ['dqn','bc','dp']}
    cases={m:sorted(e['scene_id'] for e in rows) for m,rows in groups.items()}
    equal=bool(cases['dqn']) and cases['dqn']==cases['bc']==cases['dp']
    return {'status':'PASS' if equal else 'NOT_RUN' if not all(cases.values()) else 'FAILED',
            'scene_ids_by_learned_method':cases,'scope':'All actual attempted heldout episodes including zero-action failures'}


def previews(records,out):
    from PIL import Image,ImageDraw
    chosen=[]
    for mode in ['probe','e1','collect','online','evaluate']:
        group=[r for r in records if r['_mode']==mode]
        for success in [True,False]:
            found=next((r for r in group if bool(r.get('success'))==success and (r['_directory']/'images/baseline_final_rgb.png').exists()),None)
            if found:chosen.append(found)
    if not chosen:return []
    sheet=Image.new('RGB',(960,270*((len(chosen)+2)//3)),'white');draw=ImageDraw.Draw(sheet);sources=[]
    for index,record in enumerate(chosen):
        path=record['_directory']/'images/baseline_final_rgb.png';frame=Image.open(path).convert('RGB');frame.thumbnail((320,240))
        x=index%3*320;y=index//3*270;sheet.paste(frame,(x,y+30))
        draw.text((x+3,y+3),f'{record["_mode"]} / {record["method"]} / {record["status"]}',fill='black')
        sources.append(dict(artifact(path),panel=index,episode_id=record['episode_id'],campaign=record['_campaign']))
    sheet.save(out/'actual_episode_frames.jpg',quality=85)
    (out/'preview_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    return sources


def matrix(summary,path):
    physical=summary['physical_results'];training=summary['training']
    methods=[('grasp_only','Grasp Only','FROM_SCRATCH selector / reused GraspNet','apfg/baselines/heuristic.py'),
        ('heuristic','Heuristic','FROM_SCRATCH','apfg/baselines/heuristic.py'),
        ('random_valid','Random valid','FROM_SCRATCH','apfg/baselines/heuristic.py'),
        ('dqn','Candidate Double DQN','FROM_SCRATCH; not SB3 fixed-slot DQN','apfg/baselines/dqn.py'),
        ('bc','Candidate BC','FROM_SCRATCH','apfg/baselines/bc.py'),
        ('dp','High-level Diffusion Policy','ADAPTED official U-Net + local DDPM','apfg/baselines/dp_adapted.py')]
    lines=['# Baseline implementation and measured execution matrix','',
        'Generated from actual finalized campaigns/trainer results by `scripts/collect_baseline_evidence.py`. Counts include failed and zero-action episodes; checkpoint existence alone is not training PASS.','',
        '| Method | Implementation identity | Core code | Actual training chain | Heldout physical retrieval |','|---|---|---|---|---|']
    for name,label,identity,code in methods:
        learned=[r for r in training if r['method']==name]
        train_status=', '.join(sorted({r['status'] for r in learned})) if learned else 'NOT_RUN' if name in ['dqn','bc','dp'] else 'NOT_RUN (no learned parameters)'
        rows=[r for r in physical if r['split']=='holdout' and r['method']==name]
        count=sum(r['attempted_episodes'] for r in rows);success=sum(r['physical_retrieval_successes'] for r in rows)
        physical_status=f'{success}/{count}; '+('PASS' if success==count else 'FAILED') if count else 'NOT_RUN'
        lines.append(f'| {label} | {identity} | `{code}` | {train_status} | {physical_status} |')
    source=read_json(ROOT/'apfg/baselines/_official_dp/SOURCE.json',{})
    lines+=['','Official DP network source: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy), commit `'+str(source.get('commit','NOT_VERIFIED'))+'`; source/license records are in `apfg/baselines/_official_dp/`. Only the stated network is adapted. Observation/action representation, candidate projection, scheduler and training data differ from native DP.','',
        '| Native published method | Execution identity/status | Reason |','|---|---|---|',
        '| Native Diffusion Policy | NOT_RUN | No native continuous EE/gripper action-chunk training/rollout in this shared candidate experiment. |',
        '| Native DP3 | NOT_RUN | No native synchronized high-frequency point-cloud/continuous-action dataset and executor; high-level DP is not DP3. |',
        '| GAPG / GOMP / MS-MEM / Uni-PNP / SPIN | NOT_RUN | Source/task audit only; see [primary-source review](BASELINE_SOURCE_AUDIT.md). No heuristic impersonation. |','',
        'E1 qualification and next-scale readiness are separate from optimizer/checkpoint availability. Actual per-episode outcomes, no-go reasons, losses, hashes and split audits are in `evidence/baselines/summary.json`; full observations, videos, replay and model weights remain local under `outputs/baselines/`. Static candidate agreement is not physical retrieval success.','']
    path.write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,default=ROOT/'outputs/baselines/campaigns')
    p.add_argument('--training-root',type=Path,action='append',default=[])
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/baselines/dataset/manifest.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'evidence/baselines')
    p.add_argument('--write-matrix',action='store_true',help='Update docs/BASELINE_IMPLEMENTATION_MATRIX.md from actual results')
    args=p.parse_args();out=local(args.output);out.mkdir(parents=True,exist_ok=True)
    campaigns,raw,pending=scan_campaigns(local(args.campaign_root));episodes=[compact_episode(r) for r in raw]
    roots=[local(r) for r in args.training_root] or [ROOT/'outputs/baselines/train',ROOT/'outputs/baselines/dqn']
    training=scan_training(campaigns,roots,out);physical=physical_groups(episodes)
    unit=ROOT/'outputs/baselines/all_unit_tests.log';text=unit.read_text() if unit.exists() else '';count=re.search(r'Ran (\d+) tests?',text)
    e1=[c['summary']['E1'] for c in campaigns if c['mode']=='e1' and 'E1' in c['summary']]
    frozen=read_json(ROOT/'configs/m1/e1_protocol_v3.json',{})
    summary={'collected_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'Small real Baseline development chains; no paper success-rate, novelty, or large-training readiness claim',
        'protocol':{'path':'configs/m1/e1_protocol_v3.json','frozen_protocol_sha256':frozen.get('frozen_protocol_sha256'),
            'controller_revision':frozen.get('controller_revision'),'controller_source_commit':frozen.get('controller_source_commit'),
            'cameras':frozen.get('cameras'),'baseline_partitions':frozen.get('baseline_partitions')},
        'reused_software_stack':read_json(ROOT/'evidence/m1/summary.json',{}).get('stack'),
        'campaigns':campaigns,'pending_campaigns':pending,'physical_episodes':episodes,'physical_results':physical,
        'training':training,'dataset':dataset_audit(local(args.manifest),episodes,training),'heldout_shared_cases':holdout_audit(episodes),
        'E1_qualification':e1,'CPU_tests':{'status':'PASS' if count and text.rstrip().endswith('OK') else 'FAILED' if count else 'NOT_RUN',
            'count':int(count.group(1)) if count else None,'log_sha256':digest(unit) if unit.exists() else None},
        'identity_note':'FROM_SCRATCH nonlearning/DQN/BC; adapted official U-Net high-level DP with independently implemented scheduler; native DP/DP3 and audited published systems NOT_RUN',
        'readiness':'NOT_READY for next-scale training: audit E1 benefit and actual demonstration diversity separately',
        'public_storage':'Only compact real metrics/curve CSV/previews/hashes here; large data/checkpoints/videos are local server files, not GitHub downloads'}
    (out/'summary.json').write_text(json.dumps(relative(summary),indent=2)+'\n')
    sources=previews(raw,out)
    all_files=set()
    for campaign in campaigns:
        directory=ROOT/campaign['directory']
        for name in ['summary.json','episodes.jsonl','run_config.json','gpu_memory_samples.json','training_curve.json','dqn_online.pt','heldout_preregistration.json']:
            if (directory/name).is_file():all_files.add(directory/name)
        for record in [r for r in raw if r['_campaign']==campaign['run_id']]:
            episode=record['_directory']
            for pattern in ['episode.json','policy_inference.json','telemetry.json','contacts.json','commands.json','transition_*.npz','decision_*/observation.npz','decision_*/candidates.json','decision_*/plans.json','videos/*/*.mp4','videos/*/timing.json']:
                all_files.update(p for p in episode.glob(pattern) if p.is_file())
    for run in training:
        directory=ROOT/run['directory']
        all_files.update(p for p in directory.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.pt'])
    if local(args.manifest).exists():all_files.add(local(args.manifest))
    protocol=ROOT/'configs/m1/e1_protocol_v3.json'
    if protocol.exists():all_files.add(protocol)
    (out/'artifact_manifest.json').write_text(json.dumps({'storage':'Server-local artifacts only; no binary upload implied',
        'files':[artifact(file) for file in sorted(all_files)]},indent=2)+'\n')
    if args.write_matrix:matrix(summary,ROOT/'docs/BASELINE_IMPLEMENTATION_MATRIX.md')
    print(json.dumps({'finalized_campaigns':len(campaigns),'pending_campaigns':len(pending),'actual_episodes':len(episodes),
        'zero_action_episodes':sum(e['zero_action_episode'] for e in episodes),'training_runs':len(training),'preview_panels':len(sources),
        'output':str(out.relative_to(ROOT))}))


if __name__=='__main__':main()
