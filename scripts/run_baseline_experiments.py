#!/usr/bin/env python3
"""Actual serialized SR5 interactions for E1, demonstrations, DQN and holdout."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import threading

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch
from apfg.baselines.gpu_lock import gpu_job


class GPUMemorySampler:
    """Sample device TOTAL memory (including desktop), never claim allocator peak."""
    def __init__(self):self.samples=[];self.stop=threading.Event();self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                self.samples.append({'wall_monotonic_s':time.monotonic(),'device_used_mib':int(value.splitlines()[0])})
            except Exception:pass
            self.stop.wait(1.)
    def close(self):self.stop.set();self.thread.join(timeout=6)


def append_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:stream.write(json.dumps(value)+'\n')


def select_forced(state,kind):
    from apfg.baselines.schema import TYPES
    mask=state['mask']&(state['candidates'][:,TYPES.index(kind)]>.5)
    return int(np.argmax(np.where(mask,state['candidates'][:,4],-np.inf))) if mask.any() else -1


def training_identifiers(records,run):
    return {'train_episode_ids':[r['episode_id']+'_'+run for r in records],
            'train_scene_ids':sorted({r['scene_id'] for r in records}),
            'train_object_ids':sorted({r['object_id'] for r in records}),
            'train_seeds':sorted({int(r['seed']) for r in records})}


def enforce_holdout(policy,method,tasks):
    if method=='dqn':
        meta=policy.training_metadata;objects=meta.get('train_object_ids');seeds=meta.get('train_seeds')
    else:
        meta=policy.checkpoint_metadata.get('dataset',{}).get('all_successful_train',{})
        objects=meta.get('object_ids');seeds=meta.get('seeds')
    if not objects or not seeds:raise ValueError('Checkpoint lacks auditable training partition identifiers')
    if any(t['object_id'] in objects or t['seed'] in seeds for t in tasks if t['method']==method):
        raise ValueError('TRAIN_HOLDOUT_OBJECT_OR_SEED_OVERLAP')


def episode(env,task,directory,policy=None,online_agent=None):
    from apfg.baselines.heuristic import make_policy
    transitions=[];metrics=[]
    record=dict(task,episode_id=directory.name,status='FAILED',success=False)
    started=time.monotonic()
    try:
        state=env.reset(task['seed'],task['family'],task['object_id'],directory)
        record.update(initial_state=env.initial_state,scene_stability=env.initial_stability,
            scene_sha256=env.spec['scene_content_sha256'])
        initial=directory/'decision_00/observation.npz'
        if initial.exists():
            from apfg.baselines.common_env import file_sha
            record['initial_observation_sha256']=json.loads((initial.parent/'candidates.json').read_text())['input_hash']
            record['initial_observation_file_sha256']=file_sha(initial)
        forced={'DIRECT_GRASP':'GRASP','PUSH_GRASP':'PUSH','FLIP_GRASP':'FLIP','VIEW_GRASP':'ACTIVE',
                'grasp_only':'GRASP','active_grasp':'ACTIVE','push_grasp':'PUSH','flip_grasp':'FLIP'}
        method=task['method']
        baseline=None if method in forced or policy is not None or online_agent is not None else make_policy(method,task['policy_seed'])
        for step in range(2):
            if env.terminal:break
            if method in forced:action=select_forced(state,forced[method] if step==0 else 'GRASP')
            elif online_agent is not None:action=online_agent.act(state,epsilon=None)
            elif policy is not None:action=policy.act(state)
            else:action=baseline.act(state)
            if action is None or action<0:
                env.terminal=True;env.failure_reason='NO_LEGAL_'+(forced.get(method,'CANDIDATE') if step==0 else 'GRASP');break
            state,reward,done,info,transition=env.step(int(action));transitions.append(transition)
            if online_agent is not None:
                online_agent.observe(transition)
                for _ in range(8):
                    update=online_agent.update()
                    if update:metrics.append(update)
            if done:break
        record.update(env.finish())
    except Exception as exc:
        directory.mkdir(parents=True,exist_ok=True);record.update(status='FAILED',failure_reason='RUNTIME: '+str(exc))
        (directory/'traceback.txt').write_text(traceback.format_exc())
        if env.environment is not None and hasattr(env.environment,'initial_stability'):
            try:record.update(env.finish());record.update(status='FAILED',success=False,failure_reason='RUNTIME: '+str(exc))
            except Exception:pass
    record['wall_s_including_reset']=time.monotonic()-started
    record['data_bytes']=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    try:
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
        record['gpu_memory_used_mib_after_episode']=int(memory.splitlines()[0])
    except Exception:pass
    (directory/'episode.json').write_text(json.dumps(record,indent=2)+'\n')
    return record,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['probe','e1','collect','online','evaluate'],required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/m1/e1_protocol_v3.json')
    p.add_argument('--output',type=Path);p.add_argument('--episodes',type=int,default=12)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/baselines/dataset/manifest.jsonl')
    p.add_argument('--dqn-checkpoint',type=Path);p.add_argument('--bc-checkpoint',type=Path);p.add_argument('--dp-checkpoint',type=Path)
    p.add_argument('--method',choices=['all','grasp_only','heuristic','random_valid','dqn','bc','dp'],default='all')
    p.add_argument('--probe-asset',default='ycb:008_pudding_box')
    p.add_argument('--joint-speed',type=float,default=.3);p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--gui',action='store_true');add_launcher_args(p);args=p.parse_args()
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=args.output or ROOT/'outputs/baselines/campaigns'/run;output.mkdir(parents=True,exist_ok=False)
    (output/'run_config.json').write_text(json.dumps({'args':vars(args),'command':sys.argv},indent=2,default=str)+'\n')
    tasks=[];protocol=None
    if args.mode!='probe':
        protocol=json.loads(args.protocol.read_text())
        from scripts.prepare_baseline_protocol import verify_protocol_sources
        audit=verify_protocol_sources(protocol)
        if audit['status']!='PASS':raise RuntimeError('Frozen protocol implementation mismatch: '+str(audit))
    if args.mode=='probe':
        for i in range(2):tasks.append({'scene_id':'probe_reuse','family':'Direct','seed':2026112399,'object_id':args.probe_asset,'method':'grasp_only','split':'probe','policy_seed':10+i})
    elif args.mode=='e1':
        for row in protocol['qualification_execution_order']:
            tasks.append({'scene_id':row['case_id'],'case_id':row['case_id'],'arm':row['arm'],'family':row['family'],
                'seed':row['seed'],'object_id':row['target_asset'],'method':row['arm'],'split':'qualification','policy_seed':20261123})
    elif args.mode in ('collect','online'):
        methods=['grasp_only','active_grasp','push_grasp','flip_grasp','heuristic','random_valid']
        for i in range(args.episodes):
            seed=(2026112332 if args.mode=='online' else 2026112320)+i
            if seed not in protocol['baseline_partitions']['training']['seeds']:raise ValueError('Training episode exceeds frozen small development budget')
            assets=protocol['baseline_partitions']['training']['object_ids'];asset=assets[i%len(assets)]
            tasks.append({'scene_id':f'train_Direct_{seed}_{asset.split(":")[1]}','family':'Direct','seed':seed,'object_id':asset,
                'method':'dqn' if args.mode=='online' else methods[i%len(methods)],'split':'train','policy_seed':20261123+i})
    else:
        methods=['grasp_only','heuristic','random_valid','dqn','bc','dp'] if args.method=='all' else [args.method]
        for method in methods:
            for case in protocol['baseline_partitions']['heldout']['cases']:
                tasks.append({'scene_id':case['scene_id'],'family':case['family'],'seed':case['seed'],'object_id':case['target_asset'],
                    'method':method,'split':'holdout','policy_seed':20261123})
    app=None;environment=None;records=[];online_agent=None;policies={};all_metrics=[]
    with gpu_job('Isaac '+args.mode):
        memory_sampler=GPUMemorySampler();memory_sampler.thread.start()
        try:
            app=launch(args)
            from apfg.baselines.common_env import BaselineEnvironment
            environment=BaselineEnvironment(args,app)
            if args.mode=='online':
                import torch,yaml
                torch.set_num_threads(1)
                from apfg.baselines.dqn import CandidateDQN
                online_agent=CandidateDQN(yaml.safe_load((ROOT/'configs/baselines/dqn.yaml').read_text())['network'],seed=20261123,device='cpu')
            if args.mode=='evaluate':
                import torch
                torch.set_num_threads(1)
                for name in sorted(set(t['method'] for t in tasks)):
                    if name=='dqn':
                        from apfg.baselines.dqn import CandidateDQN
                        if not args.dqn_checkpoint:raise ValueError('DQN checkpoint required')
                        policies[name]=CandidateDQN.load(args.dqn_checkpoint,device='cpu')
                    elif name=='bc':
                        from apfg.baselines.bc import BCPolicy
                        if not args.bc_checkpoint:raise ValueError('BC checkpoint required')
                        policies[name]=BCPolicy.load(args.bc_checkpoint,device='cpu')
                    elif name=='dp':
                        from apfg.baselines.dp_adapted import DiffusionHighLevelPolicy
                        if not args.dp_checkpoint:raise ValueError('DP checkpoint required')
                        policies[name]=DiffusionHighLevelPolicy.load(args.dp_checkpoint,device='cpu')
                    if name in policies:enforce_holdout(policies[name],name,tasks)
            for i,task in enumerate(tasks):
                sample_start=len(memory_sampler.samples)
                directory=output/f'episode_{i:03d}';record,metrics=episode(environment,task,directory,policies.get(task['method']),online_agent)
                samples=memory_sampler.samples[sample_start:]
                record['gpu_device_memory_sampled_peak_mib']=max((s['device_used_mib'] for s in samples),default=None)
                record['gpu_peak_scope']='1Hz sampled TOTAL device memory, includes desktop and GraspNet subprocess; not exact allocator peak'
                records.append(record);all_metrics+=metrics
                append_json(output/'episodes.jsonl',record)
                if task['split']!='probe':
                    append_json(args.manifest.with_name('episodes_manifest.jsonl'),{k:v for k,v in record.items() if k not in ['steps','initial_state','videos']})
                    for j,step in enumerate(record.get('steps',[])):
                        append_json(args.manifest,{'transition_path':step['transition_path'],'episode_id':record['episode_id']+'_'+run,
                            'scene_id':task['scene_id'],'split':task['split'],'success':record['success'],'method':task['method'],
                            'demonstrator_source':task['method']+'_actual_physical_rollout','object_id':task['object_id'],'seed':task['seed'],
                            'action_type':step['type'],'step_index':j,'data_sha256':step['data_sha256'],
                            'rgbd_before':step['observation_before'],'trajectory':str((directory/'telemetry.json').relative_to(ROOT))})
                print(json.dumps({'completed':i+1,'planned':len(tasks),'method':task['method'],'scene':task['scene_id'],
                    'status':record['status'],'success':record['success'],'reason':record.get('failure_reason'),'output':str(directory)}),flush=True)
                if online_agent is not None:
                    online_agent.training_metadata=dict(training_identifiers(records,run),source='actual online physical interactions only',split='train')
                    online_agent.save(output/'dqn_online.pt')
            if online_agent is not None:
                for _ in range(120):
                    metric=online_agent.update()
                    if metric:all_metrics.append(metric)
                online_agent.training_metadata.update(episodes=len(records),manifest=str(args.manifest.relative_to(ROOT)))
                online_agent.save(output/'dqn_online.pt')
                (output/'training_curve.json').write_text(json.dumps(all_metrics,indent=2)+'\n')
            summary={'mode':args.mode,'episodes':len(records),'successes':sum(r['success'] for r in records),
                'records_file':str((output/'episodes.jsonl').relative_to(ROOT)),'status':'PASS' if len(records)==len(tasks) else 'FAILED',
                'scope':'Small development baseline chain; not paper performance or final large-scale training'}
            if args.mode=='e1':
                from apfg.baselines.evaluation import summarize_e1
                summary['E1']=summarize_e1(records,protocol['qualification_case_order'])
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        finally:
            if environment:environment.close()
            if app:app.close()
            memory_sampler.close()
            (output/'gpu_memory_samples.json').write_text(json.dumps(memory_sampler.samples,indent=2)+'\n')
    print(json.dumps({'output':str(output),'summary':summary}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
