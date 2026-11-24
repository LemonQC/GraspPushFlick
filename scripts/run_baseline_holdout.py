#!/usr/bin/env python3
"""Frozen common environment, six matched physical holdouts and inference timing."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch
from apfg.baselines.gpu_lock import gpu_job
from scripts.run_baseline_experiments import episode,enforce_holdout,append_json,GPUMemorySampler
from scripts.prepare_baseline_protocol import verify_protocol_sources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TimedPolicy:
    """Instrumentation receives the same numeric state as the actual policy."""
    def __init__(self,policy,name,seed):self.policy=policy;self.name=name;self.seed=seed;self.decisions=[]
    def act(self,state):
        started=time.perf_counter()
        action=self.policy.act(state,seed=self.seed) if self.name in ('bc','dp') else self.policy.act(state)
        elapsed=time.perf_counter()-started
        h=hashlib.sha256()
        for key in ['obs','candidates','mask']:h.update(np.ascontiguousarray(state[key]).tobytes())
        self.decisions.append({'method':self.name,'elapsed_s':elapsed,'input_features_sha256':h.hexdigest(),
                               'selected_index':None if action is None else int(action),'legal_candidates':int(state['mask'].sum())})
        return action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/m1/e1_protocol_v3.json')
    p.add_argument('--output',type=Path);p.add_argument('--method',choices=['all','grasp_only','heuristic','random_valid','dqn','bc','dp'],default='all')
    p.add_argument('--dqn-checkpoint',type=Path);p.add_argument('--bc-checkpoint',type=Path);p.add_argument('--dp-checkpoint',type=Path)
    p.add_argument('--joint-speed',type=float,default=.3);p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--gui',action='store_true');add_launcher_args(p);args=p.parse_args()
    if args.gui and args.headless:p.error('Choose GUI or headless')
    protocol=json.loads(args.protocol.read_text());audit=verify_protocol_sources(protocol)
    if audit['status']!='PASS':raise RuntimeError('Frozen common implementation changed: '+str(audit))
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=args.output or ROOT/'outputs/baselines/campaigns'/run;out.mkdir(parents=True,exist_ok=False)
    methods=['grasp_only','heuristic','random_valid','dqn','bc','dp'] if args.method=='all' else [args.method]
    tasks=[{'scene_id':case['scene_id'],'family':case['family'],'seed':case['seed'],'object_id':case['target_asset'],
            'method':method,'split':'holdout','policy_seed':20261123}
           for method in methods for case in protocol['baseline_partitions']['heldout']['cases']]
    (out/'run_config.json').write_text(json.dumps({'args':vars(args),'command':sys.argv},default=str,indent=2)+'\n')
    app=None;env=None;records=[]
    with gpu_job('Physical matched baseline holdout'):
        checkpoints={name:getattr(args,name+'_checkpoint') for name in ['dqn','bc','dp'] if name in methods}
        for name,path in checkpoints.items():
            if path is None or not path.is_file():raise ValueError(name+' completed checkpoint required')
        registration={'status':'NOT_RUN','scope':'Fixed small physical holdout; no paper statistical claim',
            'ordered_tasks':tasks,'common_protocol_sha256':protocol['frozen_protocol_sha256'],
            'controller_revision':protocol['controller_revision'],'camera_mode':'headless' if args.headless else 'gui',
            'checkpoint_sha256':{name:digest(path) for name,path in checkpoints.items()},
            'evaluation_runner_sha256':digest(__file__),'policy_seed':20261123,
            'inference_timing':'CPU actual act() wall time per refreshed numeric observation; includes diffusion denoising/projection'}
        (out/'heldout_preregistration.json').write_text(json.dumps(registration,indent=2)+'\n')
        sampler=GPUMemorySampler();sampler.thread.start()
        try:
            app=launch(args)
            import torch
            torch.set_num_threads(1)
            from apfg.baselines.common_env import BaselineEnvironment
            from apfg.baselines.heuristic import make_policy
            env=BaselineEnvironment(args,app);policies={}
            for name in methods:
                if name=='dqn':
                    from apfg.baselines.dqn import CandidateDQN
                    policies[name]=CandidateDQN.load(checkpoints[name],device='cpu')
                elif name=='bc':
                    from apfg.baselines.bc import BCPolicy
                    policies[name]=BCPolicy.load(checkpoints[name],device='cpu')
                elif name=='dp':
                    from apfg.baselines.dp_adapted import DiffusionHighLevelPolicy
                    policies[name]=DiffusionHighLevelPolicy.load(checkpoints[name],device='cpu')
                else:policies[name]=make_policy(name,20261123)
                if name in checkpoints:enforce_holdout(policies[name],name,tasks)
            for i,task in enumerate(tasks):
                method=task['method'];timed=TimedPolicy(policies[method],method,task['policy_seed']);start=len(sampler.samples)
                # Route GraspOnly through its ordinary policy act() so its
                # decision latency is measured alongside all other methods.
                execution_task=dict(task,method='b0_grasp_only' if method=='grasp_only' else method)
                directory=out/f'episode_{i:03d}';record,_=episode(env,execution_task,directory,policy=timed)
                record.update(method=method,policy_inference=timed.decisions,mode='headless' if args.headless else 'gui',
                    gpu_device_memory_sampled_peak_mib=max((s['device_used_mib'] for s in sampler.samples[start:]),default=None))
                (directory/'episode.json').write_text(json.dumps(record,indent=2)+'\n')
                (directory/'policy_inference.json').write_text(json.dumps(timed.decisions,indent=2)+'\n')
                append_json(out/'episodes.jsonl',record);records.append(record)
                append_json(ROOT/'outputs/baselines/dataset/episodes_manifest.jsonl',{k:v for k,v in record.items() if k not in ['steps','initial_state','videos']})
                for j,step in enumerate(record.get('steps',[])):
                    append_json(ROOT/'outputs/baselines/dataset/manifest.jsonl',{'transition_path':step['transition_path'],
                        'episode_id':record['episode_id']+'_'+run,'scene_id':task['scene_id'],'split':'holdout','success':record['success'],
                        'method':method,'demonstrator_source':'physical_holdout_not_for_training','object_id':task['object_id'],
                        'seed':task['seed'],'action_type':step['type'],'step_index':j,'data_sha256':step['data_sha256'],
                        'rgbd_before':step['observation_before'],'trajectory':str((directory/'telemetry.json').relative_to(ROOT))})
                print(json.dumps({'completed':i+1,'planned':len(tasks),'method':method,'scene':task['scene_id'],
                    'status':record['status'],'success':record['success'],'reason':record['failure_reason']}),flush=True)
            summary={'mode':'evaluate','episodes':len(records),'successes':sum(r['success'] for r in records),
                'status':'PASS' if len(records)==len(tasks) else 'FAILED','scope':'Same two physical holdout cases per method; small chain test only',
                'per_method':{m:{'episodes':sum(r['method']==m for r in records),'successes':sum(r['method']==m and r['success'] for r in records)} for m in methods}}
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        finally:
            if env:env.close()
            if app:app.close()
            sampler.close();(out/'gpu_memory_samples.json').write_text(json.dumps(sampler.samples,indent=2)+'\n')
    print(json.dumps({'output':str(out),'summary':summary}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
