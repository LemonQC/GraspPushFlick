#!/usr/bin/env python3
"""Train Double DQN on recorded real online interactions, excluding holdout."""
import argparse
from contextlib import nullcontext
import csv
import datetime as dt
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import yaml
from apfg.baselines.dqn import CandidateDQN
from apfg.baselines.dqn_io import load_transitions,file_sha256
from apfg.baselines.gpu_lock import gpu_job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/baselines/dqn.yaml')
    p.add_argument('--output',type=Path);p.add_argument('--updates',type=int);p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--seed',type=int);args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text());device=args.device or cfg['device'];updates=args.updates if args.updates is not None else cfg['updates']
    seed=args.seed if args.seed is not None else cfg['seed']
    if updates<=0:p.error('updates must be positive')
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=(args.output or ROOT/'outputs/baselines/dqn'/run).resolve();out.relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=False)
    (out/'run_config.json').write_text(json.dumps({'command':sys.argv,'config':cfg,'device':device,'updates':updates,'seed':seed},indent=2)+'\n')
    start=time.monotonic();torch.set_num_threads(2)
    try:
        transitions,rows=load_transitions(args.manifest,'train')
        with gpu_job('DQN training') if device=='cuda' else nullcontext():
            agent=CandidateDQN(cfg['network'],seed,device)
            for transition in transitions:agent.observe(transition)
            agent.training_metadata={'manifest':str(args.manifest.resolve().relative_to(ROOT)),'manifest_sha256':file_sha256(args.manifest),
                'train_episode_ids':sorted({r['episode_id'] for r in rows}),'train_scene_ids':sorted({r['scene_id'] for r in rows}),
                'train_object_ids':sorted({r['object_id'] for r in rows}),'train_seeds':sorted({int(r['seed']) for r in rows}),
                'mode':'small replay optimization on actual recorded physical interaction; not a large online RL run'}
            logs=[]
            for _ in range(updates):
                metrics=agent.update()
                if metrics is None:raise RuntimeError('Insufficient actual replay for the configured min_replay gate')
                metrics['wall_s']=time.monotonic()-start;logs.append(metrics)
            checkpoint=agent.save(out/'checkpoint.pt')
            # Verify restored tensors, masks, optimizer and replay actually load.
            restored=CandidateDQN.load(checkpoint,device='cpu')
            first=transitions[0];choice=restored.act(first)
            if choice is None or not first['mask'][choice]:raise RuntimeError('Restored checkpoint violated legal action mask')
        with (out/'curve.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
        (out/'curve.json').write_text(json.dumps(logs,indent=2)+'\n')
        result={'status':'PASS','scope':'DQN training/checkpoint chain only; physical evaluation is separate','implementation':'FROM_SCRATCH',
            'algorithm':'candidate-conditioned Double DQN','actual_train_transitions':len(transitions),
            'actual_train_episodes':len({r['episode_id'] for r in rows}),'updates':len(logs),'first_loss':logs[0]['loss'],'last_loss':logs[-1]['loss'],
            'minimum_loss':min(x['loss'] for x in logs),'target_network_syncs':agent.target_syncs,
            'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':file_sha256(checkpoint),
            'replay_persisted':len(restored.replay),'restored_masked_action':choice,'wall_time_s':time.monotonic()-start,
            'training_metadata':agent.training_metadata,'physical_test':'NOT_RUN_BY_THIS_SCRIPT'}
    except Exception as exc:
        result={'status':'FAILED','reason':str(exc),'wall_time_s':time.monotonic()-start}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
