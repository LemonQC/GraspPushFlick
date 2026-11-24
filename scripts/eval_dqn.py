#!/usr/bin/env python3
"""Checkpoint inference audit on heldout observations, not physical success."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.baselines.dqn import CandidateDQN
from apfg.baselines.dqn_io import load_transitions,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',type=Path);args=p.parse_args()
    transitions,rows=load_transitions(args.manifest,'holdout');agent=CandidateDQN.load(args.checkpoint)
    training=agent.training_metadata
    for row in rows:
        for key,stored in [('episode_id','train_episode_ids'),('scene_id','train_scene_ids'),('object_id','train_object_ids'),('seed','train_seeds')]:
            if row[key] in training.get(stored,[]):raise ValueError('Checkpoint training/heldout overlap: '+key)
    predictions=[]
    for transition,row in zip(transitions,rows):
        action=agent.act(transition)
        predictions.append({'episode_id':row['episode_id'],'scene_id':row['scene_id'],'action':action,
                            'legal':action is not None and bool(transition['mask'][action])})
    out=(args.output or ROOT/'outputs/baselines/dqn_eval'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])).resolve()
    out.relative_to(ROOT);out.mkdir(parents=True,exist_ok=False)
    result={'status':'PASS' if all(r['legal'] for r in predictions) else 'FAILED',
        'scope':'Heldout checkpoint inference only; no physical success claim','physical_test':'NOT_RUN_BY_THIS_SCRIPT',
        'checkpoint_sha256':file_sha256(args.checkpoint),'heldout_observations':len(rows),'predictions':predictions}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
