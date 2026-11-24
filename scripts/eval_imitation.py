#!/usr/bin/env python3
"""Offline BC/DP prediction audit; physical eval uses the shared environment."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.baselines.dataset import TransitionDataset,project_path,file_sha256
from apfg.baselines.bc import BCPolicy
from apfg.baselines.dp_adapted import DiffusionHighLevelPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['bc','dp'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',default='holdout');p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261123)
    args=p.parse_args();torch.set_num_threads(1)
    dataset=TransitionDataset(args.manifest,args.split);model=(BCPolicy if args.method=='bc' else DiffusionHighLevelPolicy).load(project_path(args.checkpoint))
    predictions=[]
    for i,row in enumerate(dataset.rows):
        action=model.act(dataset.values[i],seed=args.seed+i)
        predictions.append({'episode_id':row['episode_id'],'step_index':row['step_index'],'action':action,
            'demonstrated_action':int(dataset.values[i]['action']),'valid':action>=0 and bool(dataset.values[i]['mask'][action])})
    out=project_path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'PASS' if all(v['valid'] for v in predictions) and predictions else 'NOT_RUN','scope':'Offline legal-candidate inference only; NOT physical success',
        'method':args.method,'dataset':dataset.summary(),'checkpoint_sha256':file_sha256(project_path(args.checkpoint)),'predictions':predictions}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
