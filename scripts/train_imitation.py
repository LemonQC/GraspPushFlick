#!/usr/bin/env python3
"""Train BC or high-level adapted DP on identical real successful episodes."""
import argparse
import contextlib
import csv
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.baselines.dataset import SuccessfulDemoDataset,project_path,file_sha256,split_demo_episodes
from apfg.baselines.bc import BCPolicy
from apfg.baselines.dp_adapted import DiffusionHighLevelPolicy
from apfg.baselines.gpu_lock import gpu_job


def batch_items(dataset,method,horizon):
    items=[]
    for i in range(len(dataset)):
        item=dataset[i]
        if method=='dp':item['sequence'],item['sequence_mask']=dataset.sequence(i,horizon)
        items.append(item)
    return items


def candidate_accuracy(model,dataset,seed):
    if not len(dataset):return None
    predicted=[model.act(dataset.values[i],seed=seed+i) for i in range(len(dataset))]
    return sum(prediction==int(dataset.values[i]['action']) for i,prediction in enumerate(predicted))/len(predicted)


def objective(model,batch,method,device,generator=None):
    tensors={k:v.to(device) for k,v in batch.items()}
    if method=='bc':return nn.functional.cross_entropy(model(tensors['obs'],tensors['candidates'],tensors['mask']),tensors['action'])
    return model.noise_loss(tensors['obs'],tensors['candidates'],tensors['mask'],tensors['sequence'],tensors['sequence_mask'],generator=generator)


def run(args):
    start=time.monotonic();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    manifest=project_path(args.manifest)
    all_success=SuccessfulDemoDataset(manifest,split='train')
    if not len(all_success):raise ValueError('No actual successful train-split demonstrations; collect physical episodes before training')
    train_ids,validation_ids=split_demo_episodes(all_success,args.validation_fraction,args.seed)
    train=SuccessfulDemoDataset(manifest,'train',train_ids);validation=SuccessfulDemoDataset(manifest,'train',validation_ids)
    output=project_path(args.output);output.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    model=(BCPolicy(args.hidden) if args.method=='bc' else DiffusionHighLevelPolicy(args.horizon,args.diffusion_steps)).to(device)
    if args.method=='dp':
        selected=torch.from_numpy(np.stack([v['candidates'][v['action']] for v in train.values])).to(device)
        model.fit_action_normalizer(selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    shuffle=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(batch_items(train,args.method,args.horizon),batch_size=min(args.batch_size,len(train)),shuffle=True,generator=shuffle,num_workers=0)
    validation_items=batch_items(validation,args.method,args.horizon)
    validation_loader=DataLoader(validation_items,batch_size=max(1,min(args.batch_size,len(validation_items))),shuffle=False,num_workers=0)
    config=vars(args).copy();config.update(manifest=str(manifest.relative_to(ROOT)),output=str(output.relative_to(ROOT)))
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    dataset_info={'all_successful_train':all_success.summary(),'optimization_train':train.summary(),'episode_validation':validation.summary(),
                  'validation_scope':'Held-out episodes from TRAIN objects/seeds; physical holdout is separate and never trained on.'}
    (output/'dataset_manifest.json').write_text(json.dumps(dataset_info,indent=2)+'\n')
    curves=[];step=0
    for epoch in range(args.epochs):
        model.train();total=0.;count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);loss=objective(model,batch,args.method,device)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite imitation loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(batch['obs']);count+=len(batch['obs']);step+=1
        model.eval();validation_loss=None
        if validation_items:
            evaluation_rng=torch.Generator(device=device).manual_seed(args.seed+104729)
            with torch.no_grad():
                values=[(float(objective(model,b,args.method,device,evaluation_rng)),len(b['obs'])) for b in validation_loader]
            validation_loss=sum(v*n for v,n in values)/sum(n for _,n in values)
        curves.append({'epoch':epoch+1,'updates':step,'train_loss':total/count,'validation_loss':validation_loss,'elapsed_s':time.monotonic()-start})
        if (epoch+1)%max(1,args.epochs//10)==0:print(json.dumps(curves[-1]),flush=True)
    with (output/'curve.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    checkpoint=output/'model.pt'
    metadata={'dataset':dataset_info,'training_seed':args.seed,'updates':step,'training_config':config,
              'implementation':'FROM_SCRATCH_BC' if args.method=='bc' else 'ADAPTED_OFFICIAL_UNET_HIGH_LEVEL_DP_LOCAL_DDPM'}
    model.save(checkpoint,metadata)
    torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'shuffle_rng':shuffle.get_state(),
                'updates':step,'epochs':args.epochs,'seed':args.seed},output/'training_state.pt')
    restored=(BCPolicy if args.method=='bc' else DiffusionHighLevelPolicy).load(checkpoint,device=device)
    sample=train.values[0]
    if args.method=='bc':
        obs=torch.tensor(sample['obs'],device=device).unsqueeze(0);candidates=torch.tensor(sample['candidates'],device=device).unsqueeze(0);mask=torch.tensor(sample['mask'],device=device).unsqueeze(0)
        with torch.no_grad():difference=float((model(obs,candidates,mask)-restored(obs,candidates,mask)).abs().max())
    else:
        obs=torch.tensor(sample['obs'],device=device).unsqueeze(0);candidates=torch.tensor(sample['candidates'],device=device).unsqueeze(0);mask=torch.tensor(sample['mask'],device=device).unsqueeze(0)
        difference=float((model.sample(obs,candidates,mask,args.seed)-restored.sample(obs,candidates,mask,args.seed)).abs().max())
    summary={'status':'PASS' if difference==0. else 'FAILED','method':args.method,'scope':'Small real-demo training-chain test, not physical policy success',
        'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':file_sha256(manifest),'successful_demo_subset_sha256':all_success.subset_sha256,
        'dataset':dataset_info,'epochs':args.epochs,'updates':step,'first_train_loss':curves[0]['train_loss'],'last_train_loss':curves[-1]['train_loss'],
        'validation_loss':curves[-1]['validation_loss'],'train_candidate_accuracy':candidate_accuracy(restored,train,args.seed),
        'episode_validation_candidate_accuracy':candidate_accuracy(restored,validation,args.seed),
        'accuracy_warning':'Static demonstrated-candidate agreement is not physical retrieval success.',
        'data_readiness':{'successful_train_episodes':len(all_success.by_episode),
            'ten_plus_successful_episode_interface_smoke':len(all_success.by_episode)>=12,
            'successful_action_support':all_success.summary()['action_types'],
            'large_scale_training_authorized_by_this_result':False,
            'reason':'E1 mechanism qualification, demonstration diversity and physical holdout results must be audited separately.'},
        'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':file_sha256(checkpoint),'checkpoint_prediction_max_abs_difference':difference,
        'wall_time_s':time.monotonic()-start,'device':str(device),'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='PASS' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['bc','dp'],required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--horizon',type=int,default=2);p.add_argument('--diffusion-steps',type=int,default=32)
    p.add_argument('--validation-fraction',type=float,default=.2);p.add_argument('--seed',type=int,default=20261123)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=1);args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:p.error('Positive epochs/batch-size required')
    context=gpu_job('imitation-'+args.method) if args.device=='cuda' else contextlib.nullcontext()
    with context:return run(args)


if __name__=='__main__':raise SystemExit(main())
