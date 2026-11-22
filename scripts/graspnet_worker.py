#!/usr/bin/env python3
"""Run the real upstream GraspNet network in a clean process, without Kit.

Input is observed RGB-D point data and calibration only. No object ground-truth
pose, synthetic grasps, top-down substitution or learning is used here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'assets/models/graspnet/checkpoint-rs.tar')
    parser.add_argument('--seed',type=int,default=20261121)
    args=parser.parse_args()
    import numpy as np
    import torch
    from apfg.download import sha256
    repo=ROOT/'third_party/graspnet-baseline'
    for p in [repo,repo/'models',repo/'utils',repo/'pointnet2',repo/'knn',ROOT/'third_party/graspnet_extensions']:
        sys.path.insert(0,str(p))
    from graspnet import GraspNet,pred_decode
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    with np.load(args.observation,allow_pickle=False) as observation:
        points=observation['points_optical'].astype(np.float32)
        transform=observation['T_world_optical'].copy()
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<50 or not np.isfinite(points).all():
        raise ValueError('RGBD_INPUT_INVALID: need >=50 finite measured optical points')
    indices=np.random.choice(len(points),20000,replace=len(points)<20000)
    net=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,
                 hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'],strict=True)
    net.cuda().eval()
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.no_grad():
        result=net({'point_clouds':torch.from_numpy(points[indices][None]).cuda()})
        rows=pred_decode(result)[0].cpu().numpy()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    if rows.ndim!=2 or rows.shape[1]!=17 or not len(rows):raise RuntimeError('NETWORK_RETURNED_NO_GRASPS')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,raw=rows,T_world_optical=transform,sampled_indices=indices)
    metadata={'status':'PASS','architecture':'upstream GraspNet baseline; is_training=False; 300 views/12 angles/4 depths',
        'checkpoint_sha256':sha256(args.checkpoint),'checkpoint_epoch':checkpoint.get('epoch'),'observation_sha256':sha256(args.observation),
        'input_points':len(points),'sampled_points':20000,'raw_candidates':len(rows),'network_time_s':elapsed,
        'score_range':[float(rows[:,0].min()),float(rows[:,0].max())],
        'width_range_m':[float(rows[:,1].min()),float(rows[:,1].max())],
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'information_mode':'RGBD workspace crop, no oracle instance segmentation or object pose',
        'warning':'Checkpoint obtained from documented community mirror; its HF LFS hash is verified, upstream Drive hash unavailable'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
