"""PRIVATE evaluation only. Never imported by tracker/candidate generator."""
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.scenes.m1 import registry
from apfg.perception.target_association import camera_cloud

def evaluate(env,visible):
    states=env.oracle_state();entries=registry();target=np.asarray(states['target']['T_world']);boxes={};neighbors={}
    for n,s in states.items():
        t=np.asarray(s['T_world']);half=np.abs(t[:3,:3])@np.asarray(entries[s['asset_id']]['geometry_dimensions_m'])/2
        boxes[n]=(t[:3,3]-half,t[:3,3]+half)
    for n,(lo,hi) in boxes.items():
        if n=='target':continue
        a,b=boxes['target'];gap=np.maximum(np.maximum(lo-b,a-hi),0)
        neighbors[n]={'aabb_gap_m':float(np.linalg.norm(gap)),'center_distance_m':float(np.linalg.norm(np.asarray(states[n]['T_world'])[:3,3]-target[:3,3]))}
    cameras={};points=[]
    for n,c in visible.rgbd.items():
        segmentation=env.scene.cameras[n].get_current_frame().get('instance_segmentation')
        if not segmentation:
            cameras[n]={'status':'NOT_RUN','reason':'Evaluator instance labels unavailable'};continue
        ids=np.asarray(segmentation['data']).squeeze();labels=segmentation['info']['idToLabels']
        target_ids=[int(i) for i,label in labels.items() if '/World/Object' in str(label)]
        gt=np.isin(ids,target_ids);pred=visible.mask[n];tp=int(np.sum(pred&gt));count=int(pred.sum());truth=int(gt.sum())
        cameras[n]={'status':'PASS','gt_target_pixels':truth,'predicted_pixels':count,'precision':tp/count if count else None,'recall':tp/truth if truth else None,'intersection_pixels':tp}
        p=camera_cloud(c);points.append(p[gt&np.isfinite(p).all(-1)])
    cloud=np.concatenate(points) if points else np.empty((0,3));voxels=np.unique(np.floor(cloud/.005).astype(int),axis=0) if len(cloud) else np.empty((0,3),int)
    guessed=env.association.get('center_m');status=env.association['status'];precision=[v['precision'] for v in cameras.values() if v.get('precision') is not None]
    identity='TARGET_LOST' if status=='LOST' else 'UNCERTAIN' if status=='UNCERTAIN' else 'WRONG_TARGET' if precision and max(precision)<.5 else 'ASSOCIATED'
    return {'scope':'PRIVATE_EVALUATOR_ONLY; source poses/GT masks never supplied to perception or selection','body_count':len(states),'neighbors':neighbors,'neighbors_gap_under_4cm':sum(v['aabb_gap_m']<.04 for v in neighbors.values()),'neighbors_gap_under_10cm':sum(v['aabb_gap_m']<.10 for v in neighbors.values()),'states':states,'association_identity':identity,'visible_centroid_to_body_origin_m':None if guessed is None else float(np.linalg.norm(np.asarray(guessed)-target[:3,3])),'cameras':cameras,'fused_gt_visible_voxels_5mm':len(voxels),'fused_gt_voxel_keys':voxels.tolist()}
