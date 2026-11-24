"""V2 shared sensor-only pool: multi-grasp, neighbor pushes, edge lifts and views.

Schema differs from baseline_candidate_v1: old models are NOT compatible.
"""
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from apfg.perception.target_association import camera_cloud,colored_components
from apfg.primitives.preparation_v2 import plan
from apfg.primitives.swept import remove_observed_robot

VERSION='rgbd_candidate_pool_v2'
SCHEMA='baseline_candidate_v2_noncompatible'
MAX_CANDIDATES=32

def visible_objects(observation,scene,target):
    pp=[];cc=[];tree=cKDTree(scene)
    for c in observation.rgbd.values():
        p=camera_cloud(c).reshape(-1,3);rgb=c['rgb'][...,:3].reshape(-1,3)/255.;keep=np.isfinite(p).all(1)
        p=p[keep];rgb=rgb[keep];dist=tree.query(p)[0];keep=(dist<.003)&(p[:,2]>.010)
        pp.append(p[keep]);cc.append(rgb[keep])
    p=np.concatenate(pp);c=np.concatenate(cc)
    if len(target):
        keep=cKDTree(target).query(p)[0]>.014;p=p[keep];c=c[keep]
    _,idx=np.unique(np.floor(p/.005).astype(int),axis=0,return_index=True);p=p[idx];c=c[idx]
    clusters=[p[i] for i in colored_components(p,c,radius=.009) if len(i)>30]
    center=np.median(target,axis=0) if len(target) else np.array([.42,0,.06])
    clusters=[v for v in clusters if np.linalg.norm(np.median(v,axis=0)[:2]-center[:2])<.25 and np.ptp(v,axis=0).max()>.025]
    return sorted(clusters,key=lambda v:np.linalg.norm(np.median(v,axis=0)-center))[:4]

class CandidateGeneratorV2:
    def __init__(self):self.entries=[]
    def build(self,env,prep_used=0):
        observation=env.observe();errors=[];feasible=[];raw=[];target=env.target_cloud.copy();scene=np.empty((0,3));optical=None
        try:
            optical,scene,target,t,_=env._measured_clouds(observation)
            feasible,raw=env.infer_grasps(observation)
        except (RuntimeError,ValueError) as exc:errors.append(str(exc))
        if not len(scene):
            points=[]
            for c in observation.rgbd.values():
                p=camera_cloud(c).reshape(-1,3);keep=np.isfinite(p).all(1)&(p[:,0]>.09)&(p[:,0]<.84)&(abs(p[:,1])<.44)&(p[:,2]>.008)&(p[:,2]<.29);points.append(p[keep])
            scene=remove_observed_robot(np.concatenate(points),env.scene.model,env.scene.ctrl.joints(),opening=float(observation.proprio['joint_positions'][env.scene.ctrl.jaw_indices].sum()))
        entries=[];model=env.scene.model;start=env.scene.ctrl.joints()
        def add(kind,parameters,subject,quality,mode='stroke',grasp_plan=None,metadata=None):
            if len(entries)>=MAX_CANDIDATES:return
            if kind=='GRASP':proposed=grasp_plan;rejection=None
            elif prep_used:proposed=None;rejection={'reason':'PREPARATION_BUDGET_EXHAUSTED'}
            else:proposed,rejection=plan(model,start,kind,parameters,scene,subject,mode)
            center=np.median(subject,axis=0).tolist() if len(subject) else None
            descriptor={'type':kind,'parameters':parameters,'subject_center_m':center,'mode':mode,'version':VERSION}
            entries.append({**descriptor,'candidate_id':hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()[:20],'valid':proposed is not None,'quality':float(quality),'plan':proposed,'rejection':rejection,'metadata':metadata or {}})
        for c,p in feasible:
            add('GRASP',{'candidate':c,'approach_m':.10,'lift_m':.15,'extraction_xyz_m':env.spec['retrieval_region']['command_goal_xyz_m']},target,c['score'],grasp_plan=p)
        center=np.median(target,axis=0) if len(target) else env.tracker.center
        if center is None:center=np.array([.42,0,.06])
        # Ring candidates are relative to the measured target and actual views.
        for angle in [-135,-90,-45,45,90,180]:
            a=np.radians(angle);eye=center+np.array([.24*np.cos(a),.24*np.sin(a),.34]);eye[2]=max(eye[2],.39)
            current=observation.rgbd['wrist']['T_world_optical'][:3,3];old=(current-center)/max(np.linalg.norm(current-center),1e-9);new=(eye-center)/np.linalg.norm(eye-center)
            benefit=1-float(old@new)
            add('ACTIVE',{'camera_xyz_m':eye.tolist(),'look_at_xyz_m':center.tolist()},target,benefit,metadata={'estimated_view_angle_change_deg':float(np.degrees(np.arccos(np.clip(old@new,-1,1))))})
        subjects=visible_objects(observation,scene,target) if len(scene) else []
        for index,subject in enumerate(subjects):
            lo,hi=np.quantile(subject,.03,axis=0),np.quantile(subject,.97,axis=0);p=(lo+hi)/2;out=p-center;out[2]=0
            if np.linalg.norm(out)<.02:continue
            out/=np.linalg.norm(out)
            directions=[out,np.array([-out[1],out[0],0])]
            for d in directions:
                # Support point on observed near face along commanded direction.
                projections=subject@d;face=subject[projections<=np.quantile(projections,.12)]
                point=np.median(face,axis=0);point[2]=max(.023,lo[2]+.45*(hi[2]-lo[2]))
                if point[2]>hi[2]-.002:continue
                add('PUSH',{'contact_point_m':point.tolist(),'direction':d.tolist(),'distance_m':.10,'speed_m_s':.035},subject,.8-index*.08,metadata={'subject':'observable_neighbor','component_index':index,'source':'calibrated RGBD colored connected component; no instance GT'})
        if len(target)>12:
            from apfg.primitives.pinch_reorientation import propose
            for parameters in propose(target):
                band=target[target[:,2]>=parameters['observed_edge_band_min_z_m']]
                add('FLIP',parameters,band,.78,mode='pinch_roll',metadata={'mode':'observed_overhang_pinch_reorientation','source':'Measured exposed-edge height band; support/identity not supplied by simulator'})
            lo,hi=np.quantile(target,.03,axis=0),np.quantile(target,.97,axis=0)
            for d in [np.array(v,float) for v in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]]:
                axis=int(np.argmax(abs(d)));point=(lo+hi)/2;point[axis]=lo[axis] if d[axis]>0 else hi[axis];point[2]=max(.023,lo[2]+.38*(hi[2]-lo[2]))
                add('PUSH',{'contact_point_m':point.tolist(),'direction':d.tolist(),'distance_m':.10,'speed_m_s':.035},target,.25,metadata={'subject':'tracked_target'})
            for d in [np.array(v,float) for v in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]]:
                axis=int(np.argmax(abs(d)));point=(lo+hi)/2;point[axis]=lo[axis] if d[axis]>0 else hi[axis];point[2]=max(.023,lo[2]+.72*(hi[2]-lo[2]))
                params={'contact_point_m':point.tolist(),'direction':d.tolist(),'speed_m_s':.025}
                # Both upright toppling and low side levering are honest trials.
                if hi[2]-lo[2]>.08:
                    p={'contact_point_m':point.tolist(),'pivot_axis':np.cross([0,0,1],d).tolist(),'stroke_m':.16,'speed_m_s':.035}
                    add('FLIP',p,target,.45,metadata={'mode':'side_topple'})
                else:
                    point[2]=max(.027,lo[2]+.5*(hi[2]-lo[2]));params['contact_point_m']=point.tolist()
                    add('FLIP',params,target,.5,mode='edge_lift',metadata={'mode':'lying_edge_lift_trial'})
        self.entries=entries
        return {'schema':SCHEMA,'generator':VERSION,'association':env.association,'errors':errors,'counts':{k:sum(e['type']==k and e['valid'] for e in entries) for k in ['GRASP','ACTIVE','PUSH','FLIP']},'proposed':len(entries),'raw_associated_grasps':len(raw),'observable_neighbor_components':len(subjects),'observation_path':str(env.last_observation_dir.relative_to(env.out))}
