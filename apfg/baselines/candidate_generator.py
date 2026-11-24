"""One sensor-only candidate pool/feature encoder shared by every Track A policy."""
import hashlib
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,TYPES,SCHEMA_VERSION
from apfg.primitives.preparation_motion import plan_preparation
from apfg.visualization.recording import pointcloud_from_depth

GENERATOR_VERSION='rgbd_candidate_pool_v1'


def observation_hash(observation):
    h=hashlib.sha256()
    for name,camera in sorted(observation.rgbd.items()):
        h.update(name.encode())
        for key in ['rgb','depth_m','K','T_world_optical']:h.update(np.ascontiguousarray(camera[key]).tobytes())
        h.update(np.ascontiguousarray(observation.mask[name]).tobytes())
    h.update(json.dumps(observation.task_hint,sort_keys=True).encode())
    h.update(np.ascontiguousarray(observation.proprio['joint_positions']).tobytes())
    return h.hexdigest()


def features(observation,target,raw_candidates,entries,prep_used,elapsed_s,history):
    """No scene/family/asset ID or evaluation callback is accepted by this API."""
    obs=np.zeros(OBS_DIM,np.float32)
    joints=np.asarray(observation.proprio['joint_positions']);obs[:6]=joints[:6]/np.pi;obs[6:8]=joints[6:8]/.04
    tcp=np.asarray(observation.proprio['T_world_tcp']);obs[8:11]=tcp[:3,3]
    if len(target):obs[11:14]=np.median(target,axis=0);obs[14:17]=np.ptp(target,axis=0)
    obs[17]=np.log1p(len(target))/np.log1p(100000)
    for i,name in enumerate(['fixed','wrist']):
        obs[18+i]=float(np.mean(observation.mask[name]));depth=observation.rgbd[name]['depth_m']
        obs[20+i]=float(np.mean(np.isfinite(depth)&(depth>.005)&(depth<2)))
    obs[22]=len(raw_candidates)/8;obs[23]=max([c['score'] for c in raw_candidates]+[0])
    obs[24]=sum(e['valid'] and e['type']=='GRASP' for e in entries)/8;obs[25]=prep_used;obs[26]=elapsed_s/60
    if history:
        obs[27+TYPES.index(history[-1]['type'])]=1;obs[31]=history[-1]['status']!='PASS'
    candidates=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);mask=np.zeros(MAX_CANDIDATES,bool)
    for i,e in enumerate(entries[:MAX_CANDIDATES]):
        candidates[i]=e['features'];mask[i]=e['valid']
    return {'obs':obs,'candidates':candidates,'mask':mask}


class CandidateGenerator:
    """Models see numeric features only; private payloads implement selected actions."""
    def __init__(self):self.entries=[];self.last_error=None

    def build(self,environment,prep_used=0,elapsed_s=0.):
        started=time.monotonic();visible=environment.visible(environment.observe());input_hash=observation_hash(visible)
        model=environment.scene.model;start=environment.scene.ctrl.joints();target=np.empty((0,3));scene=np.empty((0,3))
        raw=[];selected=None;grasp_plan=None;errors=[]
        try:
            _,scene,target,_,_=environment._measured_clouds(visible)
            selected,grasp_plan,raw=environment.infer_grasps(visible)
        except (RuntimeError,ValueError) as exc:
            errors.append(str(exc))
            # Active can still be proposed from visible workspace geometry;
            # target-absent observations never obtain oracle fallback masks.
            if not len(scene):
                points=[]
                for camera in visible.rgbd.values():
                    xyz,_=pointcloud_from_depth(camera['depth_m'],camera['rgb'],camera['K']);t=camera['T_world_optical']
                    xyz=xyz@t[:3,:3].T+t[:3,3];points.append(xyz[(xyz[:,2]>.005)&(xyz[:,2]<.28)&(xyz[:,0]>.09)&(xyz[:,0]<.84)&(abs(xyz[:,1])<.44)])
                scene=np.concatenate(points)
                from apfg.primitives.swept import remove_observed_robot
                scene=remove_observed_robot(scene,model,start,opening=float(visible.proprio['joint_positions'][6:8].sum()))
        entries=[];extent=np.ptp(target,axis=0) if len(target) else np.zeros(3)
        tcp=np.asarray(visible.proprio['T_world_tcp'])[:3,3]

        def add(kind,params,quality,benefit,plan=None,rejection=None,cost=10.):
            descriptor={'type':kind,'parameters':params,'generator_version':GENERATOR_VERSION}
            cid=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()[:20]
            vector=np.zeros(CAND_DIM,np.float32);vector[TYPES.index(kind)]=1;vector[4]=quality;vector[5]=cost/60
            if kind=='GRASP':
                pose=plan['execution_pose'];position=pose['xyz_m'];quat=pose['quat_xyzw'];direction=Rotation.from_quat(quat).apply([0,0,1])
                vector[12]=.10;vector[14]=params['candidate']['opening_m']/.08
            elif kind=='ACTIVE':
                position=params['camera_xyz_m'];direction=np.asarray(params['look_at_xyz_m'])-position;direction/=np.linalg.norm(direction);quat=plan['quat_xyzw'] if plan else [0,0,0,1]
            else:
                position=params['contact_point_m'];direction=params.get('direction',np.cross(params.get('pivot_axis',[0,1,0]),[0,0,1]));quat=plan['quat_xyzw'] if plan else [0,0,0,1]
                vector[12]=params.get('distance_m',params.get('stroke_m',0));vector[13]=params['speed_m_s']/.12
            vector[6:9]=position;vector[9:12]=direction;vector[15:19]=quat;vector[19:22]=extent
            vector[22]=np.linalg.norm(np.asarray(position)-tcp);vector[23]=benefit
            entries.append({'candidate_id':cid,'type':kind,'parameters':params,'valid':plan is not None,
                'features':vector.tolist(),'plan':plan,'rejection':rejection,'input_hash':input_hash,'generator_version':GENERATOR_VERSION})

        if selected is not None:
            add('GRASP',{'candidate':selected,'approach_m':.10,'lift_m':.15,'extraction_xyz_m':[.42,.40,.35]},selected['score'],selected['score'],grasp_plan,cost=24.)
        for eye in [[.48,.24,.42],[.20,-.08,.46]]:
            params={'camera_xyz_m':eye,'look_at_xyz_m':[.42,0,.06]}
            plan,rejection=(None,{'reason':'preparation_budget_exhausted'}) if prep_used else plan_preparation(model,start,'ACTIVE',params,scene,target)
            deficit=1-min(len(target)/2000,1)
            add('ACTIVE',params,.30+.35*deficit,deficit,plan,rejection,cost=7.)
        if len(target)>=20:
            lo,hi=np.quantile(target,.02,axis=0),np.quantile(target,.98,axis=0)
            from apfg.perception.graspnet_candidates import physical_tip_offset
            minimum_contact_height=physical_tip_offset(model)+.003
            for kind in ['PUSH','FLIP']:
                for direction in [[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0]]:
                    d=np.asarray(direction);axis=int(np.argmax(abs(d)));point=(lo+hi)/2
                    point[axis]=lo[axis] if d[axis]>0 else hi[axis]
                    point[2]=max(minimum_contact_height,lo[2]+(.16 if kind=='PUSH' else .76)*(hi[2]-lo[2]))
                    if point[2]>hi[2]-.002:continue
                    params={'contact_point_m':point.tolist(),'speed_m_s':.04}
                    if kind=='PUSH':params.update(direction=direction,distance_m=.10)
                    else:params.update(pivot_axis=np.cross([0,0,1],d).tolist(),stroke_m=.18)
                    plan,rejection=(None,{'reason':'preparation_budget_exhausted'}) if prep_used else plan_preparation(model,start,kind,params,scene,target)
                    benefit=float(np.clip(max(extent[:2])/.08-1,0,1)) if kind=='FLIP' else float(np.clip(len(scene)/max(len(target),1)/15,0,1))
                    add(kind,params,.22+.25*benefit,benefit,plan,rejection,cost=22.)
        self.entries=entries[:MAX_CANDIDATES];self.last_error='; '.join(errors) or None
        state=features(visible,target,raw,self.entries,prep_used,elapsed_s,environment.history)
        return state,{'observation':visible,'target_points':target,'scene_points':scene,'input_hash':input_hash,
                      'generation_wall_s':time.monotonic()-started,'errors':errors,'schema':SCHEMA_VERSION}
