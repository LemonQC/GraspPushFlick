"""V2 common sensor/physics interface; v1 and E1 v3 are untouched."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.spatial import cKDTree
from apfg.simulation.m1_env import M1Environment,StepResult
from apfg.simulation.clutter_hardware import ClutterHardwareScene
from apfg.perception.m1_observation import InformationMode,PolicyObservation
from apfg.perception.target_association import TargetAssociation,camera_cloud
from apfg.primitives.swept import remove_observed_robot
from apfg.evaluation.m1_metrics import pose_stability,flip_outcome
from apfg.robots.sr5 import ROOT

class ClutterEnvironment(M1Environment):
    def __init__(self,args,app,out,spec):
        args.scene_spec=spec;self.scene=ClutterHardwareScene(args,app,out);self.out=Path(out);self.spec=spec;self.mode=InformationMode.RGBD_POLICY
        self._closed=False;self.history=[];self.action_index=0;self.evaluation_records=[];self._candidate_catalog={};self._grasp_plans={};self.observation_index=0
        self.tracker=TargetAssociation(spec['target_hint']);self.scene.ctrl.phase='scene_settle';self.scene.ctrl.hold(spec['settle_s'])
        for _ in range(4):
            if pose_stability(self.scene.rows,list(self.scene.m1_bodies))['status']=='PASS':break
            self.scene.ctrl.hold(1.)
        self.initial_stability=pose_stability(self.scene.rows,list(self.scene.m1_bodies));self.settled_initial=self.oracle_state()
        self.move_view('front');self.scene.snapshot('initial_scene');self.scene.snapshot('initial_wrist','wrist');self.scene.snapshot('initial_fixed','fixed')
        self.initial_observation=self.observe()
    def observe(self):
        rgbd,proprio=self.sensor_data();joints=self.scene.ctrl.joints();opening=float(proprio['joint_positions'][self.scene.ctrl.jaw_indices].sum())
        def exclude(points):
            keep=remove_observed_robot(points,self.scene.model,joints,opening=opening)
            return np.zeros(len(points),bool) if not len(keep) else cKDTree(keep).query(points)[0]<1e-7
        masks,target,association=self.tracker.update(rgbd,float(self.scene.world.current_time),exclude=exclude)
        visible=PolicyObservation(rgbd,masks,proprio,{'type':'temporal_association_v2','status':association['status']},history=list(self.history))
        self.association=association;self.target_cloud=target
        directory=self.out/f'observation_{self.observation_index:03d}';directory.mkdir(exist_ok=True);self.observation_index+=1
        arrays={'target_points':target,'joint_positions':proprio['joint_positions'],'T_world_tcp':proprio['T_world_tcp']}
        for name,c in rgbd.items():
            for key,v in c.items():arrays[name+'_'+key]=v
            arrays[name+'_mask']=masks[name]
        np.savez_compressed(directory/'rgbd.npz',**arrays);(directory/'association.json').write_text(json.dumps(association,indent=2)+'\n')
        self.last_observation_dir=directory
        return visible
    def _measured_clouds(self,observation):
        scene=[];optical=None;t=None
        for name,c in observation.rgbd.items():
            p=camera_cloud(c);valid=np.isfinite(p).all(-1)&(p[...,0]>.09)&(p[...,0]<.84)&(abs(p[...,1])<.44)&(p[...,2]>-.008)&(p[...,2]<.29)
            scene.append(p[valid])
            if name=='wrist':
                t=c['T_world_optical'];optical=(p[valid]-t[:3,3])@t[:3,:3]
        points=remove_observed_robot(np.concatenate(scene),self.scene.model,self.scene.ctrl.joints(),opening=float(observation.proprio['joint_positions'][self.scene.ctrl.jaw_indices].sum()))
        if len(self.target_cloud)<12:raise RuntimeError('TARGET_'+self.association['status'])
        return optical,points,self.target_cloud.copy(),t,observation
    def infer_grasps(self,observation,max_checked=12,max_feasible=3):
        from apfg.perception.graspnet_candidates import ranked_candidates
        from apfg.primitives.grasp_motion import plan_grasp
        optical,world,target,t,_=self._measured_clouds(observation)
        directory=self.out/f'perception_{self.action_index:02d}';directory.mkdir(exist_ok=True)
        z=(optical@t[:3,:3].T+t[:3,3])[:,2];input_points=optical[z>.006]
        np.savez_compressed(directory/'observation.npz',points_optical=input_points,T_world_optical=t,observed_target_component=target,scene_points_world=world)
        env=os.environ.copy();env.pop('LD_PRELOAD',None);env['PYTHONPATH']=str(ROOT)
        command=[str(ROOT/'.conda/isaac-apfg/bin/python'),str(ROOT/'scripts/graspnet_worker.py'),'--observation',str(directory/'observation.npz'),'--output',str(directory/'network_predictions.npz'),'--seed',str(self.spec['seed'])]
        with (directory/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while proc.poll() is None:
                self.scene.world.render()
                if time.monotonic()-start>180:proc.terminate();proc.wait();raise RuntimeError('NETWORK_TIMEOUT')
            if proc.returncode:
                message=(directory/'worker.log').read_text()
                raise RuntimeError('NETWORK_EMPTY' if 'NETWORK_RETURNED_NO_GRASPS' in message else 'NETWORK_ERROR: raw worker log retained')
        with np.load(directory/'network_predictions.npz',allow_pickle=False) as d:raw=d['raw'].copy()
        candidates,_=ranked_candidates(raw,t,self.scene.model,60);tree=cKDTree(target);associated=[];reject=[];feasible=[]
        self._grasp_plans={}
        for c in candidates:
            seed=t[:3,:3]@np.asarray(c['raw_row'])[13:16]+t[:3,3]
            lo,hi=np.quantile(target,.01,axis=0),np.quantile(target,.99,axis=0)
            if tree.query(seed)[0]<.025 and np.all(seed>=lo-.004) and np.all(seed<=hi+.004):associated.append(c)
            else:reject.append({'index':c['index'],'reason':'outside_observed_target_surface_or_band'})
        self._candidate_catalog={c['index']:json.dumps(c,sort_keys=True) for c in associated};self._grasp_context=(world,target)
        for c in associated[:max_checked]:
            plan,attempts=plan_grasp(self.scene.model,self.scene.ctrl.joints(),c,world,target,extraction_xyz_m=self.spec['retrieval_region']['command_goal_xyz_m'])
            if plan is None:reject.append({'index':c['index'],'reason':'IK_REJECT_OR_COLLISION','attempts':attempts})
            else:
                self._grasp_plans[c['index']]=plan;feasible.append((c,plan))
                if len(feasible)>=max_feasible:break
        (directory/'candidates.json').write_text(json.dumps(associated,indent=2)+'\n');(directory/'selection.json').write_text(json.dumps({'feasible':[c for c,p in feasible],'rejections':reject,'max_checked':max_checked,'max_feasible':max_feasible},indent=2)+'\n')
        return feasible,associated
    def step_candidate(self,entry):
        from apfg.primitives.preparation_v2 import execute_plan
        from apfg.primitives.grasp_motion import execute_grasp_plan
        kind=entry['type'];self.action_index+=1;before=self.oracle_state();row_start=len(self.scene.rows);started=time.monotonic();sim=float(self.scene.world.current_time);error=None
        try:
            if kind=='GRASP':execute_grasp_plan(self.scene.ctrl,entry['plan'])
            else:execute_plan(self.scene.ctrl,entry['plan'])
        except Exception as exc:error=str(exc)
        after=self.oracle_state();rows=self.scene.rows[row_start:];stable=pose_stability(self.scene.rows,list(before))
        contact=lambda path:any(any(path in pair and any('/World/SR5_Assembly/' in p for p in pair) for pair in row['contacts']) for row in rows)
        outcome={'operation':kind,'command_status':'FAILED' if error else 'PASS','error':error,'before':before,'after':after,'stable_after':stable,'actual_target_contact':contact('/World/Object'),'cost':{'wall_s':time.monotonic()-started,'simulation_s':float(self.scene.world.current_time)-sim}}
        if kind=='FLIP':
            pairs=self.scene.rows[-1]['contacts'];support=[pair for pair in pairs if '/World/Object' in pair and not any('/World/SR5_Assembly/' in v for v in pair)]
            released=not any('/World/Object' in pair and any('/World/SR5_Assembly/' in v for v in pair) for pair in pairs)
            outcome['flip']=flip_outcome(before['target']['T_world'],after['target']['T_world'],stable['bodies'].get('target',{}).get('stable',False) and bool(support) and released,outcome['actual_target_contact'])
            outcome['flip'].update(final_support_contact_pairs=support,robot_released=released)
        if kind=='ACTIVE':
            moved=max(np.linalg.norm(np.asarray(after[n]['T_world'])[:3,3]-np.asarray(before[n]['T_world'])[:3,3]) for n in before)
            outcome['active']={'status':'PASS' if not error and moved<.002 else 'FAILED','max_object_displacement_m':float(moved)}
        if kind=='GRASP':outcome['retrieval']=self.retrieval_evaluation(rows)
        if kind=='PUSH':
            # Private evaluator maps observed subject center to closest measured
            # initial body, AFTER selection. No mapping is returned to policy.
            center=np.asarray(entry['subject_center_m']);name=min(before,key=lambda n:np.linalg.norm(np.asarray(before[n]['T_world'])[:3,3]-center))
            path=self.scene.m1_bodies[name]['path'];delta=np.asarray(after[name]['T_world'])[:3,3]-np.asarray(before[name]['T_world'])[:3,3]
            outcome['push']={'evaluated_subject':name,'subject_contact':contact(path),'displacement_m':delta.tolist(),'status':'PASS' if contact(path) and np.linalg.norm(delta[:2])>.03 else 'FAILED'}
        outcome['distractor_displacement_m']={n:float(np.linalg.norm(np.asarray(after[n]['T_world'])[:3,3]-np.asarray(before[n]['T_world'])[:3,3])) for n in before if n!='target'}
        self.history.append({'type':kind,'status':outcome['command_status']});self.evaluation_records.append(outcome)
        self.scene.snapshot(f'after_{self.action_index:02d}');self.scene.snapshot(f'after_{self.action_index:02d}_fixed','fixed');self.scene.snapshot(f'after_{self.action_index:02d}_wrist','wrist')
        self.tracker.set_action_motion(kind)
        if not error:
            if kind=='GRASP':
                self.tracker.apply_motion_prior(np.asarray(entry['plan']['goals'][-1])-np.asarray(entry['plan']['execution_pose']['xyz_m']))
            elif kind=='FLIP' and entry['plan'].get('mode')=='pinch_roll' and self.tracker.center is not None:
                from scipy.spatial.transform import Rotation
                p=entry['parameters'];roll=Rotation.from_rotvec(np.asarray(p['roll_axis'])*np.radians(p['roll_degrees']))
                placed=np.asarray(p['tcp_xyz_m']).copy();placed[2]=p['release_tcp_z_m']
                expected=roll.apply(self.tracker.center-np.asarray(p['tcp_xyz_m']))+placed
                self.tracker.apply_motion_prior(expected-self.tracker.center)
            elif kind=='PUSH' and entry.get('metadata',{}).get('subject')=='tracked_target':
                self.tracker.apply_motion_prior(np.asarray(entry['parameters']['direction'])*entry['parameters']['distance_m'])
        self.observe();return outcome
    def close(self):
        if self._closed:return
        (self.out/'association_history.json').write_text(json.dumps(self.tracker.records,indent=2)+'\n')
        super().close()
