"""M1 scene/action API on the already validated SR5 simulator and GraspNet."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from apfg.robots.sr5 import ROOT
from apfg.simulation.sr5_hardware import HardwareScene,pose_matrix
from apfg.perception.m1_observation import InformationMode,PolicyObservation,DiagnosticObservation,build_observation
from apfg.visualization.recording import pointcloud_from_depth
from apfg.primitives.m1_actions import Action,ActionType,execute_view,execute_stroke,execute_flip
from apfg.evaluation.m1_metrics import pose_stability,flip_outcome,support_state


@dataclass
class StepResult:
    observation:object
    terminal:bool
    cost:dict
    diagnostics:dict


class M1Environment:
    def __init__(self,args,app,out,spec):
        args.scene_spec=spec
        self.scene=HardwareScene(args,app,out)
        self._closed=False
        self.mode=InformationMode(args.information_mode);self.spec=spec;self.out=Path(out)
        self.history=[];self.action_index=0;self.evaluation_records=[];self._candidate_catalog={};self._grasp_plans={}
        if self.scene.recorder is None:self.scene.begin('m1')
        self.scene.ctrl.phase='scene_settle';self.scene.ctrl.hold(spec['settle_s'])
        self.initial_stability=pose_stability(self.scene.rows,list(self.scene.m1_bodies))
        self.settling_duration_s=spec['settle_s']
        # A bounded settle extension changes no body state or acceptance
        # threshold. Slightly rocking real meshes may need a few more seconds.
        for _ in range(4):
            if self.initial_stability['status']=='PASS':break
            self.scene.ctrl.hold(1.);self.settling_duration_s+=1.
            self.initial_stability=pose_stability(self.scene.rows,list(self.scene.m1_bodies))
        self.settled_initial=self.oracle_state()
        # Fixed workspace view, the same starting observation for every family.
        self.move_view('front')
        self.scene.snapshot('initial_scene');self.scene.snapshot('initial_wrist','wrist');self.scene.snapshot('initial_fixed','fixed')
        self.initial_observation=self.observe()

    def sensor_data(self):
        cameras={}
        for name in ['fixed','wrist']:
            c=self.scene.cameras[name]
            rgb=c.get_rgba();depth=c.get_depth()
            if rgb is None or depth is None:raise RuntimeError('Real rendered RGB-D unavailable')
            cameras[name]={'rgb':rgb[...,:3].copy(),'depth_m':depth.copy(),'K':c.get_intrinsics_matrix().copy(),
                           'T_world_optical':pose_matrix(*c.get_world_pose(camera_axes='ros'))}
        return cameras,{'joint_positions':self.scene.robot.get_joint_positions().copy(),
                         'joint_velocities':self.scene.robot.get_joint_velocities().copy(),
                         'T_world_tcp':self.scene.ctrl.tcp().copy()}

    def oracle_state(self):
        return {name:{'asset_id':d['spec']['asset_id'],'T_world':pose_matrix(*d['body'].get_world_pose()).tolist()}
                for name,d in self.scene.m1_bodies.items()}

    def observe(self):
        return build_observation(self.sensor_data,self.mode,self.spec['target_hint'],self.history,
                                 self.oracle_state if self.mode==InformationMode.ORACLE_DIAGNOSTIC else None)

    @staticmethod
    def visible(observation):
        return observation.visible if isinstance(observation,DiagnosticObservation) else observation

    def move_view(self,name):
        eye={'front':[.48,-.24,.42],'back':[.48,.24,.42],'left':[.20,-.08,.46]}[name]
        return execute_view(self.scene.ctrl,self.scene.model,eye,[.42,0,.06])

    def _measured_clouds(self,observation):
        visible=self.visible(observation);c=visible.rgbd['wrist'];t=c['T_world_optical']
        optical,colors=pointcloud_from_depth(c['depth_m'],c['rgb'],c['K'])
        world=optical@t[:3,:3].T+t[:3,3]
        keep=(world[:,0]>.09)&(world[:,0]<.84)&(abs(world[:,1])<.44)&(world[:,2]>-.008)&(world[:,2]<.28)
        from apfg.primitives.swept import remove_observed_robot
        # The network remains a wrist-RGBD detector. Obstacle/path checks use
        # every provided measured view, including obstacles outside wrist FOV.
        fused=[world[keep]]
        for name,other in visible.rgbd.items():
            if name=='wrist':continue
            p,_=pointcloud_from_depth(other['depth_m'],other['rgb'],other['K']);ot=other['T_world_optical']
            p=p@ot[:3,:3].T+ot[:3,3]
            region=(p[:,0]>.09)&(p[:,0]<.84)&(abs(p[:,1])<.44)&(p[:,2]>-.008)&(p[:,2]<.28)
            fused.append(p[region])
        scene_world=remove_observed_robot(np.concatenate(fused),self.scene.model,self.scene.ctrl.joints())
        target,_=pointcloud_from_depth(c['depth_m'],c['rgb'],c['K'],visible.mask['wrist'])
        target=target@t[:3,:3].T+t[:3,3]
        target=target[(target[:,0]>.09)&(target[:,0]<.84)&(abs(target[:,1])<.44)&(target[:,2]>.006)&(target[:,2]<.28)]
        if len(target)<20:raise RuntimeError('VISIBLE_TARGET_CUE_INSUFFICIENT: no oracle segmentation fallback')
        # Grow the fair RGB cue only through connected measured surface points.
        # Touching objects may merge: this is explicitly not perfect instance GT.
        surface=scene_world[scene_world[:,2]>.006]
        _,indices=np.unique(np.floor(surface/.004).astype(np.int64),axis=0,return_index=True);voxels=surface[indices]
        pairs=cKDTree(voxels).query_pairs(.006,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(voxels),len(voxels)))
        _,labels=connected_components(graph,directed=False)
        seed_dist,seed_indices=cKDTree(voxels).query(target)
        if np.any(seed_dist<.008):
            votes=np.bincount(labels[seed_indices[seed_dist<.008]]);component=int(votes.argmax())
            target=voxels[labels==component]
        return optical[keep],scene_world,target,t,visible

    def infer_grasps(self,observation):
        from apfg.perception.graspnet_candidates import ranked_candidates
        from apfg.primitives.grasp_motion import plan_grasp
        inference_start=time.monotonic()
        optical,world,target,t,visible=self._measured_clouds(observation)
        directory=self.out/f'perception_{self.action_index:02d}';directory.mkdir(exist_ok=True)
        z=(optical@t[:3,:3].T+t[:3,3])[:,2];input_points=optical[z>.006]
        np.savez_compressed(directory/'observation.npz',points_optical=input_points,T_world_optical=t,
                            observed_target_component=target,scene_points_world=world)
        env=os.environ.copy();env.pop('LD_PRELOAD',None);env['PYTHONPATH']=str(ROOT)
        command=[str(ROOT/'.conda/isaac-apfg/bin/python'),str(ROOT/'scripts/graspnet_worker.py'),
                 '--observation',str(directory/'observation.npz'),'--output',str(directory/'network_predictions.npz'),
                 '--seed',str(self.spec['seed'])]
        with (directory/'worker.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while process.poll() is None:
                self.scene.world.render()
                if time.monotonic()-start>180:process.terminate();process.wait();raise RuntimeError('GraspNet timeout')
            if process.returncode:raise RuntimeError('GraspNet inference failed; raw worker log retained')
        with np.load(directory/'network_predictions.npz',allow_pickle=False) as d:raw=d['raw'].copy()
        candidates,_=ranked_candidates(raw,t,self.scene.model,30)
        associated=[];rejections=[];tree=cKDTree(target)
        for candidate in candidates:
            rawrow=np.asarray(candidate['raw_row']);seed_world=t[:3,:3]@rawrow[13:16]+t[:3,3]
            if tree.query(seed_world)[0]>.030:
                rejections.append({'index':candidate['index'],'reason':'outside_visible_target_component'});continue
            associated.append(candidate)
        visible.grasp_candidates=associated
        self._candidate_catalog={c['index']:json.dumps(c,sort_keys=True) for c in associated}
        self._grasp_plans={};self._grasp_context=(world,target)
        (directory/'candidates.json').write_text(json.dumps(associated,indent=2)+'\n')
        selected=None;plan=None
        for candidate in associated[:8]:
            proposed,attempts=plan_grasp(self.scene.model,self.scene.ctrl.joints(),candidate,world,target,
                extraction_xyz_m=self.spec['retrieval_region']['command_goal_xyz_m'])
            if proposed is None:
                rejections.append({'index':candidate['index'],'reason':'no_checked_execution_variant','variant_attempts':attempts});continue
            selected=candidate;plan=proposed;self._grasp_plans[candidate['index']]=plan;break
        (directory/'selection.json').write_text(json.dumps({'rejections':rejections,'selected':selected,'plan':plan},indent=2)+'\n')
        self.last_perception_cost={'operation':'GRASPNET_AND_ONCE_SWEPT_CHECK','wall_time_s':time.monotonic()-inference_start,
                                   'simulation_time_s':0.,'action_count':0}
        return selected,plan,associated

    def compose_contact(self,kind,observation):
        if self.mode==InformationMode.ORACLE_DIAGNOSTIC:
            state=self.oracle_state()['target'];t=np.asarray(state['T_world'])
            from apfg.scenes.m1 import registry
            from itertools import product
            d=np.asarray(registry()[state['asset_id']]['geometry_dimensions_m'])
            corners=np.array(list(product(*[[-v/2,v/2] for v in d])))@t[:3,:3].T+t[:3,3]
            lo,hi=corners.min(axis=0),corners.max(axis=0);source='ORACLE_DIAGNOSTIC: measured target pose and source extents'
        else:
            _,_,points,_,_=self._measured_clouds(observation)
            lo,hi=np.quantile(points,.02,axis=0),np.quantile(points,.98,axis=0);source='RGBD_POLICY: measured target cue component percentiles only'
        z=float(lo[2]+(.76 if kind==ActionType.FLIP else .16)*(hi[2]-lo[2]))
        return {'contact_point_m':[float(lo[0]),float((lo[1]+hi[1])/2),z],
                **({'pivot_axis':[0,1,0],'stroke_m':.18,'speed_m_s':.04} if kind==ActionType.FLIP else {'direction':[1,0,0],'distance_m':.10,'speed_m_s':.04})},source

    def step(self,action):
        action=action if isinstance(action,Action) else Action(action['type'],action.get('parameters',{}))
        self.action_index+=1;start=time.monotonic();sim_start=float(self.scene.world.current_time)
        before=self.oracle_state();begin_row=len(self.scene.rows);output={};error=None
        if action.type!=ActionType.GRASP:self._candidate_catalog={};self._grasp_plans={}
        try:
            p=dict(action.parameters)
            if action.type==ActionType.ACTIVE:output=execute_view(self.scene.ctrl,self.scene.model,**p)
            elif action.type==ActionType.PUSH:output=execute_stroke(self.scene.ctrl,**p)
            elif action.type==ActionType.FLIP:output=execute_flip(self.scene.ctrl,**p)
            elif action.type==ActionType.GRASP:
                from apfg.primitives.grasp_motion import plan_grasp,execute_grasp_plan
                candidate=p['candidate']
                if self.mode==InformationMode.RGBD_POLICY and self._candidate_catalog.get(candidate['index'])!=json.dumps(candidate,sort_keys=True):
                    raise ValueError('RGBD_POLICY grasp must use an unchanged candidate from the actual measured GraspNet observation')
                params={'approach_m':float(p.get('approach_m',.10)),'lift_m':float(p.get('lift_m',.15)),
                        'extraction_xyz_m':np.asarray(p['extraction_xyz_m'],float).tolist()}
                plan=self._grasp_plans.get(candidate['index'])
                if plan is None or plan['parameters']!=params:
                    if not hasattr(self,'_grasp_context'):raise ValueError('Measured-cloud grasp planning context required')
                    world,target=self._grasp_context
                    plan,attempts=plan_grasp(self.scene.model,self.scene.ctrl.joints(),candidate,world,target,**params)
                    if plan is None:raise ValueError('Custom grasp parameters have no checked execution branch')
                output=execute_grasp_plan(self.scene.ctrl,plan)
        except Exception as exc:error=str(exc)
        after=self.oracle_state();rows=self.scene.rows[begin_row:]
        stable=pose_stability(self.scene.rows,list(self.scene.m1_bodies))
        target_contact=any(any('/World/Object' in pair and any('/World/SR5_Assembly/' in v for v in pair) for pair in row['contacts']) for row in rows)
        outcome={'command_status':'FAILED' if error else 'PASS','error':error,'operation':action.type.value,'stable_after':stable,
                 'before':before,'after':after,'actual_target_contact':target_contact}
        outcome['distractor_displacement_m']={name:float(np.linalg.norm(np.asarray(after[name]['T_world'])[:3,3]-np.asarray(before[name]['T_world'])[:3,3]))
                                             for name in before if name!='target'}
        outcome['catastrophic_distractor_drop']=any(np.asarray(row['m1_objects'][name]['T_world'])[2,3]<-.05
            for row in rows for name in before if name!='target')
        if action.type==ActionType.FLIP:
            support_pairs=[pair for pair in self.scene.rows[-1]['contacts'] if '/World/Object' in pair and not any('/World/SR5_Assembly/' in v for v in pair)]
            robot_still_contacting=any('/World/Object' in pair and any('/World/SR5_Assembly/' in v for v in pair) for pair in self.scene.rows[-1]['contacts'])
            supported=stable['bodies'].get('target',{}).get('stable',False) and bool(support_pairs) and not robot_still_contacting
            outcome['flip']=flip_outcome(before['target']['T_world'],after['target']['T_world'],supported,target_contact)
            outcome['flip'].update(final_support_contact_pairs=support_pairs,robot_released=not robot_still_contacting)
        if action.type==ActionType.PUSH:
            delta=np.asarray(after['target']['T_world'])[:3,3]-np.asarray(before['target']['T_world'])[:3,3]
            outcome['push']={'status':'PASS' if np.linalg.norm(delta[:2])>.03 and target_contact else 'FAILED','delta_m':delta.tolist()}
        if action.type==ActionType.ACTIVE:
            displacement=max(np.linalg.norm(np.asarray(after[k]['T_world'])[:3,3]-np.asarray(before[k]['T_world'])[:3,3]) for k in before)
            outcome['active']={'status':'PASS' if not error and displacement<.002 else 'FAILED','max_object_motion_m':float(displacement)}
        if action.type==ActionType.GRASP:outcome['retrieval']=self.retrieval_evaluation(rows)
        self.evaluation_records.append(outcome)
        public={'operation_status':outcome['command_status'],'execution_error':error,'controller_result':output}
        self.history.append({'type':action.type.value,'parameters':action.parameters,'status':outcome['command_status']})
        observation=self.observe()
        if self.mode==InformationMode.ORACLE_DIAGNOSTIC:public['oracle_evaluation']=outcome
        return StepResult(observation,action.type==ActionType.GRASP,
            {'wall_time_s':time.monotonic()-start,'simulation_time_s':float(self.scene.world.current_time)-sim_start,'action_count':1},public)

    def retrieval_evaluation(self,rows):
        from apfg.scenes.m1 import registry
        from itertools import product
        dims=np.asarray(registry()[self.scene.target_asset]['geometry_dimensions_m'])
        local=np.array(list(product(*[[-v/2,v/2] for v in dims])))
        hold=[r for r in rows if r['phase']=='grasp_extracted_hold']
        initial_z=np.asarray(self.settled_initial['target']['T_world'])[2,3]
        tests=[]
        for r in hold:
            t=np.asarray(r['m1_objects']['target']['T_world']);corners=local@t[:3,:3].T+t[:3,3]
            bilateral=all(any('/World/Object' in pair and any('/ral_'+s+'_finger' in v for v in pair) for pair in r['contacts']) for s in ['left','right'])
            tests.append(bool(corners[:,1].min()>.30 and t[2,3]-initial_z>.10 and bilateral))
        catastrophe=any(np.asarray(r['m1_objects'][name]['T_world'])[2,3]<-.05 for r in rows for name in self.scene.m1_bodies if name!='target')
        return {'status':'PASS' if len(tests)>=30 and all(tests) and not catastrophe else 'FAILED','hold_frames':len(tests),'fully_extracted_contact_frames':sum(tests),
                'catastrophic_distractor_drop':catastrophe,
                'criterion':'entire source-extent target beyond y=.30m, >.10m rise, real bilateral target contact, sustained 1s; evaluator-only GT',
                'final_support':support_state(self.oracle_state()['target']['T_world'])}

    def close(self):
        if self._closed:return
        self.scene.close()
        (self.out/'evaluation_oracle.json').write_text(json.dumps(self.evaluation_records,indent=2)+'\n')
        self._closed=True
