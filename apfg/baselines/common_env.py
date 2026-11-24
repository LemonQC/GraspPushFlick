"""Track A reset/observe/legal_actions/step API on the measured SR5 simulator."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from apfg.baselines.candidate_generator import CandidateGenerator
from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,SCHEMA_VERSION
from apfg.robots.sr5 import ROOT
from apfg.scenes.m1 import generate
from apfg.primitives.m1_actions import Action


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BaselineEnvironment:
    """Only policy_state() tensors enter learned models. Evaluation stays private."""
    def __init__(self,args,app):
        self.args=args;self.app=app;self.environment=None;self.episode_counter=0
        self.generator=CandidateGenerator();self.state=None;self.terminal=True;self.success=False

    def _dispose(self):
        from isaacsim.core.api import World
        if self.environment is not None:
            env=self.environment
            if hasattr(env,'scene'):
                scene=env.scene
                if not getattr(env,'_closed',False):
                    try:env.close()
                    except (AttributeError,RuntimeError):scene.close()
                scene.subscription=None
                for camera in scene.cameras.values():camera.pause()
                scene.cameras.clear()
                scene.world.stop()
            self.environment=None
        World.clear_instance();gc.collect()
        # All scene/body resets occur here, BETWEEN episodes only.
        from isaacsim.core.utils.stage import create_new_stage
        create_new_stage()

    def reset(self,seed,scene_id,target_asset,out,scene_spec=None):
        self._dispose()
        from apfg.simulation.m1_env import M1Environment
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.spec=scene_spec or generate(scene_id,int(seed),target_asset)
        (self.out/'scene.json').write_text(json.dumps(self.spec,indent=2)+'\n')
        self.args.information_mode='RGBD_POLICY';self.args.scene_spec=self.spec
        self.prep_used=0;self.success=False;self.terminal=False;self.failure_reason=None
        self.initial_state={};self.initial_stability={'status':'NOT_RUN'}
        self.steps=[];self.generation_costs=[];self.reset_wall=time.monotonic()
        self.environment=M1Environment.__new__(M1Environment)
        self.environment.__init__(self.args,self.app,self.out,self.spec)
        self.episode_counter+=1
        self.initial_state=self.environment.oracle_state()  # private evaluator record
        from apfg.evaluation.m1_metrics import pose_stability
        settling=self.environment.initial_stability
        current=pose_stability(self.environment.scene.rows,list(self.environment.scene.m1_bodies))
        self.initial_stability=current if settling['status']=='PASS' else settling
        self.episode_sim_start=float(self.environment.scene.world.current_time)
        if self.initial_stability['status']!='PASS':
            self.terminal=True;self.failure_reason='INITIAL_UNSTABLE';self.state=self.empty_state()
        else:self._refresh()
        return self.observe()

    @staticmethod
    def empty_state():return {'obs':np.zeros(OBS_DIM,np.float32),'candidates':np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32),'mask':np.zeros(MAX_CANDIDATES,bool)}

    def _refresh(self):
        elapsed=float(self.environment.scene.world.current_time)-self.episode_sim_start
        self.state,diagnostic=self.generator.build(self.environment,self.prep_used,elapsed)
        self.generation_costs.append({'step':len(self.steps),'wall_s':diagnostic['generation_wall_s'],'errors':diagnostic['errors']})
        self._save_decision(diagnostic)

    def _save_decision(self,diagnostic):
        index=len(self.steps);directory=self.out/f'decision_{index:02d}';directory.mkdir(exist_ok=True)
        visible=diagnostic['observation'];arrays={}
        for name,c in visible.rgbd.items():
            for key in ['rgb','depth_m','K','T_world_optical']:arrays[name+'_'+key]=c[key]
        arrays.update(target_points=diagnostic['target_points'],scene_points=diagnostic['scene_points'])
        arrays.update(self.state)
        np.savez_compressed(directory/'observation.npz',**arrays)
        public_pool=[{k:v for k,v in e.items() if k!='plan'} for e in self.generator.entries]
        (directory/'candidates.json').write_text(json.dumps({'schema':SCHEMA_VERSION,'input_hash':diagnostic['input_hash'],'candidates':public_pool},indent=2)+'\n')
        (directory/'plans.json').write_text(json.dumps([e['plan'] for e in self.generator.entries],indent=2)+'\n')
        self.observation_reference=str((directory/'observation.npz').relative_to(ROOT))

    def observe(self):return {k:v.copy() for k,v in self.state.items()}

    def legal_actions(self):
        return [{'index':i,'candidate_id':e['candidate_id'],'type':e['type'],'valid':e['valid']} for i,e in enumerate(self.generator.entries)]

    def step(self,index):
        from apfg.baselines.reward import compute_reward
        if self.terminal:raise RuntimeError('EPISODE_ALREADY_TERMINAL')
        if not isinstance(index,(int,np.integer)) or not 0<=index<len(self.generator.entries) or not self.state['mask'][index]:
            raise ValueError('ACTION_MASK_VIOLATION')
        before=self.observe();entry=copy.deepcopy(self.generator.entries[index]);kind=entry['type'];started=time.monotonic()
        if kind!='GRASP' and self.prep_used>=1:raise ValueError('PREPARATION_BUDGET_EXHAUSTED')
        result=self.environment.step(Action(kind,entry['parameters']),execution_plan=entry['plan'] if kind!='GRASP' else None)
        evaluation=self.environment.evaluation_records[-1]
        if kind!='GRASP':self.prep_used+=1
        if kind=='GRASP':
            self.success=evaluation['retrieval']['status']=='PASS';self.terminal=True
            if not self.success:self.failure_reason='GRASP_MISS_OR_INCOMPLETE_RETRIEVAL'
        failed=evaluation['command_status']!='PASS'
        if failed:self.terminal=True;self.failure_reason='IK_PATH_OR_TRACKING: '+str(evaluation['error'])
        if evaluation['catastrophic_distractor_drop']:
            self.terminal=True;self.success=False;self.failure_reason='SEVERE_DISTRACTOR_DROP'
        if float(self.environment.scene.world.current_time)-self.episode_sim_start>120 or time.monotonic()-self.reset_wall>600:
            self.terminal=True;self.success=False;self.failure_reason='EPISODE_TIMEOUT'
        disturbance=sum(evaluation['distractor_displacement_m'].values())
        reward,terms=compute_reward(success=self.success,terminal=self.terminal,preparation=kind!='GRASP',
            simulation_time_s=result.cost['simulation_time_s'],disturbance_m=disturbance,command_failed=failed)
        record={'index':int(index),'candidate_id':entry['candidate_id'],'type':kind,'parameters':entry['parameters'],
                'reward':reward,'reward_terms':terms,'cost':result.cost,'physical_evaluation':evaluation,
                'observation_before':self.observation_reference}
        self.steps.append(record)
        if self.terminal:self.state=self.empty_state()
        else:
            self._refresh()
            if not self.state['mask'].any():self.terminal=True;self.failure_reason='NO_LEGAL_CANDIDATE_AFTER_PREPARATION'
        record['total_step_wall_s']=time.monotonic()-started
        transition={'obs':before['obs'],'candidates':before['candidates'],'mask':before['mask'],'action':int(index),'reward':float(reward),
                    'next_obs':self.state['obs'],'next_candidates':self.state['candidates'],'next_mask':self.state['mask'],'done':bool(self.terminal)}
        file=self.out/f'transition_{len(self.steps)-1:02d}.npz';np.savez_compressed(file,**transition)
        record['transition_path']=str(file.relative_to(ROOT));record['data_sha256']=file_sha(file)
        return self.observe(),float(reward),self.terminal,{'operation_status':evaluation['command_status'],'failure_reason':self.failure_reason},transition

    def is_success(self):return bool(self.success)

    def render(self):self.environment.scene.world.render()

    def finish(self):
        env=self.environment
        env.scene.snapshot('baseline_final');env.scene.snapshot('baseline_final_wrist','wrist')
        env.close()
        recording_ok=bool(env.scene.video_reports) and not env.scene.frame_gaps and all(v['status']=='PASS' for group in env.scene.video_reports.values() for v in group.values())
        result={'status':'PASS' if self.success else 'FAILED','success':self.success,'failure_reason':self.failure_reason,
            'initial_state':self.initial_state,'scene_stability':self.initial_stability,'steps':self.steps,
            'generation_costs':self.generation_costs,'videos':env.scene.video_reports,
            'episode_wall_s':time.monotonic()-self.reset_wall,'simulation_time_s':float(env.scene.world.current_time)-env.scene.start_time,
            'controller_revision':'parallel_jaw_execution_v1','candidate_generator':'rgbd_candidate_pool_v1',
            'policy_information':'RGBD_POLICY sensor-derived features; no family/seed/asset/GT in model state',
            'reset_ordinal_same_isaac_process':self.episode_counter}
        result['recording_validation']='PASS' if recording_ok else 'FAILED'
        if not recording_ok:result['status']='FAILED';result['recording_error']='VIDEO_CAPTURE_OR_CONTENT_FAILURE'
        if not self.success and not result['failure_reason']:result['failure_reason']='NO_LEGAL_GRASP_OR_REQUESTED_ACTION'
        (self.out/'baseline_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def close(self):self._dispose()
