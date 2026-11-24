#!/usr/bin/env python3
"""Bounded GUI development tests of versioned clutter/association/action support."""
import argparse
import datetime as dt
import hashlib
import shutil
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch
from apfg.baselines.gpu_lock import gpu_job
from scripts.run_baseline_experiments import GPUMemorySampler

def save(path,value):path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',default='Direct:INSPECT,C1:PUSH,C2:FLIP,C3:ACTIVE,Mixed:INSPECT');p.add_argument('--seed',type=int,default=2026112401);p.add_argument('--output',type=Path)
    p.add_argument('--flip-mode',choices=['any','stroke','edge_lift','pinch_roll'],default='any');p.add_argument('--candidate-rank',type=int,default=0);p.add_argument('--joint-speed',type=float,default=.3);p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480);p.add_argument('--gui',action='store_true');add_launcher_args(p);args=p.parse_args()
    if args.gui and args.headless:p.error('Choose one mode')
    cases=[x.split(':') for x in args.cases.split(',')]
    if len(cases)>10:p.error('Development budget <=10 episodes per invocation; no training')
    from apfg.scenes.clutter_v2 import generate
    tasks=[{'family':f,'requested':a,'seed':args.seed,'spec':generate(f,args.seed)} for f,a in cases]
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=args.output or ROOT/'outputs/clutter_v2/campaigns'/run;out.mkdir(parents=True,exist_ok=False)
    source_files=['apfg/scenes/clutter_v2.py','apfg/simulation/clutter_hardware.py','apfg/simulation/clutter_env.py','apfg/perception/target_association.py','apfg/baselines/candidate_generator_v2.py','apfg/primitives/preparation_v2.py','apfg/primitives/pinch_reorientation.py','apfg/evaluation/clutter_v2.py','scripts/run_clutter_v2.py','configs/scenes/v2/common.json']
    save(out/'development_preregistration.json',{'scope':'Finite development diagnostics, not formal E1 or training','command':sys.argv,'tasks':tasks,'candidate_rank':args.candidate_rank,'mode':'headless' if args.headless else 'gui','source_sha256':{n:digest(ROOT/n) for n in source_files},'controller':'parallel_jaw_execution_v1','old_schema_models':'NOT_LOADED_INCOMPATIBLE','created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    for name in source_files:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    app=None;holder=None;records=[];sampler=GPUMemorySampler()
    with gpu_job('Clutter v2 finite GUI validation'):
        # No queued process may launch changed code under stale registration.
        for name in source_files:
            if digest(ROOT/name)!=digest(out/'source_snapshot'/name):raise RuntimeError('REGISTERED_SOURCE_CHANGED_WHILE_QUEUED: '+name)
        sampler.thread.start()
        try:
            app=launch(args)
            from apfg.baselines.common_env import BaselineEnvironment
            from apfg.simulation.clutter_env import ClutterEnvironment
            from apfg.baselines.candidate_generator_v2 import CandidateGeneratorV2
            from apfg.evaluation.clutter_v2 import evaluate
            holder=BaselineEnvironment(args,app)
            for i,task in enumerate(tasks):
                directory=out/f'episode_{i:03d}';directory.mkdir();started=time.monotonic();env=None;record={'episode':i,'family':task['family'],'requested':task['requested'],'seed':args.seed,'status':'FAILED','retrieval':'NOT_RUN','actions':[]}
                try:
                    holder._dispose();env=ClutterEnvironment.__new__(ClutterEnvironment);holder.environment=env;env.__init__(args,app,directory,task['spec']);generator=CandidateGeneratorV2()
                    from apfg.evaluation.m1_metrics import pose_stability
                    env.initial_stability=pose_stability(env.scene.rows,list(env.scene.m1_bodies))
                    record['initial_stability']=env.initial_stability
                    initial=generator.build(env);save(directory/'pool_00.json',{'diagnostics':initial,'entries':generator.entries});before=evaluate(env,env.observe());save(directory/'evaluation_before.json',before);record['before']=initial
                    if env.initial_stability['status']!='PASS':raise RuntimeError('INITIAL_UNSTABLE')
                    request=task['requested']
                    if request!='INSPECT':
                        available=[e for e in generator.entries if e['type']==request and e['valid'] and (request!='FLIP' or args.flip_mode=='any' or e['mode']==args.flip_mode)];available.sort(key=lambda e:e['quality'],reverse=True)
                        if len(available)<=args.candidate_rank:raise RuntimeError('NO_LEGAL_'+request)
                        entry=available[args.candidate_rank];record['selected_candidate']=entry['candidate_id'];record['actions'].append(env.step_candidate(entry))
                        preparation_evaluation=evaluate(env,env.observe());save(directory/'evaluation_after_preparation.json',preparation_evaluation)
                        after_pool=generator.build(env,prep_used=request!='GRASP');save(directory/'pool_01.json',{'diagnostics':after_pool,'entries':generator.entries});record['after']=after_pool
                        if request!='GRASP' and record['actions'][-1]['command_status']=='PASS':
                            grasps=[e for e in generator.entries if e['type']=='GRASP' and e['valid']]
                            if grasps:record['actions'].append(env.step_candidate(grasps[0]));record['retrieval']=record['actions'][-1]['retrieval']['status']
                            else:record['failure_reason']='NO_LEGAL_GRASP_AFTER_'+request
                        elif request=='GRASP':record['retrieval']=record['actions'][-1]['retrieval']['status']
                    final_evaluation=evaluate(env,env.observe());save(directory/'evaluation_final.json',final_evaluation)
                    post=preparation_evaluation if request!='INSPECT' else final_evaluation;save(directory/'evaluation_after.json',post)
                    a=set(map(tuple,before['fused_gt_voxel_keys']));b=set(map(tuple,post['fused_gt_voxel_keys']))
                    record['observation_evaluation']={'before_fused_gt_visible_voxels':len(a),'after_fused_gt_visible_voxels':len(b),'novel_visible_voxels':len(b-a),'before_cameras':before['cameras'],'after_cameras':post['cameras'],'identity_before':before['association_identity'],'identity_after':post['association_identity'],'density_neighbors_under_4cm':before['neighbors_gap_under_4cm'],'body_count':before['body_count']}
                    record['status']='PASS' if request=='INSPECT' or record['retrieval']=='PASS' else 'FAILED'
                    if record['status']=='FAILED' and not record.get('failure_reason'):
                        record['failure_reason']='CONTACT_FAILURE: '+str(record['actions'][-1].get('error')) if record['actions'] and record['actions'][-1]['command_status']!='PASS' else 'GRASP_MISS_OR_INCOMPLETE_RETRIEVAL'
                except Exception as exc:
                    record['failure_reason']=str(exc);(directory/'traceback.txt').write_text(traceback.format_exc())
                finally:
                    if env and hasattr(env,'scene'):
                        try:
                            env.close();record['videos']=env.scene.video_reports;record['recording_status']='PASS' if env.scene.video_reports and not env.scene.frame_gaps and all(v['status']=='PASS' for g in env.scene.video_reports.values() for v in g.values()) else 'FAILED'
                        except Exception as exc:record['close_error']=str(exc)
                    record['wall_s']=time.monotonic()-started;save(directory/'result.json',record);records.append(record);save(out/'summary.json',{'scope':'Bounded development; all attempts including failures','episodes':records});print(json.dumps({'completed':i+1,'planned':len(tasks),'out':str(directory),'family':task['family'],'status':record['status'],'retrieval':record['retrieval'],'reason':record.get('failure_reason'),'counts':record.get('before',{}).get('counts')}),flush=True)
        finally:
            if holder:holder.close()
            sampler.close();save(out/'gpu_memory_samples.json',sampler.samples)
            save(out/'completion.json',{'completed':len(records),'planned':len(tasks),'all_episode_records_flushed':True,'recording_validation':[r.get('recording_status') for r in records]})
            if app:app.close()
    print(json.dumps({'output':str(out),'completed':len(records)}),flush=True)

if __name__=='__main__':main()
