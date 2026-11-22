#!/usr/bin/env python3
"""Small-budget M1 development smokes; no training and no E2/E3 execution."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import traceback
import uuid
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch
from apfg.scenes.m1 import generate,FAMILIES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=FAMILIES,default='Direct');p.add_argument('--seed',type=int,default=2026112201)
    p.add_argument('--target-asset',default='ycb:004_sugar_box');p.add_argument('--scene-file',type=Path)
    p.add_argument('--action',choices=['observe','active','push','flip','grasp','push_grasp','flip_grasp','active_grasp'],default='observe')
    p.add_argument('--action-params',type=Path);p.add_argument('--information-mode',choices=['RGBD_POLICY','ORACLE_DIAGNOSTIC'],default='RGBD_POLICY')
    p.add_argument('--gui',action='store_true');p.add_argument('--interactive',action='store_true')
    p.add_argument('--joint-speed',type=float,default=.3);p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    add_launcher_args(p);args=p.parse_args()
    if args.gui and args.headless:p.error('Choose GUI or headless')
    spec=json.loads(args.scene_file.read_text()) if args.scene_file else generate(args.family,args.seed,args.target_asset)
    if spec['family']=='C4' and args.action!='observe':p.error('C4/Mix supports scene inspection only in M1; E3 is NOT_RUN')
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'outputs/m1/runs'/run;out.mkdir(parents=True)
    (out/'scene.json').write_text(json.dumps(spec,indent=2)+'\n');(out/'run_config.json').write_text(json.dumps({'args':vars(args),'command':sys.argv},default=str,indent=2)+'\n')
    wall_start=time.monotonic()
    app=None;environment=None;result={'status':'NOT_RUN','run_id':run,'family':spec['family'],'scope':'M1 development smoke, NOT formal E1/E2/E3',
        'information_mode':args.information_mode,'mode':'headless' if args.headless else 'gui','artifacts_directory':str(out.relative_to(ROOT))}
    try:
        app=launch(args)
        import numpy as np
        from apfg.simulation.m1_env import M1Environment
        from apfg.primitives.m1_actions import Action,ActionType,view_tcp
        from apfg.perception.m1_observation import DiagnosticObservation
        from apfg.visualization.grasp_candidates import draw_grasps
        environment=M1Environment(args,app,out,spec);observation=environment.initial_observation
        visible=environment.visible(observation)
        initial_pixels={n:int(m.sum()) for n,m in visible.mask.items()}
        costs=[];policy_records=[]
        if args.action.startswith('active'):
            tcp,q=view_tcp(environment.scene.model,[.48,.24,.42],[.42,0,.06])
            params=json.loads(args.action_params.read_text()) if args.action_params else {'camera_xyz_m':[.48,.24,.42],'look_at_xyz_m':[.42,0,.06]}
            step=environment.step(Action(ActionType.ACTIVE,params));observation=step.observation;costs.append(step.cost);policy_records.append(step.diagnostics)
        elif args.action.startswith(('push','flip')):
            kind=ActionType.FLIP if args.action.startswith('flip') else ActionType.PUSH
            params,source=environment.compose_contact(kind,observation)
            if args.action_params:params=json.loads(args.action_params.read_text());source='Explicit user action file'
            result['action_parameter_source']=source
            step=environment.step(Action(kind,params));observation=step.observation;costs.append(step.cost);policy_records.append(step.diagnostics)
        if args.action.endswith('grasp'):
            candidate,plan,candidates=environment.infer_grasps(observation)
            costs.append(environment.last_perception_cost)
            draw_grasps(environment.scene.stage,candidates,candidates.index(candidate) if candidate else None)
            environment.scene.ctrl.phase='candidate_display';environment.scene.ctrl.hold(.5)
            environment.scene.snapshot('grasp_candidates')
            if candidate is None:raise RuntimeError('NO_TARGET_GRASP_PASSED_BOUNDED_FULL_ARM_CHECK')
            result['selected_candidate']=candidate;result['grasp_plan']=plan
            step=environment.step(Action(ActionType.GRASP,{'candidate':candidate,'approach_m':.10,'lift_m':.15,
                'extraction_xyz_m':spec['retrieval_region']['command_goal_xyz_m']}));observation=step.observation;costs.append(step.cost);policy_records.append(step.diagnostics)
        environment.scene.snapshot('final_scene');environment.scene.snapshot('final_wrist','wrist')
        environment.close()
        final_visible=environment.visible(observation)
        result.update(status='PASS',scene_stability=environment.initial_stability,
            actual_settling_duration_s=environment.settling_duration_s,
            initial_state=environment.settled_initial,final_state=environment.oracle_state(),
            initial_visible_hint_pixels=initial_pixels,final_visible_hint_pixels={n:int(m.sum()) for n,m in final_visible.mask.items()},
            action_evaluation=environment.evaluation_records,costs=costs,videos=environment.scene.video_reports,
            policy_observation_fields=list(vars(final_visible)),policy_diagnostics=policy_records)
        result['performance']={'physics_steps':int(environment.scene.world.current_time_step_index)-environment.scene.start_step,
            'simulation_time_s':float(environment.scene.world.current_time)-environment.scene.start_time,
            'scene_wall_time_s':time.monotonic()-environment.scene.wall_start,'startup_and_run_wall_time_s':time.monotonic()-wall_start,
            'rendering':'RTX RGBD 640x480 or CLI resolution, 30fps; PhysX CPU 480Hz; single simulated scene; no training throughput claim'}
        gates=[environment.initial_stability['status']=='PASS']
        for e in environment.evaluation_records:
            gates.append(e['command_status']=='PASS')
            for name in ['active','push','flip','retrieval']:
                if name in e:gates.append(e[name]['status']=='PASS')
        gates.append(all(v['status']=='PASS' for group in environment.scene.video_reports.values() for v in group.values()))
        result['status']='PASS' if all(gates) else 'FAILED'
        (out/'policy_history.json').write_text(json.dumps(policy_records,indent=2)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['status'],'family':spec['family'],'action':args.action,'output':str(out)}),flush=True)
        if args.interactive and not args.headless:
            import omni.ui as ui
            info=ui.Window('M1 research scene',width=420,height=190)
            with info.frame:
                with ui.VStack():
                    ui.Label(f'{spec["family"]} | seed {spec["seed"]} | {args.information_mode}')
                    ui.Label('Target: '+args.target_asset)
                    ui.Label('Development smoke: '+result['status'])
                    ui.Label('Family is an unverified mechanism hypothesis, not a required skill label.',word_wrap=True)
            print('M1_VIEW_READY: completed; close window to exit',flush=True)
            while app.is_running():environment.scene.world.render()
    except Exception as exc:
        result.update(status='FAILED',reason=str(exc));(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        if environment:
            environment.close()
            result.setdefault('scene_stability',environment.initial_stability)
            result.setdefault('action_evaluation',environment.evaluation_records)
            result.setdefault('videos',environment.scene.video_reports)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['status'],'reason':result.get('reason'),'output':str(out)}),flush=True)
        if app:app.close()
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
