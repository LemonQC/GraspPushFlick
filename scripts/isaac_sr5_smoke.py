#!/usr/bin/env python3
"""Run real SR5 hardware smoke tests; GUI/headless use identical scene/control."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--test',choices=['all','joints','gripper','camera','grasp','push','move','commands','prediction'],default='all')
    p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--joint-speed',type=float,default=.15)
    p.add_argument('--xyz',type=float,nargs=3,default=[.45,0,.30])
    p.add_argument('--quat-xyzw',type=float,nargs=4,default=[1,0,0,0])
    p.add_argument('--frame',choices=['world','base'],default='world')
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--command-file',type=Path,help='JSON list of explicit move_tcp/move_joints/set_opening/grasp/push commands')
    p.add_argument('--predictions',type=Path,help='External .npz with filtered Nx17 GraspNet predictions and matching T_world_optical; no model inference is implied')
    p.add_argument('--prediction-index',type=int,default=0)
    add_launcher_args(p);args=p.parse_args()
    if args.gui and args.headless:p.error('--gui and --headless are mutually exclusive')
    if args.test=='commands' and args.command_file is None:p.error('--test commands requires --command-file')
    if args.test=='prediction' and args.predictions is None:p.error('--test prediction requires --predictions')
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'outputs/isaac_sr5_hardware/runs'/run;out.mkdir(parents=True)
    app=None;result={'status':'NOT_RUN'}
    (out/'run_config.json').write_text(json.dumps({'args':vars(args),'actual_command':sys.argv,'run_id':run},default=str,indent=2)+'\n')
    try:
        app=launch(args)
        from apfg.simulation.sr5_hardware import run_tests
        result=run_tests(args,app,out)
    except Exception as exc:
        result.update(status='FAILED',reason=str(exc))
        (out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        result.update(run_id=run,mode='headless' if args.headless else 'gui',artifacts_directory=str(out.relative_to(ROOT)))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['status'],'output':str(out),'reason':result.get('reason'),'checks':result.get('checks')}),flush=True)
        if app:app.close()
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
