#!/usr/bin/env python3
"""Observed RGB-D -> actual GraspNet -> visible 6D poses -> physical SR5 grasp."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apfg.simulation.launcher import add_launcher_args,launch


def demo(args,app,out):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from apfg.simulation.sr5_hardware import HardwareScene,pose_matrix
    from apfg.visualization.recording import pointcloud_from_depth,write_ply
    from apfg.perception.graspnet_candidates import ranked_candidates,observed_collision
    from apfg.visualization.grasp_candidates import draw_grasps,information_window
    from apfg.download import sha256
    scene=HardwareScene(args,app,out);ctrl=scene.ctrl;result={'status':'NOT_RUN'};window=None
    try:
        scene.begin('graspnet_6d')
        if args.observation=='oblique':
            # Fixed workspace viewpoint, not object-ground-truth-dependent.
            eye=np.array([.48,-.24,.42]);target=np.array([.4,0,.06])
            z=target-eye;z/=np.linalg.norm(z)
            x=np.array([1.,0.,0.]);x-=x@z*z;x/=np.linalg.norm(x);y=np.cross(z,x)
            r=np.column_stack([x,y,z])
            source=scene.model.fk(np.zeros(6))
            offset=(np.linalg.inv(source['ral_tcp'])@source['ral_optical'])[:3,3]
            ctrl.move_tcp(eye-r@offset,Rotation.from_matrix(r).as_quat(),phase='rgbd_observe_oblique')
        ctrl.phase='rgbd_settle';ctrl.hold(1.)
        c=scene.cameras['wrist'];rgb,depth=c.get_rgba(),c.get_depth();K=c.get_intrinsics_matrix()
        transform=pose_matrix(*c.get_world_pose(camera_axes='ros'))
        points,colors=pointcloud_from_depth(depth,rgb,K)
        world=points@transform[:3,:3].T+transform[:3,3]
        # Known tabletop workspace bounds, no instance labels/GT object poses.
        workspace=(world[:,0]>.24)&(world[:,0]<.67)&(abs(world[:,1])<.22)&(world[:,2]>-.008)&(world[:,2]<.26)
        inference=workspace&(world[:,2]>.006)
        if np.count_nonzero(inference)<50:raise RuntimeError('INSUFFICIENT_MEASURED_OBJECT_POINTS')
        observation=out/'observation.npz'
        np.savez_compressed(observation,points_optical=points[inference],colors=colors[inference],
            scene_points_world=world[workspace],T_world_optical=transform,K=K,rgb=rgb[...,:3],depth_m=depth)
        write_ply(out/'observed_cloud_world.ply',world[inference],colors[inference]);scene.snapshot('observed_wrist','wrist')
        predicted=out/'network_predictions.npz'
        env=os.environ.copy();env.pop('LD_PRELOAD',None)
        env['PYTHONPATH']=str(ROOT)
        env.pop('ISAAC_PATH',None)
        worker=[str(ROOT/'.conda/isaac-apfg/bin/python'),str(ROOT/'scripts/graspnet_worker.py'),
                '--observation',str(observation),'--output',str(predicted),'--checkpoint',str(args.checkpoint),'--seed',str(args.seed)]
        with (out/'network_worker.log').open('w') as log:
            process=subprocess.Popen(worker,env=env,stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic()
            while process.poll() is None:
                # Keep native GUI responsive, without advancing physics or
                # contaminating the frozen observation with pose overlays.
                scene.world.render()
                if time.monotonic()-started>180:
                    process.terminate();process.wait();raise RuntimeError('GRASPNET_INFERENCE_TIMEOUT')
            if process.returncode:raise RuntimeError(f'GRASPNET_WORKER_FAILED: see {out.relative_to(ROOT)}/network_worker.log')
        with np.load(predicted,allow_pickle=False) as predictions:raw=predictions['raw'].copy()
        candidates,rejections=ranked_candidates(raw,transform,scene.model,args.top_k)
        if not candidates:raise RuntimeError('NO_VALID_PGIA_WIDTH_MODEL_CANDIDATES')
        selected=None;selection=[]
        eligible=range(min(8,len(candidates))) if args.candidate_rank is None else [args.candidate_rank]
        for rank in eligible:
            if not 0<=rank<len(candidates):raise ValueError('Candidate rank out of range')
            candidate=candidates[rank]
            observed=observed_collision(candidate,world[workspace],scene.model)
            record={'rank':rank,'raw_index':candidate['index'],'observed_cloud_filter':observed}
            if observed['collision']:
                record['status']='REJECTED_OBSERVED_COLLISION';selection.append(record);continue
            try:
                xyz=np.array(candidate['xyz_m']);q=candidate['quat_xyzw'];axis=Rotation.from_quat(q).apply([0.,0.,1.])
                pre=xyz-.10*axis
                hover=pre.copy();hover[2]=max(.35,pre[2]+.20)
                seed=scene.model.ik(hover,q,ctrl.joints())
                seed=scene.model.ik(pre,q,seed)
                seed=scene.model.ik(xyz,q,seed)
                scene.model.ik(xyz+[0,0,.15],q,seed)
                record['status']='IK_REACHABLE';selected=rank
            except ValueError as error:record.update(status='REJECTED_IK',reason=str(error))
            selection.append(record)
            if selected is not None:break
        (out/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
        (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        draw_grasps(scene.stage,candidates,selected)
        if not args.headless:
            window=information_window(candidates[selected] if selected is not None else None,len(candidates))
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([.95,-.85,.70]),target=np.array([.4,0,.16]))
        # Record a readable close view of actual 3D pose overlays as well as
        # the fixed and wrist streams, rather than a tiny distant gripper.
        from apfg.simulation.sr5_hardware import look_at
        eye=np.array([.9,-.7,.6]);scene.cameras['third'].set_world_pose(position=eye,orientation=look_at(eye,[.4,0,.14]))
        ctrl.phase='show_model_candidates';ctrl.hold(2.)
        scene.snapshot('predicted_grasps');scene.snapshot('predicted_wrist','wrist');scene.snapshot('predicted_top','fixed')
        result={'status':'PASS','inference':'PASS','pose_visualization':'PASS','physical_grasp':'NOT_RUN',
            'information_mode':'RGBD_ONLY_WORKSPACE_CROP; no target GT or instance segmentation supplied to network/selection',
            'asset_id':args.asset_id or 'known_cube','raw_candidates':len(raw),'displayed_candidates':len(candidates),'rejections':rejections,
            'selected_rank':selected,'selection':selection,'checkpoint_sha256':sha256(args.checkpoint),'observation_sha256':sha256(observation),
            'candidate_source':'actual upstream GraspNet forward pass; unchanged raw pose rotations; explicit source finger-tip/TCP mapping',
            'candidates':candidates}
        if not args.preview_only:
            if selected is None:
                result.update(status='FAILED',physical_grasp='NOT_RUN',reason='No candidate passed one observed-cloud filter and bounded IK selection')
            else:
                chosen=candidates[selected];xyz=np.asarray(chosen['xyz_m']);quat=chosen['quat_xyzw'];axis=Rotation.from_quat(quat).apply([0.,0.,1.])
                initial=scene.cube.get_world_pose()[0].copy()  # evaluation only
                try:
                    ctrl.set_opening(.08,phase='predicted_open')
                    pre=xyz-.10*axis;hover=pre.copy();hover[2]=max(.35,pre[2]+.20)
                    ctrl.move_tcp(hover,quat,phase='predicted_hover')
                    ctrl.move_tcp(pre,quat,phase='predicted_pregrasp')
                    for u in np.linspace(.2,1.,5):ctrl.move_tcp(xyz-(1-u)*.10*axis,quat,phase='predicted_approach')
                    ctrl.set_opening(0.,phase='predicted_close')
                    ctrl.move_tcp(xyz+[0,0,.15],quat,phase='predicted_lift')
                    ctrl.phase='predicted_hold';ctrl.hold(1.)
                    held=[r for r in scene.rows if r['phase']=='predicted_hold']
                    rise=min(r['object_T_world'][2][3] for r in held)-initial[2]
                    contact=sum(all(any('/ral_'+side+'_finger' in str(p) for p in r['contacts']) for side in ['left','right']) for r in held)
                    success=rise>.08 and contact>=len(held)*.7
                    result.update(status='PASS' if success else 'FAILED',physical_grasp='PASS' if success else 'FAILED',
                        physical_evaluation={'minimum_sustained_lift_m':float(rise),'hold_frames':len(held),'bilateral_contact_frames':contact,
                            'initial_object_xyz_m':initial.tolist(),'final_object_xyz_m':scene.cube.get_world_pose()[0].tolist(),'artificial_attachment':False})
                except Exception as error:
                    result.update(status='FAILED',physical_grasp='FAILED',execution_error=str(error))
                    (out/'execution_traceback.txt').write_text(traceback.format_exc())
                scene.snapshot('execution_final')
        scene.end()
        result['videos']=scene.video_reports
        result['camera_capture_gap_steps']=scene.frame_gaps
        result['recording_validation']='PASS' if not scene.frame_gaps and all(v['status']=='PASS' for group in scene.video_reports.values() for v in group.values()) else 'FAILED'
        if result['recording_validation']!='PASS':result.update(status='FAILED',recording_error='Real sensor video missing frames or failed decoding')
        result['seed']=args.seed
        (out/'graspnet_result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:scene.close()
    print(json.dumps({'status':result['status'],'inference':result.get('inference'),'physical_grasp':result.get('physical_grasp'),'selected_rank':result.get('selected_rank'),'output':str(out)}),flush=True)
    if args.interactive and not args.headless:
        print('INTERACTIVE_GRASPNET_VIEW_READY',flush=True)
        while app.is_running():scene.world.render()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gui',action='store_true');p.add_argument('--interactive',action='store_true');p.add_argument('--preview-only',action='store_true')
    p.add_argument('--asset-id',default='ycb:004_sugar_box',help='Registry YCB asset; use empty string for diagnostic cube')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'assets/models/graspnet/checkpoint-rs.tar')
    p.add_argument('--observation',choices=['oblique','top'],default='oblique')
    p.add_argument('--top-k',type=int,default=20);p.add_argument('--candidate-rank',type=int)
    p.add_argument('--joint-speed',type=float,default=.15);p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    add_launcher_args(p);args=p.parse_args()
    if args.gui and args.headless:p.error('--gui and --headless are mutually exclusive')
    if not args.checkpoint.is_file():p.error(f'GraspNet checkpoint missing: {args.checkpoint}')
    if not 1<=args.top_k<=50:p.error('--top-k must be 1..50')
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'outputs/sr5_graspnet'/run;out.mkdir(parents=True)
    (out/'run_config.json').write_text(json.dumps({'args':vars(args),'command':sys.argv},default=str,indent=2)+'\n')
    app=None;result={'status':'NOT_RUN'}
    try:
        app=launch(args);result=demo(args,app,out)
    except Exception as error:
        result.update(status='FAILED',reason=str(error));(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        # Preserve the already completed experiment if Ctrl-C interrupts only
        # the optional interactive viewer after files/videos were finalized.
        if result['status']=='NOT_RUN' and (out/'graspnet_result.json').is_file():
            result=json.loads((out/'graspnet_result.json').read_text())
        result.update(run_id=run,mode='headless' if args.headless else 'gui',artifacts_directory=str(out.relative_to(ROOT)))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['status'],'reason':result.get('reason'),'output':str(out)}),flush=True)
        if app:app.close()
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
