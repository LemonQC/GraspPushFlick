"""Shared GUI/headless M0 diagnostic runner. No task policy or training."""
import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['physics', 'robot', 'camera', 'objects', 'clutter'], default='clutter')
    parser.add_argument('--scene', choices=['s1', 's2'], default='s1')
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--asset-id')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--attempt-launch', action='store_true', help='Try Kit despite failed CUDA probe; for diagnosis only')
    # Import the pinned launcher only after identifying output location.
    sys.path.insert(0, str(ROOT/'third_party/IsaacLab/source/isaaclab'))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out = ROOT/'outputs/environment_check/runs'/run_id
    out.mkdir(parents=True)
    mode = 'headless' if args.headless else 'gui'
    result = {'run_id': run_id, 'test': args.test, 'mode': mode, 'scene': args.scene, 'seed': args.seed,
              'asset_id': args.asset_id,
              'command': sys.argv, 'status': 'NOT_RUN', 'artifacts_directory': str(out.relative_to(ROOT))}
    from apfg.download import sha256
    result['asset_registry_sha256'] = sha256(ROOT/'assets/manifests/object_registry.json')
    result['version_config_sha256'] = sha256(ROOT/'configs/versions.json')
    latest = ROOT/'outputs/environment_check'/f'{mode}_{args.test}_smoke.json'
    app = None
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        cuda_status = cuda.cuInit(0)
        result['cuda_init_returncode'] = cuda_status
        if cuda_status and not args.attempt_launch:
            result.update(status='BLOCKED', reason=f'cuInit(0)={cuda_status}; GPU rendering unavailable. Check /dev/nvidia-uvm before retrying.')
            return 2
        if not args.enable_cameras:
            raise ValueError('M0 requires --enable_cameras in GUI and headless modes')
        if args.steps < 240 or args.fps <= 0 or 120 % args.fps:
            raise ValueError('Use >=240 physics steps and fps dividing 120')
        args.kit_args = f'--portable-root {ROOT}/cache/omniverse/portable --/app/renderer/skipWhileMinimized=false'
        app = AppLauncher(args).app
        result.update(run_simulation(args, app, out))
        return 0 if result['status']=='PASS' else 1
    except Exception as e:
        result.update(status='FAILED', reason=str(e))
        (out/'traceback.txt').write_text(traceback.format_exc())
        return 1
    finally:
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        latest.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if app is not None:
            app.close()


def run_simulation(args, app, out):
    import carb
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from pxr import UsdGeom, UsdLux
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from apfg.visualization.recording import Recorder, save_rgbd, pointcloud_from_depth, write_ply

    carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
    carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/args.fps)
    world.scene.add_default_ground_plane()
    stage = get_current_stage()
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1000)
    world.scene.add(FixedCuboid('/World/Table', position=np.array([.5, 0., .7]), scale=np.array([1.4, 1.1, .08]), color=np.array([.32, .26, .20])))
    robot = None
    if args.test in ['robot', 'camera', 'clutter']:
        add_reference_to_stage(str(ROOT/'assets/isaac/Isaac/Robots/Franka/franka.usd'), '/World/Panda')
        robot = world.scene.add(SingleArticulation('/World/Panda', name='panda', position=np.array([-.05, 0., .74])))
    objects = []
    if args.test == 'physics':
        objects.append(world.scene.add(DynamicCuboid('/World/Drop', name='drop', position=np.array([.5, 0., 1.6]), scale=np.array([.08]*3), mass=.2)))
    entries = json.loads((ROOT/'assets/manifests/object_registry.json').read_text())['objects']
    selected = []
    if args.test in ['camera', 'clutter', 'objects']:
        if args.test == 'objects':
            if not args.asset_id:
                raise ValueError('Independent object smoke requires --asset-id ycb:<object>')
            selected = [e for e in entries if e['asset_id']==args.asset_id]
            if not selected:
                raise ValueError('Asset ID absent from registry')
        else:
            wanted = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '011_banana', '025_mug']
            selected = [next(e for e in entries if e['asset_id']=='ycb:'+name) for name in wanted]
        rng = np.random.default_rng(args.seed)
        for i, entry in enumerate(selected):
            path = f'/World/Object_{i}'
            add_reference_to_stage(str(ROOT/entry['usd_path']), path)
            add_update_semantics(stage.GetPrimAtPath(path), entry['asset_id'])
            # Initial positions are non-overlapping. All settling thereafter uses PhysX.
            xy = [.35+.23*(i%3), -.22+.32*(i//3)]
            z = 1.05
            if args.scene == 's2' and args.test == 'clutter':
                xy = [.56+float(rng.uniform(-.08, .08)), float(rng.uniform(-.08, .08))]
                z = 1.05+.38*i
            q = Rotation.from_euler('z', rng.uniform(-np.pi, np.pi)).as_quat()
            objects.append(world.scene.add(SingleRigidPrim(path, name=f'object_{i}', position=np.array([*xy, z]), orientation=np.array([q[3], *q[:3]]))))

    def look_at(position, target):
        x = np.asarray(target)-np.asarray(position)
        x = x/np.linalg.norm(x)
        y = np.cross([0., 0., 1.], x)
        y = y/np.linalg.norm(y)
        z = np.cross(x, y)
        q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
        return np.array([q[3], *q[:3]])

    cameras = {}
    for name, pos, target in [('third', [2., -1.7, 1.9], [.5, 0., .9]), ('fixed', [.51, -.01, 2.5], [.5, 0., .75])]:
        cameras[name] = Camera('/World/'+name, resolution=(args.width, args.height), frequency=args.fps,
                               position=np.array(pos), orientation=look_at(pos, target))
    if robot:
        cameras['wrist'] = Camera('/World/Panda/panda_hand/WristCamera', resolution=(args.width, args.height), frequency=args.fps)
        cameras['wrist'].set_local_pose(translation=np.array([.06, 0., .07]), orientation=np.array([1., 0., 0., 0.]))
    contact_view = None
    if objects:
        paths = [o.prim_path for o in objects]
        contact_view = world.scene.add(RigidPrim(prim_paths_expr=paths, name='contact_view', track_contact_forces=True))
    world.reset()
    for camera in cameras.values():
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        camera.add_instance_segmentation_to_frame({'colorize': False})
    if robot:
        home = np.array([0., -.6, 0., -2., 0., 1.6, .8, .04, .04])
        robot.set_joint_positions(home)  # initialization only; no resets in rollout
        ee = SingleXFormPrim('/World/Panda/panda_hand')
    recorder = Recorder(out/'videos', cameras, args.fps)
    telemetry = []
    initial = [o.get_world_pose()[0].copy() for o in objects]
    completed_steps = 0
    try:
        for step in range(args.steps):
            if not app.is_running():
                break
            target = None
            if robot:
                t = step/120
                target = home.copy()
                target[:7] += .10*np.sin(1.1*t+np.arange(7)*.5)
                target[7:] = .04 if int(t/2)%2==0 else .005
                robot.apply_action(ArticulationAction(joint_positions=target))
            capture = step % (120//args.fps) == 0
            world.step(render=capture)
            completed_steps = step+1
            if capture:
                row = {'step': step, 'time_s': (step+1)/120,
                       'objects': [{'position': o.get_world_pose()[0].tolist(), 'orientation': o.get_world_pose()[1].tolist(),
                                    'linear_velocity': o.get_linear_velocity().tolist(), 'angular_velocity': o.get_angular_velocity().tolist()} for o in objects]}
                if contact_view is not None:
                    row['net_contact_forces_n'] = contact_view.get_net_contact_forces(dt=1/120).tolist()
                if robot:
                    row.update(command=target.tolist(), joints=robot.get_joint_positions().tolist(), ee_position=ee.get_world_pose()[0].tolist(),
                               ee_orientation=ee.get_world_pose()[1].tolist(), wrist_position=cameras['wrist'].get_world_pose()[0].tolist(),
                               wrist_orientation=cameras['wrist'].get_world_pose()[1].tolist())
                telemetry.append(row)
                # Rendering warm-up is excluded and its timestamps remain explicit.
                if step >= 30:
                    recorder.append({n: c.get_rgba() for n, c in cameras.items()}, step, (step+1)/120)
        videos = recorder.close()
    except BaseException:
        for writer in recorder.writers.values():
            writer.close()
        raise
    (out/'telemetry.json').write_text(json.dumps(telemetry, indent=2)+'\n')
    calibration = {}
    point_count = 0
    segmentation_valid = True
    final_images = []
    valid_cameras = True
    for name, camera in cameras.items():
        rgb, depth = camera.get_rgba(), camera.get_depth()
        valid = rgb is not None and rgb.size > 0 and np.std(rgb[..., :3]) > 1 and depth is not None and np.any(np.isfinite(depth) & (depth>0))
        valid_cameras &= bool(valid)
        if valid:
            save_rgbd(out, name+'_camera', rgb, depth, camera.get_intrinsics_matrix(), camera.get_world_pose(camera_axes='ros'))
            final_images.append(Image.fromarray(rgb[..., :3].astype(np.uint8)))
            frame = camera.get_current_frame()
            segmentation = frame.get('instance_segmentation')
            if segmentation is not None:
                ids = np.asarray(segmentation['data']).squeeze()
                np.save(out/(name+'_instance_segmentation.npy'), ids)
                label_rgb = np.stack([(ids*37)%256, (ids*73)%256, (ids*19)%256],axis=-1).astype(np.uint8)
                Image.fromarray(label_rgb).save(out/(name+'_instance_segmentation.png'))
                (out/(name+'_instance_labels.json')).write_text(json.dumps(segmentation['info'], default=str, indent=2))
                labels = segmentation['info'].get('idToLabels', {})
                target_ids = [int(k) for k, v in labels.items() if '/World/Object_0' in str(v)]
                if name == 'fixed' and target_ids:
                    points, colors = pointcloud_from_depth(depth, rgb, camera.get_intrinsics_matrix(), np.isin(ids, target_ids))
                    write_ply(out/'target_camera_frame.ply', points, colors)
                    point_count = len(points)
            else:
                segmentation_valid = False
            calibration[name] = {'valid_rgbd': bool(valid)}
    final = [o.get_world_pose()[0].copy() for o in objects]
    stable = all(np.linalg.norm(o.get_linear_velocity()) < .02 and np.linalg.norm(o.get_angular_velocity()) < .1 and .70 < p[2] < 2 for o, p in zip(objects, final))
    fell = all(a[2]-b[2]>.1 for a, b in zip(initial, final))
    tail = [t for t in telemetry if t['time_s'] >= completed_steps/120-1]
    sustained_stability = bool(tail) and all(np.linalg.norm(o['linear_velocity']) < .02 and np.linalg.norm(o['angular_velocity']) < .1
        for t in tail for o in t['objects'])
    contact_confirmed = bool(objects) and all(any(np.linalg.norm(t.get('net_contact_forces_n', [])[i]) > .05 for t in tail) for i in range(len(objects)))
    video_valid = all(v['status']=='PASS' for v in videos.values())
    physical_valid = (not objects) or (stable and fell and sustained_stability and contact_confirmed)
    result = {'status': 'FAILED',
              'physics_steps': completed_steps, 'camera_rgbd_valid': bool(valid_cameras), 'videos': videos,
              'objects': [e['asset_id'] for e in selected], 'settled_snapshot': bool(stable), 'fell': bool(fell),
              'initial_positions': [p.tolist() for p in initial], 'final_positions': [p.tolist() for p in final],
              'contact_validation': 'PASS' if contact_confirmed else ('FAILED' if objects else 'NOT_RUN'),
              'sustained_stability_1s': bool(sustained_stability), 'calibration': calibration,
              'target_point_count': point_count, 'instance_segmentation_valid': bool(segmentation_valid)}
    checks = {'steps': completed_steps == args.steps, 'rgbd': bool(valid_cameras), 'video': video_valid, 'physics': bool(physical_valid)}
    if robot and telemetry:
        joints = np.array([t['joints'] for t in telemetry])
        result.update(joint_range_rad=np.ptp(joints[:,:7], axis=0).tolist(), gripper_range_m=np.ptp(joints[:,7:],axis=0).tolist(),
                      final_joint_error_rad=(np.array(telemetry[-1]['command'])-joints[-1]).tolist())
        ee_positions = np.array([t['ee_position'] for t in telemetry])
        wrist_positions = np.array([t['wrist_position'] for t in telemetry])
        relative_positions = []
        relative_rotations = []
        for row in telemetry:
            eq, wq = row['ee_orientation'], row['wrist_orientation']
            er = Rotation.from_quat([*eq[1:], eq[0]])
            wr = Rotation.from_quat([*wq[1:], wq[0]])
            relative_positions.append(er.inv().apply(np.array(row['wrist_position'])-np.array(row['ee_position'])))
            relative_rotations.append(er.inv()*wr)
        orientation_drift = max((relative_rotations[0].inv()*r).magnitude() for r in relative_rotations)
        attachment_error = float(np.max(np.ptp(np.array(relative_positions), axis=0)))
        result.update(wrist_attachment_position_drift_m=attachment_error, wrist_attachment_orientation_drift_rad=float(orientation_drift))
        checks['robot_motion'] = bool(np.all(np.ptp(joints[:,:7],axis=0)>.01) and np.all(np.ptp(joints[:,7:],axis=0)>.02)
                                     and np.count_nonzero(np.ptp(ee_positions,axis=0)>.01)>=2
                                     and np.max(np.abs(np.array(result['final_joint_error_rad'])[:7])) < .15)
        checks['wrist_sync'] = bool(attachment_error < .001 and orientation_drift < .005 and np.linalg.norm(np.ptp(wrist_positions,axis=0))>.01)
    if args.test in ['camera', 'clutter']:
        checks['target_pointcloud'] = point_count > 20 and segmentation_valid
    if final_images:
        mosaic = Image.new('RGB', (args.width*len(final_images), args.height))
        for i, im in enumerate(final_images):
            mosaic.paste(im, (i*args.width, 0))
        mosaic.save(out/'multiview.png')
    result.update(checks=checks, status='PASS' if all(checks.values()) else 'FAILED')
    if args.interactive and not args.headless:
        while app.is_running():
            world.step(render=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
