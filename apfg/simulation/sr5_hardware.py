"""Shared GUI/headless SR5 assembly and actual contact-physics smoke tests."""
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from apfg.robots.sr5 import SR5Model, ARM, JAWS, ROOT, digest
from apfg.primitives.sr5_control import SR5Controller


def pose_matrix(p, q_wxyz):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.r_[q_wxyz[1:], q_wxyz[0]]).as_matrix()
    t[:3, 3] = p
    return t


def look_at(position, target):
    x = np.asarray(target)-position
    x /= np.linalg.norm(x)
    y = np.cross([0,0,1],x); y /= np.linalg.norm(y)
    q = Rotation.from_matrix(np.column_stack([x,y,np.cross(x,y)])).as_quat()
    return np.r_[q[3],q[:3]]


class HardwareScene:
    def __init__(self, args, app, out):
        import carb
        from pxr import UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Gf
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim,SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.semantics import add_update_semantics
        from isaacsim.sensors.camera import Camera
        from apfg.visualization.recording import Recorder
        self.args, self.app, self.out = args, app, Path(out)
        self.model = SR5Model()
        self.audit = json.loads((ROOT/'outputs/isaac_sr5_hardware/assembly_audit.json').read_text())
        if self.audit['status']!='PASS':
            raise ValueError('Import assembly audit must PASS first')
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=1/480, rendering_dt=1/30)
        # The pinned camera experience disables Fabric transform publication,
        # while the inherited Fabric extension can leave the
        # rendered meshes at their bind pose. Use supported USD publication;
        # this transfers measured PhysX transforms, never commanded fake poses.
        self.world.get_physics_context().enable_fabric(False)
        self.world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True,update_velocities_to_usd=True)
        self.stage = get_current_stage()
        light = UsdLux.DomeLight.Define(self.stage, '/World/Light'); light.CreateIntensityAttr(600.)
        mat = PhysicsMaterial('/World/ContactMaterial', static_friction=.7, dynamic_friction=.6, restitution=0.)
        self.world.scene.add(FixedCuboid('/World/Table', name='table', position=np.array([.4,0,-.055]),
                                        scale=np.array([1.35,1.05,.11]), color=np.array([.3,.27,.23]),physics_material=mat))
        self.world.scene.add(FixedCuboid('/World/Floor', name='floor',position=np.array([0,0,-.80]),scale=np.array([8.,8.,.05]),color=np.array([.22,.24,.27])))
        add_reference_to_stage(str(ROOT/self.audit['usd_path']), '/World/SR5_Assembly')
        self.robot = self.world.scene.add(SingleArticulation('/World/SR5_Assembly',name='sr5',position=self.model.base))
        # 4.5 importer fixes its dummy URDF 'world' body with a joint whose
        # body0 is the global scene. Move that world anchor as well as the
        # Xform at initialization; otherwise PhysX pulls the base back to zero.
        anchor_path=self.audit['articulation_roots'][0].replace('/xMateSR5','/World/SR5_Assembly',1)
        UsdPhysics.FixedJoint(self.stage.GetPrimAtPath(anchor_path)).CreateLocalPos0Attr(Gf.Vec3f(*self.model.base))
        self.link_paths = {n:p.replace('/xMateSR5','/World/SR5_Assembly',1) for n,p in self.audit['link_paths'].items()}
        self.frames = {n:SingleXFormPrim(self.link_paths[n]) for n in ['ral_tcp','ral_camera','ral_optical','ral_gripper','ral_left_finger','ral_right_finger','ral_plate','xMateSR5_link6']}
        # A known centred calibration fixture, identical across reproducible seeds.
        self.scene_spec=getattr(args,'scene_spec',None)
        self.m1_bodies={}
        self.target_asset=getattr(args,'asset_id',None)
        if self.scene_spec:
            from apfg.scenes.m1 import validate,registry
            validate(self.scene_spec);entries=registry()
            for item in self.scene_spec['objects']:
                entry=entries[item['asset_id']]
                path='/World/Object' if item['role']=='target' else '/World/M1_'+item['id']
                add_reference_to_stage(str(ROOT/entry['usd_path']),path)
                q=item['orientation_xyzw']
                rigid=self.world.scene.add(SingleRigidPrim(path,name='m1_'+item['id'],position=np.asarray(item['position_m']),orientation=np.r_[q[3],q[:3]]))
                self.m1_bodies[item['id']]={'body':rigid,'path':path,'spec':item}
                add_update_semantics(self.stage.GetPrimAtPath(path),item['asset_id'])
                if item['role']=='target':
                    self.cube=rigid;self.object_initial=np.asarray(item['position_m']);self.target_asset=item['asset_id']
        elif self.target_asset:
            entries=json.loads((ROOT/'assets/manifests/object_registry.json').read_text())['objects']
            entry=next(e for e in entries if e['asset_id']==self.target_asset)
            self.object_initial=np.array([.4,0.,entry['geometry_dimensions_m'][2]/2+.008])
            add_reference_to_stage(str(ROOT/entry['usd_path']),'/World/Object')
            self.cube=self.world.scene.add(SingleRigidPrim('/World/Object',name='observed_ycb',position=self.object_initial))
        else:
            self.object_initial = np.array([.4,0.,.025])
            self.cube = self.world.scene.add(DynamicCuboid('/World/Object',name='test_cube',position=self.object_initial,
                scale=np.array([.04,.04,.04]),mass=.10,color=np.array([.85,.25,.07]),physics_material=mat))
        add_update_semantics(self.stage.GetPrimAtPath('/World/Object'),self.target_asset or 'hardware_test_cube')
        self.cameras = {}
        for name,pos,target in [('third',[1.7,-1.9,1.4],[.25,0,.42]),('fixed',[.42,0,1.7],[.4,0,0])]:
            self.cameras[name] = Camera('/World/'+name,resolution=(args.width,args.height),frequency=30,
                position=np.array(pos),orientation=look_at(np.array(pos,float),target))
        self.cameras['wrist'] = Camera(self.link_paths['ral_optical']+'/D435iSensor',resolution=(args.width,args.height),frequency=30)
        self.cameras['wrist'].set_local_pose(translation=np.zeros(3),orientation=np.array([1.,0.,0.,0.]),camera_axes='ros')
        # Source ral_optical uses +Z forward, +X right, +Y down. ROS->USD
        # rotation is handled explicitly by the pinned Camera API, no mount edits.
        for name,c in self.cameras.items():
            c.set_clipping_range(.005,20.)
            c.set_focal_length(2.)
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()).startswith('/World/SR5_Assembly'):
                from isaacsim.core.prims import SingleGeometryPrim
                SingleGeometryPrim(str(prim.GetPath())).apply_physics_material(mat)
        self.active_pairs = set(); self.contact_events = []; self.phase='initializing'
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        def contact(headers,data):
            for h in headers:
                a,b=str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                if not any(p=='/World/Object' or p.startswith('/World/M1_') for p in (a,b)):continue
                pair=tuple(sorted([a,b]))
                if h.num_contact_data:self.active_pairs.add(pair)
                else:self.active_pairs.discard(pair)
                self.contact_events.append({'time_s':float(self.world.current_time),'phase':self.phase,'a':a,'b':b,'count':h.num_contact_data,'event':str(h.type)})
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(contact)
        self.world.reset()
        self.robot.set_world_pose(position=self.model.base,orientation=np.array([1.,0.,0.,0.]))  # initialization only
        self.robot.set_solver_position_iteration_count(64)
        self.robot.set_solver_velocity_iteration_count(8)
        for c in self.cameras.values():
            c.initialize(); c.add_distance_to_image_plane_to_frame(); c.add_instance_segmentation_to_frame({'colorize':False})
        c=self.cameras['wrist']
        vertical = 2*2.*math.tan(math.radians(self.model.config['camera_fov_vertical_deg'])/2)
        c.set_vertical_aperture(vertical)
        c.set_horizontal_aperture(vertical*args.width/args.height)
        # Explicit elbow-up home branch. The other planar IK branch puts the
        # elbow below the work surface at low TCP heights; do not disable its
        # table collisions or increase motor force to pass through the table.
        seed_q=np.array([-.25,.5,-1.5,0,-.5,1.3])
        home=self.model.ik([.4,0,.30],[1.,0.,0.,0.],seed_q)
        q=np.zeros(8)
        for n,v in zip(ARM,home):q[self.robot.get_dof_index(n)]=v
        for n in JAWS:q[self.robot.get_dof_index(n)]=.04
        self.robot.set_joint_positions(q)  # initialization only
        self.home = home
        self.ctrl=SR5Controller(self.robot,self.model,self.tick,args.joint_speed)
        self.rows=[];self.video_reports={};self.recorder=None;self.group='initialization';self.start_time=float(self.world.current_time)
        self.start_step=int(self.world.current_time_step_index)
        self.frame_gaps=[];self.wall_start=time.monotonic()
        if self.scene_spec:self.begin('m1')
        self.ctrl.phase='settle';self.ctrl.hold(1.)
        if not args.headless:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([1.7,-1.9,1.4]),target=np.array([.25,0,.42]))

    def tick(self, reference, phase):
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom,Usd
        if not self.app.is_running():raise RuntimeError('GUI_CLOSED_DURING_TEST')
        self.phase=phase
        self.robot.apply_action(ArticulationAction(joint_positions=reference))
        # Explicit substeps + publish + render: the mixed Kit timestep path can
        # leave a recently sleeping object's USD transform a few frames old.
        # Publish the simulator's measured transforms before generating frames.
        for _ in range(16):
            self.world.step(render=False)
        from omni.physx import get_physx_interface
        get_physx_interface().update_transformations(False,True,True)
        self.world.render()
        actual=self.robot.get_joint_positions()
        frames={n:c.get_rgba() for n,c in self.cameras.items()}
        poses={n:pose_matrix(*f.get_world_pose()).tolist() for n,f in self.frames.items()}
        row={'step':int(self.world.current_time_step_index),'time_s':float(self.world.current_time),'group':self.group,'phase':phase,
             'q':actual.tolist(),'qref':reference.tolist(),'qd':self.robot.get_joint_velocities().tolist(),
             'frames_world':poses,'object_T_world':pose_matrix(*self.cube.get_world_pose()).tolist(),
             'object_velocity_m_s':self.cube.get_linear_velocity().tolist(),
             'contacts':[list(p) for p in sorted(self.active_pairs)],
             'render_usd_tcp':np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath(self.link_paths['ral_tcp'])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.tolist(),
             'render_usd_object':np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Object')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.tolist(),
             'T_world_optical':pose_matrix(*self.cameras['wrist'].get_world_pose(camera_axes='ros')).tolist()}
        if self.m1_bodies:
            # Evaluation log only. The M1 RGBD_POLICY boundary never receives
            # this row or a reference to the underlying scene/rigid bodies.
            row['m1_objects']={name:{'T_world':pose_matrix(*data['body'].get_world_pose()).tolist(),
                'linear_velocity_m_s':data['body'].get_linear_velocity().tolist()} for name,data in self.m1_bodies.items()}
        self.rows.append(row)
        if self.recorder:
            if all(f is not None and f.ndim==3 and f.size for f in frames.values()):
                self.recorder.append(frames,row['step'],row['time_s'])
            else:self.frame_gaps.append(row['step'])

    def begin(self, group):
        from apfg.visualization.recording import Recorder
        self.end()
        self.group=group
        self.recorder=Recorder(self.out/'videos'/group,self.cameras,30)

    def end(self):
        if self.recorder:
            self.video_reports[self.group]=self.recorder.close()
            self.recorder=None

    def snapshot(self, name, camera='third'):
        from apfg.visualization.recording import save_rgbd
        d=self.out/'images';d.mkdir(exist_ok=True)
        c=self.cameras[camera]
        save_rgbd(d,name,c.get_rgba(),c.get_depth(),c.get_intrinsics_matrix(),c.get_world_pose(camera_axes='ros'))

    def assembly_views(self):
        from pxr import UsdGeom,Gf
        self.snapshot('front')
        self.snapshot('top','fixed')
        c=self.cameras['third'];saved=c.get_world_pose()
        for name,eye,target in [('side',[-1.0,-1.5,1.0],[.2,0,.45]),
                                 ('tool_closeup',[.68,-.42,.55],[.4,0,.37])]:
            eye=np.array(eye,float)
            c.set_world_pose(position=eye,orientation=look_at(eye,target))
            self.ctrl.hold(.1);self.snapshot(name)
        paths=[]
        for name in ['ral_tcp','ral_optical']:
            t=pose_matrix(*self.frames[name].get_world_pose())
            points=[]
            for i in range(3):points.extend([Gf.Vec3f(*t[:3,3]),Gf.Vec3f(*(t[:3,3]+.06*t[:3,i]))])
            path='/World/DebugAxes_'+name;paths.append(path)
            axes=UsdGeom.BasisCurves.Define(self.stage,path)
            axes.CreateTypeAttr('linear');axes.CreateCurveVertexCountsAttr([2,2,2]);axes.CreatePointsAttr(points)
            axes.CreateWidthsAttr([.002]*6)
            axes.CreateDisplayColorAttr([(1.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,1.,0.),(0.,0.,1.),(0.,0.,1.)])
            axes.GetDisplayColorPrimvar().SetInterpolation('vertex')
        self.ctrl.hold(.1);self.snapshot('tcp_axes')
        for path in paths:self.stage.RemovePrim(path)
        c.set_world_pose(position=saved[0],orientation=saved[1]);self.ctrl.hold(.1)

    def measure(self):
        return {n:pose_matrix(*f.get_world_pose()) for n,f in self.frames.items()}

    def opening(self):
        poses=self.measure()
        axis=poses['ral_gripper'][:3,0]
        # Source finger box width .012m: subtract the two inner half-widths.
        return float((poses['ral_left_finger'][:3,3]-poses['ral_right_finger'][:3,3])@axis-.012)

    def camera_check(self):
        from apfg.visualization.recording import pointcloud_from_depth
        c=self.cameras['wrist'];rgb,depth=c.get_rgba(),c.get_depth()
        seg=c.get_current_frame().get('instance_segmentation')
        if depth is None or not seg:return {'status':'FAILED','reason':'Missing real RGB-D or segmentation'}
        ids=np.asarray(seg['data']).squeeze()
        target=[int(k) for k,v in seg['info'].get('idToLabels',{}).items() if '/World/Object' in str(v)]
        mask=np.isin(ids,target)
        points,colors=pointcloud_from_depth(depth,rgb,c.get_intrinsics_matrix(),mask)
        t=pose_matrix(*c.get_world_pose(camera_axes='ros'))
        world=points@t[:3,:3].T+t[:3,3]
        obj=pose_matrix(*self.cube.get_world_pose())
        local=(world-obj[:3,3])@obj[:3,:3]
        outside=np.maximum(np.abs(local)-.02,0.)
        surface_distance=np.where(np.any(outside>0,axis=1),np.linalg.norm(outside,axis=1),np.min(.02-np.abs(local),axis=1))
        error=float(np.quantile(surface_distance,.95)) if len(points) else None
        self.snapshot('camera','wrist')
        np.savez_compressed(self.out/'camera_validation.npz',points_optical=points,points_world=world,colors=colors,
                             K=c.get_intrinsics_matrix(),T_world_optical=t,T_world_object=obj)
        return {'status':'PASS' if len(points)>50 and error<.005 else 'FAILED','target_pixels':len(points),'surface_distance_p95_m':error,
                'depth_definition':'optical axis +Z metres','optical_convention':'+X right, +Y down, +Z forward',
                'T_world_optical':t.tolist(),'K':c.get_intrinsics_matrix().tolist(),'geometry':'known 40mm cube, actual physics pose; GT only for evaluation'}

    def close(self):
        self.end()
        (self.out/'telemetry.json').write_text(json.dumps(self.rows)+'\n')
        (self.out/'commands.json').write_text(json.dumps(self.ctrl.commands,indent=2)+'\n')
        (self.out/'contacts.json').write_text(json.dumps(self.contact_events)+'\n')
        self.stage.Export(str(self.out/'scene_snapshot.usd'))


def run_tests(args,app,out):
    scene=HardwareScene(args,app,out)
    ctrl=scene.ctrl
    checks={}
    tests = ['joints','gripper','camera','grasp','push'] if args.test=='all' else [args.test]
    if args.test=='move':tests=['move']
    try:
        for test in tests:
            scene.begin(test)
            try:
                if test=='joints':
                    scene.assembly_views()
                    results=[]
                    for i in range(6):
                        target=scene.home.copy();target[i]+=.06
                        results.append(ctrl.move_joints(target,phase=f'joint_{i+1}_positive'))
                        results.append(ctrl.move_joints(scene.home,phase=f'joint_{i+1}_return'))
                    checks['H2']=dict(status='PASS',moves=results)
                elif test=='gripper':
                    openings=[]
                    for target,label in [(.08,'gripper_open'),(0.,'gripper_closed'),(.08,'gripper_reopen')]:
                        ctrl.set_opening(target,phase=label)
                        measured=scene.opening();scene.snapshot(label)
                        openings.append({'command_m':target,'actual_inner_gap_m':measured,'joint_positions':scene.robot.get_joint_positions()[ctrl.jaw_indices].tolist()})
                    checks['H3']={'status':'PASS' if all(abs(x['command_m']-x['actual_inner_gap_m'])<.002 for x in openings) else 'FAILED','openings':openings}
                elif test=='camera':
                    ctrl.move_tcp([.4,0,.30],[1,0,0,0],phase='camera_observe')
                    camera=scene.camera_check()
                    initial=scene.measure()
                    result=ctrl.move_tcp([.45,0,.30],[1,0,0,0],phase='tcp_50mm')
                    final=scene.measure()
                    delta=final['ral_tcp'][:3,3]-initial['ral_tcp'][:3,3]
                    relative0=np.linalg.inv(initial['ral_tcp'])@initial['ral_camera']
                    relative1=np.linalg.inv(final['ral_tcp'])@final['ral_camera']
                    fixed_error=float(np.max(np.abs(relative1-relative0)))
                    checks['H4']={'status':'PASS' if camera['status']=='PASS' and np.linalg.norm(delta-[.05,0,0])<.004 and fixed_error<.001 else 'FAILED',
                                  'tcp_50mm_delta_m':delta.tolist(),'camera_attachment_matrix_error':fixed_error,'camera':camera,'move':result}
                    scene.snapshot('tcp_moved');scene.snapshot('top','fixed')
                    ctrl.move_tcp([.4,0,.30],[1,0,0,0],phase='camera_return')
                elif test=='grasp':
                    initial=scene.cube.get_world_pose()[0].copy()
                    ctrl.grasp([.4,0,.027],lift_m=.15)
                    rows=[r for r in scene.rows if r['phase']=='grasp_hold' and r['group']=='grasp']
                    heights=[r['object_T_world'][2][3] for r in rows]
                    bilateral=[all(any('/ral_'+side+'_finger' in str(p) for p in r['contacts']) for side in ['left','right']) for r in rows]
                    relative=[(np.linalg.inv(np.asarray(r['frames_world']['ral_tcp']))@np.asarray(r['object_T_world']))[:3,3] for r in rows]
                    drift=float(np.linalg.norm(np.ptp(relative,axis=0))) if relative else float('inf')
                    rise=float(min(heights)-initial[2]) if heights else 0.
                    checks['H5']={'status':'PASS' if rise>.10 and sum(bilateral)>=len(rows)*.7 and drift<.008 else 'FAILED',
                        'initial_object_xyz_m':initial.tolist(),'held_object_xyz_m':scene.cube.get_world_pose()[0].tolist(),
                        'minimum_sustained_lift_m':rise,'hold_duration_s':len(rows)/30,'bilateral_contact_frames':sum(bilateral),'hold_frames':len(rows),
                        'object_tcp_relative_position_drift_m':drift,'control_information':'KNOWN_CENTRED_ORACLE_FIXTURE; no learned grasp prediction or object pose fed to controller',
                        'artificial_attachment':False}
                    scene.snapshot('grasp_lifted')
                    ctrl.move_tcp([.4,0,.027],[1,0,0,0],phase='grasp_place')
                    ctrl.set_opening(.08,phase='grasp_release')
                    ctrl.move_tcp([.4,0,.30],[1,0,0,0],phase='grasp_retreat')
                elif test=='push':
                    initial=scene.cube.get_world_pose()[0].copy()
                    ctrl.push([.32,0,.027],[.51,0,.027])
                    final=scene.cube.get_world_pose()[0].copy()
                    contact=any(r['group']=='push' and r['phase']=='push_contact' and any('/World/SR5_Assembly/' in str(p) for p in r['contacts']) for r in scene.rows)
                    checks['H7']={'status':'PASS' if final[0]-initial[0]>.05 and contact else 'FAILED','initial_xyz_m':initial.tolist(),'final_xyz_m':final.tolist(),
                                  'delta_m':(final-initial).tolist(),'physical_robot_object_contact':contact,'object_pose_resets_after_initialization':0}
                    scene.snapshot('push_final')
                elif test=='move':
                    result=ctrl.move_tcp(args.xyz,args.quat_xyzw,frame=args.frame)
                    scene.snapshot('coordinate_move')
                    checks['coordinate_move']={'status':'PASS',**result}
                elif test=='commands':
                    actions=json.loads(args.command_file.read_text())
                    if not isinstance(actions,list):raise ValueError('Command file must be a JSON list')
                    outputs=[]
                    for i,action in enumerate(actions):
                        op=action.get('op');kwargs={k:v for k,v in action.items() if k!='op'}
                        if op not in ('move_tcp','move_joints','set_opening','grasp','push'):raise ValueError(f'Unsupported operation {op}')
                        outputs.append({'index':i,'op':op,'result':getattr(ctrl,op)(**kwargs)})
                    checks['coordinate_commands']={'status':'PASS','results':outputs,'input_sha256':digest(args.command_file)}
                    scene.snapshot('coordinate_commands')
                elif test=='prediction':
                    from apfg.perception.graspnet_pose import prediction_to_tcp
                    source=np.load(args.predictions,allow_pickle=False)
                    if isinstance(source,np.lib.npyio.NpzFile):
                        try:
                            rows=source['filtered']
                            t=source['T_world_optical']
                        finally:source.close()
                    else:
                        raise ValueError('Prediction .npz must include filtered Nx17 and matching T_world_optical; .npy lacks required calibration')
                    actual=pose_matrix(*scene.cameras['wrist'].get_world_pose(camera_axes='ros'))
                    if np.max(np.abs(t-actual))>.002:raise ValueError('STALE_CAMERA_TRANSFORM: external prediction does not match this observation pose')
                    if not 0<=args.prediction_index<len(rows):raise ValueError('Prediction index out of range')
                    tcp=prediction_to_tcp(rows[args.prediction_index],t)
                    (Path(out)/'selected_external_grasp.json').write_text(json.dumps(tcp,indent=2)+'\n')
                    ctrl.grasp(tcp['xyz_m'],tcp['quat_xyzw'])
                    checks['external_prediction_execution']={'status':'PASS','tcp':tcp,'input_sha256':digest(args.predictions),
                        'note':'Completed pose commands; this flag does not certify physical grasp success or model inference provenance'}
                    scene.snapshot('external_prediction_execution')
                print(json.dumps({'test':test,'checks':checks}),flush=True)
            except Exception as exc:
                gate={'joints':'H2','gripper':'H3','camera':'H4','grasp':'H5','push':'H7','move':'coordinate_move','commands':'coordinate_commands','prediction':'external_prediction_execution'}[test]
                checks[gate]={'status':'FAILED','reason':str(exc)}
                (Path(out)/(test+'_traceback.txt')).write_text(traceback.format_exc())
            finally:scene.end()
        # Cross-check source FK against actual USD frames throughout execution.
        frame_errors=[];angles=[]
        for row in scene.rows:
            q=np.asarray(row['q'])[ctrl.arm_indices]
            fk=scene.model.fk(q)
            for n in ['ral_tcp','ral_camera','ral_plate','xMateSR5_link6']:
                t=np.asarray(row['frames_world'][n]);expected=fk[n]
                frame_errors.append(np.linalg.norm(t[:3,3]-expected[:3,3]))
                angles.append((Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_matrix(expected[:3,:3])).magnitude())
        pmax=float(max(frame_errors));amax=float(max(angles))
        poses=scene.measure()
        cosine=abs(float(poses['ral_gripper'][:3,0]@poses['ral_camera'][:3,1]))
        checks['H0']={'status':scene.audit['status'],'source_urdf_sha256':scene.audit['source_urdf_sha256'],'usd_path':scene.audit['usd_path']}
        checks['H1']={'status':'PASS' if pmax<.001 and amax<.005 and cosine>.99999 else 'FAILED',
                      'actual_jaw_camera_abs_cosine':cosine,'max_source_FK_vs_USD_position_error_m':pmax,'max_source_FK_vs_USD_angle_error_rad':amax,
                      'geometry_fidelity':'SR5 vendor source and real plate CAD; PGIA/D435i source APPROX boxes; mount extrinsics not calibrated'}
        checks['H6']={'status':'NOT_RUN','reason':'No GraspNet checkpoint/inference worker in the supplied hardware snapshot. Hardware fixture actions are explicitly known-centred.'}
        render_tcp_error=max(np.linalg.norm(np.asarray(r['render_usd_tcp'])[:3,3]-np.asarray(r['frames_world']['ral_tcp'])[:3,3]) for r in scene.rows)
        render_object_error=max(np.linalg.norm(np.asarray(r['render_usd_object'])[:3,3]-np.asarray(r['object_T_world'])[:3,3]) for r in scene.rows)
        checks['render_state_sync']={'status':'PASS' if render_tcp_error<.001 and render_object_error<.001 else 'FAILED',
                                     'max_tcp_Usd_vs_PhysX_position_error_m':float(render_tcp_error),'max_object_Usd_vs_PhysX_position_error_m':float(render_object_error),
                                     'mode':'PhysX USD publication enabled; Fabric disabled; no commanded visual transforms'}
        steps=int(scene.world.current_time_step_index)-scene.start_step
        elapsed=float(scene.world.current_time)-scene.start_time
        checks['simulation_clock']={'status':'PASS' if abs(steps/480-elapsed)<1e-4 and len(scene.rows)*16==steps else 'FAILED',
                                    'measured_steps':steps,'measured_time_s':elapsed,'recorded_control_ticks':len(scene.rows)}
        checks['rgbd']={'status':'PASS' if all(c.get_depth() is not None and np.any(np.isfinite(c.get_depth())&(c.get_depth()>0)) and np.std(c.get_rgba()[...,:3])>1 for c in scene.cameras.values()) else 'FAILED'}
    finally:
        scene.close()
    cameras_valid=all(v['status']=='PASS' for group in scene.video_reports.values() for v in group.values())
    result={'status':'PASS' if all(c['status']=='PASS' for g,c in checks.items() if g!='H6') and cameras_valid else 'FAILED',
            'checks':checks,'seed':args.seed,'robot':'SR5-5/0.9C + source PGIA-140-80-W/D435i approximation + real plate',
            'physics_hz':480,'control_and_video_hz':30,'joint_speed_limit_rad_s':args.joint_speed,
            'simulated_time_s':float(scene.world.current_time),'wall_time_s':time.monotonic()-scene.wall_start,
            'physics_steps':int(scene.world.current_time_step_index),'camera_capture_gaps':scene.frame_gaps,
            'videos':scene.video_reports,'initial_config':{'q_home_rad':scene.home.tolist(),'object_xyz_m':scene.object_initial.tolist(),'cube_size_m':.04,'cube_mass_kg':.10,'base_xyz_m':scene.model.base.tolist()},
            'coordinate_interface':'metres/radians; world or base position; quaternion XYZW; TCP=source ral_tcp',
            'test_scope':'hardware contact primitive validation; no new push/grasp algorithm, no training'}
    (Path(out)/'smoke_results.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.interactive and not args.headless:
        print('INTERACTIVE_VIEW_READY: close the Isaac window to exit',flush=True)
        while app.is_running():scene.world.step(render=True)
    return result
