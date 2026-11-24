"""V2 initialization adapter; frozen HardwareScene v1 remains byte-identical.

The source initializer is copied to change only scene validation. Physical
stepping, cameras, recording and close inherit the validated implementation.
"""
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


from apfg.simulation.sr5_hardware import HardwareScene


class ClutterHardwareScene(HardwareScene):
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
            from apfg.scenes.clutter_v2 import validate
            from apfg.scenes.m1 import registry
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
        for name,pos,target in [('third',[1.05,-1.0,.85],[.4,0,.13]),('fixed',[.42,0,1.7],[.4,0,0])]:
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
            set_camera_view(eye=np.array([1.05,-1.0,.85]),target=np.array([.4,0,.13]))
