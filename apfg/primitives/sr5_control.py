"""Simulation-only SR5 coordinate control through physical joint drives.

One URDF IK solve per Cartesian request, then speed-limited joint interpolation.
No state resets, attachment constraints, shadow physics or per-step preflight.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.robots.sr5 import ARM, JAWS


class MotionError(RuntimeError):
    pass


class SR5Controller:
    def __init__(self, robot, model, tick, speed=.15):
        if not 0 < speed <= .5:
            raise ValueError('Joint speed must be in (0,.5] rad/s')
        self.robot, self.model, self.tick, self.speed = robot, model, tick, speed
        self.arm_indices = np.array([robot.get_dof_index(n) for n in ARM])
        self.jaw_indices = np.array([robot.get_dof_index(n) for n in JAWS])
        self.reference = robot.get_joint_positions().copy()
        self.commands = []
        self.phase = 'init'

    def joints(self):
        return self.robot.get_joint_positions()[self.arm_indices].copy()

    def tcp(self):
        return self.model.fk(self.joints())['ral_tcp']

    def hold(self, seconds=.5):
        for _ in range(max(1, math.ceil(seconds*30))):
            self.tick(self.reference.copy(), self.phase)

    def move_joints(self, q, phase='move_joints',minimum_duration_s=0.):
        q = np.asarray(q, float)
        if q.shape!=(6,) or not np.isfinite(q).all() or np.any(q<self.model.lower) or np.any(q>self.model.upper):
            raise ValueError('JOINT_LIMIT: six finite arm angles within source limits required')
        self.phase = phase
        start = self.joints()
        n = max(6, math.ceil(1.5*np.max(np.abs(q-start))/self.speed*30),math.ceil(minimum_duration_s*30))
        self.commands.append({'command':'move_joints','phase':phase,'goal_rad':q.tolist(),'duration_s':n/30,'speed_limit_rad_s':self.speed})
        for i in range(1,n+1):
            u = i/n
            self.reference[self.arm_indices] = start+(u*u*(3-2*u))*(q-start)
            self.tick(self.reference.copy(), self.phase)
        self.hold(.4)
        error = float(np.max(np.abs(self.joints()-q)))
        if error > .025:
            raise MotionError(f'JOINT_TRACKING_ERROR: {error:.6f}rad')
        return {'max_joint_error_rad':error}

    def move_tcp(self, xyz, quat_xyzw=None, frame='world', phase='move_tcp',speed_m_s=None):
        xyz = np.asarray(xyz, float)
        if frame not in ('world','base'):
            raise ValueError('frame must be world or base')
        goal = xyz + self.model.base if frame=='base' else xyz.copy()
        quat = Rotation.from_matrix(self.tcp()[:3,:3]).as_quat() if quat_xyzw is None else np.asarray(quat_xyzw,float)
        solution = self.model.ik(goal, quat, self.joints())
        self.commands.append({'command':'move_tcp','phase':phase,'xyz_m':goal.tolist(),'quat_xyzw':quat.tolist(),
                              'input_frame':frame,'method':'single_URDF_IK_then_joint_ramp_no_collision_planner'})
        if speed_m_s is not None and not .003<=speed_m_s<=.15:raise ValueError('Nominal TCP segment speed must be .003.. .15 m/s')
        minimum=0. if speed_m_s is None else 1.5*float(np.linalg.norm(self.tcp()[:3,3]-goal))/speed_m_s
        result = self.move_joints(solution, phase,minimum_duration_s=minimum)
        measured = self.tcp()
        error = float(np.linalg.norm(measured[:3,3]-goal))
        rotation_error = float((Rotation.from_quat(quat).inv()*Rotation.from_matrix(measured[:3,:3])).magnitude())
        result.update(actual_xyz_m=measured[:3,3].tolist(), position_error_m=error, orientation_error_rad=rotation_error)
        if error > .008 or rotation_error > .04:
            raise MotionError(f'TCP_FINAL_ERROR: {result}')
        return result

    def set_opening(self, opening_m, seconds=1., phase='gripper'):
        if not np.isfinite(opening_m) or not 0<=opening_m<=.08:
            raise ValueError('PGIA total opening must be 0..0.080 m')
        self.phase = phase
        initial = self.reference[self.jaw_indices].copy()
        self.commands.append({'command':'set_opening','phase':phase,'total_opening_m':float(opening_m),'per_jaw_m':float(opening_m/2)})
        n = max(6, math.ceil(seconds*30))
        for i in range(1,n+1):
            u=i/n
            self.reference[self.jaw_indices] = initial+u*u*(3-2*u)*(opening_m/2-initial)
            self.tick(self.reference.copy(),self.phase)
        self.hold(.5)
        return self.robot.get_joint_positions()[self.jaw_indices].tolist()

    def grasp(self, xyz, quat_xyzw=(1.,0.,0.,0.), pregrasp_m=.15, lift_m=.15):
        """Known externally supplied TCP centre; no object state enters control."""
        xyz = np.asarray(xyz,float)
        axis = Rotation.from_quat(quat_xyzw).apply([0.,0.,1.])
        self.set_opening(.08, phase='grasp_open')
        self.move_tcp(xyz-axis*pregrasp_m,quat_xyzw,phase='grasp_pre')
        self.move_tcp(xyz,quat_xyzw,phase='grasp_approach')
        self.set_opening(0.,phase='grasp_close')
        self.move_tcp(xyz+[0,0,lift_m],quat_xyzw,phase='grasp_lift')
        self.phase='grasp_hold'
        self.hold(1.)

    def push(self, start_xyz, end_xyz, quat_xyzw=(1.,0.,0.,0.)):
        start, end = np.asarray(start_xyz,float), np.asarray(end_xyz,float)
        self.set_opening(0.,phase='push_close')
        self.move_tcp(start+[0,0,.15],quat_xyzw,phase='push_pre')
        self.move_tcp(start,quat_xyzw,phase='push_approach')
        # Geometric waypoints along the requested line are motion commands,
        # not repeated safety scans or physical previews.
        for u in np.linspace(0,1,9)[1:]:
            self.move_tcp(start+u*(end-start),quat_xyzw,phase='push_contact')
        self.phase='push_hold';self.hold(.5)
