import copy
from itertools import product
import unittest
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation

from apfg.robots.sr5 import SR5Model,SNAPSHOT,nearest_revolute_equivalent
from apfg.primitives.grasp_motion import execution_variants,execute_grasp_plan
from apfg.perception.graspnet_candidates import observed_collision


class RotationContinuity(unittest.TestCase):
    def test_periodic_joint_representation_uses_shorter_valid_travel(self):
        actual=nearest_revolute_equivalent(np.radians([-165,170]),np.radians([45,-170]),
                                          np.radians([-360,-180]),np.radians([360,180]))
        np.testing.assert_allclose(np.degrees(actual),[195,170])
        with self.assertRaises(ValueError):nearest_revolute_equivalent([3],[0],[-1],[1])

    def test_explicit_finger_exchange_preserves_raw_pose_and_approach(self):
        candidate={'index':7,'xyz_m':[.4,0,.1],'quat_xyzw':Rotation.from_euler('z',179,degrees=True).as_quat().tolist()}
        original=copy.deepcopy(candidate);variants=execution_variants(candidate,np.eye(4))
        self.assertEqual(candidate,original);self.assertEqual(variants[0]['jaw_roll_about_approach_deg'],180)
        self.assertAlmostEqual(variants[0]['rotation_from_current_deg'],1.,places=8)
        raw=Rotation.from_quat(candidate['quat_xyzw']).as_matrix();executed=Rotation.from_quat(variants[0]['quat_xyzw']).as_matrix()
        np.testing.assert_allclose(executed[:,2],raw[:,2],atol=1e-12)
        np.testing.assert_allclose(executed[:,0],-raw[:,0],atol=1e-12)

    def test_execution_uses_checked_joints_and_rejects_stale_start(self):
        controller=Mock();controller.joints.return_value=np.zeros(6);controller.commands=[]
        pose={'quat_xyzw':[0,0,0,1]}
        path=[np.zeros(6).tolist(),np.full(6,.1).tolist(),np.full(6,.2).tolist()]
        plan={'joint_path':path,'execution_pose':pose,'raw_prediction_index':4,'controller_revision':'test',
              'raw_tcp_pose':pose,'goals':[[.4,0,.1],[.4,0,.25]],'phases':['approach','lift']}
        execute_grasp_plan(controller,plan)
        self.assertEqual(controller.move_tcp.call_args_list[0].kwargs['joint_goal'],path[1])
        self.assertEqual(controller.move_tcp.call_args_list[1].kwargs['joint_goal'],path[2])
        controller.joints.return_value=np.ones(6)
        with self.assertRaisesRegex(ValueError,'STALE'):execute_grasp_plan(controller,plan)


@unittest.skipUnless((SNAPSHOT/'robot/assembled_portable.urdf').is_file(),'Private source assembly required')
class SourceAssemblySymmetry(unittest.TestCase):
    def test_joint_wrap_does_not_change_source_fk(self):
        model=SR5Model();q=np.array([0,.3,-1.2,.4,-.8,-2.9]);seed=q.copy();seed[5]=1.
        equivalent=nearest_revolute_equivalent(q,seed,model.lower,model.upper)
        for name,transform in model.fk(q).items():np.testing.assert_allclose(transform,model.fk(equivalent)[name],atol=1e-12)

    def test_camera_collision_is_not_assumed_jaw_symmetric(self):
        model=SR5Model();xyz=np.array([.4,0,.35])
        # Measured-cloud fixture inside the source D435i housing on only one
        # side. A finger exchange moves the housing to the opposite side.
        points=xyz+np.array(list(product([-.01,.01],[-.081,-.069],[-.10,-.09])))
        original={'xyz_m':xyz.tolist(),'quat_xyzw':[0,0,0,1]}
        exchanged=dict(original,quat_xyzw=Rotation.from_euler('z',180,degrees=True).as_quat().tolist())
        self.assertTrue(observed_collision(original,points,model)['collision'])
        self.assertFalse(observed_collision(exchanged,points,model)['collision'])


if __name__=='__main__':unittest.main()
