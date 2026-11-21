import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from apfg.robots.sr5 import SR5Model,SNAPSHOT
from apfg.perception.graspnet_pose import prediction_to_tcp


class SR5KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SNAPSHOT/'robot/assembled_portable.urdf').is_file():
            raise unittest.SkipTest('Private SR5 snapshot is required for source-model kinematics checks')
        cls.model=SR5Model()

    def test_zero_flange_from_source_offsets(self):
        t=self.model.fk(np.zeros(6))
        np.testing.assert_allclose(t['xMateSR5_link6'][:3,3],[-.16,.136,1.2315],atol=1e-9)
        np.testing.assert_allclose(t['ral_tcp'][:3,3],[-.16,.136,1.347],atol=1e-9)

    def test_two_jaw_directions_and_total_gap(self):
        for opening in (0.,.04,.08):
            t=self.model.fk(np.zeros(6),opening=opening)
            axis=t['ral_gripper'][:3,0]
            gap=(t['ral_left_finger'][:3,3]-t['ral_right_finger'][:3,3])@axis-.012
            self.assertAlmostEqual(gap,opening,places=10)
            self.assertAlmostEqual(abs(axis@t['ral_camera'][:3,1]),1.,places=10)

    def test_ik_reaches_external_coordinates_and_base_frame(self):
        q=self.model.ik([.4,0,.3],[1,0,0,0],[0,1.9,1.3,0,3.7,1.3])
        t=self.model.fk(q)['ral_tcp']
        np.testing.assert_allclose(t[:3,3],[.4,0,.3],atol=.001)
        self.assertLess((Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_quat([1,0,0,0])).magnitude(),.01)

    def test_unreachable_and_nonfinite_are_rejected(self):
        for xyz in ([10,10,10],[float('nan'),0,.3]):
            with self.assertRaises(ValueError):self.model.ik(xyz,[1,0,0,0],[0,1.9,1.3,0,3.7,1.3])


class GraspNetMappingTests(unittest.TestCase):
    def row(self):
        return np.r_[.9,.04,.02,.03,np.eye(3).ravel(),[.1,.2,.4],-1]

    def test_transformed_approach_closing_and_translation(self):
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',[.3,-.4,.7]).as_matrix();t[:3,3]=[.2,-.1,.8]
        pose=prediction_to_tcp(self.row(),t)
        r=Rotation.from_quat(pose['quat_xyzw']).as_matrix()
        np.testing.assert_allclose(r[:,2],t[:3,0],atol=1e-9)
        np.testing.assert_allclose(r[:,0],t[:3,1],atol=1e-9)
        np.testing.assert_allclose(pose['xyz_m'],t[:3,:3]@[.1,.2,.4]+t[:3,3])

    def test_width_and_invalid_rotation_not_silently_corrected(self):
        for index,value in [(1,.081),(4,-1.)]:
            row=self.row();row[index]=value
            with self.assertRaises(ValueError):prediction_to_tcp(row,np.eye(4))


if __name__=='__main__':unittest.main()
