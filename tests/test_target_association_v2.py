"""Synthetic UNIT TEST inputs only; never experiment evidence."""
import unittest
import numpy as np
from apfg.perception.target_association import TargetAssociation
from apfg.perception.m1_observation import visible_mask
from apfg.scenes.clutter_v2 import generate,validate


def camera(offsets=(0,),color=(190,170,25)):
    n=300;rgb=np.zeros((n,n,3),np.uint8)+70;depth=np.ones((n,n))
    for shift in offsets:
        rgb[135:165,140+shift:160+shift]=color;depth[135:165,140+shift:160+shift]=.92
    t=np.diag([1.,-1.,-1.,1.]);t[:3,3]=[.42,0,1]
    return {'rgb':rgb,'depth_m':depth,'K':np.array([[500,0,150],[0,500,150],[0,0,1.]]),'T_world_optical':t}

class AssociationV2(unittest.TestCase):
    def tracker(self):return TargetAssociation({'type':'initial_rgbd_region','world_xy_region':[.40,.44,-.025,.025]})
    def test_follows_translation_outside_frozen_roi(self):
        tracker=self.tracker();c=camera();m,p,r=tracker.update({'fixed':c,'wrist':c},0)
        self.assertEqual(r['status'],'TRACKED');initial=np.median(p,0)
        c=camera((70,));m,p,r=tracker.update({'fixed':c,'wrist':c},1)
        self.assertEqual(r['status'],'TRACKED');self.assertGreater(np.median(p,0)[0]-initial[0],.10)
        old=visible_mask(c['rgb'],c['depth_m'],c['K'],c['T_world_optical'],{'type':'workspace_prompt','world_xy_region':[.40,.44,-.025,.025]})
        self.assertEqual(old.sum(),0);self.assertGreater(m['wrist'].sum(),100)
    def test_equal_color_competitors_are_uncertain(self):
        tracker=self.tracker();c=camera();tracker.update({'fixed':c,'wrist':c},0)
        c=camera((-40,40));m,p,r=tracker.update({'fixed':c,'wrist':c},1)
        self.assertEqual(r['status'],'UNCERTAIN');self.assertEqual(len(p),0);self.assertGreaterEqual(len(r['hypotheses']),2)
    def test_lost_has_no_stale_roi_fallback(self):
        tracker=self.tracker();c=camera();tracker.update({'fixed':c,'wrist':c},0)
        c=camera((),color=(0,0,0));m,p,r=tracker.update({'fixed':c,'wrist':c},1)
        self.assertEqual(r['status'],'LOST');self.assertFalse(m['wrist'].any())
    def test_poison_oracle_not_accepted(self):
        tracker=self.tracker();c=camera()
        with self.assertRaises(TypeError):tracker.update({'fixed':c,'wrist':c},0,oracle_provider=lambda:1/0)
        tracker.update({'fixed':c,'wrist':c},0)
        self.assertEqual(len(tracker.records),1)
        tracker.update({'fixed':c,'wrist':c},0)
        self.assertEqual(len(tracker.records),1)
    def test_texture_patches_on_one_surface_are_not_two_objects(self):
        tracker=self.tracker();c=camera()
        c['rgb'][135:165,140:160]=[190,170,25]
        c['rgb'][135:165,148:152]=[20,45,180]
        m,p,r=tracker.update({'fixed':c,'wrist':c},0)
        self.assertEqual(r['status'],'TRACKED');self.assertGreater(m['wrist'].sum(),500)
    def test_command_motion_prior_is_not_an_instance_mask(self):
        tracker=self.tracker();c=camera();tracker.update({'fixed':c,'wrist':c},0)
        c=camera();c['T_world_optical'][0,3]+=.36
        tracker.apply_motion_prior([.36,0,0]);m,p,r=tracker.update({'fixed':c,'wrist':c},1)
        self.assertEqual(r['status'],'TRACKED');self.assertGreater(np.median(p,0)[0],.75)
        blank=camera(());blank['T_world_optical'][0,3]+=.36
        tracker.apply_motion_prior([0,0,0]);m,p,r=tracker.update({'fixed':blank,'wrist':blank},2)
        self.assertEqual(r['status'],'LOST');self.assertFalse(m['wrist'].any())
    def test_contact_action_permits_unknown_visual_displacement(self):
        tracker=self.tracker();c=camera();tracker.update({'fixed':c,'wrist':c},0)
        tracker.set_action_motion('PUSH');c=camera((90,));m,p,r=tracker.update({'fixed':c,'wrist':c},1)
        self.assertEqual(r['status'],'TRACKED');self.assertGreater(np.median(p,0)[0],.55)
    def test_pinch_reorientation_needs_observed_gap_and_rolls_about_horizontal_axis(self):
        from apfg.primitives.pinch_reorientation import propose
        from scipy.spatial.transform import Rotation
        x,y,z=np.meshgrid(np.linspace(.332,.508,20),np.linspace(-.047,.047,10),np.linspace(.0335,.083,8))
        points=np.stack([x.ravel(),y.ravel(),z.ravel()],1)
        params=propose(points);self.assertEqual(len(params),2)
        for p in params:
            rotation=Rotation.from_rotvec(np.asarray(p['roll_axis'])*np.pi/2)*Rotation.from_quat(p['quat_xyzw'])
            np.testing.assert_allclose(rotation.apply([0,0,1]),p['approach_direction'],atol=1e-10)
            np.testing.assert_allclose(rotation.apply([0,-1,0]),[0,0,1],atol=1e-10)
        points[:,2]-=.0335;self.assertEqual(propose(points),[])
    def test_common_sensor_observation_and_clouds_ignore_poison_oracle(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from apfg.simulation.clutter_env import ClutterEnvironment
        from apfg.robots.sr5 import SR5Model,ROOT
        env=ClutterEnvironment.__new__(ClutterEnvironment);env.tracker=self.tracker();env.history=[];env.observation_index=0
        c=camera();j=np.r_[np.zeros(6),.04,.04];model=SR5Model()
        env.scene=SimpleNamespace(model=model,ctrl=SimpleNamespace(joints=lambda:j[:6],jaw_indices=np.array([6,7])),world=SimpleNamespace(current_time=0))
        env.sensor_data=lambda:({'fixed':c,'wrist':c},{'joint_positions':j,'T_world_tcp':model.fk(j[:6])['ral_tcp']})
        def poisoned():raise AssertionError('Oracle accessed by sensor interface')
        env.oracle_state=poisoned
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            env.out=Path(directory);visible=env.observe();_,scene,target,_,_=env._measured_clouds(visible)
            self.assertEqual(env.association['status'],'TRACKED');self.assertGreater(len(target),20);self.assertGreater(len(scene),len(target))
    def test_observed_overhang_does_not_merge_a_narrow_lower_support(self):
        from apfg.perception.target_association import observed_overhang_band
        x,y,z=np.meshgrid(np.linspace(.332,.508,20),np.linspace(-.047,.047,10),np.linspace(.084,.133,8));upper=np.stack([x.ravel(),y.ravel(),z.ravel()],1)
        x,y,z=np.meshgrid(np.linspace(.37,.47,12),np.linspace(-.03,.03,10),np.linspace(.009,.084,8));lower=np.stack([x.ravel(),y.ravel(),z.ravel()],1)
        result,band=observed_overhang_band(np.concatenate([upper,lower]));self.assertIsNotNone(band);self.assertGreater(result[:,2].min(),.075)
        whole,band=observed_overhang_band(upper);self.assertIsNone(band);self.assertEqual(len(whole),len(upper))
    def test_dense_scene_and_v1_frozen(self):
        from apfg.scenes.m1 import generate as old
        for family in ['Direct','C1','C2','C3','Mixed']:
            s=generate(family);self.assertEqual(len(s['objects']),7);validate(s)
        self.assertEqual(len(old('Direct',2026112401)['objects']),4)

if __name__=='__main__':unittest.main()
