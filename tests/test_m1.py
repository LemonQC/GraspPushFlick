import unittest
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation

from apfg.perception.m1_observation import build_observation,InformationMode,PolicyObservation,DiagnosticObservation
from apfg.primitives.m1_actions import Action
from apfg.evaluation.m1_metrics import flip_outcome
from apfg.scenes.m1 import generate,validate,registry,suitability,ROOT


class InformationBoundary(unittest.TestCase):
    def sensor(self):
        return {'wrist':{'rgb':np.array([[[200,180,20],[50,50,100]]],dtype=np.uint8),
                         'depth_m':np.ones((1,2))*.3,'K':np.eye(3),'T_world_optical':np.eye(4)}},{'joint_positions':np.zeros(8)}
    hint={'type':'visible_color_prompt','color_rule':{'red_min':90,'green_min':80,'blue_to_min_rg_max':.72}}

    def test_rgbd_never_calls_or_returns_oracle(self):
        def forbidden():raise AssertionError('GT leak')
        obs=build_observation(self.sensor,InformationMode.RGBD_POLICY,self.hint,[],forbidden)
        self.assertIsInstance(obs,PolicyObservation)
        self.assertFalse(hasattr(obs,'oracle_state'))
        self.assertEqual(obs.mask['wrist'].tolist(),[[True,False]])
        self.assertNotIn('family',asdict(obs));self.assertNotIn('objects',asdict(obs))

    def test_oracle_requires_explicit_mode_and_provider(self):
        with self.assertRaises(ValueError):build_observation(self.sensor,'ORACLE_DIAGNOSTIC',self.hint,[])
        obs=build_observation(self.sensor,'ORACLE_DIAGNOSTIC',self.hint,[],lambda:{'target_pose':'evaluation-only'})
        self.assertIsInstance(obs,DiagnosticObservation);self.assertEqual(obs.oracle_state['target_pose'],'evaluation-only')

    def test_action_rejects_gt_and_unknown_parameters(self):
        with self.assertRaises(ValueError):Action('PUSH',{'gt_target_pose':[0,0,0]})

    def test_explicit_yellow_prompt_excludes_red_occluder(self):
        hint={'type':'visible_color_prompt','color_rule':dict(self.hint['color_rule'],r_g_min=.75,r_g_max=1.35)}
        def sensors():
            rgbd,proprio=self.sensor();rgbd['wrist']['rgb']=np.array([[[240,210,20],[240,90,20]]],dtype=np.uint8);return rgbd,proprio
        obs=build_observation(sensors,'RGBD_POLICY',hint,[])
        self.assertEqual(obs.mask['wrist'].tolist(),[[True,False]])


class PhysicalFlipMetric(unittest.TestCase):
    def rotated(self,axis,angle):
        t=np.eye(4);t[:3,:3]=Rotation.from_euler(axis,angle,degrees=True).as_matrix();return t
    def test_yaw_does_not_count_as_flip(self):
        self.assertEqual(flip_outcome(np.eye(4),self.rotated('z',120),True,True)['status'],'FAILED')
    def test_nonvertical_rotation_requires_real_contact_and_stability(self):
        end=self.rotated('y',90)
        self.assertEqual(flip_outcome(np.eye(4),end,True,True)['status'],'PASS')
        self.assertEqual(flip_outcome(np.eye(4),end,False,True)['status'],'FAILED')
        self.assertEqual(flip_outcome(np.eye(4),end,True,False)['status'],'FAILED')


@unittest.skipUnless((ROOT/'assets/manifests/object_registry.json').exists(),'Local asset registry required')
class SceneGeneration(unittest.TestCase):
    def test_seeded_families_do_not_encode_skill_labels(self):
        for family in ['Direct','C1','C2','C3','C4']:
            a=generate(family,2026112201);self.assertEqual(a,generate(family,2026112201));validate(a)
            self.assertEqual(len(a['objects']),4)
            self.assertNotIn('correct_action',a);self.assertEqual(a['mechanism_qualification'].split(':')[0],'UNVERIFIED')
    def test_seed_changes_pose_and_large_target_is_not_suitable(self):
        self.assertNotEqual(generate('Direct',1)['objects'][0]['position_m'],generate('Direct',2)['objects'][0]['position_m'])
        self.assertEqual(suitability(registry()['ycb:036_wood_block'])['status'],'UNSUITABLE')

    def test_mixed_schema_does_not_accept_action_labels(self):
        with self.assertRaises(ValueError):generate('C4',1,mix={'correct_action':'FLIP'})
        with self.assertRaises(ValueError):generate('C4',1,mix={'clearance':False,'orientation':True,'visibility':False})

    def test_modified_scene_cannot_reuse_frozen_content_hash(self):
        spec=generate('Direct',1);spec['objects'][0]['position_m'][0]+=.001
        with self.assertRaisesRegex(ValueError,'hash mismatch'):validate(spec)


if __name__=='__main__':unittest.main()
