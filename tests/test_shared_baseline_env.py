"""Information-boundary fixtures; never experiment or demonstration data."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from apfg.baselines.candidate_generator import CandidateGenerator,features,observation_hash
from apfg.baselines.common_env import BaselineEnvironment
from apfg.perception.m1_observation import PolicyObservation
from apfg.robots.sr5 import SR5Model,SNAPSHOT


def visible_fixture():
    rgbd={name:{'rgb':np.full((8,8,3),100,np.uint8),'depth_m':np.full((8,8),.5,np.float32),
                 'K':np.eye(3),'T_world_optical':np.eye(4)} for name in ['fixed','wrist']}
    return PolicyObservation(rgbd,{name:np.ones((8,8),bool) for name in rgbd},
        {'joint_positions':np.zeros(8),'T_world_tcp':np.eye(4)}, {'type':'workspace_prompt'})


class PolicyInputBoundary(unittest.TestCase):
    def test_numeric_observation_has_no_identity_or_evaluation_fields(self):
        state=features(visible_fixture(),np.ones((30,3))*.1,[],[],0,0,[])
        self.assertEqual(set(state),{'obs','candidates','mask'})
        self.assertEqual(state['obs'].shape,(32,));self.assertEqual(state['candidates'].shape,(24,24))
        self.assertTrue(all(np.isfinite(value).all() for value in state.values()))

    def test_hash_covers_visible_task_mask(self):
        observation=visible_fixture();before=observation_hash(observation)
        observation.mask['wrist'][0,0]=False
        self.assertNotEqual(before,observation_hash(observation))

    @unittest.skipUnless((SNAPSHOT/'robot/assembled_portable.urdf').exists(),'Source model required')
    def test_candidate_generation_does_not_read_poisoned_object_truth(self):
        class Environment:
            def __init__(self):
                self.scene=SimpleNamespace(model=SR5Model(),ctrl=SimpleNamespace(joints=lambda:np.zeros(6)))
                self.history=[]
            def oracle_state(self):raise AssertionError('Object GT reached candidate generator')
            def observe(self):return visible_fixture()
            def visible(self,x):return x
            def _measured_clouds(self,x):
                target=np.tile([.42,0,.1],(30,1));return target,target,target,np.eye(4),x
            def infer_grasps(self,x):return None,None,[]
        with patch('apfg.baselines.candidate_generator.plan_preparation',return_value=(None,{'reason':'unit_fixture'})):
            state,diagnostic=CandidateGenerator().build(Environment())
        self.assertEqual(set(state),set(BaselineEnvironment.empty_state()))
        self.assertFalse(state['mask'].any())
