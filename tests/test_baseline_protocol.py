import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from apfg.baselines.schema import MAX_CANDIDATES,CAND_DIM
from apfg.baselines.heuristic import HeuristicPolicy,RandomValidPolicy,GraspOnlyPolicy
from apfg.baselines.reward import compute_reward
from apfg.baselines.evaluation import audit_pair,summarize_e1,ARMS

ROOT=Path(__file__).resolve().parents[1]


def state():
    c=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);m=np.zeros(MAX_CANDIDATES,bool)
    c[0,0]=1;c[0,4]=.7;c[1,1]=1;c[1,4]=.2;m[:2]=True
    return {'candidates':c,'mask':m}


class SharedNonlearningPolicies(unittest.TestCase):
    def test_mask_and_empty_pool(self):
        s=state();s['candidates'][2,4]=1000
        self.assertEqual(HeuristicPolicy().act(s),0)
        s['mask'][:]=False
        for p in [HeuristicPolicy(),RandomValidPolicy(3),GraspOnlyPolicy()]:self.assertEqual(p.act(s),-1)

    def test_random_uniform_support_and_grasp_restriction(self):
        s=state();p=RandomValidPolicy(77);choices=[p.act(s) for _ in range(600)]
        self.assertEqual(set(choices),{0,1});self.assertGreater(sum(x==0 for x in choices),200);self.assertLess(sum(x==0 for x in choices),400)
        s['mask'][0]=False
        self.assertEqual(GraspOnlyPolicy().act(s),-1);self.assertEqual(HeuristicPolicy().act(s),1)

    def test_policy_never_reads_oracle_or_family(self):
        class Poison(dict):
            def __getitem__(self,key):
                if key not in ['candidates','mask']:raise AssertionError('Policy leaked forbidden key '+key)
                return super().__getitem__(key)
        for p in [HeuristicPolicy(),RandomValidPolicy(1),GraspOnlyPolicy()]:self.assertIn(p.act(Poison(state())),[0,1])


class SharedSparseReward(unittest.TestCase):
    def test_preparation_cannot_earn_retrieval_bonus(self):
        with self.assertRaises(ValueError):compute_reward(success=True,terminal=False)
        r,t=compute_reward(preparation=True,simulation_time_s=10)
        self.assertLess(r,0);self.assertEqual(t['terminal_retrieval_bonus'],0)
        r,t=compute_reward(success=True,terminal=True,simulation_time_s=10)
        self.assertAlmostEqual(r,.99)


class PhysicalPairing(unittest.TestCase):
    def body(self):return {'target':{'asset_id':'test','T_world':np.eye(4).tolist()},'obstacle':{'asset_id':'other','T_world':np.eye(4).tolist()}}
    def test_distractor_drift_invalidates_pair(self):
        states=[self.body() for _ in range(4)];self.assertEqual(audit_pair(states)['pair_status'],'VALID_PAIR')
        states[-1]['obstacle']['T_world'][0][3]=.003
        self.assertEqual(audit_pair(states)['pair_status'],'INVALID_PAIR')

    def test_rotating_initial_target_and_missing_arm_rejected(self):
        states=[self.body() for _ in range(4)];t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',.02).as_matrix();states[-1]['target']['T_world']=t.tolist()
        self.assertEqual(audit_pair(states)['status'],'FAILED');self.assertEqual(audit_pair(states[:3])['status'],'FAILED')

    def test_opposite_deviations_and_missing_state_are_not_valid_pairs(self):
        states=[self.body() for _ in range(4)]
        states[1]['target']['T_world'][0][3]=.0015;states[2]['target']['T_world'][0][3]=-.0015
        self.assertEqual(audit_pair(states)['pair_status'],'INVALID_PAIR')
        states[2]={'status':'FAILED','failure_reason':'Environment initialization failed'}
        self.assertEqual(audit_pair(states)['pair_status'],'INVALID_PAIR')

    def test_direct_success_is_not_preparation_necessity(self):
        records=[]
        for case in ['a','b','c','d']:
            for arm in ARMS:records.append({'case_id':case,'arm':arm,'initial_state':self.body(),'status':'PASS','success':True})
        result=summarize_e1(records,['a','b','c','d'])
        self.assertEqual(result['status'],'PASS');self.assertFalse(result['go_for_remaining_48'])
        records[0]['success']=False
        self.assertTrue(summarize_e1(records,['a','b','c','d'])['go_for_remaining_48'])

    def test_frozen_protocol_cannot_be_overwritten(self):
        path=ROOT/'scripts/prepare_baseline_protocol.py';spec=importlib.util.spec_from_file_location('protocol',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            file=Path(tmp)/'v3.json';a={'missing_implementation_files':[],'value':1}
            self.assertTrue(module.freeze_protocol(file,a));self.assertFalse(module.freeze_protocol(file,a))
            with self.assertRaises(RuntimeError):module.freeze_protocol(file,dict(a,value=2))


if __name__=='__main__':unittest.main()
