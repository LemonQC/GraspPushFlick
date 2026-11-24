import unittest
from types import SimpleNamespace
import numpy as np

from scripts.run_baseline_holdout import TimedPolicy
from scripts.run_baseline_experiments import enforce_holdout


class HeldoutPolicyBoundary(unittest.TestCase):
    def test_timing_passes_only_numeric_state_and_preserves_diffusion_seed(self):
        class Policy:
            def act(self,state,seed):
                assert set(state)=={'obs','candidates','mask'}
                assert seed==42
                return 2
        state={'obs':np.zeros(32),'candidates':np.zeros((24,24)),'mask':np.zeros(24,bool)};state['mask'][2]=True
        wrapped=TimedPolicy(Policy(),'dp',42)
        self.assertEqual(wrapped.act(state),2)
        self.assertGreaterEqual(wrapped.decisions[0]['elapsed_s'],0)
        self.assertEqual(wrapped.decisions[0]['legal_candidates'],1)
        self.assertEqual(state['mask'].sum(),1)

    def test_physical_holdout_rejects_missing_or_overlapping_training_identifiers(self):
        tasks=[{'method':'dqn','object_id':'heldout','seed':10}]
        with self.assertRaises(ValueError):enforce_holdout(SimpleNamespace(training_metadata={}),'dqn',tasks)
        with self.assertRaises(ValueError):
            enforce_holdout(SimpleNamespace(training_metadata={'train_object_ids':['heldout'],'train_seeds':[1]}),'dqn',tasks)
        enforce_holdout(SimpleNamespace(training_metadata={'train_object_ids':['train'],'train_seeds':[1]}),'dqn',tasks)
