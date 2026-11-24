"""CPU numerical fixtures; these are not physical experiment results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES
from apfg.baselines.dqn import CandidateDQN,CandidateQ,DQNConfig,ReplayBuffer,double_dqn_targets
from apfg.baselines.dqn_io import manifest_rows,load_transitions,file_sha256

ROOT=Path(__file__).resolve().parents[1]


def state():
    rng=np.random.default_rng(8)
    return {'obs':rng.normal(size=OBS_DIM).astype(np.float32),
            'candidates':rng.normal(size=(MAX_CANDIDATES,CAND_DIM)).astype(np.float32),
            'mask':np.array([True,False,True]+[False]*(MAX_CANDIDATES-3))}


def transition(reward=1.,done=True):
    current=state();following=state()
    return dict(current,action=0,reward=reward,done=done,next_obs=following['obs'],
                next_candidates=following['candidates'],next_mask=following['mask'])


class DQNMaskAndTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_double_target_uses_online_argmax_and_terminal_masks(self):
        rewards=torch.tensor([1.,2.,3.]);done=torch.tensor([False,False,True])
        online=torch.tensor([[100.,4.,3.],[1.,8.,4.],[3.,2.,1.]])
        target=torch.tensor([[90.,2.,100.],[20.,30.,40.],[99.,99.,99.]])
        mask=torch.tensor([[False,True,True],[False,False,False],[True,True,True]])
        result=double_dqn_targets(rewards,done,online,target,mask,.9)
        torch.testing.assert_close(result,torch.tensor([2.8,2.,3.]))

    def test_candidates_are_permutation_equivariant(self):
        agent=CandidateDQN({'hidden_dim':16},seed=7);value=state()
        permutation=np.random.default_rng(3).permutation(MAX_CANDIDATES)
        reordered=dict(value,candidates=value['candidates'][permutation],mask=value['mask'][permutation])
        np.testing.assert_allclose(agent.values(reordered),agent.values(value)[permutation],rtol=1e-5,atol=1e-6)

    def test_greedy_and_random_policies_cannot_choose_invalid_slots(self):
        agent=CandidateDQN({'hidden_dim':16},seed=2);value=state()
        self.assertIn(agent.act(value,epsilon=0),[0,2])
        choices=[agent.act(value,epsilon=1.) for _ in range(60)]
        self.assertEqual(set(choices),{0,2})
        value['mask'][:]=False;self.assertIsNone(agent.act(value))

    def test_extra_oracle_metadata_never_enters_the_network(self):
        agent=CandidateDQN({'hidden_dim':16},seed=1);value=state()
        class Poison:
            def __repr__(self):raise AssertionError('Oracle field touched')
            def __array__(self):raise AssertionError('Oracle field converted')
        before=agent.values(value)
        after=agent.values(dict(value,oracle_state=Poison(),family=Poison(),seed=Poison()))
        np.testing.assert_array_equal(before,after)

    def test_replay_copies_numeric_state_and_rejects_illegal_action(self):
        replay=ReplayBuffer(2);item=transition();original=item['obs'].copy();replay.append(item)
        item['obs'][:]=20;np.testing.assert_array_equal(replay.storage[0]['obs'],original)
        item['action']=1
        with self.assertRaisesRegex(ValueError,'illegal'):replay.append(item)


class DQNTrainingAndCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True);torch.set_num_threads(1)

    def test_real_backward_td_updates_and_exact_resume(self):
        agent=CandidateDQN({'hidden_dim':24,'learning_rate':.003,'batch_size':4,'min_replay':2,
                            'target_update_interval':5},seed=19)
        for _ in range(4):agent.observe(transition())
        before={k:v.detach().clone() for k,v in agent.online.state_dict().items()}
        logs=[agent.update() for _ in range(45)]
        self.assertLess(np.mean([r['loss'] for r in logs[-5:]]),np.mean([r['loss'] for r in logs[:5]]))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in agent.online.state_dict().items()))
        self.assertEqual(agent.target_syncs,9)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='dqn_unit_') as directory:
            path=Path(directory)/'checkpoint.pt';agent.save(path)
            loaded=CandidateDQN.load(path)
            self.assertEqual(len(loaded.replay),4);self.assertEqual(loaded.environment_steps,4)
            self.assertEqual(loaded.act(state()),agent.act(state()))
            # Both RNG streams and optimizer momentum are resumed, not weights alone.
            expected=agent.update();actual=loaded.update()
            self.assertAlmostEqual(expected['loss'],actual['loss'],places=7)
            for k,v in agent.online.state_dict().items():torch.testing.assert_close(v,loaded.online.state_dict()[k])

    def test_qualification_is_excluded_and_corrupt_transition_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='dqn_data_unit_') as directory:
            directory=Path(directory);path=directory/'step.npz';np.savez_compressed(path,**transition())
            base={'transition_path':str(path.relative_to(ROOT)),'episode_id':'train_ep','scene_id':'train_scene',
                  'object_id':'train_object','seed':1,'data_sha256':file_sha256(path),'split':'train'}
            qualification=dict(base,split='qualification',episode_id='qualification_ep')
            manifest=directory/'manifest.jsonl';manifest.write_text(json.dumps(base)+'\n'+json.dumps(qualification)+'\n')
            data,rows=load_transitions(manifest);self.assertEqual(len(data),1);self.assertEqual(rows[0]['split'],'train')
            bad=dict(base,data_sha256='0'*64);manifest.write_text(json.dumps(bad)+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_transitions(manifest)

    def test_train_holdout_object_or_seed_leak_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='dqn_split_unit_') as directory:
            manifest=Path(directory)/'manifest.jsonl'
            base={'transition_path':'irrelevant','episode_id':'train_ep','scene_id':'train_scene',
                  'object_id':'same_object','seed':1,'data_sha256':'irrelevant','split':'train'}
            holdout=dict(base,split='holdout',episode_id='holdout_ep',scene_id='holdout_scene',seed=2)
            manifest.write_text(json.dumps(base)+'\n'+json.dumps(holdout)+'\n')
            with self.assertRaisesRegex(ValueError,'object_id overlap'):manifest_rows(manifest)


if __name__=='__main__':unittest.main()
