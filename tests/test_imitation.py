"""Synthetic encoding fixtures verify loaders/models, never physical outcomes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES
from apfg.baselines.dataset import SuccessfulDemoDataset,TransitionDataset,validate_partitions,split_demo_episodes
from apfg.baselines.bc import BCPolicy
from apfg.baselines.dp_adapted import DiffusionHighLevelPolicy

ROOT=Path(__file__).resolve().parents[1]


class DemonstrationBoundaries(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='imitation_fixture_',dir=ROOT/'tmp');self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.rows=[]

    def add(self,episode,step,success=True,split='train',obj='fixture_train',seed=7):
        file=self.directory/f'{episode}_{step}.npz'
        candidates=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);candidates[0,0]=1;candidates[1,1]=1
        mask=np.zeros(MAX_CANDIDATES,bool);mask[:2]=True
        np.savez_compressed(file,obs=np.zeros(OBS_DIM,np.float32),candidates=candidates,mask=mask,action=np.array(step%2),reward=np.array(0.),
            next_obs=np.zeros(OBS_DIM,np.float32),next_candidates=candidates,next_mask=mask,done=np.array(step==1),
            oracle_pose=np.full((4,4),np.nan))
        self.rows.append({'transition_path':str(file.relative_to(ROOT)),'episode_id':episode,'scene_id':episode,
            'split':split,'success':success,'method':'FIXTURE_NOT_EXPERIMENT','demonstrator_source':'UNIT_TEST_ONLY',
            'object_id':obj,'seed':seed,'action_type':['GRASP','ACTIVE'][step%2],'step_index':step,
            'data_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})

    def manifest(self):
        file=self.directory/'manifest.jsonl';file.write_text(''.join(json.dumps(r)+'\n' for r in self.rows));return file

    def test_success_filter_hash_and_sequence_do_not_cross_episodes(self):
        self.add('a',0);self.add('a',1);self.add('b',0);self.add('failed',0,False)
        data=SuccessfulDemoDataset(self.manifest())
        self.assertEqual(len(data),3);self.assertEqual(len(data.by_episode),2)
        sequence,mask=data.sequence(1,2)
        self.assertEqual(mask.tolist(),[1.,0.]);self.assertEqual(set(data[0]),{'obs','candidates','mask','action','reward','next_obs','next_candidates','next_mask','done'})
        self.assertEqual(data.subset_sha256,SuccessfulDemoDataset(self.manifest()).subset_sha256)
        train,val=split_demo_episodes(data,seed=9);self.assertFalse(set(train)&set(val));self.assertEqual(len(train)+len(val),2)

    def test_train_holdout_object_or_seed_overlap_is_rejected(self):
        self.add('a',0);self.add('test',0,split='holdout',obj='fixture_train',seed=9)
        with self.assertRaisesRegex(ValueError,'overlap'):SuccessfulDemoDataset(self.manifest())
        self.rows[-1]['object_id']='fixture_unseen';self.rows[-1]['seed']=7
        with self.assertRaisesRegex(ValueError,'overlap'):validate_partitions(self.rows)

    def test_modified_transition_cannot_keep_old_sha(self):
        self.add('a',0);manifest=self.manifest();file=ROOT/self.rows[0]['transition_path'];file.write_bytes(file.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):TransitionDataset(manifest)


class ImitationModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def state(self):
        candidates=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);candidates[5,0]=1;candidates[9,1]=1
        mask=np.zeros(MAX_CANDIDATES,bool);mask[5]=True;mask[9]=True
        return {'obs':np.zeros(OBS_DIM,np.float32),'candidates':candidates,'mask':mask}

    def test_bc_mask_and_checkpoint_reproduce(self):
        model=BCPolicy();state=self.state();index=model.act(state);self.assertIn(index,[5,9])
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='bc_fixture_',dir=ROOT/'tmp') as temp:
            path=Path(temp)/'model.pt';model.save(path);self.assertEqual(index,BCPolicy.load(path).act(state))
        state['mask'][:]=False;self.assertEqual(model.act(state),-1)

    def test_bc_selection_tracks_candidate_features_not_dynamic_slot(self):
        torch.manual_seed(8);model=BCPolicy();state=self.state();selected=model.act(state)
        permutation=np.roll(np.arange(MAX_CANDIDATES),3)
        reordered={'obs':state['obs'],'candidates':state['candidates'][permutation],'mask':state['mask'][permutation]}
        actual=model.act(reordered)
        self.assertEqual(int(permutation[actual]),selected)

    def test_official_unet_backward_reverse_and_checkpoint_are_real(self):
        torch.manual_seed(5);model=DiffusionHighLevelPolicy(diffusion_steps=8)
        state=self.state();obs=torch.tensor(state['obs']).unsqueeze(0);candidates=torch.tensor(state['candidates']).unsqueeze(0);mask=torch.tensor(state['mask']).unsqueeze(0)
        targets=candidates[:,5].unsqueeze(1).repeat(1,2,1);model.fit_action_normalizer(targets)
        loss=model.noise_loss(obs,candidates,mask,targets);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.denoiser.parameters()))
        self.assertLess(float(model.abar[-1]),1e-4)
        first=model.sample(obs,candidates,mask,seed=19);self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first,model.sample(obs,candidates,mask,seed=19)))
        with tempfile.TemporaryDirectory(prefix='dp_fixture_',dir=ROOT/'tmp') as temp:
            path=Path(temp)/'model.pt';model.save(path);restored=DiffusionHighLevelPolicy.load(path)
            self.assertTrue(torch.equal(first,restored.sample(obs,candidates,mask,seed=19)))
        self.assertIn(model.act(state,seed=19),[5,9]);state['mask'][:]=False;self.assertEqual(model.act(state),-1)


if __name__=='__main__':unittest.main()
