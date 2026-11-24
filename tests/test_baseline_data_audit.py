"""Temporary schema fixtures only; none are research data or demonstrations."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from apfg.baselines.data_audit import audit_dataset,file_sha256
from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,SCHEMA_VERSION

ROOT=Path(__file__).resolve().parents[1]


class BaselineDataAudit(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.context=tempfile.TemporaryDirectory(prefix='baseline_audit_unit_',dir=ROOT/'tmp');self.addCleanup(self.context.cleanup)
        self.directory=Path(self.context.name);self.manifest=self.directory/'manifest.jsonl';self.episodes=self.directory/'episodes_manifest.jsonl'

    def write_jsonl(self,path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def episode_fixture(self,index=0,split='train',object_id='unit_object',steps=1):
        directory=self.directory/'campaign'/f'episode_{index:03d}';directory.mkdir(parents=True)
        record={'episode_id':directory.name,'scene_id':'unit_scene_'+str(index),'object_id':object_id,'seed':index+1,
            'method':'unit_fixture','split':split,'wall_s_including_reset':float(index+3),'status':'PASS' if steps else 'FAILED',
            'success':bool(steps),'failure_reason':None if steps else 'NO_LEGAL_GRASP'}
        if not steps:
            (directory/'episode.json').write_text(json.dumps(dict(record,steps=[])))
            return record,None
        obs=np.zeros(OBS_DIM,np.float32);target=np.array([[.40,0,.02],[.42,0,.02]],np.float64)
        obs[11:14]=np.median(target,axis=0);obs[14:17]=np.ptp(target,axis=0)
        candidates=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);candidates[0,0]=1
        mask=np.zeros(MAX_CANDIDATES,bool);mask[0]=True
        transition={'obs':obs,'candidates':candidates,'mask':mask,'action':np.int64(0),'reward':np.float32(1.),
            'next_obs':np.zeros_like(obs),'next_candidates':np.zeros_like(candidates),'next_mask':np.zeros_like(mask),'done':np.bool_(True)}
        file=directory/'transition_00.npz';np.savez_compressed(file,**transition)
        decision=directory/'decision_00';decision.mkdir()
        arrays={name:transition[name] for name in ['obs','candidates','mask']};arrays.update(target_points=target,scene_points=target)
        rgb=np.zeros((480,640,3),np.uint8);rgb[:240]=190
        for camera in ['fixed','wrist']:
            arrays[camera+'_rgb']=rgb;arrays[camera+'_depth_m']=np.ones((480,640),np.float32)
            arrays[camera+'_K']=np.array([[100,0,320],[0,100,240],[0,0,1]],float);arrays[camera+'_T_world_optical']=np.eye(4)
        observation=decision/'observation.npz';np.savez_compressed(observation,**arrays)
        candidate={'type':'GRASP','candidate_id':'unit_candidate','features':candidates[0].tolist(),'valid':True,'input_hash':'a'*64}
        (decision/'candidates.json').write_text(json.dumps({'schema':SCHEMA_VERSION,'input_hash':'a'*64,'candidates':[candidate]}))
        trajectory=directory/'telemetry.json'
        trajectory.write_text(json.dumps([{'q':[0]*8,'frames_world':{'ral_tcp':np.eye(4).tolist()},
            'T_world_optical':np.eye(4).tolist(),'time_s':i/30} for i in range(2)]))
        (directory/'episode.json').write_text(json.dumps(dict(record,steps=[{'type':'GRASP','physical_evaluation':{'retrieval':{'status':'PASS'}}}])))
        row={'transition_path':str(file.relative_to(ROOT)),'episode_id':directory.name+'_unit_campaign','scene_id':record['scene_id'],
            'object_id':object_id,'seed':record['seed'],'method':record['method'],'split':split,'success':True,
            'demonstrator_source':'explicit_unit_fixture_not_physical_data','action_type':'GRASP','step_index':0,
            'data_sha256':file_sha256(file),'rgbd_before':str(observation.relative_to(ROOT)),'trajectory':str(trajectory.relative_to(ROOT))}
        return record,row

    def audit(self):return audit_dataset(self.manifest,self.episodes,episode_roots=[self.directory])

    def test_empty_data_is_not_a_passed_training_run(self):
        result=self.audit();self.assertEqual(result['status'],'NOT_RUN');self.assertEqual(result['data_readiness'],'DATA_PENDING')

    def test_zero_transition_failures_remain_in_denominator(self):
        record,_=self.episode_fixture(steps=0);self.write_jsonl(self.episodes,[record]);self.write_jsonl(self.manifest,[])
        result=self.audit();self.assertEqual(result['status'],'PASS');self.assertEqual(result['counts']['episodes'],1)
        self.assertEqual(result['counts']['zero_transition_episodes'],1);self.assertEqual(result['by_split']['train']['failures'],1)
        self.assertFalse(result['next_round_large_scale_training_ready'])

    def test_real_saved_shapes_mask_semantics_and_hash_are_checked(self):
        record,row=self.episode_fixture();self.write_jsonl(self.episodes,[record]);self.write_jsonl(self.manifest,[row])
        result=self.audit();self.assertEqual(result['status'],'PASS');self.assertEqual(result['counts']['validated_transitions'],1)
        row['action_type']='PUSH';self.write_jsonl(self.manifest,[row])
        broken=self.audit();self.assertEqual(broken['status'],'FAILED')
        self.assertTrue(any('semantic mismatch' in e.get('detail','') for e in broken['errors']))

    def test_zero_transition_object_overlap_is_rejected(self):
        train,_=self.episode_fixture(0,'train','overlap',0);holdout,_=self.episode_fixture(1,'holdout','overlap',0)
        self.write_jsonl(self.episodes,[train,holdout]);self.write_jsonl(self.manifest,[])
        result=self.audit();self.assertEqual(result['status'],'FAILED')
        self.assertTrue(any(e['code']=='EPISODE_TRAIN_HOLDOUT_OVERLAP' for e in result['errors']))

    def test_forbidden_gt_array_is_rejected_even_with_updated_file_hash(self):
        record,row=self.episode_fixture();path=ROOT/row['transition_path']
        with np.load(path) as file:arrays={k:file[k].copy() for k in file.files}
        arrays['oracle_state']=np.zeros(7);np.savez_compressed(path,**arrays);row['data_sha256']=file_sha256(path)
        self.write_jsonl(self.episodes,[record]);self.write_jsonl(self.manifest,[row])
        result=self.audit();self.assertEqual(result['status'],'FAILED')
        self.assertTrue(any('Unexpected policy transition fields' in e.get('detail','') for e in result['errors']))


if __name__=='__main__':unittest.main()
