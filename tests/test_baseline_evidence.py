"""Report fixtures only: none are experiment data or training demonstrations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('baseline_evidence',ROOT/'scripts/collect_baseline_evidence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BaselineEvidenceAccounting(unittest.TestCase):
    def test_zero_action_failures_remain_in_physical_denominator(self):
        base={'split':'holdout','method':'dqn','wall_s_including_reset':10.,'simulation_time_s':2.,
              'transition_count':1,'zero_action_episode':False,'scene_id':'a','object_id':'test','seed':2,
              'status':'PASS','success':True,'failure_class':None,'actual_disk_bytes':20,
              'gpu_device_memory_sampled_peak_mib':None}
        failed=dict(base,transition_count=0,zero_action_episode=True,status='FAILED',success=False,
                    scene_id='b',failure_class='NO_LEGAL_GRASP')
        result=module.physical_groups([base,failed])[0]
        self.assertEqual(result['attempted_episodes'],2)
        self.assertEqual(result['physical_retrieval_successes'],1)
        self.assertEqual(result['zero_action_episodes'],1)
        self.assertEqual(result['actual_transitions'],1)

    def test_pending_campaign_does_not_count_as_completed(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            campaign=Path(tmp)/'pending';campaign.mkdir()
            (campaign/'episodes.jsonl').write_text('{}\n')
            campaigns,episodes,pending=module.scan_campaigns(Path(tmp))
            self.assertFalse(campaigns);self.assertFalse(episodes)
            self.assertEqual(pending[0]['status'],'NOT_RUN');self.assertEqual(pending[0]['completed_records'],1)

    def test_dataset_partition_overlap_is_failed(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            directory=Path(tmp);file=directory/'fixture.bin';file.write_bytes(b'explicit report test fixture')
            rows=[]
            for split in ['train','holdout']:
                rows.append({'split':split,'episode_id':split,'scene_id':split,'object_id':'same_object',
                    'seed':12,'success':False,'action_type':'GRASP','transition_path':str(file.relative_to(ROOT)),
                    'data_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            manifest=directory/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=module.dataset_audit(manifest,[],[])
            self.assertEqual(result['train_holdout_disjoint_status'],'FAILED')
            self.assertEqual(result['train_holdout_overlap']['object_id'],['same_object'])


if __name__=='__main__':unittest.main()
