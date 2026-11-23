"""Encoding fixtures test the recorder, not research experiment evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from apfg.visualization.recording import Recorder
from apfg.visualization.video_validation import validate_video

ROOT=Path(__file__).resolve().parents[1]


class VideoContentValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True)

    def record(self,frame):
        context=tempfile.TemporaryDirectory(prefix='video_test_',dir=ROOT/'tmp')
        self.addCleanup(context.cleanup)
        recorder=Recorder(context.name,['camera'])
        for i in range(3):recorder.append({'camera':frame},i,i/30)
        return Path(context.name)/'camera.mp4',recorder.close()['camera']

    def test_readable_all_black_video_fails(self):
        path,result=self.record(np.zeros((48,64,3),np.uint8))
        self.assertEqual(result['status'],'FAILED');self.assertEqual(result['black_frames'],3)

    def test_uniform_bright_video_fails_content_gate(self):
        _,result=self.record(np.full((48,64,3),150,np.uint8))
        self.assertEqual(result['status'],'FAILED')

    def test_actual_frame_count_must_match_sampling(self):
        frame=np.zeros((48,64,3),np.uint8);frame[:24]=190
        path,result=self.record(frame)
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['frames'],3)
        self.assertEqual(validate_video(path,expected_frames=4)['status'],'FAILED')


if __name__=='__main__':unittest.main()
