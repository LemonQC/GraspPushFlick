import math
import unittest
from apfg.evaluation.stability import evaluate_stability


class Stability(unittest.TestCase):
    def trajectory(self,moving=False,rotating=False):
        return [{'time_s':i/30,'objects':[{'position':[i*.01 if moving else 0,0,.8],
                'orientation':[math.cos(i*.01/2),0,0,math.sin(i*.01/2)] if rotating else [1,0,0,0],
                'linear_velocity':[1,0,0],'angular_velocity':[1,0,0]}]} for i in range(31)]

    def test_stationary_poses_with_stale_sdk_velocity(self):
        self.assertTrue(evaluate_stability(self.trajectory(),1)['stable'])

    def test_translating_body_rejected(self):
        self.assertFalse(evaluate_stability(self.trajectory(moving=True),1)['stable'])

    def test_rotating_body_rejected(self):
        self.assertFalse(evaluate_stability(self.trajectory(rotating=True),1)['stable'])

    def test_short_window_rejected(self):
        self.assertFalse(evaluate_stability(self.trajectory()[:5],1)['stable'])
