"""Transparent Track-A policies over exactly the shared masked candidate pool.

No policy reads scene IDs, asset IDs, seeds, family names or oracle fields.
"""
import numpy as np

from apfg.baselines.schema import CAND_DIM,MAX_CANDIDATES


def policy_arrays(state):
    candidates=np.asarray(state['candidates'],dtype=np.float32)
    mask=np.asarray(state['mask'],dtype=bool)
    if candidates.shape!=(MAX_CANDIDATES,CAND_DIM) or mask.shape!=(MAX_CANDIDATES,):
        raise ValueError('Shared candidate/mask dimensions required')
    if not np.isfinite(candidates[mask]).all():raise ValueError('Legal candidate features must be finite')
    return candidates,mask


class HeuristicPolicy:
    """score = observable quality + .25 benefit - .10 cost - .10 prep.

    Quality is raw GraspNet score for grasps and the declared observable
    geometric proxy for other skills; benefit/cost come from the common
    generator. These uncalibrated proxies are a baseline, not predicted task
    success or information gain. Ties follow fixed pool order.
    """
    def act(self,state):
        candidates,mask=policy_arrays(state)
        if not mask.any():return -1
        preparation=1.-candidates[:,0]
        scores=candidates[:,4]+.25*candidates[:,23]-.10*candidates[:,5]-.10*preparation
        return int(np.argmax(np.where(mask,scores,-np.inf)))


class GraspOnlyPolicy:
    """Highest-score legal GraspNet candidate, same execution/collision checks."""
    def act(self,state):
        candidates,mask=policy_arrays(state)
        mask=mask&(candidates[:,0]>.5)
        if not mask.any():return -1
        return int(np.argmax(np.where(mask,candidates[:,4],-np.inf)))


class RandomValidPolicy:
    """Uniform legal candidates, with an explicit independent policy RNG."""
    def __init__(self,seed=0):self.rng=np.random.default_rng(seed)
    def act(self,state):
        _,mask=policy_arrays(state);indices=np.flatnonzero(mask)
        return int(self.rng.choice(indices)) if len(indices) else -1


def make_policy(name,seed=0):
    if name.lower() in ('heuristic','b1_heuristic'):return HeuristicPolicy()
    if name.lower() in ('grasp_only','b0_grasp_only'):return GraspOnlyPolicy()
    if name.lower() in ('random_valid','random','b2_random_valid'):return RandomValidPolicy(seed)
    raise ValueError('Unknown non-learning baseline: '+name)
