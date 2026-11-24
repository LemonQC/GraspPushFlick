"""Shared sparse retrieval reward; evaluator scalars never become observations."""
import math


def compute_reward(*,success=False,terminal=False,preparation=False,simulation_time_s=0.,
                   disturbance_m=0.,command_failed=False,time_weight=.001,
                   preparation_weight=.01,disturbance_weight=.02,failure_weight=.05):
    if success and not terminal:raise ValueError('Only terminal full retrieval can earn success reward')
    numbers=[simulation_time_s,disturbance_m,time_weight,preparation_weight,disturbance_weight,failure_weight]
    if not all(math.isfinite(float(v)) and v>=0 for v in numbers):raise ValueError('Finite nonnegative raw costs/weights required')
    terms={'terminal_retrieval_bonus':1. if success else 0.,
           'preparation_cost':-preparation_weight*bool(preparation),
           'simulation_time_cost':-time_weight*simulation_time_s,
           'disturbance_cost':-disturbance_weight*min(disturbance_m,1.),
           'command_failure_cost':-failure_weight*bool(command_failed)}
    total=float(sum(terms.values()))
    return total,dict(terms,total=total,raw_simulation_time_s=float(simulation_time_s),
        raw_disturbance_m=float(disturbance_m),physical_full_retrieval=bool(success),
        scope='evaluator reward only; no family-based skill labels or grasp-count shaping')
