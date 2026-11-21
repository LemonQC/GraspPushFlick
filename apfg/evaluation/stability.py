"""Measure settling from actual pose changes, retaining SDK velocities separately."""
import math


def norm(values):
    return math.sqrt(sum(v*v for v in values))


def angle(a,b):
    dot=abs(sum(x*y for x,y in zip(a,b)))/(norm(a)*norm(b))
    return 2*math.acos(min(1.,max(0.,dot)))


def evaluate_stability(telemetry, object_count):
    if not telemetry:
        return {'stable':False,'reason':'no measured poses'}
    tail=[r for r in telemetry if r['time_s']>=telemetry[-1]['time_s']-1.]
    coverage=tail[-1]['time_s']-tail[0]['time_s']
    speed=angular_speed=displacement=rotation=0.
    for i in range(object_count):
        final=tail[-1]['objects'][i]
        for row in tail:
            obj=row['objects'][i]
            displacement=max(displacement,norm([a-b for a,b in zip(obj['position'],final['position'])]))
            rotation=max(rotation,angle(obj['orientation'],final['orientation']))
        for left,right in zip(tail,tail[1:]):
            delta=right['time_s']-left['time_s']
            if delta<=0:raise ValueError('Non-increasing measured simulation time')
            a,b=left['objects'][i],right['objects'][i]
            speed=max(speed,norm([x-y for x,y in zip(a['position'],b['position'])])/delta)
            angular_speed=max(angular_speed,angle(a['orientation'],b['orientation'])/delta)
    return {'stable':coverage>=.9 and speed<.02 and angular_speed<.1 and displacement<.002 and rotation<.01,
            'coverage_s':coverage,'max_pose_derived_speed_m_s':speed,'max_pose_derived_angular_speed_rad_s':angular_speed,
            'max_displacement_m':displacement,'max_rotation_rad':rotation,
            'source':'successive measured rigid-body poses; SDK velocities are saved unchanged as separate diagnostics'}
