"""Early driver/Vulkan/PhysX probe using the pinned official AppLauncher."""
import argparse
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'third_party/IsaacLab/source/isaaclab'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.kit_args = f'--portable-root {root}/cache/omniverse/portable --/app/fastShutdown=true --/app/renderer/skipWhileMinimized=false'
app = AppLauncher(args).app
try:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/30)
    world.scene.add_default_ground_plane()
    cube = world.scene.add(DynamicCuboid('/World/FallingBody', position=np.array([0., 0., 1.]), scale=np.array([.1]*3), mass=.2))
    camera = Camera('/World/ProbeCamera', position=np.array([2., 0., 1.5]), resolution=(640, 480),
                    orientation=euler_angles_to_quat(np.array([0., 25., 180.]), degrees=True))
    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    samples = []
    for step in range(360):
        world.step(render=True)
        if step % 10 == 0:
            samples.append({'step': step, 'position': cube.get_world_pose()[0].tolist(), 'velocity': cube.get_linear_velocity().tolist()})
    rgb = camera.get_rgba()
    valid_rgb = rgb is not None and rgb.size > 0 and np.std(rgb[..., :3]) > 1
    passed = samples[0]['position'][2] > .8 and abs(samples[-1]['position'][2] - .05) < .02 and np.linalg.norm(samples[-1]['velocity']) < .05 and valid_rgb
    result = {'status': 'PASS' if passed else 'FAILED', 'physics_steps': 360, 'physics_dt': 1/120,
              'samples': samples, 'rgb_shape': list(rgb.shape) if rgb is not None else None,
              'rendered_rgb_valid': bool(valid_rgb), 'mode': 'headless' if args.headless else 'gui'}
    output = root / 'outputs/environment_check' / ('headless_probe.json' if args.headless else 'gui_probe.json')
    output.write_text(json.dumps(result, indent=2))
    print('APFG_PROBE_RESULT', json.dumps(result), flush=True)
finally:
    app.close()
