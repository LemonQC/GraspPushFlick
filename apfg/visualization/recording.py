"""Record only sensor frames and validate the encoded stream by decoding it."""
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
from PIL import Image


class Recorder:
    def __init__(self, directory, camera_names, fps=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writers = {n: imageio.get_writer(str(self.directory / (n+'.mp4')), fps=fps,
                         codec='libx264', quality=7, macro_block_size=1) for n in camera_names}
        self.times = []

    def append(self, rgb_by_name, step, simulation_time):
        for name, writer in self.writers.items():
            rgb = np.asarray(rgb_by_name[name], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                raise ValueError(f'Invalid real camera RGB frame: {name} {rgb.shape}')
            writer.append_data(rgb[..., :3])
        self.times.append({'step': step, 'simulation_time_s': simulation_time, 'wall_monotonic_s': time.monotonic()})

    def close(self):
        for writer in self.writers.values():
            writer.close()
        reports = {}
        for name in self.writers:
            path = self.directory / (name+'.mp4')
            if not path.exists():
                reports[name] = {'path':str(path),'frames':0,'duration_s':0,'status':'FAILED','reason':'No sensor frames captured before exit'}
                continue
            reader = imageio.get_reader(str(path))
            metadata = reader.get_meta_data()
            count = sum(1 for _ in reader)
            reader.close()
            duration = count / self.fps
            reports[name] = {'path': str(path), 'frames': count, 'duration_s': duration,
                             'fps': metadata.get('fps'), 'status': 'PASS' if count > 1 and duration > 0 else 'FAILED'}
        (self.directory/'timing.json').write_text(json.dumps({'sampling': self.times, 'videos': reports}, indent=2)+'\n')
        return reports


def save_rgbd(directory, name, rgb, depth, intrinsic, pose):
    directory = Path(directory)
    rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
    Image.fromarray(rgb).save(directory/(name+'_rgb.png'))
    depth = np.asarray(depth, dtype=np.float32)
    np.save(directory/(name+'_depth_m.npy'), depth)
    mm = np.clip(np.nan_to_num(depth, nan=0, posinf=0)*1000, 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(directory/(name+'_depth.png'))
    (directory/(name+'_calibration.json')).write_text(json.dumps({
        'K': np.asarray(intrinsic).tolist(), 'position_m': pose[0].tolist(), 'orientation_wxyz': pose[1].tolist(),
        'camera_axes': 'ROS: +Z optical, +X image right, +Y image down',
        'depth_definition': 'distance_to_image_plane; optical-axis Z in meters; PNG uint16 millimeters, 0 invalid'}, indent=2)+'\n')


def pointcloud_from_depth(depth, rgb, intrinsic, mask=None):
    depth = np.asarray(depth)
    y, x = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
    if mask is not None:
        valid &= mask
    k = np.asarray(intrinsic)
    z = depth[valid]
    points = np.stack(((x[valid]-k[0, 2])*z/k[0, 0], (y[valid]-k[1, 2])*z/k[1, 1], z), axis=-1)
    return points, np.asarray(rgb)[valid, :3]


def write_ply(path, points, colors):
    with Path(path).open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, colors):
            f.write(' '.join(map(str, [*p, *map(int, c)]))+'\n')
