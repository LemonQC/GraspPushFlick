"""Pinned Isaac Lab AppLauncher configuration shared by hardware tools."""
import ctypes
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def add_launcher_args(parser):
    sys.path.insert(0, str(ROOT/'third_party/IsaacLab/source/isaaclab'))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)


def launch(args):
    from isaaclab.app import AppLauncher
    code = ctypes.CDLL('libcuda.so.1').cuInit(0)
    if code:
        raise RuntimeError(f'CUDA_BLOCKED: cuInit={code}')
    args.enable_cameras = True
    if not args.experience:
        args.experience = str(ROOT/'configs/apps'/('apfg.python.headless.rendering.kit' if args.headless else 'apfg.python.rendering.kit'))
    args.kit_args = (f'--portable-root {ROOT}/cache/omniverse/portable --/app/renderer/skipWhileMinimized=false '
                     f'--ext-folder {ROOT}/configs/apps --/app/tokens/apfg_root={ROOT} '
                     f'--/persistent/isaac/asset_root/default={ROOT}/assets/isaac '
                     '--/app/extensions/registryEnabled=false --/telemetry/enableAnonymousData=false --/telemetry/enableAnonymousAppName=false')
    return AppLauncher(args).app
