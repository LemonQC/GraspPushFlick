"""Serialize project GPU jobs without modifying system-wide configuration."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

@contextmanager
def gpu_job(name='baseline'):
    path=ROOT/'cache/baselines/gpu.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.seek(0);stream.truncate();stream.write(name+'\n');stream.flush()
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
