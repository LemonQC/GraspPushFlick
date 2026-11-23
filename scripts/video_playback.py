#!/usr/bin/env python3
"""Project-local, on-demand VP8 copies for desktops without an H.264 decoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.visualization.video_validation import validate_video


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(source):
    import imageio_ffmpeg
    source=Path(source).resolve();source.relative_to(ROOT)
    source_hash=sha256(source)
    out=ROOT/'outputs/video_playback'/(source_hash[:16]+'_'+source.stem);out.mkdir(parents=True,exist_ok=True)
    target=out/'preview.webm';report=out/'validation.json'
    if target.exists() and report.exists():
        cached=json.loads(report.read_text())
        if cached['source_sha256']==source_hash and cached['preview_sha256']==sha256(target) and cached['preview_validation']['status']=='PASS':return target
        raise RuntimeError('Cached preview changed; retain it for audit instead of overwriting')
    original=validate_video(source)
    if original['status']!='PASS':raise RuntimeError('Original video failed content validation: '+json.dumps(original))
    partial=out/'preview.partial.webm'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source),
        '-map','0:v:0','-an','-c:v','libvpx','-deadline','realtime','-cpu-used','4','-crf','10','-b:v','2M',
        '-pix_fmt','yuv420p','-threads','2',str(partial)]
    subprocess.run(command,check=True)
    converted=validate_video(partial,expected_frames=original['frames'])
    if converted['status']!='PASS':raise RuntimeError('VP8 conversion failed validation: '+json.dumps(converted))
    partial.replace(target);converted['path']=str(target.relative_to(ROOT));original['path']=str(source.relative_to(ROOT))
    report.write_text(json.dumps({'source':str(source.relative_to(ROOT)),'source_sha256':source_hash,
        'preview':str(target.relative_to(ROOT)),'preview_sha256':sha256(target),
        'original_validation':original,'preview_validation':converted,
        'purpose':'Playback-compatible re-encoding of original rendered frames; not a new simulation or replacement evidence',
        'encoder':'project-local imageio-ffmpeg, libvpx VP8 / WebM; original fps and resolution'},indent=2)+'\n')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',nargs='?',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--camera',choices=['third','fixed','wrist'],default='third');args=p.parse_args()
    source=args.video
    if source is None:
        completed=[d for d in sorted((ROOT/'outputs/m1/runs').iterdir()) if (d/'result.json').is_file()]
        if not completed:p.error('No completed M1 run; provide an existing project-local MP4')
        source=completed[-1]/'videos/m1'/(args.camera+'.mp4')
    if source.is_dir():source=source/'videos/m1'/(args.camera+'.mp4')
    target=prepare(source)
    print(str(target),flush=True)
    if not args.prepare_only:
        if not os.environ.get('DISPLAY'):p.error('No desktop DISPLAY; copy is ready, open it in the existing desktop session')
        # XImage avoids depending on a remote desktop's GL video surface. All
        # decoding uses already installed VP8, without system package changes.
        env=os.environ.copy();env['GST_REGISTRY']=str(ROOT/'cache/gstreamer/registry.bin')
        Path(env['GST_REGISTRY']).parent.mkdir(parents=True,exist_ok=True)
        return subprocess.call(['gst-launch-1.0','-q','playbin','uri='+target.as_uri(),
                                'video-sink=ximagesink','audio-sink=fakesink'],env=env)
    return 0


if __name__=='__main__':raise SystemExit(main())
