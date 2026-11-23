"""Decode actual videos and check content as well as container readability."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def validate_video(path, expected_frames=None):
    path=Path(path)
    if not path.is_file():
        return {'path':str(path),'frames':0,'status':'FAILED','reason':'Video file missing'}
    count=black=informative=0
    means=[];width=height=None
    try:
        reader=imageio.get_reader(str(path))
        try:
            metadata=reader.get_meta_data()
            for frame in reader:
                height,width=frame.shape[:2]
                # Pixel statistics cover a regular grid in EVERY decoded frame.
                pixels=frame[::8,::8,:3].astype(np.float32)
                is_black=float(np.mean(pixels.max(axis=-1)<8))>=.995
                contrast=float(np.percentile(pixels,98)-np.percentile(pixels,2))
                black+=int(is_black);informative+=int(not is_black and contrast>=5)
                means.append(float(pixels.mean()));count+=1
        finally:
            reader.close()
    except Exception as exc:
        return {'path':str(path),'frames':count,'status':'FAILED','reason':'Decode failed: '+str(exc)}
    fps=float(metadata.get('fps',0.) or 0.)
    reasons=[]
    if count<2:reasons.append('Fewer than two decoded frames')
    if fps<=0:reasons.append('Missing/invalid frame rate')
    if expected_frames is not None and count!=expected_frames:reasons.append('Decoded frame count differs from captured samples')
    if informative<2 or (count and black/count>=.95):reasons.append('No usable scene content or predominantly black frames')
    return {'path':str(path),'frames':count,'duration_s':count/fps if fps else 0.,'fps':fps,
            'width':width,'height':height,'black_frames':black,'informative_frames':informative,
            'black_frame_fraction':black/count if count else 1.,
            'mean_pixel_value_min':min(means) if means else None,'mean_pixel_value_max':max(means) if means else None,
            'content_check':'Regular pixel grid on every decoded frame; dark/flat content detected, not proof of semantic motion',
            'status':'FAILED' if reasons else 'PASS',**({'reason':'; '.join(reasons)} if reasons else {})}
