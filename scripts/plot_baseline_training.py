#!/usr/bin/env python3
"""Plot actual recorded optimizer losses using the existing bundled Matplotlib."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read_curve(directory):
    path=directory/'curve.csv'
    with path.open(newline='') as stream:
        return path,[{k:float(v) if v else np.nan for k,v in row.items()} for row in csv.DictReader(stream)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dqn-run',type=Path,required=True);p.add_argument('--bc-run',type=Path,required=True)
    p.add_argument('--dp-run',type=Path,required=True);p.add_argument('--bc-overfit-run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'evidence/baselines/training_curves.png');args=p.parse_args()
    # No installation or global path edits. The simulator already bundles a
    # compatible Python3.10 Matplotlib + dependencies; append them only here.
    sys.path.append(str(ROOT/'third_party/isaacsim/exts/omni.isaac.core_archive/pip_prebundle'))
    for path in sorted((ROOT/'third_party/isaacsim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')):sys.path.append(str(path))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dqn_path=args.dqn_run/'training_curve.json';dqn=json.loads(dqn_path.read_text())
    bc_path,bc=read_curve(args.bc_run);dp_path,dp=read_curve(args.dp_run);overfit_path,overfit=read_curve(args.bc_overfit_run)
    if not all([dqn,bc,dp,overfit]):raise ValueError('Actual nonempty training curves required')
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),layout='constrained')
    axes[0,0].plot([r['update'] for r in dqn],[r['loss'] for r in dqn],color='#3568a0',linewidth=1)
    axes[0,0].set(title='Double DQN: real interaction replay',xlabel='Optimizer update',ylabel='Huber TD loss')
    for ax,rows,title,ylabel in [(axes[0,1],bc,'BC: primary 200-update checkpoint','Cross entropy'),
                                 (axes[1,0],dp,'High-level DP: official U-Net adaptation','Noise-prediction MSE')]:
        ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='Optimization train',linewidth=1)
        ax.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label='One-episode validation',linewidth=.8,alpha=.7)
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.legend(fontsize=8)
    axes[1,1].semilogy([r['epoch'] for r in overfit],[max(r['train_loss'],1e-12) for r in overfit],color='#328053')
    axes[1,1].set(title='BC overfit diagnostic (not the test checkpoint)',xlabel='Epoch',ylabel='Training cross entropy (log)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Real small-data training chains; objectives/scales differ; no physical-success claim',fontsize=11)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=140);plt.close(fig)
    sources=[dqn_path,bc_path,dp_path,overfit_path]
    args.output.with_suffix('.json').write_text(json.dumps({'plot':'Actual raw optimizer traces; no synthetic/interpolated experiment data',
        'matplotlib_version':matplotlib.__version__,'inputs':[{'path':str(path.resolve().relative_to(ROOT)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]},indent=2)+'\n')
    print(str(args.output))


if __name__=='__main__':main()
