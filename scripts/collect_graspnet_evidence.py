#!/usr/bin/env python3
"""Compact, provenance-linked records of actual network and physics runs."""
import json
from pathlib import Path
import sys
import hashlib
import datetime as dt
import re

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs=[]
    for folder in sorted((ROOT/'outputs/sr5_graspnet').glob('*')):
        if not (folder/'graspnet_result.json').exists():continue
        d=json.loads((folder/'graspnet_result.json').read_text())
        cfg=json.loads((folder/'run_config.json').read_text())
        d.update(run_id=folder.name,mode='headless' if cfg['args']['headless'] else 'gui',artifacts_directory=str(folder.relative_to(ROOT)),actual_command=cfg['command'])
        runs.append((folder,d,cfg))
    successes=[r for r in runs if r[1].get('physical_grasp')=='PASS']
    headless=[r for r in successes if r[1]['mode']=='headless'];gui=[r for r in successes if r[1]['mode']=='gui']
    previews=[r for r in runs if r[2]['args']['preview_only'] and r[1]['status']=='PASS']
    if not headless or not gui or not previews:raise SystemExit('Need actual successful headless, GUI and pose preview runs')
    selected=[headless[-1],gui[-1],previews[-1]]
    public=ROOT/'evidence/sr5_graspnet';public.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((ROOT/'assets/manifests/graspnet_model.json').read_text())
    summaries=[];manifest=[]
    for folder,d,cfg in selected:
        chosen=d['candidates'][d['selected_rank']] if d['selected_rank'] is not None else None
        s={k:d.get(k) for k in ['run_id','mode','status','inference','pose_visualization','physical_grasp','physical_evaluation','raw_candidates','displayed_candidates','selected_rank','information_mode','checkpoint_sha256','actual_command','artifacts_directory']}
        s.update(selected_candidate=chosen,network=json.loads((folder/'network_predictions.json').read_text()),videos=d['videos'])
        if not all(v['status']=='PASS' and v['frames']>1 and abs(v['fps']-30)<1e-6 for group in s['videos'].values() for v in group.values()):
            raise SystemExit(f'Real encoded videos not validated: {folder}')
        for group in s['videos'].values():
            for v in group.values():v['path']=str(Path(v['path']).relative_to(ROOT))
        summaries.append(s)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.npz','.json','.mp4','.png','.ply','.usd']:
                manifest.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path),'availability':'project-local; full artifact not uploaded'})
    failures=[{'run_id':p.name,'mode':d['mode'],'status':d['status'],'physical_grasp':d.get('physical_grasp'),'reason':d.get('execution_error') or d.get('reason')} for p,d,c in runs if d['status']=='FAILED']
    test_log=ROOT/'outputs/environment_check/graspnet_unit_tests.txt'
    test_text=test_log.read_text();match=re.search(r'Ran (\d+) tests',test_text)
    if not match or not test_text.rstrip().endswith('OK'):raise SystemExit('Actual CPU test log missing or not PASS')
    summary={'status':'PASS','scope':'Real RGBD GraspNet inference, native 3D pose display, single-YCB-object physical grasp; no training or general success-rate claim',
        'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_model':metadata,'runs':summaries,'failed_runs_retained':failures,
        'cpu_tests':{'count':int(match.group(1)),'status':'PASS','log_sha256':digest(test_log)},
        'limits':['Community checkpoint-rs mirror LFS hash verified; equality to inaccessible official Drive original not independently verified',
                  'Source PGIA/camera geometry and extrinsics remain approximations',
                  'Observed-cloud and fixed-table filter only; not full-arm or occluded collision certification',
                  'One prediction/action per run; displayed predictions remain at the capture-time pose']}
    (public/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (public/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previews_meta=[]
    for folder,source,name in [(previews[-1][0],'predicted_grasps','candidates'),(previews[-1][0],'predicted_wrist','wrist_candidates'),(headless[-1][0],'execution_final','side_grasp')]:
        original=folder/f'images/{source}_rgb.png';target=public/(name+'.jpg')
        im=Image.open(original).convert('RGB');im.thumbnail((800,600));im.save(target,quality=88)
        previews_meta.append({'path':str(target.relative_to(ROOT)),'sha256':digest(target),'source_path':str(original.relative_to(ROOT)),'source_sha256':digest(original)})
    (public/'previews.json').write_text(json.dumps(previews_meta,indent=2)+'\n')
    lines=['# SR5 GraspNet 6-DoF 实测报告','',
        '状态：真实模型推理 PASS，三维候选显示 PASS，Headless / GUI 物理抓取 PASS。此前固定姿态硬件夹持不是本报告的 GraspNet 抓取。','',
        '| Run ID | 模式 | 原始候选/显示 | 选中倾角 | 物理结果 |','|---|---|---|---|---|']
    for s in summaries:
        tilt=s['selected_candidate']['tilt_from_world_down_deg'] if s['selected_candidate'] else float('nan')
        lines.append(f'| `{s["run_id"]}` | {s["mode"]} | {s["raw_candidates"]}/{s["displayed_candidates"]} | {tilt:.2f}° | {s["physical_grasp"]} |')
    side=summaries[0];evaluation=side['physical_evaluation']
    lines+=['',f'侧向抓取记录最小持续抬升 {evaluation["minimum_sustained_lift_m"]:.6f} m，保持 30 帧/1 秒，双指接触 {evaluation["bilateral_contact_frames"]}/30 帧；没有人工附着或物体位姿重置。',
        '实际模型输入来自腕部 RGB-D 工作区点云，未使用实例分割或物体真值。预测原始姿态、端平面/TCP 几何映射、筛选理由、控制及接触均保存在原始运行目录。','',
        '查看候选：','```bash','cd /data/GraspPushFlicks','bash scripts/run_sr5_graspnet.sh --gui --preview-only --interactive --joint-speed .3','```',
        '执行一次真实预测抓取：','```bash','bash scripts/run_sr5_graspnet.sh --gui --interactive --joint-speed .3','```','',
        '完整 MP4、RGB-D 和网络数组仅保存在服务器 `outputs/sr5_graspnet/<run_id>/`，公开 manifest 仅给出相对路径与 SHA256，不提供远程视频下载。',
        '失败运行保留：'+', '.join(f['run_id'] for f in failures)+'.','',
        '本次官方 Google Drive 下载配额已用尽，使用文档记录的公开 checkpoint-rs 镜像。LFS 哈希和模型参数匹配已验证，未独立验证与官方 Drive 原件哈希相等。详细来源、近似外参和执行范围见 `docs/SR5_GRASPNET_ZH.md`。']
    (ROOT/'docs/SR5_GRASPNET_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','run_ids':[s['run_id'] for s in summaries],'failed_runs':len(failures)}))


if __name__=='__main__':main()
