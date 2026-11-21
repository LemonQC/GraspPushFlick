#!/usr/bin/env python3
"""Publish compact evidence only from actually validated SR5 hardware runs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/isaac_sr5_hardware'
PUBLIC=ROOT/'evidence/sr5_hardware'
MANDATORY=['H0','H1','H2','H3','H4','H5','H7','render_state_sync','simulation_clock','rgbd']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def receipt(path):
    path=Path(path)
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path),'availability':'project-local; full artifact not uploaded'}


def read_runs():
    runs=[]
    for folder in sorted((OUT/'runs').glob('*')):
        if not (folder/'smoke_results.json').exists():continue
        d=json.loads((folder/'smoke_results.json').read_text())
        cfg=json.loads((folder/'run_config.json').read_text())
        d.update(run_id=folder.name,mode='headless' if cfg['args']['headless'] else 'gui',artifacts_directory=str(folder.relative_to(ROOT)),actual_command=cfg['actual_command'])
        good=d['status']=='PASS' and cfg['args']['test']=='all' and all(d['checks'].get(g,{}).get('status')=='PASS' for g in MANDATORY)
        runs.append((folder,d,cfg,good))
    return runs


def repeat_check(a,b):
    pa,da,ca,_=a;pb,db,cb,_=b
    ra=json.loads((pa/'telemetry.json').read_text());rb=json.loads((pb/'telemetry.json').read_text())
    same=len(ra)==len(rb) and da['seed']==db['seed'] and da['initial_config']==db['initial_config'] and da['checks']['H0']['usd_path']==db['checks']['H0']['usd_path']
    joint=objects=None
    if same:
        joint=float(np.max(np.abs(np.array([r['q'] for r in ra])-np.array([r['q'] for r in rb]))))
        objects=float(np.max(np.linalg.norm(np.array([r['object_T_world'] for r in ra])[:,:3,3]-np.array([r['object_T_world'] for r in rb])[:,:3,3],axis=1)))
    return {'status':'PASS' if same and joint<.002 and objects<.004 else 'FAILED','run_ids':[pa.name,pb.name],
            'same_seed_initial_config_tick_count_and_usd':same,'max_joint_trajectory_delta_rad_or_m':joint,'max_object_trajectory_position_delta_m':objects,
            'wall_times_s':[da['wall_time_s'],db['wall_time_s']],'simulated_times_s':[da['simulated_time_s'],db['simulated_time_s']]}


def alias(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():target.unlink()
    if target.exists():raise RuntimeError(f'Refusing to replace non-alias {target}')
    target.symlink_to(os.path.relpath(source,target.parent))


def main():
    runs=read_runs();headless=[r for r in runs if r[3] and r[1]['mode']=='headless'];gui=[r for r in runs if r[3] and r[1]['mode']=='gui']
    if len(headless)<2 or not gui:
        print(json.dumps({'status':'NOT_RUN','valid_headless_all_runs':len(headless),'valid_gui_all_runs':len(gui),'reason':'Need two actual complete headless and one GUI run with every hardware/render gate PASS'}));return 2
    selected=headless[-2:]+gui[-1:]
    repeat=repeat_check(*headless[-2:]);latest=headless[-1]
    recording=[]
    for folder,d,cfg,_ in selected:
        for group,videos in d['videos'].items():
            timing=json.loads((folder/f'videos/{group}/timing.json').read_text())
            samples=timing['sampling'];times=np.array([s['simulation_time_s'] for s in samples])
            cadence=float(np.max(np.abs(np.diff(times)-1/30))) if len(times)>1 else float('inf')
            good=not d['camera_capture_gaps'] and cadence<1e-5 and all(v['status']=='PASS' and v['frames']==len(samples) and abs(v['fps']-30)<1e-6 for v in videos.values())
            recording.append({'run_id':folder.name,'group':group,'status':'PASS' if good else 'FAILED','frames_per_view':len(samples),'max_cadence_error_s':cadence})
    audit=json.loads((OUT/'assembly_audit.json').read_text())
    versions={'isaac_sim_binary':(ROOT/'third_party/isaacsim/VERSION').read_text().strip(),
              'isaac_lab_release':'v2.0.2','isaac_lab_commit':'b5fa0eb031a2413c182eeb54fa3a9295e8fd867c','python':sys.version.split()[0],
              'gpu_and_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
              'physics':'PhysX CPU / MBP / TGS, verified from saved physicsScene; 480 Hz; RTX 3090 rendering; USD publication; fixed base; self collision enabled',
              'simulation_dependencies':'project .conda/isaac-apfg and third_party/isaacsim; no original PyBullet server dependency'}
    gates={g:latest[1]['checks'][g]['status'] for g in MANDATORY}
    gates['H8']=repeat['status'];gates['H6']='NOT_RUN'
    coordinates=[r for r in runs if r[2]['args']['test']=='commands' and r[1]['checks'].get('coordinate_commands',{}).get('status')=='PASS']
    rejected=[r for r in runs if r[2]['args']['test']=='move' and 'IK_UNREACHABLE' in r[1]['checks'].get('coordinate_move',{}).get('reason','')]
    coordinate_validation={'status':'NOT_RUN'}
    if coordinates and rejected:
        rejected_commands=json.loads((rejected[-1][0]/'commands.json').read_text())
        coordinate_validation={'status':'PASS' if not rejected_commands else 'FAILED','positive_run_id':coordinates[-1][0].name,
            'positive_results':coordinates[-1][1]['checks']['coordinate_commands'],'unreachable_run_id':rejected[-1][0].name,
            'unreachable_request_result':rejected[-1][1]['checks']['coordinate_move'],'rejected_move_commands_executed':len(rejected_commands)}
    summary={'status':'PASS' if repeat['status']=='PASS' and all(x['status']=='PASS' for x in recording) else 'FAILED','scope':'SR5 hardware fixture verification; H6 GraspNet model inference remains NOT_RUN',
             'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gates':gates,'repeatability':repeat,'versions':versions,
             'recording_validation':recording,
             'coordinate_interface_validation':coordinate_validation,
             'source_urdf_sha256':audit['source_urdf_sha256'],'usd_sha256':audit['usd_sha256'],'usd_path':audit['usd_path'],
             'assumption_status':audit['assumption_status'],
             'graspnet_pose_interface':{'status':'PASS','validation':'Two executed CPU tests for upstream optical/TCP axis mapping and invalid width/rotation rejection; model inference NOT_RUN'},
             'cpu_tests':{'status':'PASS' if (OUT/'unit_tests.txt').read_text().rstrip().endswith('OK') else 'FAILED','count':13,'log_sha256':sha(OUT/'unit_tests.txt')},
             'runs':[r[1] for r in selected],
             'historical_run_counts':{'total_with_smoke_results':len(runs),'failed_with_smoke_results':sum(r[1]['status']=='FAILED' for r in runs)},
             'limitations':['PGIA and D435i source box approximations; mount/TCP/optical extrinsics uncalibrated',
                            'SR5 mesh convex hull collisions; no real-world friction/force identification',
                            'Known-centred 40mm/100g fixture, not general learned grasp or clutter extraction',
                            'No GraspNet checkpoint or inference worker in hardware snapshot; pose mapping interface only']}
    PUBLIC.mkdir(parents=True,exist_ok=True)
    (PUBLIC/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'logs').mkdir(exist_ok=True)
    (OUT/'logs/smoke_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'software_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    (PUBLIC/'assembly_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    p=latest[0]
    alias(p/'images',OUT/'images')
    for group,name in [('joints','arm_joint_smoke'),('gripper','gripper_smoke'),('grasp','grasp_smoke'),('push','push_smoke')]:
        for camera in ['third','wrist']:
            alias(p/f'videos/{group}/{camera}.mp4',OUT/f'videos/{name}{"_wrist" if camera=="wrist" else ""}.mp4')
    manifest=[]
    for folder,d,cfg,_ in selected:
        for name in ['smoke_results.json','telemetry.json','commands.json','contacts.json','camera_validation.npz','run_config.json','scene_snapshot.usd']:
            if (folder/name).is_file():manifest.append(receipt(folder/name))
        for f in sorted((folder/'videos').rglob('*')):
            if f.is_file():manifest.append(receipt(f))
        for f in sorted((folder/'images').glob('*')):
            if f.is_file():manifest.append(receipt(f))
    manifest.extend([receipt(OUT/'assembly_audit.json'),receipt(ROOT/audit['usd_path']),receipt(OUT/'unit_tests.txt')])
    source_assembly=json.loads((ROOT/'assets/imported_sr5/isaac_sr5_export/assembly_current.json').read_text())
    metadata={'model':'SR5-5/0.9C + PGIA-140-80-W + D435i','source_urdf':audit['source_urdf'],
              'source_urdf_sha256':audit['source_urdf_sha256'],'vendor_source_sha256':source_assembly['source_sha256'],
              'plate_source_sha256':source_assembly['plate_sha256'],'source_to_usd':audit['source_to_usd'],
              'local_usd':audit['usd_path'],'local_usd_sha256':audit['usd_sha256'],'assumption_status':audit['assumption_status'],
              'redistribution':'Meshes, URDF snapshot and USD kept project-local; upstream redistribution rights not inferred from supplied snapshot'}
    (ROOT/'assets/manifests/sr5_assembly.json').write_text(json.dumps(metadata,indent=2)+'\n')
    previews=[]
    for source,name in [('front','assembly'),('tool_closeup','tool'),('tcp_axes','tcp_axes'),('grasp_lifted','grasp'),('push_final','push'),('camera','wrist')]:
        original=p/f'images/{source}_rgb.png';dest=PUBLIC/(name+'.jpg')
        im=Image.open(original).convert('RGB');im.thumbnail((800,600));im.save(dest,quality=85)
        previews.append({'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'source_path':str(original.relative_to(ROOT)),'source_sha256':sha(original),'encoding':'JPEG85 from actual sensor RGB'})
    sheet=Image.new('RGB',(960,760),'#20252a');draw=ImageDraw.Draw(sheet)
    for i,(name,label) in enumerate([('assembly','SR5 assembly'),('tool','Source PGIA / D435i / real plate'),('grasp','Physical lift: 40mm / 100g fixture'),('push','Physical push, no object pose resets')]):
        im=Image.open(PUBLIC/(name+'.jpg'));im.thumbnail((480,350));x=(i%2)*480;y=(i//2)*380
        sheet.paste(im,(x,y));draw.text((x+10,y+354),label,fill='white')
    sheet.save(PUBLIC/'contact_sheet.jpg',quality=85)
    (PUBLIC/'previews.json').write_text(json.dumps(previews,indent=2)+'\n')
    (PUBLIC/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=['# SR5 硬件仿真实测报告','',f'硬件范围状态：{summary["status"]}。GraspNet 模型推理 H6：NOT_RUN。','',
          '实际版本：Isaac Sim 4.5 / Isaac Lab v2.0.2 / Python 3.10 / RTX 3090。源 URDF 未修改，PGIA/相机外形与外参仍为源快照近似。','',
          '| Gate | 状态 |','|---|---|']+[f'| {g} | {v} |' for g,v in gates.items()]
    rows+=['','| 运行 | Run ID | 状态 |','|---|---|---|']+[f'| {d["mode"]} | `{d["run_id"]}` | {d["status"]} |' for _,d,_,_ in selected]
    h=latest[1]['checks'];rows+=['',f'夹持最小持续抬升 {h["H5"]["minimum_sustained_lift_m"]:.6f} m，保持 {h["H5"]["hold_duration_s"]:.2f} s，双指接触 {h["H5"]["bilateral_contact_frames"]}/{h["H5"]["hold_frames"]} 帧。',
        f'推动位移 {h["H7"]["delta_m"]} m，有真实机器人接触。',
        f'深度反投影表面距离 p95={h["H4"]["camera"]["surface_distance_p95_m"]:.6f} m。',
        f'重复运行关节最大差 {repeat["max_joint_trajectory_delta_rad_or_m"]}；物体轨迹最大位置差 {repeat["max_object_trajectory_position_delta_m"]} m。','',
        '实际命令：','```bash','bash scripts/python.sh scripts/import_sr5.py --headless','bash scripts/run_sr5.sh --headless --test all','bash scripts/run_sr5.sh --gui --test all --interactive','bash scripts/run_sr5.sh --headless --test commands --command-file configs/sr5_coordinate_example.json','```','',
        '详细装配、坐标及近似参数见 docs/SR5_HARDWARE_ZH.md。视频及原始 RGB-D/遥测仅在项目本地；evidence/sr5_hardware/artifact_manifest.json 记录相对路径与 SHA256，不能当作远程下载链接。','',
        '失败、旧绑定姿态渲染和固定锚点问题的运行均保留；最终运行显式验证 USD 渲染状态与 PhysX 一致。未训练模型，未验证一般物体/杂乱提取或实机控制。']
    report='\n'.join(rows)+'\n'
    (ROOT/'docs/SR5_HARDWARE_REPORT_ZH.md').write_text(report)
    (OUT/'README_RUN.md').write_text(report)
    print(json.dumps({'status':summary['status'],'run_ids':[r[0].name for r in selected],'repeatability':repeat},ensure_ascii=False))
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
