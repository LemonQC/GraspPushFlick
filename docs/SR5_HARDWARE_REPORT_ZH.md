# SR5 硬件仿真实测报告

硬件范围状态：PASS。GraspNet 模型推理 H6：NOT_RUN。

实际版本：Isaac Sim 4.5 / Isaac Lab v2.0.2 / Python 3.10 / RTX 3090。源 URDF 未修改，PGIA/相机外形与外参仍为源快照近似。

| Gate | 状态 |
|---|---|
| H0 | PASS |
| H1 | PASS |
| H2 | PASS |
| H3 | PASS |
| H4 | PASS |
| H5 | PASS |
| H7 | PASS |
| render_state_sync | PASS |
| simulation_clock | PASS |
| rgbd | PASS |
| H8 | PASS |
| H6 | NOT_RUN |

| 运行 | Run ID | 状态 |
|---|---|---|
| headless | `20261121T073140Z_1cc305cc` | PASS |
| headless | `20261121T073346Z_dc62fe18` | PASS |
| gui | `20261121T073141Z_b7341988` | PASS |

夹持最小持续抬升 0.149991 m，保持 1.00 s，双指接触 30/30 帧。
推动位移 [0.14199185371398926, -0.0004918311024084687, 9.443610906600952e-07] m，有真实机器人接触。
深度反投影表面距离 p95=0.000420 m。
重复运行关节最大差 0.0；物体轨迹最大位置差 0.0 m。

实际命令：
```bash
bash scripts/python.sh scripts/import_sr5.py --headless
bash scripts/run_sr5.sh --headless --test all
bash scripts/run_sr5.sh --gui --test all --interactive
bash scripts/run_sr5.sh --headless --test commands --command-file configs/sr5_coordinate_example.json
```

详细装配、坐标及近似参数见 docs/SR5_HARDWARE_ZH.md。视频及原始 RGB-D/遥测仅在项目本地；evidence/sr5_hardware/artifact_manifest.json 记录相对路径与 SHA256，不能当作远程下载链接。

失败、旧绑定姿态渲染和固定锚点问题的运行均保留；最终运行显式验证 USD 渲染状态与 PhysX 一致。未训练模型，未验证一般物体/杂乱提取或实机控制。
