# GraspPushFlick / TRO-APFG

指定目标的三维杂乱环境取出研究：Active、Push、真实 Flip、6-DoF Grasp。已建立 M0 环境、SR5 硬件仿真与 M1 接口，并按 com3.md 开展 Baseline 小规模真实训练和物理测试，不宣称最终算法贡献。

**Baseline 实际链路已运行，规模化训练 NOT_READY。** E1 v3 首批16次全部完成、初态配对有效，但没有准备动作的额外取出收益，剩余48次未扩展。DQN完成4次真实online交互/6条Replay/160次更新；BC和高层DP在相同6个成功episode上各完成200次更新。统一GUI的两个heldout实例结果：Grasp Only 1/2、Heuristic 1/2、Random 0/2、DQN 2/2、BC 1/2、DP 0/2。这是链路smoke，不是论文方法排名。详见 [完整报告](docs/BASELINE_PARALLEL_REPORT_ZH.md)、[实现身份矩阵](docs/BASELINE_IMPLEMENTATION_MATRIX.md) 和 `evidence/baselines/summary.json`。

```bash
# 直接查看DQN在汤罐上的真实 Active→Grasp 完整取出录像
bash scripts/play_video.sh outputs/baselines/campaigns/20261124T025638Z_c38f8690/episode_007/videos/m1/third.mp4
# 训练与物理测评入口；完整精确参数和模型路径见报告
bash scripts/train_baseline.sh --help
bash scripts/eval_baseline.sh --help
```

**M1 专家场景与统一接口已实际运行。** Direct、C1空间不足、C2不利姿态、C3信息不足，以及C4混合配置均可查看；每类是待验证的机制假设。独立推动、约94.8°真实翻转（显式ORACLE诊断）、RGBD GraspNet完整取出已通过开发smoke。原17次运行含4次失败，见 [历史M1报告](docs/M1_REPORT_ZH.md)、[原E1协议](docs/E1_PROTOCOL_ZH.md) 与 `evidence/m1/summary.json`。本轮新增E1v3和训练结果另存Baseline报告；E2/E3与规模化训练仍NOT_RUN。

```bash
# 查看本轮专家设计的候选场景；也可换成 Direct / C2 / C3 / C4
bash scripts/run_m1.sh --gui --family C1 --action observe --interactive
# 真实GraspNet六维候选、实际夹持并完整移出任务区
bash scripts/run_m1.sh --gui --family Direct --action grasp --interactive
# 非yaw物理翻转；本次使用显式诊断接触点
bash scripts/run_m1.sh --gui --family Direct --action flip --information-mode ORACLE_DIAGNOSTIC --interactive
```

每次自动记录第三人称/固定/腕部视频、RGB-D与物理遥测到项目内 `outputs/m1/runs/<run_id>/`。影片仅在服务器，GitHub含真实关键帧与哈希清单。

若 Ubuntu 默认视频播放器显示黑屏：当前系统 GStreamer 缺少 H.264 解码器。使用 `bash scripts/play_video.sh` 查看最新已完成场景；也可传入具体 MP4 路径。脚本使用已有项目内 FFmpeg 生成 VP8/WebM 播放副本，以 X11 窗口播放，不改系统软件或原始录像。详见 [播放诊断与验证](docs/VIDEO_PLAYBACK_ZH.md)。

抓取前多余转腕已修正：原始GraspNet预测保持不变，单独选择经过完整装配检查的较近双指执行朝向，并选择限位内较近的2π关节表示。GUI实际调整由170.2°降至21.8°；M1侧向抓取由178.4°降至61.1°，真实夹持与完整取出分别通过。见 [控制修正实测](docs/GRASP_MOTION_ZH.md)。

SR5 + PGIA-140-80-W + D435i 的装配、坐标运动与物理夹持/推动入口见 [硬件说明](docs/SR5_HARDWARE_ZH.md)，实际分项结果见 `evidence/sr5_hardware/summary.json`。真实 SR5/转接板保持源网格；PGIA/相机外形及外参为源快照近似。

**真实 GraspNet 6-DoF 推理和三维抓取位姿显示**使用独立入口，见 [GraspNet 说明](docs/SR5_GRASPNET_ZH.md) 与 `evidence/sr5_graspnet/summary.json`。此前的硬件 `--test grasp` 是已知坐标夹持测试，不能当作 GraspNet 感知抓取。

```bash
bash scripts/run_sr5_graspnet.sh --gui --preview-only --interactive --joint-speed .3
bash scripts/run_sr5_graspnet.sh --gui --interactive --joint-speed .3
```

```bash
bash scripts/run_sr5.sh --gui --test all --interactive
bash scripts/run_sr5.sh --headless --test move --xyz 0.45 0.04 0.32 --quat-xyzw 1 0 0 0
```

**APFG_M0_READY：RTX 3090 / Isaac Sim 4.5 替代组合的 M0 核心验收与 GitHub 同步均已通过。** 最新验收状态见 `evidence/m0_status.json` 和 [M0 报告](docs/M0_REPORT_ZH.md)。此前 CUDA/UVM 故障已在本次复查中恢复，项目没有升级或重载驱动。

实际验证组合：RTX 3090 24 GiB、Isaac Sim 4.5 二进制、Isaac Lab v2.0.2、Python 3.10.22、PyTorch 2.5.1/cu118。原定 5.1 pip 组合受本机 Ubuntu 20.04 / GLIBC 2.31 限制，不能把本轮结果称为 5.1 验证。

已准备 30 种官方 YCB，13 种完成独立物理验证。Franka 七轴/双指、固定和腕部 RGB-D、目标点云、S1 密集桌面与 S2 自然掉落围框堆叠均实际运行。GUI 和 Headless 共享代码；物理 120 Hz、视频 30 FPS，并核对 SDK 的真实步数与时间。GUI 场景使用默认 1280×720，部分 Headless 测试显式使用 640×480。

```bash
cd /data/GraspPushFlicks
bash scripts/run_gui.sh --test clutter --scene s1 --interactive
bash scripts/run_gui.sh --test clutter --scene s2 --interactive
bash scripts/run_gui.sh --test robot --interactive
bash scripts/run_headless.sh --test camera --offline
bash scripts/run_m0.sh
```

GUI 继承当前桌面的 DISPLAY/XAUTHORITY，支持原生旋转视角、相机选择、暂停和退出。S2 使用记录在案的真实静态围框，物体通过重力和接触形成堆叠；没有 rollout 中的物体状态重置。

所有安装、资产、缓存、完整视频与原始日志在项目内。`project_env.sh` 不修改父进程 HOME 或 .bashrc；Kit 使用 portable-root，仿真子进程使用独立 HOME。CUDA、GL 和 OptiX 缓存分别显式配置。内存 IPC、字符设备和线程命名操作与持久磁盘写入分开记录。

完整结果在 `outputs/environment_check/runs/<run_id>/`。便捷文件包括 `robot_motion.mp4`、`object_physics.mp4`、`clutter_preview.mp4`、`wrist_camera.mp4` 和 RGB-D 图片；原始运行文件不覆盖。完整视频仅保存在服务器，GitHub 提交代码、来源/哈希、精简指标和以下真实关键帧。

![Franka 与真实 YCB 桌面](evidence/previews/tabletop.jpg)
![自然形成的三维堆叠](evidence/previews/pile.jpg)

[环境](docs/ENVIRONMENT_SETUP_ZH.md) · [资产](docs/ASSET_MANIFEST_ZH.md) · [复现](docs/REPRODUCIBILITY.md) · [研究范围](docs/BASELINE_SCOPE_ZH.md)

### 局部密集场景与动作后目标关联 v2（2026-11-24）

新增7件真实YCB局部场景、传感器时序关联、近邻移障Push、多Grasp/Active/Flip候选；保留旧E1 v3全部冻结源码和模型。实测有Active后取出、真实移障和91.6°卧倒翻转后取出，但成功Flip场景的Direct也成功，机制资格仍未齐备，不扩大训练。详见[完整开发报告](docs/CLUTTER_V2_ASSOCIATION_REPORT_ZH.md)及[真实证据](evidence/clutter_v2/summary.json)。

```bash
bash scripts/play_clutter_v2.sh --list
bash scripts/play_clutter_v2.sh active
bash scripts/play_clutter_v2.sh flip --run-id 20261124T075922Z_9e0cf087 --camera fixed
```

完整视频仅在服务器 `outputs/clutter_v2/`，GitHub只有关键帧/清单；v2候选schema不兼容旧基线checkpoint，未硬加载或重训。
