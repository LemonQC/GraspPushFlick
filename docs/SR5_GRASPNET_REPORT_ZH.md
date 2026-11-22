# SR5 GraspNet 6-DoF 实测报告

状态：真实模型推理 PASS，三维候选显示 PASS，Headless / GUI 物理抓取 PASS。此前固定姿态硬件夹持不是本报告的 GraspNet 抓取。

| Run ID | 模式 | 原始候选/显示 | 选中倾角 | 物理结果 |
|---|---|---|---|---|
| `20261122T033838Z_f29ff73c` | headless | 368/20 | 86.95° | PASS |
| `20261122T034038Z_e813e7a9` | gui | 372/20 | 15.86° | PASS |
| `20261122T034626Z_e99eeb0e` | gui | 372/20 | 15.86° | NOT_RUN |

侧向抓取记录最小持续抬升 0.150216 m，保持 30 帧/1 秒，双指接触 30/30 帧；没有人工附着或物体位姿重置。
实际模型输入来自腕部 RGB-D 工作区点云，未使用实例分割或物体真值。预测原始姿态、端平面/TCP 几何映射、筛选理由、控制及接触均保存在原始运行目录。

查看候选：
```bash
cd /data/GraspPushFlicks
bash scripts/run_sr5_graspnet.sh --gui --preview-only --interactive --joint-speed .3
```
执行一次真实预测抓取：
```bash
bash scripts/run_sr5_graspnet.sh --gui --interactive --joint-speed .3
```

完整 MP4、RGB-D 和网络数组仅保存在服务器 `outputs/sr5_graspnet/<run_id>/`，公开 manifest 仅给出相对路径与 SHA256，不提供远程视频下载。
失败运行保留：20261122T033519Z_f295530c.

本次官方 Google Drive 下载配额已用尽，使用文档记录的公开 checkpoint-rs 镜像。LFS 哈希和模型参数匹配已验证，未独立验证与官方 Drive 原件哈希相等。详细来源、近似外参和执行范围见 `docs/SR5_GRASPNET_ZH.md`。
