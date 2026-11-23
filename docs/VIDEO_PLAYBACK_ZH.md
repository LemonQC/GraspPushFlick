# 视频黑屏诊断与播放修复

本次复查定位到桌面解码缺失：系统默认 Totem 使用 GStreamer，实际加载 M1 MP4 时报告 `Missing decoder: H.264 (High Profile)`。系统没有 `avdec_h264`、`vaapih264dec`、`openh264dec`，已有 VP8 解码器。项目内 FFmpeg 能正常解码相同文件。因此本次没有重新安装环境、修改系统解码器或重跑物理实验。

逐帧复查了当前18次M1运行的54个MP4，共33,168帧，未检出黑帧，所有文件内容检查PASS。范围包括用户新增的GUI C1运行 `20261123T014238Z_6b829661`。这是视频内容审计，不是新增实验成功率；不能替代各运行原有的物理验收与失败记录。

## 直接播放

```bash
# 默认显示最新已完成M1场景的第三人称录像
bash scripts/play_video.sh
# 指定真实GraspNet完整取出录像
bash scripts/play_video.sh outputs/m1/runs/20261122T095317Z_45c90433/videos/m1/third.mp4
# 查看最新腕部录像
bash scripts/play_video.sh --camera wrist
# 仅生成副本，打印路径；可以在桌面播放器打开打印出的WebM文件并暂停/拖动
bash scripts/play_video.sh outputs/m1/runs/20261123T014238Z_6b829661/videos/m1/third.mp4 --prepare-only
```

脚本从实际原始帧生成VP8/WebM，保持分辨率、帧率与帧数；它不是新的仿真证据。原始MP4及其SHA256保持不变。副本按原件内容哈希缓存于项目内 `outputs/video_playback/<hash>_<camera>/preview.webm`，验证结果及原件/副本哈希在同目录 `validation.json`。直接播放使用已有 `gst-launch-1.0`、VP8与 `ximagesink`，不依赖远程桌面的GL视频表面，继承当前DISPLAY，不修改桌面配置。

已生成用户C1与完整取出运行的第三人称/固定/腕部六个副本；六个均保持原帧数并通过内容验证。系统GStreamer已实际解码WebM通过（退出码0），且原生X11播放窗口已实际显示SR5/YCB场景，截图来自该窗口。播放脚本正常播放到片尾后退出；需要暂停和拖动时，可使用已生成的WebM文件进入桌面播放器。

## 录像验收改动

原验收只检查可解码、帧数与时长，无法拒绝“可解码的全黑视频”。现在 `Recorder.close()` 调用 `validate_video()`：逐帧解码，在每帧规则像素网格上检测黑色/纯色、统计有效画面、亮度和尺寸，并检查解码帧数与实际捕获时间戳数量一致。全黑、无有效画面、帧数不符或解码失败均标FAILED。少数相机遮挡产生的黑帧单独记录，不自动解释为整个场景失败。这不是机器人运动语义识别或物理验收的替代品。

28项回归测试通过，包含全黑、全灰、帧数不符的实际编码小样测试。此类小样仅是单元测试输入，不放入实验结果，也不用于生成科研证据。

```bash
source scripts/project_env.sh
"$APFG_PYTHON" -m unittest discover -s tests -v
"$APFG_PYTHON" - <<'PY'
from apfg.visualization.video_validation import validate_video
print(validate_video('outputs/m1/runs/20261123T014238Z_6b829661/videos/m1/third.mp4'))
PY
```

精简结果/文件哈希见 `evidence/video_playback/summary.json`；完整54文件审计、解码日志和回归日志留在 `outputs/video_audit/`。旧的 `evidence/m1/` 仍描述提交 `79a77fd151fca4be1afe6d05696ab4108f1e973e` 的M1里程碑，本轮不改写其历史报告或原始运行文件。
