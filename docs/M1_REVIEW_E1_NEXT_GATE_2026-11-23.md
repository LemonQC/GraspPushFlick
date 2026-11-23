# M1 审查与下一步执行决策（2026-11-23）

> **REVIEW_ONLY / NEXT_GATE_NOT_RUN**。已核查 main @ `d51265bc76aab1d1cee45a16995a510d04f61a1b`。不是新的实验结果；任何实验失败与成功均以服务器实际运行和日志为准。

## 1. 实际证据

- M1 的 `docs/M1_REPORT_ZH.md` 有 17 次开发运行（含 4 次失败），四类候选场景、C4 配置及可重复生成、分离的 RGBD/ORACLE 观测。
- 已有真实物理 Push 和 ORACLE 下的非 yaw Flip（约 94.8°），但翻转后完整取出 NOT_RUN；Active 相机动了约 480 mm，目标可见颜色提示计数 8144→4012，**不能声称信息增益**。
- 真实 GraspNet + SR5 在 Direct 场景有完整取出 PASS；新 `parallel_jaw_execution_v1` 改善抓取前不必要旋转，GUI/Headless 再验证通过；33 CPU 回归测试 PASS。
- `docs/E1_PROTOCOL_ZH.md` + `configs/m1/e1_protocol_v2.json` 有 4 families × 2 seeds × 2 objects × 4 arms = 64 项，但仍全部 NOT_RUN。该文件冻结于控制器修复之前，不得静默覆盖。
- `scripts/run_m1.py` 提供 `grasp/push_grasp/flip_grasp/active_grasp`，但它们只是固定动作序列及参数组合，不是学习策略；其仿真控制和记录可以用于 E1。
- `apfg/primitives/m1_actions.py`: Flip 当前主要通过 `execute_stroke` 产生非 yaw 倾倒，并非已训练的独立动态模型/闭环翻转策略。C1 物体放置时的 7 mm 间隙不能直接判定落定后的抓取受阻；C2 初始绕 Y 90°不能证明落定后无法直接抓取；C3 初始化遮挡不能证明 Active 带来有效新信息。
- 当前非糖盒目标使用 workspace region 提示；黄色糖盒则使用颜色提示。对于同一 episode 各 arm 必须严格相同，但跨物体的提示差异和遮挡物误合并应在报告中标出。RGBD 策略不得读取 oracle 真实位姿/隐藏碰撞几何。

## 2. 决策：优先 E1 机制资格审查，不进入训练

只从**同一个稳定的物理初始化场景**出发，用同一观测、同一机器人和同一 GraspNet 候选预算比较：
- DIRECT_GRASP
- PUSH_GRASP
- FLIP_GRASP
- VIEW_GRASP

关键是完整取出判据，而非动作是否动了。额外 View 必须计时、计动作数，不得免费获得另一个相机视角。Push 既可能平移又可能倾倒，不能在所有被称为 Push 的运行中默认为纯平移。Flip 需报告实际姿态与新支撑状态、下一步 Grasp 是否成功；操作完成并不能推出最终目标更可抓。

**当前建议流程（属于 E1 内部审核，不是额外大型实验）：**
1. 将当前抓取控制器 revision `parallel_jaw_execution_v1`、commit SHA、相机提示、RGB-D、物理资产与候选预算写入 `configs/m1/e1_protocol_v3.json`；保留原 v1/v2 不变。采用稳定的预登记 case order（例如每 family 第一个已经冻结的 seed/target），不要根据结果修改顺序/选种子。
2. 实现自动配对 E1 runner：完整保存每个 arm 的落定初态及哈希、camera/observation、真实接触、候选及拒绝原因、真实取出结果和时长。检查不同 arm 落定初态的数值差异，若超阈值则判为 INVALID_PAIR 而非公平比较。
3. 对 predeclared 的 4 families × 1 种对象/种子 × 4 arms = 16 次执行作为 E1 校验段，不用开发单例、ORACLE Flip 或抓取姿态回归 run 充数。16个必须都做，不能按好坏选择；已不稳定场景照原样保留 INVALID 状态、不能为了使某技能有优势而更改它。若有严重控制、参数或场景失效，修复并登记新的协议/版本后整组重新比较，旧组保留。
4. 若校验段能够形成有效成对数据且说明上述机制可被区分，再按冻结 case list 做余下48次；若不成立，暂停 E1，不直接开始训练。最终只把执行充分的结果称为 E1，不声称完整64次之前已完成。
5. 生成逐实例对比表、4x4 配对 outcome matrix、失败taxonomy；完整视频本地、精选图/轻量 JSON 到 `evidence/e1/`，并 push GitHub。不要按场景名字生成应采用的技能标签，也不要因 Direct 失败就假设 Flip 必须成功。

## 3. 下一步科学决策门槛

- **C1：** 测量落定后的真实可抓空间与 GraspNet 可执行候选；至少找到存在合法移动/推动可解但直接抓取受阻的物理实例。若 Direct 全成功或 Push 只靠偶然倾倒获益，现有 C1 不是清晰的 Push 研究对象。
- **C2：** 同一物理实例必须展示 Direct 不易取出、真实 Flip 产生稳定非 yaw 姿态变化、Flip→Grasp 能完成取出。若 Direct 同样简单，Flip 研究必要性不足；若只有 ORACLE 能翻而 RGBD 不能生成安全接触，则先解决感知/原语，不先训练高层。
- **C3：** 检查 View 前后目标可见三维点、有效抓取候选/路径可行数、最终取出与动作代价。像素数下降本身不够，假阳性/串色要显式暴露。若只是改变相机视角却不能影响决定、就没有 Active 贡献。
- **Direct：** 四技能策略不能依赖永远额外动作来成功，必须可直接抓。
- **所有场景：** 若成功/失败主要由机械臂路径碰撞、相机/抓取器校准、抓取原语不稳定导致，应先修物理基线，不将控制 bug 当成需要复杂学习的科学问题。

## 4. 后续训练入口（E1 合格后才考虑）

先保留现有 GraspNet 网络不重训。不同场景收集 `(o_t, action_type, continuous_parameters, o_{t+1}, measured_outcome, eventual_retrieval)`，动作包括 View、Push、Flip、Grasp。保留动态物理后果和真实失败，在相同可观察信息下先比较直接的候选评分/简单 MPC/受控启发式；若规则/MPC 已达到同样取出率，先停止追求四技能堆叠，不批量生成数据。

可能的学习对象是 **action-conditioned outcome/value predictor + parameterized high-level selection**，而非重训低层抓取网络，也不是 C1→Push/C2→Flip/C3→View 的监督技能分类器。正式创新机制需相对 GAPG、GOMP、MS-MEM、Uni-PNP 等做原文排重与同协议强基线审查。

## 5. Codex 本轮任务边界

不要重新安装环境或下载大数据；保持现有 `/data/GraspPushFlicks` 项目和脚本、原实验、RTX3090/Isaac4.5/Lab2.0.2。先对齐 E1 v3 新控制版本，写自动配对的 E1 runner 和数据验证，执行预登记16次**作为 E1 一部分**。不做 E2/E3，不训练网络，不重新命名论文算法。若前16次全部有效且机制初步可区分，继续冻结的剩余48次；否则停止，报告 NO_GO_REASON。全部原始视频/资产在项目目录，报告和轻量证据到 GitHub；给出 commit SHA、实际每arm结果、失败原因、磁盘占用及是否能启动下一阶段。
