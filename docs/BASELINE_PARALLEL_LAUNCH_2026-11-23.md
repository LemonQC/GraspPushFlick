# APFG Baseline Parallel Launch v1（2026-11-23）
**Status: DESIGN_ONLY / BASELINES_NOT_RUN.** 目标：在已经验证的 M1 SR5 / Isaac 4.5 / RTX 3090 环境中并行构建启发式、DQN、BC、Diffusion Policy (DP) 基线；不宣称已训练或 T-RO 创新。来源审计见本项目 `docs/M1_REPORT_ZH.md`, `docs/E1_PROTOCOL_ZH.md`, `docs/GRASP_MOTION_ZH.md`。

## 0. 先确认已存在的事实
- M1 独立 Active/Push/Flip/Grasp smoke 均有实测，17 个 M1 开发运行含4个失败；E1 正式比较未执行。
- 当前 Git main 审查基于 `d51265bc76aab1d1cee45a16995a510d04f61a1b` 的控制器：`parallel_jaw_execution_v1`。有过手腕大幅旋转问题，**不可将旧版本结果与新版本混入同一 baseline 对比**。
- Flip 目前为接触点参数化的物理推动导致的倾倒，不是学习得到的闭环姿态策略。Active 在开发样例里目标像素数并未证明信息改善。不可把 family C1/C2/C3 当成真实最优技能标签。
- 用户要求所有新环境、依赖、缓存、模型、数据、录像留在当前项目目录；源码、配置、可复现报告及小型证据同步 `LemonQC/GraspPushFlick`，大文件和视频不进 Git。
- 不重装或破坏已有 Isaac 4.5 + Isaac Lab 2.0.2 + Python 3.10 + torch 2.5.1 + SR5；增量依赖隔离，在不冲突的情况下复用原环境。

## 1. Baseline 分层，不要混淆

### Track A：**严格同候选、同动作能力、同信息** 的技能层比较（优先主表）
1. `B0_GRASP_ONLY`: 当前 GraspNet 最优可行候选直接完整取出。
2. `B1_HEURISTIC`: 明示几何/观测规则评分 **同一套候选动作**，可选 Direct、Active、Push、Flip；连续接触/方向/行程由候选池给定。不能偷看 family 标签或 oracle，也不许规则按测试 seed 分支。
3. `B2_RANDOM_VALID`: 合法候选均匀随机 + 相同失败/时间上限，作为调试下界。
4. `B3_DQN`: **离散候选 DQN/Double DQN**，对当步合法候选计算价值；动作 mask 禁用不合法候选；动作池需具有跨时间稳定编码或使用 `Q(obs,candidate)` 结构。若采用候选条件 Double-DQN，要明确不是未改的固定离散维度 SB3 DQN，提供算法公式、TD target、target-network、replay、epsilon schedule、seed。
5. `B4_BC`: 使用 **同一示范集** 的模仿学习，候选分类；对示范动作的参数监督可选，但必须使用相同合法动作支持。
6. `B5_DIFFUSION_HIGH_LEVEL`: 从与 BC 相同的示范中学习条件高层动作序列或连续动作参数；执行前经完全相同的合法候选映射/检查。它是**DP 高层适配实现**，不是照搬论文默认端到端连续末端轨迹的“原版 DP”；保留网络/条件/去噪/时间窗的差异说明、模型 checkpoint 来源。

### Track B：**原生动作形式** 的已发表方法对照（在可用数据和工程适配后）
- Diffusion Policy 原版/合理保真改写（https://github.com/real-stanford/diffusion_policy）：历史 RGB 或其他观测 + 连续 EE/夹爪 action chunk，需专用高频多模态示范；不可以把任意高层类别网络命名为“官方 DP 复现”。
- 3D Diffusion Policy (DP3, https://github.com/YanjieZe/3D-Diffusion-Policy)：点云 + 本体状态到连续动作；同样要有高频连续轨迹，**不等于**直接能操纵4个离散操作类型。若做不了，应明确 NOT_RUN 和缺少的数据/接口。
- P-DQN（混合离散技能类别 + 连续参数）和参数化 SAC/PPO 是第二批。不要仅为增加算法数量投入大量无意义训练。
- 已发表强竞争工作的适配审计：GAPG、GOMP(2026)、MS-MEM(2026)、Uni-PNP(2026)、SPIN；记录任务差异、源环境和未复现原因。**不要在未做官方实现或忠实改写的情况下把启发式替代品标记为这些论文结果。**

Track A 严控候选池、视角、预算和控制器；Track B 允许原生动作空间，必须说明不再是完全隔离动作表示的对比。

## 2. 统一环境和动作/观测规范（所有 Track A 共用）
- 同一 `reset(seed,scene_id)`, `observe()`, `legal_actions()`, `step(action)`, `is_success()`, `render()`。
- 每个 episode 按物理落定后真实稳定状态初始化；训练/评测**只能在 episode 之间 reset**，不得 rollout 中重置物体和虚构抓取约束。
- 观测只含RGB-D / K/T / robot proprio / 目标可见提示 / 历史 / **当前可见点云**；所有候选方法使用同等信息。Oracle 仅场景生成、物理标签、诊断、评价，绝不可作为 RGBD 策略隐藏输入。测试一个 poison-oracle 回归。
- 每个候选结构：`type`、接触/视点/六维抓取及参数、valid-mask、估计可达与碰撞、candidate_id、生成器版本与输入哈希。动态候选池保持逻辑一致；候选数可限制但不允许按方法不同。初步最多24–32个候选/决策，实际按吞吐量和有效率调整并冻结；候选中至少包含合法Grasp/Active/Push/Flip（没有合法的不强行填充）。
- 物理动作：Active 真正移动相机；Push 参数化；Flip 真实非yaw姿态/支撑变化；Grasp 复用相同权重 GraspNet、`parallel_jaw_execution_v1` 同一候选执行器。
- Terminal: 真实**完整目标取出+保持**，不是只抬高或只完成准备动作。失败分类：no visible target, no legal candidate, IK/path, grasp miss, flip unstable, dropped target, exceeded max prep actions, timeout, severe distractor movement, etc. 使用相同最多准备动作数（初步1用于与E1对齐，长序列另一个协议）。
- 主奖励：最终完整取出 +1；失败0或固定负值；准备动作时间和非目标扰动小罚；记录各项原值。塑形信号如抓取数变化必须单独做消融，不能按 family C1/C2/C3 制造标签。
- 每步记录RGBD/点云或其确定性引用、动作候选池、原始动作选择、执行结果、接触/姿态及物理评价、时间、显存峰值。决策前和动作后的图像必须真实刷新。
- 实验组别：Direct、C1、C2、C3，Mix C4 最初只验证生成，不当作正式泛化成绩；另建立 held-out 对象与场景 seed。**训练与正式测评 seed、对象实例互斥；YCB/GraspNet 同一对象不能跨训练测试伪作“unseen”。**
- 准备动作的机会成本是真实动作、仿真时长、推理/规划时延；所有方法一律同口径。

## 3. 数据生成与 DP 资格条件
- **DP/BC必须先有 demonstrator 数据**。来源可为成功的几何/启发式/规划 rollouts + 人工遥操作补充；每个示范应含完整状态观察、动作类型、连续参数、原生关节与 TCP 轨迹、夹爪状态、RGBD、相机 K/T、最终真实任务成败、精确 timing / version。保留失败用于 offline Q/诊断，但监督 BC/DP 的正例来源必须明示；示范质量不佳，DP 训练失败**不能**归咎于 DP 方法能力。
- 为避免 DP 只是克隆同一条启发式策略，保留异质专家/人工纠错及不同成功序列；标记来源分层、验证多模态分布和是否包含真实 View/Push/Flip transitions。
- 对于 high-level DP，允许结构化动作 token + 连续参数、再映射合法候选；清楚披露“DP 高层改写”，不要宣传为完整原始 DP3。
- 对于 native DP/DP3，必须采集足够频率的连续关节/TCP/夹爪轨迹；只记录每次宏动作结束后的单张图/类别不足以训练原生 DP3。
- 输出 dataset manifest + partition IDs + episode hashes + failed/success counts + length histograms + action diversity + actual observation dimensions。先产生十数条**真实成功**和若干失败作数据接口 smoke，下一阶段再增量扩充，禁止预先杜撰大数据规模或预估成功率。
- 若 16 个配对试验没有出现有效准备动作后取出收益，继续数据协议/系统验证可以，但不得大规模生成只会‘直接抓’的演示数据让 DP 虚假取得高分。

## 4. 并行开发、串行 GPU 的资源策略（实机3090、CPU PhysX）
- **可并行实现三个代码分支**：环境/规则与E1；DQN/replay；demo录制+BC/DP dataset和trainer。GPU/Kit 运行需要资源协调：同时启动多个 Isaac RTX GUI 进程与大型训练可能 OOM；用项目内共享显存/进程锁，默认同一时刻一个 GPU 负载任务；训练可复用离线缓存、轮流申请 GPU。‘并行验证’指开发和实验工作流同时启动，不是无限 GPU 并发。
- 首先测 `env reuse`（一个Isaac进程多episode reset）、头less低分辨率决策帧、低频视频（只审计代表性episode，所有失败保留轻量遥测）；统计sims/sec, episode_wall_s, transitions/hour, disk per episode, GPU peak。不能改变真实物理步长/碰撞设置以虚假提高成功率。
- 每条路线设单独 `train.py` / `eval.py` 和独立 config/checkpoints/replay；不允许从同一评测集取训练标签。
- DQN先最小 replay/TD回归、mask 单元测试，启动非常小的 sanity online train；BC在少量真实示范上过拟合 sanity；DP先完整 dataset loader/forward/backward/denoise+checkpoint可训练性，再在真实有一定数量有效演示后进行有限训练。**三者评估时至少使用完全相同 holdout scene IDs。**
- E1 v2 已冻结但**先于 `parallel_jaw_execution_v1` 控制器修改**。先另存 E1 v3，注明控制器 SHA 与不可比点；不覆盖 v1/v2；用固定16-run paired subset 检查实际机制，再决定是否继续余下48。

## 5. 交付与代码组织
建议增量新增：
- `apfg/baselines/common_env.py`, `candidate_generator.py`, `reward.py`, `dataset.py`, `evaluation.py`;
- `apfg/baselines/heuristic.py`, `dqn.py`, `bc.py`, `dp_adapted.py`;
- `configs/baselines/common.yaml`, `heuristic.yaml`, `dqn.yaml`, `bc.yaml`, `dp.yaml`；
- `scripts/run_baseline_smoke.sh`, `scripts/train_baseline.sh`, `scripts/eval_baseline.sh`；
- `docs/BASELINE_IMPLEMENTATION_MATRIX.md` 标注 OFFICIAL, ADAPTED, FROM_SCRATCH, NOT_RUN，源commit和差异；
- `docs/BASELINE_PARALLEL_REPORT_ZH.md`，`evidence/baselines/`（只有可审计的真结果）。
兼容现有 `apfg/simulation/m1_env.py`、`scripts/run_m1.py`，不擅改之前研究成功标准。
- 仿真大数据、replay、checkpoint、完整MP4仅保存在当前目录 `outputs/baselines/`，不提交 Git；小型实验记录与预览上传 GitHub。
- 所有统计必须带 denominators、scene IDs、seed、动作预算、success/failure、walltime、版本与测试结果；结果只标记 PASS/FAILED/BLOCKED/NOT_RUN，不把静态预测准确率称作物理成功。

## 6. 初始实验进度门槛（不是最终论文标准）
Gate-0: shared env/API、示范数据格式、action-mask及物理/观测边界测试 PASS。
Gate-1: 规则 baseline 和 E1 v3 固定16-run 验证，全部含失败。若同实例对比失效或控制问题主导，则停止正式E1，并报告问题，不能随意挑 seed。
Gate-2: DQN mini-training（TD loss下降不是必然成功，需要至少真实动作采样、replay更新、heldout物理rollout及日志）+ BC small-demo overfit + DP一次真反向传播和复现预测。分别做可用性测试；不要求三者成绩都好。
Gate-3: 如E1和数据有效，再按照预登记的数据量逐渐扩展DQN训练与BC/DP，做相同holdout test；多随机种子和置信区间仅在正式评测时报告。
未过关者继续如实标记 `NOT_READY`，不捏造研究收益，不先开发所谓 Ours。先明确近期 GAPG、GOMP、Uni-PNP、MS-MEM、SPIN 的方法重合与任务差异。
