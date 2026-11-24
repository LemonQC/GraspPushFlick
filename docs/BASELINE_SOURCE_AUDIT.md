# 已发表方法的一手来源与适配审查（2026-11-23）

本文件是来源/任务审查，不是这些方法在本项目的实验结果。各方法实际复现均 **NOT_RUN**。没有把本项目启发式、DQN或DP高层改写重新命名为论文方法，没有下载其权重/数据或安装其环境。

| 方法与一手来源 | 已核验的核心任务/形式 | 本项目相关性与适配缺口 | 代码/版本与本轮状态 |
|---|---|---|---|
| [GAPG官方代码](https://github.com/xiaolijz/GAPG)，[论文2603.21195](https://arxiv.org/abs/2603.21195) | 点云几何评估抓取，预测推动对后续可抓空间的作用，面向杂乱中的目标操作 | 与Push→Grasp目标高度相关；不能把普通“有候选就抓”称为GAPG。需把其抓取/推动评价网络与原生数据流程移植到同一SR5执行器，或单列原生动作对照。官方README测试栈为Python3.8/PyBullet3.2.7，另需PyTorch3D等依赖；不替换现有Isaac环境。 | HEAD `1bfd4c040715e606ba7a2668697b9acd189dfff8`，仅`git ls-remote`核对；没有复制代码/运行模型。**NOT_RUN**。 |
| [GOMP官方项目](https://zlicheng.com/gomp_page/)，[官方代码](https://github.com/Colmar-zlicheng/GOMP) | 学习物体状态上的可抓取性场，用于非抓取重配置与何时转入抓取；ECCV2026 | 与“不利姿态→重配置→抓取”直接重合；需要任务终止从物体进入可抓区域/抓取提升，适配到指定目标完整取出。原版教师/学生与其几何数据、SE(3)-Dif参考抓取不能被现有GraspNet启发式替代。 | HEAD `a7f1b99f1e87e9562246f856b77ad939a95cce92`；官方README要求Python3.11、IsaacSim5.0、IsaacLab `e4d644f2b1ffccd184b909e8d65a83d86847a6ea`及额外数据/权重。与当前4.5/2.0.2不同，未安装。**NOT_RUN**。 |
| [MS-MEM论文2609.02493v1](https://arxiv.org/html/2609.02493v1) | 统一评估主动视点、推动、抓取；证据式不确定性地图、信息收益与旁侧扰动约束 | 与Active+Push+Grasp及扰动约束明显重合；评价重点含场景映射/不确定性，不能直接当作本项目完整指定目标取出分数。移植需要真实belief/不确定性表示、动作后预测和信息收益准则。简单RGB像素数变化不是其复现。 | 所查论文/官方关联页面未提供可核验的实现commit；没有假设其仿真器/代码栈。**NOT_RUN**，待官方可复现实现与任务接口审查。 |
| [Uni-PNP论文2610.21735](https://arxiv.org/abs/2610.21735)，[官方项目](https://lgswdn.github.io/Uni-PNP/) | 联合抓取与非抓取的物体位姿到达；接触动力学表征预训练、两阶段RL；跨物体/夹爪评测 | 抓前倾倒、重配置、借桌面再定向等行为与Flip/Grasp组合直接相关；其目标是位姿到达，且跨夹爪泛化不是本项目研究目标，不能混成同候选池隔离对照。 | 官方项目当前标记“Code (Coming soon)”。本轮无官方代码/权重可复现记录，不臆造commit。**NOT_RUN**。 |
| [SPIN论文2502.18015v3](https://arxiv.org/html/2502.18015v3)，[官方Skill-RRT项目](https://sites.google.com/view/skill-rrt) | Skill-RRT规划长时序抓取/非抓取操作，学习技能连接器，以示范蒸馏为策略；包括倾倒以便抓取 | 与物理翻转、抓取技能串联和示范蒸馏直接重合；“增加Flip”和“用DP克隆技能序列”本身不足以构成新贡献。原作使用IsaacGym批量规划，需技能适用性/连接器/示范链路；不能拿本项目高层DP替代整套Skill-RRT而称官方复现。 | 论文版本v3（2025-05-07），官方项目CoRL2025；所查官方项目页未链接可核验实现commit。**NOT_RUN**，原生规划/技能数据链路尚未适配。 |

GOMP另有同名的“Grasped Object Manifold Projection”装配工作，本审查明确指专家要求的2026 graspability-field方法。SPIN也有同名的移动操作主动视觉工作，本表指Skill Planning to INference，不混用两者。

这些来源说明任务机制已存在相当重合，尚不能对本项目声称T-RO新颖性。近期应先完成真实E1同实例资格审查、检查Active/Push/Flip是否对完整取出带来可区分收益，再判断强方法移植的优先级。Track A的共同候选池比较隔离高层选择能力；Track B的原生动作/原生网络比较应单列表格，不能在改变感知/动作能力后仍宣称“严格同条件”。

审查命令包括两个官方仓库的`git ls-remote ... HEAD`，不克隆环境/数据；公共来源链接和固定SHA见 `configs/baselines/source_audit.json`。该文件不意味着源代码已经获得本地执行验证或重分发许可。
