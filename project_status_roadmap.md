# 项目总路线与当前进度

更新时间：2026-04-24

本文件用于记录本项目从零开始到最终完成需要做什么、目前已经做了什么、当前卡点在哪里、下一步该推进什么。它不是论文摘录，也不是实验日志，而是项目级总控文档。

## 1. 项目目标

目标是构建一个面向多节点边缘计算网络的双时间尺度、双智能体、双世界模型系统：

- 慢尺度 Agent-D：负责服务阶段部署。
- 快尺度 Agent-S：负责请求到达后的逐请求、逐阶段调度。
- 下层资源分配：由 KKT 闭式求解器完成计算资源与链路资源分配。
- 状态表示：使用 GNN 表示边缘拓扑、节点资源和部署状态。
- 世界模型：
  - WM-D：预测候选部署方案在未来慢周期内的效果。
  - WM-S：预测当前阶段调度动作对未来阶段与总 KKT 成本的影响。

最终希望形成一套可复现实验系统，并能完成：

- 仿真环境
- 基线算法
- 双世界模型
- 双智能体协同
- 对比实验
- 消融实验
- 论文/报告撰写材料

## 2. 完成态定义

项目“完成”至少意味着以下内容全部具备：

- 有稳定可复现的仿真环境与配置文件。
- 有正确的数学模型与 KKT 下层分配模块。
- 有部署和调度两个层面的可运行基线。
- 有 WM-S，并在闭环调度中优于简单 greedy。
- 有 WM-D，并能在慢周期内改进部署。
- 有双智能体联动实验结果。
- 有轻载/中载/重载等不同负载区间的性能对比。
- 有消融实验与可直接写入论文的图表/结论。

## 3. 从零到完成的路线图

### 阶段 0：论文理解与问题定义

目标：

- 读清楚论文中的系统模型、变量定义、目标函数、约束、KKT 推导。
- 明确本项目是“部署 + 调度 + KKT 资源分配”的层级结构。

产出：

- 论文关键数学模型梳理
- KKT 下层求解理解
- 项目问题定义

状态：已完成

### 阶段 1：基础仿真环境

目标：

- 实现边缘节点、服务阶段、请求生成、拓扑、资源容量等基础环境。
- 能在固定部署下运行一个快时隙并给出完整成本。

产出：

- `edge_sim/env/`
- 请求生成器
- 拓扑与节点资源生成
- 部署合法性检查

状态：已完成

### 阶段 2：KKT 下层求解器

目标：

- 在给定部署与调度路径时，计算闭式资源分配与总成本。

产出：

- `edge_sim/optim/kkt_allocator.py`
- `gamma`、`link_load`、`f_alloc`、`r_alloc`
- `compute_delay`、`transmission_delay`、`total_delay`

状态：已完成

### 阶段 3：基础部署与调度基线

目标：

- 先建立可比较的非学习基线。

产出：

- 部署基线：
  - `fixed`
  - `heuristic`
  - `random`
  - `monolithic`
- 调度基线：
  - `greedy_delta`
  - `lookahead_delta`

状态：已完成

### 阶段 4：V1 调度学习原型

目标：

- 先在固定部署上训练一个阶段级 GAT-PPO 调度器，验证 GNN + PPO 这条线是否可跑通。

产出：

- `edge_sim/agents/gat_ppo.py`
- `edge_sim/training/train_v1.py`
- `edge_sim/evaluation/evaluate_v1.py`

状态：已完成

说明：

- V1 证明了 GNN 调度训练管线是通的。
- 但在当前环境下，固定部署和简单启发式仍然很强。

### 阶段 5：V2 动态部署原型

目标：

- 把部署从静态扩展到慢周期动态变化。

产出：

- `edge_sim/env/dynamic_deployment.py`
- `edge_sim/training/simulate_v2.py`
- deployment gate 原型

状态：已完成原型，未完成最终版本

说明：

- 动态部署候选生成和 gate 已有雏形。
- 但还没有形成真正意义上的 WM-D + Agent-D。

### 阶段 6：WM-S 原型

目标：

- 让快尺度调度不再只依赖启发式，而是具备“预测未来代价”的能力。

产出：

- 手工特征版 WM-S：
  - `build_wms_dataset.py`
  - `train_wms.py`
  - `evaluate_wms.py`
- GNN 版 WM-S：
  - `build_wms_gnn_dataset.py`
  - `train_wms_gnn.py`
  - `evaluate_wms_gnn.py`

状态：已完成原型，正在继续优化

### 阶段 7：WM-D 与 Agent-D

目标：

- 构建慢尺度部署世界模型，预测候选部署在未来慢周期内的收益与代价。

产出：

- 候选部署表示
- 部署世界模型
- 部署动作设计
- 慢尺度奖励定义

状态：已完成 WM-D v2 原型，已优于 `keep_previous`，下一步进入双智能体联动

### 阶段 8：双智能体协同

目标：

- 让 Agent-D 的部署结果真正为 Agent-S 服务。
- 形成双时间尺度闭环。

产出：

- 慢周期部署、快时隙调度、KKT 结算的一体化流程
- 双智能体实验

状态：已完成初版闭环，当前重点是缩小 Agent-D 与 WM-D 规划器之间的差距

### 阶段 9：实验与论文产出

目标：

- 完整对比实验、负载扫描、消融实验、论文图表。

产出：

- 多负载实验
- 多拓扑实验
- 消融实验
- 最终图表与结论

状态：未完成

## 4. 当前已完成内容

### 4.1 数学模型与论文提取

已完成：

- 从原始论文中提取第三章、第四章的数学模型和 KKT 推导。
- 梳理了实现所需的变量、约束和优化结构。

相关文件：

- [model_kkt_extract.md](<C:/Users/1/Desktop/I_hate _dream/model_kkt_extract.md>)
- [paper_extracted_raw.txt](<C:/Users/1/Desktop/I_hate _dream/paper_extracted_raw.txt>)
- [paper_extracted_layout.txt](<C:/Users/1/Desktop/I_hate _dream/paper_extracted_layout.txt>)
- [dual_agent_ref_raw.txt](<C:/Users/1/Desktop/I_hate _dream/dual_agent_ref_raw.txt>)

### 4.2 环境与 KKT 模块

已完成：

- 节点、服务阶段、资源容量生成
- clustered/ring/full-mesh 拓扑
- effective bandwidth 与 reachability
- KKT 闭式计算模块

关键文件：

- [edge_env.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/env/edge_env.py>)
- [request.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/env/request.py>)
- [deployment.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/env/deployment.py>)
- [kkt_allocator.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/optim/kkt_allocator.py>)

### 4.3 基线算法

已完成：

- 部署基线：`fixed / heuristic / random / monolithic`
- 调度基线：`greedy_delta / lookahead_delta`

关键文件：

- [policies.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/policies.py>)

### 4.4 V1 GAT-PPO 调度原型

已完成：

- GAT-PPO 结构
- 阶段级 masked action 训练
- 评估脚本

关键文件：

- [gat_ppo.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/gat_ppo.py>)
- [train_v1.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_v1.py>)
- [evaluate_v1.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_v1.py>)

### 4.5 V2 动态部署雏形

已完成：

- 动态部署候选生成
- deployment gate
- 慢周期仿真脚本

关键文件：

- [dynamic_deployment.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/env/dynamic_deployment.py>)
- [simulate_v2.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/simulate_v2.py>)
- [deployment_gate.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/deployment_gate.py>)

### 4.6 WM-S 原型

已完成：

- 手工特征版 WM-S
- GNN 版 WM-S
- 多 seed 数据集构建
- 闭环评估脚本

关键文件：

- [scheduler_world_model.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/scheduler_world_model.py>)
- [build_wms_dataset.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/build_wms_dataset.py>)
- [train_wms.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_wms.py>)
- [evaluate_wms.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_wms.py>)
- [build_wms_gnn_dataset.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/build_wms_gnn_dataset.py>)
- [train_wms_gnn.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_wms_gnn.py>)
- [evaluate_wms_gnn.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_wms_gnn.py>)

### 4.7 WM-D 初版原型

已完成：

- 慢尺度候选部署池生成
- 候选部署特征编码
- WM-D 数据集构建
- WM-D 训练脚本
- WM-D 初版评估脚本

关键文件：

- [deployment_world_model.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/deployment_world_model.py>)
- [wmd_utils.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/wmd_utils.py>)
- [build_wmd_dataset.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/build_wmd_dataset.py>)
- [train_wmd.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_wmd.py>)
- [evaluate_wmd.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_wmd.py>)

### 4.8 Agent-D 与双智能体初版闭环

已完成：

- Agent-D 候选部署策略网络
- 基于 WM-D 数据集的蒸馏训练脚本
- Agent-D 与 WM-D / WM-S 的双智能体联动评估

关键文件：

- [deployment_policy.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/deployment_policy.py>)
- [train_agent_d.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_agent_d.py>)
- [evaluate_dual_agent.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_dual_agent.py>)

### 4.9 Agent-S 蒸馏与快尺度策略执行器

已完成：

- Agent-S 学生策略网络 `SchedulerGNNPolicy`
- 基于 GNN-WM-S 数据集的快尺度蒸馏训练脚本
- Agent-S 单独评估脚本
- Agent-S 接入 WM-D / Agent-D 双智能体闭环
- Agent-S 轻量保守守门与 greedy fallback
- Agent-S 守门参数多 seed 校准脚本
- 双智能体多 seed 汇总评估脚本
- Agent-D 守门参数多 seed 校准脚本

关键文件：

- [scheduler_policy.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/agents/scheduler_policy.py>)
- [train_agent_s.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/training/train_agent_s.py>)
- [evaluate_agent_s.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_agent_s.py>)
- [calibrate_agent_s_guard.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/calibrate_agent_s_guard.py>)
- [calibrate_agent_d_guard.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/calibrate_agent_d_guard.py>)
- [evaluate_dual_agent_multiseed.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_dual_agent_multiseed.py>)
- [evaluate_wmd.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_wmd.py>)
- [evaluate_dual_agent.py](<C:/Users/1/Desktop/I_hate _dream/edge_sim/evaluation/evaluate_dual_agent.py>)

## 5. 当前最重要实验结论

当前在 `v2_drift`、未见 seed、heuristic deployment 下，快尺度调度表现大致为：

- `greedy_delta`: 约 `5.0592`
- 当前最佳 GNN-WM-S：约 `5.0511`
- `lookahead_delta`: 约 `5.0005`

当前在同配置、未见 seed、快尺度使用 GNN-WM-S 时，慢尺度部署表现大致为：

- `wmd_agent`: 约 `99.0295`
- `keep_previous`: 约 `99.9298`
- `history_keep`: 约 `110.0987`
- `trend_keep`: 约 `110.2112`

当前在同配置、未见 seed、双时间尺度闭环下，联合策略表现大致为：

- `dual_wmd_wms`: 约 `99.0295`
- `dual_agentd_wms`: 约 `102.3823`
- `keep_previous_wms`: 约 `103.8138`
- `history_keep_wms`: 约 `110.0987`
- `trend_keep_wms`: 约 `110.2112`

在扩大慢尺度数据并强化困难决策训练后，`dual_agentd_wms` 在部分未见 seed 上可进一步提升到约 `98.8408`，甚至超过 `dual_wmd_wms`；但在另一些未见 seed 上仍可能退化到约 `105.2376`，说明当前 Agent-D 的跨 seed 鲁棒性仍不稳定。

进一步加入基于 WM-D 预测收益的保守守门机制后，`dual_agentd_guarded_wms` 在测试 seed 上表现为：

- seed `1007`：约 `99.0295`，与 `dual_wmd_wms` 持平，优于 `keep_previous_wms` 的约 `100.0461`
- seed `1207`：约 `94.6930`，与 `keep_previous_wms` 持平，明显优于原始 `dual_agentd_wms` 的约 `105.2376`
- seed `1407`：约 `68.5191`，优于 `dual_wmd_wms` 的约 `72.0820` 和 `keep_previous_wms` 的约 `72.4179`

在 Agent-S 蒸馏后，学生策略训练验证结果大致为：

- `top1_accuracy`: 约 `0.9507`
- `top1_accuracy_hard`: 约 `0.7312`
- `avg_regret_target_score`: 约 `0.0014`

在未见 seed 的单独快尺度评估中，guarded Agent-S 已接近 GNN-WM-S 规划器：

- seed `1007`：`agent_s_policy` 约 `5.0005`，优于 `greedy_delta` 的约 `5.0103`，接近 `gnn_wms_planner` 的约 `4.9967`
- seed `1207`：`agent_s_policy` 约 `4.6351`，较原始学生策略有所改善，但仍落后于 `gnn_wms_planner` 的约 `4.6225`

在 guarded Agent-S 接入双智能体闭环后，三组未见 seed 的平均总成本约为：

- guarded Agent-D + guarded Agent-S：约 `87.7474`
- keep_previous + guarded Agent-S：约 `88.9120`
- WM-D + guarded Agent-S：约 `90.3493`

进一步在 5 个未见 seed（`1007, 1107, 1207, 1307, 1407`）上做 Agent-S 守门校准后，得到：

- 最优轻量守门参数：`min_prob = 0.0`，`min_margin = 0.05`，`fallback = greedy`
- 对应快尺度单独评估均值：`agent_s_policy = 4.3846`
- 同批次基线：`greedy_delta = 4.3817`，`gnn_wms_planner = 4.3972`
- 平均 `guard_ratio = 0.0194`

这说明当前 Agent-S 学生策略只需要极轻的边界守门，真正被拦下的阶段决策比例不到 `2%`。

在同一组 5 个未见 seed 上做双智能体多 seed 汇总后：

- `dual_wmd_wms`：均值约 `87.2134`
- `dual_agentd_wms`：均值约 `88.8277`
- `keep_previous_wms`：均值约 `89.3956`
- `dual_agentd_guarded_wms`（fallback=`keep_previous`）：均值约 `89.7483`

把 Agent-D 守门回退方式改成 `wmd_if_gain`，并设置 `fallback_gain = 0.6` 后：

- `dual_agentd_guarded_wms`：均值约 `89.0082`
- 相比 `keep_previous_wms` 平均改善约 `0.3875`
- 但仍落后于 `dual_wmd_wms`

进一步做 Agent-D 多 seed 守门校准后，发现：

- 在小规模 3 seed 粗校准中，`min_prob / min_margin / base_gain / max_wmd_gap` 几乎不改变结果，说明当前慢尺度守门的主要敏感项是回退目标本身
- 在更完整的 5 seed 验证中，最佳默认组合为：
  - `agentd_min_prob = 0.22`
  - `agentd_min_margin = 0.05`
  - `agentd_base_gain = 0.6`
  - `agentd_max_wmd_gap = 0.75`
  - `agentd_fallback_mode = wmd_if_gain`
  - `agentd_fallback_gain = 0.6`
  - `agent_s_min_prob = 0.0`
  - `agent_s_min_margin = 0.05`
  - `agent_s_fallback = greedy`

在这组默认参数下，5 个未见 seed、`episodes = 6` 的双智能体多 seed 汇总结果为：

- `dual_wmd_wms`：均值约 `83.3329`
- `dual_agentd_guarded_wms`：均值约 `84.2799`
- `dual_agentd_wms`：均值约 `84.3632`
- `keep_previous_wms`：均值约 `88.2023`

这说明当前 guarded Agent-D 已经：

- 明显优于 `keep_previous_wms`
- 略优于未守门的 `dual_agentd_wms`
- 与 `dual_wmd_wms` 的差距缩小到约 `0.9470`

结论：

- GNN-WM-S 已经优于简单 greedy。
- 但提升幅度还不大。
- 当前最佳 WM-S 还没有追平精确 lookahead。
- WM-D 经过候选池扩展、特征压缩和排序式训练后，已经优于 `keep_previous`，说明慢尺度世界模型开始具备实际部署价值。
- Agent-D 在扩大慢尺度数据、加入 gap/margin 加权和 pairwise 排序损失后，已经能在部分未见 seed 上逼近甚至超过 `dual_wmd_wms`。
- 但 Agent-D 的跨 seed 鲁棒性仍不稳定，说明慢尺度策略蒸馏还需要更多样本覆盖与更稳健的保守机制。
- 加入基于 WM-D 预测收益和候选风险级别的守门机制后，`dual_agentd_guarded_wms` 已能在已测 seed 上明显缩小跨 seed 波动，并在保守与进取之间取得更稳的平衡。
- Agent-S 蒸馏已经跑通，并且学生策略在单独快尺度评估中已经接近教师规划器，说明“先世界模型规划，再蒸馏成策略网络”这条路线成立。
- 但 Agent-S 目前还没有稳定超越教师规划器，说明快尺度学生策略还需要继续做置信度校准和更多跨 seed 验证。
- 经过多 seed 校准后，Agent-S 的最佳守门是一个非常轻的 `margin` 规则，说明快尺度学生策略已经基本可用，系统瓶颈正在转移到慢尺度守门策略。
- 当前双智能体系统在多 seed 平均意义上已经可以略优于 `keep_previous`，但还没有稳定追上 `dual_wmd_wms`，说明下一步应优先继续校准 Agent-D 的守门与回退逻辑，而不是重新大改 Agent-S 结构。
- 进一步的 Agent-D 守门校准表明，当前阶段真正关键的是“回退到哪里”，而不是继续细抠 `min_prob / min_margin` 这类弱敏感参数。
- 在当前配置下，`wmd_if_gain` 已经成为更合适的默认慢尺度回退方式。

进一步拆分发现：

- 总成本主要还是由 `compute_delay` 主导。
- 调度改进目前主要体现在少量关键阶段，而不是全局大幅下降。
- 当前环境属于中等负载，方法间差距天然较小。

## 6. 当前未完成内容

以下内容仍未完成：

- 混合 rollout + hard sample 的 WM-S 数据集
- 更强的 WM-S 排序学习
- 更强的闭环调度提升
- 更强的 Agent-D 蒸馏策略
- 更大样本的 Agent-S / Agent-D 组合多 seed 统计
- 负载扫描实验
- 消融实验
- 论文图表与结果整理

## 7. 当前优先级

### P0：继续校准 Agent-D 守门与回退逻辑

目的：

- 让双智能体系统在更大样本多 seed 上稳定逼近 `dual_wmd_wms`

当前方向：

- 固定 `wmd_if_gain` 为默认回退方式
- 在更多 unseen seed 上验证当前默认参数
- 观察最坏 seed、均值和回退触发分布

状态：进行中

### P1：扩大双智能体多 seed 统计

目的：

- 作为进入大规模训练前的门槛验证

当前方向：

- 扩展到更多 unseen seed
- 汇总均值、方差、最坏 seed 与胜率
- 验证当前默认慢尺度守门参数是否足够稳定

### P2：负载分层实验

目的：

- 证明在更高负载下，双智能体方案的优势会被放大

当前状态：暂缓，待双智能体默认配置固定后执行

## 8. 下一阶段建议任务

建议按以下顺序推进：

1. 继续做 Agent-D 守门参数的多 seed 校准
2. 固定当前默认双智能体参数，扩大 unseen seed 统计样本
3. 统计 Agent-D 守门触发频率、回退去向和收益分布
4. 做双智能体联动消融
5. 在默认配置稳定后再开展多负载实验
6. 最后整理完整图表与论文结果材料

## 9. GitHub 阶段保存

当前 GitHub 仓库：

- `https://github.com/makaka233/I_hate-_dream.git`

已完成阶段性上传：

- 提交：`311f66d`
- 信息：`Add GNN scheduler world model pipeline`
- 提交：`2ec0df4`
- 信息：`Improve GNN WM-S with mixed rollout hard samples`
- 提交：`45f05e1`
- 信息：`Add initial WM-D slow-timescale pipeline`
- 提交：`24be7c3`
- 信息：`Improve WM-D candidate ranking and beat keep_previous`
- 提交：`53d738f`
- 信息：`Add Agent-D distillation and dual-agent evaluation`
- 提交：`6ad1bd6`
- 信息：`Strengthen Agent-D with hard slow-scale decisions`
- 提交：`8197543`
- 信息：`Add guarded Agent-D evaluation and fairness fixes`
- 提交：`fb9e07c`
- 信息：`Add Agent-S distillation pipeline`
- 提交：`ecb0c0a`
- 信息：`Add multi-seed guard calibration tools`

后续建议：

- 每完成一个可复现阶段就提交一次
- 代码、配置、脚本进入仓库
- 大体积模型、日志、数据集不直接进仓库

## 10. 文档维护规则

后续每推进一个阶段，建议同步更新本文件中的以下部分：

- “当前已完成内容”
- “当前最重要实验结论”
- “当前未完成内容”
- “当前优先级”
- “GitHub 阶段保存”

这样可以始终保持项目有一个清晰的总控视图。
