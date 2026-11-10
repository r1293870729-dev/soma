# 本地 Agent 执行入口：SOMA 多损失函数自动权重优化

开始任务前，先完整阅读：
- docs/multi_loss/experiment_protocol.md
- AGENTS.md
- docs/EXPERIMENT_AUDIT.md
- docs/EXPERIMENT_PLAN.md
- docs/PROTOCOL_CONFLICTS.md

严格按 docs/multi_loss/experiment_protocol.md 执行，不得自行扩大任务范围。

## 当前执行阶段：Phase 0 — 只读资产盘点

本轮只允许读取、扫描和生成审计文档。禁止修改历史训练代码、历史信号实现、历史配置、checkpoint、数据划分；禁止启动正式训练或下载数据集。

### 必须完成

1. 扫描 Git 仓库、历史源码包（若本地存在）、配置、训练日志和 checkpoint。
2. 定位旧版统一 Psi/Xi-B 的真实源码、类/函数入口、公式、依赖和配置；确认是否为历史 stage-keyed EMA / shrinkage whitening / MP edge / B-mass 路径。
3. 单独列出仓库中任何 B_mass_fixed_white_empirical_v1 或其他非历史实现，不能把它们当成旧版实现。
4. 从实际训练代码追踪每个损失项：定义、系数、计算位置、共享参数、梯度、detach 行为、数值范围和训练入口。
5. 找到主任务验证指标、训练/验证/测试划分、seed、batch size、优化器、训练步数和 checkpoint 选择规则。
6. 检查 B 信号能否在不改变模型更新的情况下只读记录；标出所需的 logger 接口，但本阶段不实现。
7. 生成损失依赖图和训练流程图。
8. 明确列出缺失文件、不可访问的本地归档、路径假设和无法验证的事项。

### 输出

- docs/multi_loss/loss_inventory.md
- docs/multi_loss/legacy_b_implementation_audit.md
- docs/multi_loss/data_and_split_audit.md
- docs/multi_loss/phase0_blockers.md

### 停止条件

如果找不到历史旧版 B 实现，或者不能从源码确认真实损失定义、主任务指标和数据划分，立即停止，不得创建替代信号，不得猜测损失项，不得启动训练。

完成后向项目负责人汇报审计结论和阻塞项，等待批准后才能进入 Phase 1。后续阶段仍须遵守实验协议中的公平对照、统计和停止规则。


## 来源文件与方法顺序补充（必须遵守）

项目负责人指定的来源文件为 `multi_loss_optimization_guide(1).html`。如果该文件在本地可用，必须先阅读；其损失权重方法和术语按该指南执行，并以仓库 `docs/multi_loss/experiment_protocol.md` 的“来源依据”章节为准。

指南中的设定是三个损失项、共享模型参数、目标为最终验证集任务指标。示例损失数值 0.5、10、0.02 仅用于说明尺度差异，不是 SOMA 真实损失值，也不能据此命名真实损失。

实验比较顺序按指南固定为：
- Exp-1：原始等权重；
- Exp-2：初始尺度归一化；
- Exp-3：手动权重搜索；
- Exp-4：Uncertainty Weighting；
- Exp-5：GradNorm；
- Exp-6：仅在多 batch、多训练阶段发现持续梯度方向冲突后，开展 GradNorm + PCGrad。

GradNorm 使用指南中的相对训练速度、共享层梯度范数和 alpha=1.5 起始设定；Uncertainty Weighting 使用 exp(-s_i)*L_i+s_i，但不得对任意自定义正则项宣称严格概率解释；PCGrad 是梯度修改而非权重估计。

本地审计必须核实 SOMA 的真实损失数量、名称、定义、共享参数层和主任务验证指标。指南未提供这些 SOMA 专属信息，不得推测或补造。若实际结构与“三损失共享参数”前提不同，报告差异并暂停方法适配，等待负责人确认。
