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
