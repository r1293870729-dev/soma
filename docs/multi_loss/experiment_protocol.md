# SOMA 多损失函数自动权重优化实验任务书

> 状态：计划阶段，尚未开始正式实验。独立扩展实验，不替代既有 SOMA 主实验。
> 信号约束：必须调用历史实验实际使用的旧版统一 Psi/Xi-B 实现；B/mass 仅作结构监测，不作为训练损失。

## 一、项目目标

在现有 SOMA 项目中开展多损失函数自动权重分配实验，比较人工固定权重、初始尺度归一化、Uncertainty Weighting、GradNorm 和 PCGrad 等策略。目标是找到能够改善 SOMA 主任务验证性能、训练稳定性和结构适应表现的损失权重分配策略。

实验必须使用项目历史实验中实际使用的旧版统一 Psi/Xi-B 实现。禁止自行创建新版 B 信号实现，禁止替换 B 的计算公式。

## 二、最高优先级约束

1. 先扫描现有 Git 仓库、历史实验代码包、配置文件和训练记录。
2. 找到历史实际使用的旧版统一 Psi/Xi-B 实现，并确认源文件、类名、函数入口、计算公式和依赖。
3. 不得使用新编写的 B 实现替代历史实现。
4. 不得修改 B 的白化、协方差、MP 谱边缘、窗口、归一化或 mass 计算方式。
5. B 仅用于原有结构监测和实验分析，不得将 B 作为可反向传播的训练损失。
6. 不得覆盖历史实验、历史报告和已有 checkpoint。
7. 所有新代码放在独立实验目录。
8. 不得擅自修改 SOMA 的模型结构、主任务定义和数据划分。
9. 任何实验不得使用测试集调整权重、超参数或选择最优模型。
10. 不得把工程 smoke test 结果描述成正式科学实验结果。

特别注意：仓库中若存在 B_mass_fixed_white_empirical_v1 实现，它不是历史旧版 stage-keyed EMA/MP runner 的自动替代品。必须从历史源码归档中定位本次指定的旧实现。若历史代码无法找到，立即停止正式实验并报告缺失资产。

## 三、第一阶段：识别真实损失函数

扫描历史训练代码，找到所有参与总损失计算的项目。对每个损失记录：

- 原始名称和代码位置。
- 数学定义和计算方式。
- 是否属于主任务。
- 是否属于辅助任务或正则约束。
- 是否共享模型参数。
- 是否经过同一个共享特征层。
- 原始人工权重。
- 损失数值的典型范围。
- 梯度范数。
- 是否存在负值、零值或数值不稳定。
- 是否使用 detach 或 stop_gradient。

生成 docs/multi_loss/loss_inventory.md，生成实际损失关系图和原始训练流程图。不得将示例中的三个损失直接当成真实 SOMA 损失。如果真实损失不止三个，按实际数量扩展算法；如果不足三个，也按实际情况处理。在确认损失定义前，不得进入正式训练。

## 四、第二阶段：建立实验代码

新建 experiments/multi_loss_optimization/，建议结构：

    experiments/multi_loss_optimization/
        README.md
        configs/baseline.yaml
        configs/normalized.yaml
        configs/uncertainty.yaml
        configs/gradnorm.yaml
        configs/pcgrad.yaml
        controllers/fixed.py
        controllers/normalized.py
        controllers/uncertainty.py
        controllers/gradnorm.py
        controllers/pcgrad.py
        diagnostics/gradient_norm.py
        diagnostics/gradient_cosine.py
        diagnostics/loss_logger.py
        diagnostics/b_signal_logger.py
        runners/run_experiment.py
        runners/run_sweep.py
        analysis/compare_results.py
        analysis/plot_loss_weights.py
        analysis/plot_gradient_diagnostics.py
        analysis/plot_b_trajectory.py
        tests/test_weight_controllers.py
        tests/test_gradient_diagnostics.py
        tests/test_b_legacy_compatibility.py

上述目录是新实验模块，不得覆盖原有代码。

## 五、第三阶段：实现五种实验方法

### Exp-0：Fixed Weight Baseline

严格沿用历史人工权重。

### Exp-1：Initial Scale Normalization

参考公式：

    L_total = Σ λ_i * L_i / (L_i_initial + epsilon)

参考损失必须通过固定预热数据估计并冻结，不能每个 batch 重新初始化。对接近零或可能为负的损失进行专门处理，不允许直接使用 abs() 掩盖损失定义问题。

### Exp-2：Uncertainty Weighting

采用：

    L_total = Σ [exp(-s_i) * L_i + s_i]

s_i 为可学习参数。检查概率模型假设是否适用于对应损失。如果不适用，明确将其作为经验性可学习加权方法。记录每个训练阶段的实际权重。

### Exp-3：GradNorm

采用共享参数梯度范数与相对训练速度动态平衡。参考 alpha=1.5，但将 alpha 设置为独立超参数。

实现注意：
- 使用真正共享的模型参数。
- 对每个损失分别计算梯度。
- 不允许对 GradNorm 权重和模型参数混用优化器。
- 保证权重非负。
- 约束权重总和为损失项数量。
- 正确处理 autograd 图和梯度保留。
- 记录权重变化、梯度范数和目标梯度范数。

### Exp-4：GradNorm + PCGrad

先完成梯度冲突诊断。计算：

    cos(g_i, g_j) = dot(g_i, g_j) / (norm(g_i) * norm(g_j))

如果没有持续负余弦冲突，不要直接启用 PCGrad。如果冲突明显，实施梯度投影，并与单独 GradNorm 比较。确保 PCGrad 只修改用于模型更新的梯度，不篡改原始损失记录。

## 六、第四阶段：旧版 B 信号集成

通过独立 logger 调用历史旧版 B 实现。记录：
- B/mass 原始数值。
- 活动模态数量。
- 谱特征值。
- 原有信号状态。
- 原有窗口状态。
- 训练 step。
- 模型 checkpoint ID。
- B 实现版本和配置哈希。

禁止修改 B 的原始计算、将 B 放入总训练损失、对 B 反向传播、用新版本 B 替换旧版本 B，或为了某种权重策略修改 B 的参考基线。B 日志必须能够和各损失权重日志按训练 step 一一对应。

## 七、第五阶段：公平对照实验

第一轮：
- E0：历史人工固定权重，3 seeds。
- E1：初始尺度归一化，3 seeds。
- E2：Uncertainty Weighting，3 seeds。
- E3：GradNorm，3 seeds。
- E4：GradNorm + PCGrad，仅在冲突诊断满足条件后开展，3 seeds。

所有方法保持相同模型结构、训练数据、数据划分、初始化 seed、训练步数、优化器及学习率策略、batch size、checkpoint 选择规则、B 信号实现和配置。对随机种子使用配对设计。人工权重搜索必须使用独立验证集，记录搜索范围和选择规则。

## 八、第六阶段：记录与分析

每个实验输出：
1. 每项原始训练损失。
2. 每项验证损失。
3. 自动权重随训练步数变化的曲线。
4. 每项梯度范数。
5. 梯度余弦相似度矩阵。
6. B/mass 时间序列。
7. 主任务验证指标。
8. 最佳 checkpoint。
9. 训练时间和显存峰值。
10. 完整配置和随机种子。

每次运行保存 config.yaml、metrics.csv、weights.csv、gradients.csv、b_signal.csv、environment.json、checkpoint、run_summary.json。禁止只保存最终平均数值。

## 九、第七阶段：统计分析

使用配对随机种子比较不同策略，报告均值、标准差、配对差值、置信区间、效应量、训练成本和失败运行数量。主要以预先指定的 SOMA 主任务验证指标为选择依据。B/mass 用于独立结构诊断，不用它代替主任务性能。如果自动权重降低某个辅助损失，却导致主任务变差，应如实报告。

## 十、第八阶段：交付成果

生成：

    reports/multi_loss/
        experiment_protocol.md
        loss_inventory.md
        experiment_results.md
        statistical_comparison.md
        failure_analysis.md
        final_recommendation.md

生成图表：各损失训练曲线、自动权重变化曲线、梯度范数变化曲线、梯度冲突矩阵、B/mass 变化曲线、主任务验证性能对比、不同 seed 的稳定性对比。

最终给出：
1. 哪种策略表现最好。
2. 相比历史人工权重提升多少。
3. 哪些损失被自动提高或降低。
4. 是否存在梯度冲突。
5. B/mass 是否出现可解释的结构变化。
6. 训练成本增加多少。
7. 是否值得用于正式 SOMA 实验。
8. 是否需要进一步扩大 seeds。

## 十一、执行权限与停止规则

第一步只允许扫描和分析，不允许修改历史代码。第二步创建独立实验模块和单元测试。第三步运行小规模 smoke test。第四步通过测试后开展第一轮配对实验。第五步根据第一轮结果决定是否开展 PCGrad 和第二轮实验。

如发现以下情况立即停止相关实验：
- 历史旧版 B 实现无法定位。
- 实际损失定义不明确。
- 主任务和辅助损失关系不明确。
- 梯度出现 NaN 或 Inf。
- 权重持续振荡或坍缩。
- 数据划分发生变化。
- 测试集被用于调参。
- 实验与历史基线不再公平。

每个阶段先提交审计报告，不允许擅自跳过失败步骤。最终目标不是证明某一种算法一定更好，而是通过可复现的对照实验，找出适合 SOMA 的自动损失权重分配策略。
