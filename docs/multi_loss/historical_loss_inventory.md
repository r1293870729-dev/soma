# SOMA 历史训练入口真实损失函数盘点

审计日期：2026-11-11  
依据：用户提供的 `soma_psi_code_handoff_20260903(1).zip`（320 个 ZIP 条目，排除 macOS 元数据后静态检索 Python 源码）及 `soma已有资料.zip` 中的 HAM 源码。本文区分“训练目标中实际参与反向传播的损失”“独立优化器损失”“验证/诊断指标”和“结构选择代价”。未运行训练。

## 1. 执行摘要

SOMA 历史代码不是一个统一的“三损失模型”，而是多个实验入口和阶段：

| 历史入口 | 实际联合训练损失 | 其他独立损失/指标 | 自动权重候选结论 |
|---|---|---|---|
| 01 历史 A/B/C 谱信号实验 | 标准模式：重构 + beta-KL；特定 `loss_plus_xi` 模式再加 lambda_xi·Xi | probe CE 单独训练；谱信号通常监测 | 先以 rec/KL 为候选；Xi 仅限特定模式且需验证可微性/旧报告一致性 |
| 02 Xi 数据集泛化 | 重构 + beta-KL | 冻结 latent 的 linear-probe CE 独立训练 | 只对 rec/KL 做联合权重实验；probe CE 不得混入 |
| 03 统一旧版 Xi-B | 重构 + beta-KL | linear-probe CE 独立训练；Xi-B 作为残差结构监测 | 四数据集信号验证的干净基线；不是三损失模型 |
| 04 I1–I4 残差干预 | 与 01 同源 runner：标准路径 rec + beta-KL；特定模式可加入 lambda_xi·Xi | probe CE / 干预指标独立 | 按具体 run manifest 的 mode 判定；不可将不同 mode 合并 |
| 05 主动 SOMA trainer | 重构、structured-prior KL、可选 disentanglement、separation、usage、orthogonality、symmetry；另有结构复杂度 penalty | 结构信号/门控统计；候选结构评估目标 | 先按实际 TOML 配置筛选；complexity 是离散结构代价，不适合普通共享参数 GradNorm |
| 05 VAE baselines | VAE: rec+beta·KL；FactorVAE 另加 gamma·TC；DIP-VAE 另加 dip_lambda·DIP | Factor discriminator 有独立 BCE 更新 | 按模型 variant 单独研究，判别器 BCE 不能与生成器损失混为一组 |
| HAM pretrain | reconstruction MSE only | 无 | 不适用多损失自动加权 |
| HAM 主训练 | reconstruction + KL + invariance + equivariance + TC penalty，经 winner-take-all 后更新生成器/节点参数 | discriminator BCE 单独更新判别器 | 是明确的多损失候选，但参数共享结构和 WTA 路由需分组诊断 |

**结论：** 当前最符合“很多损失、人工设置权重”的真实历史入口是 05 主动 SOMA 和 HAM 主训练；03 的四数据集旧版 Xi-B runner 本身只有重构与 KL 两个联合损失，不能直接作为老师所说的三损失或多损失自动权重模型。

## 2. 01 / 02 / 03 / 04：VAE 谱信号与数据集实验族

### 2.1 共同的 VAE 训练目标

源码位置：
- `03_psi_xib_unified/train_xib_unified.py`，训练循环约第 472–478 行
- `03_psi_xib_unified/models.py`，`kl_standard_normal` 和 `reconstruction_loss`
- `02_dataset_generalization_xi/train_dataset_generalization.py`，训练目标约第 217–219 行
- `01_spectral_signals_abc/run_pw_signal_experiment.py`，约第 972–981 行
- `04_i1_i4_residual_interventions/run_pw_signal_experiment.py`，与 01 同源实现

联合目标：

```
L_VAE = L_reconstruction + beta_kl * L_KL
```

- 重构项：按数据 likelihood 选择 Bernoulli BCE 或 Gaussian 配置下 sigmoid 输出的 MSE。
- KL 项：对每个样本 latent 维度求和，再对 batch 求平均。
- `beta_kl` 是 KL 系数，不是独立可学习参数；训练中与重构项共同反向传播。

03 的四类数据输入：
- controlled：MLPVAE。
- dSprites：ConvVAE，Bernoulli/BCE。
- 3D Shapes：ConvVAE，代码实际使用 sigmoid 输出 MSE。
- CIFAR/CIFAR-C：ConvVAE，代码实际使用 sigmoid 输出 MSE。

### 2.2 Probe CE 不是 VAE 联合损失

01/02/03 中的 linear probe 在冻结或提取好的 latent 表征上单独训练分类器，调用 cross_entropy 并对 probe optimizer 反向传播。该 CE 不回传到 VAE encoder/decoder，因此不能把它当作与 reconstruction、KL 共同更新同一模型参数的第三项损失。

Probe 的训练/验证 loss 与 accuracy 是表征评估指标，不是 VAE 总目标。

### 2.3 01 / 04 的特殊模式：loss_plus_xi

01 和 04 的 runner 在特定 `mode == "loss_plus_xi"` 分支中执行：

```
L = L_reconstruction + beta_kl * L_KL + lambda_xi * Xi
```

此时 Xi 才会被加入目标并参与反向传播。默认/普通 loss-only 路径仍是 rec + beta-KL。此模式与纯监测 Xi-A/B/C 的实验不是同一训练设定；必须按 run config/manifest 区分，不能把它的结果与 loss-only 报告混合。

### 2.4 适合自动权重的范围

若研究对象是 03 四数据集的旧版 Xi-B 泛化验证：
- 可做的只有两项联合权重实验：reconstruction 与 beta-KL。
- 这不是老师提出的“多个损失项”的首选模型。
- Xi-B 保持独立监测，禁止加入目标。

若研究 01/04 的 loss_plus_xi：
- 可另设三项实验（rec、KL、Xi），但必须明确选择了 loss_plus_xi 历史模式。
- 需要核验 Xi 的可微路径、信号定义和该模式对应的历史报告；不能用它替代旧版 B 的 loss-only 信号验证结论。

## 3. 05 主动 SOMA trainer

源码位置：
- `05_soma_psi_gated/soma/training/trainer.py`，`_loss_components` 约第 336–407 行
- `05_soma_psi_gated/soma/runtime/losses.py`
- `05_soma_psi_gated/soma/config.py`
- `05_soma_psi_gated/configs/controlled_stream_soma_trigger_controls_nll_sigma01_randomseed_20260902.toml`

### 3.1 代码支持的损失组件

```
stage_one =
    reconstruction
  + beta_kl * KL_structured_prior
  + lambda_dis * disentanglement
  + lambda_sep * separation
  + lambda_use * usage
  + lambda_orth * orthogonality
  + lambda_sym * symmetry

total_objective = stage_one + lambda_cmp * complexity
```

各项定义：
- reconstruction：Bernoulli BCE 或 Gaussian NLL（按配置）；不是 03 runner 的 MSE 实现。
- KL：common latent 对节点混合先验的 KL + side latent 对各向同性高斯先验的 KL。
- disentanglement：common/side latent 的 total-correlation 与 joint mutual-information 估计，按配置启用。
- separation：common/side posterior 均值的归一化交叉协方差惩罚。
- usage：节点 responsibilities 使用率约束。
- orthogonality：common 与 side encoder Jacobian 的局部几何正交惩罚。
- symmetry：对指定变换前后的 common 不变、side 按表示矩阵等变的约束。
- complexity：节点数、有效边数、common dimension 的离散结构复杂度代价。

### 3.2 实际历史 controlled run 配置

文件：`configs/controlled_stream_soma_trigger_controls_nll_sigma01_randomseed_20260902.toml`

| 项目 | 配置系数 | 本次配置是否启用 |
|---|---:|---|
| reconstruction | 隐式 1 | 是 |
| structured KL | beta_kl=0.01 | 是 |
| disentanglement | lambda_dis=0 | 否 |
| separation | lambda_sep=0.1 | 是 |
| usage | lambda_use=0.1 | 是 |
| orthogonality | lambda_orth=0 | 否 |
| symmetry | lambda_sym=0 | 否 |
| complexity | lambda_cmp=0.1 | 结构代价启用，但不属于普通可微共享参数损失 |

因此，该具体历史配置有 **4 项数据驱动、参与模型梯度的训练目标**：reconstruction、KL、separation、usage；另有 complexity 结构代价。不能笼统说它只有“三个损失”，也不能把 config.py 的默认值当作该 run 的真实设置。

### 3.3 05 自动权重适配判断

优先候选：
- reconstruction
- structured KL
- separation
- usage

这些项在同一次 `_loss_components` 前向中由同一个 SOMA model 计算，并进入 stage_one。正式实现前仍需分别测量它们对共享 encoder、node heads、decoder 的梯度；不同损失对各子模块的覆盖可能不同。

不建议直接对 complexity 用 GradNorm：
- `StructureState.complexity` 基于节点数、阈值化有效边数和维度数。
- StructureState 的结构张量在构造时 detach。
- 这是结构候选评分/离散结构代价，不是常规对连续共享网络参数可微的损失。
- 可以保留原结构选择流程，单独做结构代价敏感性分析；不要将它伪装成可自动学习的梯度任务。

orthogonality 和 symmetry 在其他配置中可能启用，但本次指定 controlled run 的系数为 0；不可把未启用项纳入本次 active-loss inventory。

## 4. 05 VAE baseline variants：必须按 variant 分开

源码：`05_soma_psi_gated/soma/experiments/vae_baselines.py`

```
base = reconstruction + beta * KL

factor_vae:
    objective = base + gamma * TC

dip_vae:
    objective = base + dip_lambda * DIP

vae / ada_gvae:
    objective = reconstruction + effective_beta * KL
```

FactorVAE 的 discriminator 另用真实/打乱 latent 的 BCE 训练 discriminator 参数；生成器侧 TC 项则通过 discriminator 对 latent 反向传播。两者属于不同优化器/参数组，不应将 discriminator BCE 与 generator objective 合并做同一组自动权重。

DIP-VAE penalty 在该文件中通过 latent mean 的协方差偏离单位阵构造。必须按 variant 单独跑，不得把 VAE、FactorVAE、DIP-VAE、Ada-GVAE 的损失项合成一个总清单。

## 5. HAM：pretrain 与主训练是不同阶段

源码：
- `HAM/pretrain.py`
- `HAM/ham_trainer.py`
- `HAM/losses.py`
- `HAM/config.py`

### 5.1 Pretrain 阶段

仅使用 reconstruction MSE：

```
L_pretrain = MSE(reconstruction, input)
```

没有多损失自动权重问题。不能把主训练损失倒灌到 pretrain。

### 5.2 HAM 主训练生成器目标

每个 active node 计算：

```
F_u = loss_rec + loss_kl + loss_inv + loss_eqv + tc_penalty
loss_G = mean(min_u F_u)
```

Config.py 中权重：
- LAMBDA_REC = 10.0
- LAMBDA_KL = 0.01
- LAMBDA_INV = 5.0
- LAMBDA_EQV = 5.0
- LAMBDA_TC = 5.0

五项含义：
1. Reconstruction MSE：重构 clean image。
2. KL：common/side latent 的 KL。
3. Invariance：common factor 对外观破坏保持不变。
4. Equivariance：side factor 按变换矩阵保持等变。
5. TC penalty：通过 discriminator logits 的 softplus 均值，作用于生成器 latent。

HAM 使用 winner-take-all：每个样本只选择能量最低的 node。梯度诊断必须区分 shared_encoder、winning node heads/decoder 与 discriminator 路径，不能把所有节点参数简单视为完全共享。

### 5.3 HAM discriminator 独立目标

```
L_disc = 0.5 * (BCE(real_joint, 1) + BCE(permuted_marginals, 0))
```

该损失单独更新 discriminator optimizer，不是与五个生成器项共同优化同一参数集合的第六项。

### 5.4 HAM 自动权重适配判断

可以把五个生成器项作为一个候选研究对象，但需：
- 只对 generator objective 的五项做初步权重/梯度诊断；
- discriminator BCE 独立处理；
- 在共享 encoder 参数上计算共同梯度，在 node-specific heads/decoder 上报告各自梯度覆盖；
- 考虑 WTA 造成的 winner 路由和梯度稀疏；
- 检查 TC penalty 对 generator 的梯度是否稳定；
- 先确认实际 HAM 训练数据与本次四数据集计划是否兼容。当前 HAM 历史数据/输入处理不能未经验证就视为 CIFAR、dSprites、3D Shapes、Controlled 四个统一实验入口。

## 6. 哪些损失适合自动权重：推荐分层

### 首选候选 A：05 主动 SOMA 的真实 controlled run

对本次 TOML 配置，先诊断四个可微训练项：
1. reconstruction
2. structured KL
3. separation
4. usage

不要把 complexity 加入 GradNorm。它是结构代价而非同类的连续参数梯度任务。

优势：同一个主动 SOMA 训练目标中确实有多个不同功能的损失，符合“减少人工权重设置”的问题。

限制：这是 controlled 数据集的历史配置。要推广到 CIFAR/dSprites/3D Shapes，必须先有相同主动 SOMA 架构和数据管线的可运行证据；不能假设 03 的 VAE runner 已有这些损失。

### 首选候选 B：HAM 主训练五项

五个 generator terms 可以独立评估自动权重，但应作为 HAM 单独实验，不与 05 SOMA 的四项合并。其 WTA 节点路由使梯度分析更复杂。

### 次级候选：01/04 的 loss_plus_xi 三项模式

只在研究目标明确要测试 Xi 进入训练目标时考虑；它与旧版 B loss-only 信号泛化验证是不同实验。

### 不适合作为老师所说多损失自动权重主实验的入口

- 03 统一旧版 Xi-B：仅 reconstruction + beta-KL 两项；B 是监测信号，不是 loss。
- 02 数据集泛化：同样两项；probe CE 单独训练。
- 01/04 的 loss-only：同样两项；Xi 仅监测。
- HAM pretrain：只有 reconstruction 一项。
- FactorVAE discriminator BCE：独立优化 discriminator 参数，不能并入生成器多损失权重集合。

## 7. 四数据集目标与自动权重研究的边界

用户指定的四数据集：
- CIFAR
- dSprites
- 3D Shapes
- Controlled

当前 03 统一旧版 Xi-B runner 是四数据集信号泛化验证的共同入口，保持 loss-only rec+beta-KL，并记录旧版 B。它适合回答“旧版信号是否跨数据集工作”。

多损失自动权重需要另选真正的多损失模型入口：
- 若选择 05 主动 SOMA，现有已确认的实际多损失配置是 controlled stream；其他三个视觉数据集的主动 SOMA 对应训练配置/模型工厂还需确认。
- 若选择 HAM，需单独核验其数据管线和四数据集兼容性。
- 不得将 05/HAM 损失移植到 03 只为制造三个或更多损失。

## 8. Phase 0.5 必须补充的审计工作

在正式自动权重实现前：
1. 为每个目标入口和每个 seed 导出 resolved config，不能只读默认 config。
2. 对每个 active loss 记录未加权原始值、配置系数、加权值。
3. 在真正共享的参数子集上分别测量梯度范数。
4. 记录每对损失梯度余弦相似度，跨 batch 和训练阶段汇总。
5. 对 winner-take-all / node-specific 模块分开记录梯度覆盖。
6. 明确主任务 validation metric；禁止用测试集调权重。
7. 先决定多损失主实验究竟选择 05 主动 SOMA 还是 HAM；不要把两者合并。
8. 四数据集旧版 B 泛化验证仍按 03 loss-only 独立开展，B 不进入自动权重目标。

## 9. 最终建议

**最贴近老师原话的研究主线：先以 05 主动 SOMA 的 controlled 多损失配置为候选，针对 reconstruction / structured KL / separation / usage 四项开展梯度诊断；另把 HAM 五项生成器目标作为独立候选。**

但如果论文的主目标是“四数据集验证旧版 B”，则 03 仍是信号实验入口，不能把它说成多损失自动权重实验。

当前仍需负责人确认多损失主实验的选择（05 或 HAM），并补齐其他目标数据集上的模型/配置和主任务验证指标。此报告为静态源码盘点，不代表任何自动权重方法已运行或有效。
