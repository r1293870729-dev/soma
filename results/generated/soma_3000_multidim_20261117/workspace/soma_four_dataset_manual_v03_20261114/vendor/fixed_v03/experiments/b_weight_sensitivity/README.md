> 本次完整预算训练入口为根目录 run_formal.sh；先读根目录 README_FIRST.md 和 docs/FORMAL_WEIGHT_SENSITIVITY_ADDENDUM.md。下文保留上一版 80-step 工程入口说明。

# 03 旧版 Xi-B：损失权重敏感性工程模块

本次用户选择唯一入口为历史 03 runner（rec+KL），残差明确保留 **xhat−x**。
原始模型、损失和 B 源码由 SHA-256 固定并直接调用，历史文件不改写。
05/HAM 的损失、新版固定白化 B、probe CE、复杂度和判别器目标均不进入本实验目标。

**当前只完成静态审计与待验证实现。当前机器缺少 PyTorch，依赖源 DNS 超时；
smoke、单元回归和梯度回放均未执行。不要把代码存在或 AST 通过当作运行通过。**
正式配对多 seed 训练、PCGrad 和 test 评估均未启动。

## 文件

- weighting.py：固定系数、初始尺度、经验 uncertainty、GradNorm；正/有限 reference、完整 target detach、独立参数组检查。
- legacy_b_monitor.py：原 XiBSignal 的 detached CSV logger；要求 checkpoint/source/config 身份，拒绝重复 step、跨 run 复用与已有 CSV。
- historical_loader.py：验证五个原文件哈希，避免 sys.path 与裸名 utils 污染；直接执行原字节，不写历史 pycache。
- gradient_diagnostics.py：命名共享参数组上的原始/加权范数、余弦与零/未连接状态；不更新 .grad。
- engineering_smoke.py：仅一个受限 seed0/80-step/CPU run，不能启动正式 sweep。
- replay_gradient_diagnostics.py：只读取该 smoke 的自有 checkpoint，weights_only=True；零 optimizer step、零 B 状态更新。
- tests/test_engineering_contracts.py：14 个待运行回归用例，含旧 B 的逐值对照。
- configs/source_pin.json：03 的五个源文件 SHA-256。
- configs/engineering_smoke.json：源码默认+缩短预算；不是找回的历史 resolved config。
- configs/arms.example.json：原提供示例，不能直接作为正式参数或 seeds。
- reports/b_weight_sensitivity/：位于仓库根目录，含审计与执行状态。

## 准备环境

在单独的 venv 或已有解释器中安装 requirements-engineering.txt。
顶层依赖配对为 torch2.6.0/torchvision0.21.0/numpy2.0.2。
当前安装未完成，因此没有伪造完整运行环境锁。

将原 03 目录作为外部只读输入。目录必须包含：
data.py、models.py、signals_xib.py、train_xib_unified.py、utils.py，
且其内容与 configs/source_pin.json 一致。

以下命令在仓库根目录运行，将 /path/to/03_psi_xib_unified 换为实际路径。
不需要下载任何外部数据集；controlled 由原历史生成器产生。

## 回归检查

~~~sh
BWS_LEGACY_ROOT=/path/to/03_psi_xib_unified \
python -m unittest discover -s experiments/b_weight_sensitivity/tests -v
~~~

如果未设置原目录，历史源兼容测试会标记 skipped；不得把跳过当作兼容通过。
本次交付没有运行上述测试。

## 唯一工程 smoke

~~~sh
python -m experiments.b_weight_sensitivity.engineering_smoke \
  --legacy-root /path/to/03_psi_xib_unified \
  --out results/generated/bws03-smoke-20261111-s0-v1
~~~

拒绝已有输出路径。预算锁定为 stage_steps=[32,24,24]、80个模型更新。
使用原始 MLPVAE/build_model、实际 rec/KL 函数和 AdamW；系数为源码默认1与0.001，
但缺少历史 resolved_args，故组名为 engineering_fixed，不能声称完整历史重现。

保留每步 pre-update 模型/优化器/RNG/输入快照及清单；
比较 adapter 与未改原 XiBSignal 的每个返回字段，
检查 logger 不改变模型、梯度或全局 torch RNG；
训练和 B 日志按实际 checkpoint_id/step 对齐。

## 只读梯度回放

只有 smoke 完整通过后运行：

~~~sh
python -m experiments.b_weight_sensitivity.replay_gradient_diagnostics \
  --legacy-root /path/to/03_psi_xib_unified \
  --smoke results/generated/bws03-smoke-20261111-s0-v1 \
  --out results/generated/bws03-gradient-replay-20261111-s0-v1
~~~

共享组固定为 MLPVAE.encoder；逐 checkpoint 用相同输入和 RNG 重放 raw loss，
报告原始及原系数加权梯度，不训练模型，不推进 B。
这是一个短 seed 的工程诊断，不能证明跨 seed 持续冲突或启用 PCGrad。

## 正式实验停止点

负责人尚需批准实际历史/新基线系数、数据角色、初始校准、权重初始化、
共享组、预算、事件真值范围及报警阈值/持续/去重/匹配合同。
不把 MP 内部 edge 当成外部 B 报警阈值，
不把 B 幅度变大当作识别提高，不用 test 调参。

没有正式配对比较时，对“识别是否更好、是否牺牲验证性能、
误报是否改变、哪种方法最稳定”的回答只能是：证据不足，尚不能判断。
