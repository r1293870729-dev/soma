# 03 Psi/Xi-B 统一多数据集代码

这一组代码把前面的受控流形实验和公开数据集实验统一到一个入口里，并且只保留 Xi-B。代码和文档里通常把 Xi-B 称为 `Psi`。

## 这组代码解决的问题

前一组 `02_dataset_generalization_xi` 同时计算 Xi-A、Xi-B、Xi-C。这个目录进一步收敛到一个更简单的问题：如果只看 Psi，它在 controlled、dSprites、3D Shapes、CIFAR/CIFAR-C 上如何变化。

同时，这里更明确地区分了：

- VAE 训练数据流。
- probe 诊断数据流。

这样 probe accuracy 不会直接使用 VAE 正在训练的同一批样本。

## 支持的数据集和默认潜变量维度

```text
controlled: z = 4
dsprites:   z = 8, 16, 32
shapes3d:   z = 16, 32, 64
cifar:      z = 16, 32, 64
```

## 建议数据相对路径

```text
../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
../data/shapes3d/3dshapes.h5
../data/cifar10
../data/CIFAR-C
```

controlled 数据由代码生成，不需要外部数据。

## 这组代码里有什么

```text
README.md                  # 原统一 runner 说明
data.py                    # 四类数据集的数据流和 train/probe 划分
models.py                  # controlled 与图像数据集的模型
signals_xib.py             # Psi/Xi-B 信号计算
train_xib_unified.py        # 统一训练入口
utils.py                   # 随机种子、CSV 和工具函数
README_HANDOFF.md          # 本交接说明
```

## 如何运行

进入本目录：

```bash
cd 03_psi_xib_unified
```

运行 controlled：

```bash
python train_xib_unified.py \
  --dataset controlled \
  --seeds 0 1 2 \
  --latent-dims 4 \
  --output-dir runs/xib_unified/controlled \
  --device cpu \
  --resume
```

运行 dSprites：

```bash
python train_xib_unified.py \
  --dataset dsprites \
  --seeds 0 1 2 \
  --latent-dims 8 16 32 \
  --dsprites-path ../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
  --output-dir runs/xib_unified/dsprites \
  --device cpu \
  --resume
```

运行 3D Shapes：

```bash
python train_xib_unified.py \
  --dataset shapes3d \
  --seeds 0 1 2 \
  --latent-dims 16 32 64 \
  --shapes3d-path ../data/shapes3d/3dshapes.h5 \
  --output-dir runs/xib_unified/shapes3d \
  --device cpu \
  --resume
```

运行 CIFAR/CIFAR-C：

```bash
python train_xib_unified.py \
  --dataset cifar \
  --seeds 0 1 2 \
  --latent-dims 16 32 64 \
  --cifar10-root ../data/cifar10 \
  --cifarc-root ../data/CIFAR-C \
  --output-dir runs/xib_unified/cifar \
  --device cpu \
  --resume
```

## 常用参数

- `--train-samples-per-stage`：每阶段 VAE 训练流样本数，默认 4096。
- `--probe-samples-per-stage`：每阶段 probe 流样本数，默认 4096。
- `--probe-train-fraction`：probe 内部训练/测试比例，默认 0.7。
- `--x-window`：`W/B`，默认 8。
- `--baseline-warmup-batches`：每阶段 baseline warmup batch 数，默认 20。
- `--alpha`：协方差 shrinkage 系数，默认 0.1。

## 交接提醒

这组代码适合用来单独研究 Psi 曲线和 probe 曲线，但还没有 SOMA 结构编辑。SOMA 部分请看 `05_soma_psi_gated`。

