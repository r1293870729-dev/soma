# 四数据集3000步多维度实验（2026-11-17）

本提交归档已经完成的42/42个父条件，无训练失败；不改变历史源码、模型更新或已保存结果。它是开发诊断与历史协议重建，不代表手册Q1–Q3最终测试、正式统计验收或独立reviewer的accepted。

| 数据集 | 潜维数 | seed | 父条件 |
|---|---|---|---:|
| controlled | 2/4/8/16/32 | 0/1/2 | 15 |
| dSprites | 8/16/32 | 10/11/12 | 9 |
| Shapes3D | 16/32/64 | 10/11/12 | 9 |
| CIFAR/CIFAR-C | 16/32/64 | 10/11/12 | 9 |

所有主轨迹均为1000/1000/1000步。每条图像父轨迹在更新1000后分叉同模型、完整Adam状态的S1稳定对照，续跑2000步；两个分支横轴均截止3000，稳定分支不是独立seed。新增主模型计算预算213000更新，不把探针拟合、克隆校验算成新的独立父轨迹。

controlled保留circle/torus/local历史runner、rec+0.001KL+0.001旧Ξ及原当前阶段θ普通准确率。seed0原15个θ点逐项核验，0/1/3000为离线补测；该探针是历史训练池80/20回测，不能解读为独立固定旧任务测试。图像保留rec+0.0001KL、独立固定探针角色及μ/z读出；dSprites训练重构项是BCE，另记录像素MSE，其他图像训练重构项为MSE。

旧Ξ使用历史stage-keyed EMA/收缩白化/MP最大特征值绝对超额。新Ψ使用固定参考、迹归一化、独立经验edge和全部超边缘相对质量；分别保存参考0及参考1000，同轨迹旁路记录，参考1000从1001批起才使用。两种信号的坐标轴和图像读出器不同，不能按峰高直接排名。

## 图表入口

- [四数据集、全部维度、信号+MSE+全部下游探针总合图](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/figures/合图_信号_MSE_全部探针_20261117/00_四数据集_全部维度_信号_MSE_全部探针_总合图.png)
- [清晰单数据集图及原seed图](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/reports/combined_20261117/README.md)
- [原始结果、校准、模型快照与诊断](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/results/all_3000_001)
- [本次全部训练代码](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/src) · [全部绘图、监测及校验代码](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/tools)
- [配置、冻结哈希](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/configs) · [协议及探针定义](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/docs)
- [完整进度与资源账本](../../results/generated/soma_3000_multidim_20261117/workspace/soma_all_datasets_3000_multidim_20261116/reports/FINAL_STATUS.json)
- [逐文件SHA256清单](../../results/generated/soma_3000_multidim_20261117/asset_manifest.json)

合图为3seed均值与最小—最大范围，范围不是置信区间；原seed版单列保留。所有已记录探针，包括S3条件方向与原始辅助方向，都在合图中。未平滑、未补造未来测点。MSE采用线性轴。原7月源码/环境仍不可用，因此历史部分不能宣称原版字节级复现。

## 完整取回

```bash
git clone --branch experiment/3000-multidim-20261117 https://github.com/r1293870729-dev/soma.git
cd soma
git lfs install
git lfs pull
```

本目录的publication_summary.json登记文件数与字节。大文件为真正Git LFS对象，不能只下载GitHub的Source ZIP就假定已取得模型。上传完成状态以远端提交与传输回执为准。

原始数据集是单独输入，不属于本次实验输出；本归档保留原始数据哈希、原图角色和划分、观测特征缓存、必要固定参考、官方ResNet18读出器权重以及本次全部模型/探针快照。raw_inputs.json列出dSprites、Shapes3D、CIFAR/CIFAR-C输入与数值缓存，未静默改样本或重新下载数据集。

便携校验/重画入口与环境文件见本目录。归档原文件按原字节保存，路径迁移由独立adapter处理；重画写到独立输出目录，不回写归档。重新训练需要原始数据输入，不能用只有表格的目录假装可训练。


## 校验、重画与新目录重跑

```bash
python -m pip install -r experiments/multidim_3000_20261117/requirements-plot.txt
python experiments/multidim_3000_20261117/run_archive.py --verify
python experiments/multidim_3000_20261117/run_archive.py --plot --output ../soma_replot_3000
```

重画入口会逐项保留原测点，并强制核验导出的135406行数值CSV与原文件SHA256完全一致。PNG/SVG输出到新目录，不覆盖归档。Linux查看中文图需要已安装的中文字体；本次原图在Mac使用Arial Unicode MS。

训练另安装requirements-train.txt，需要Python3.12兼容环境。如下命令会显式开始新训练，当前发布操作不会执行它们：

```bash
python experiments/multidim_3000_20261117/run_archive.py --train controlled --workdir ../new_controlled_3000
python experiments/multidim_3000_20261117/run_archive.py --train dsprites --workdir ../new_dsprites_3000 --data-root /path/to/input_project --dsprites-images /path/to/dsprites_imgs.npy
python experiments/multidim_3000_20261117/run_archive.py --train shapes3d --workdir ../new_shapes3d_3000 --data-root /path/to/input_project --shapes-images /path/to/shapes3d_imgs.npy
python experiments/multidim_3000_20261117/run_archive.py --train cifar --workdir ../new_cifar_3000 --data-root /path/to/input_project
```

input_project沿用原数据目录结构：data/cifar10、data/CIFAR-C（motion_blur、frost、labels）；dSprites/Shapes3D数值图像输入单独指定。原图ID、角色与训练抽样配置仍使用本归档字节。输入数值缓存可以由归档的prepare.py从对应原始数据生成，但不可用其他样本或新划分代替本次输入。

已验证：全部5315个文件的SHA256、从发布副本重画、135406行绘图数值逐字节一致。迁移训练入口保留原训练代码与参数；本次上传没有另开全量训练。
