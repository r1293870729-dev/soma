# 四数据集多维度实验：2/4维补充与逐步切换图

本次归档补充图像数据集的18个低维训练条件，并对已有全部60个条件在结构切换前后逐步补测。包括全部新增代码、模型与探针快照、原始指标、分析报告和新可视化。此前42个条件及历史依赖保留在[上一批归档](../multidim_3000_20261117/README.md)。本次发布没有启动训练或修改原结果。

| 数据集 | 潜维数 | seed | 完整条件 |
|---|---|---|---:|
| controlled | 2 / 4 / 8 / 16 / 32 | 0 / 1 / 2 | 15 |
| dSprites | 2 / 4 / 8 / 16 / 32 | 10 / 11 / 12 | 15 |
| Shapes3D | 2 / 4 / 16 / 32 / 64 | 10 / 11 / 12 | 15 |
| CIFAR / CIFAR-C | 2 / 4 / 16 / 32 / 64 | 10 / 11 / 12 | 15 |

每条变化流为1000+1000+1000步。45条图像轨迹在1000次更新后以同模型和完整Adam状态分叉S1稳定流，再训练2000步，两分支横轴均截止3000。低维补充增加90,000次模型更新；逐步补测另有227,700次确定性重放更新，**没有增加实验方法或独立条件**。

## 新可视化

四张总图均为全部五种维数、每维全部三个seed，信号、MSE和探针在同一张图中。切换窗口为970–1030和1970–2030，各61个真实测点；每步画点、不平滑、不插值，探针使用真实指标和标明刻度的局部纵轴。阴影是seed最小—最大范围，不是置信区间。

| 数据集 | 总图 PNG | 矢量 SVG |
|---|---|---|
| controlled | [全部维数](figures/controlled_all_dimensions.png) | [SVG](figures/controlled_all_dimensions.svg) |
| dSprites | [全部维数](figures/dsprites_all_dimensions.png) | [SVG](figures/dsprites_all_dimensions.svg) |
| Shapes3D | [全部维数](figures/shapes3d_all_dimensions.png) | [SVG](figures/shapes3d_all_dimensions.svg) |
| CIFAR / CIFAR-C | [全部维数](figures/cifar_all_dimensions.png) | [SVG](figures/cifar_all_dimensions.svg) |

- [完整图册：32张逐步图，含每次切换及单维数全部任务](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/reports/GALLERY.md)
- [全部逐步图下载包](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/四数据集_逐步切换窗口图.zip)
- [controlled全部维数下载包](../../results/generated/soma_dense_switch_20261121/workspace/downloads/controlled_all_dimensions.zip)
- [700–3000步四数据集概览](figures/all_datasets_overview.png) · [SVG](figures/all_datasets_overview.svg)
- [宽窗口、全部因子、μ/z、冻结头及分析图册](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/reports/GALLERY.md)
- [完整训练结果分析与四张汇总图](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/reports/final_analysis_20261119/REPORT.md)

## 指标和结论边界

图像训练固定重构权重1、KL权重0.0001；dSprites训练重构项为BCE，另画像素MSE，其余图像训练重构项为MSE。controlled保留历史rec+0.001KL+0.001旧Ξ，θ为当前阶段训练池80/20回测的普通准确率。两类探针不作为同一个指标横向排名。

图像“固定S1”固定评估原图池，但每个时点重拟合线性头；“当前阶段”更换评估支持；“冻结S1头”固定到达步900时的标准化和分类器，下降可能包含表示坐标漂移。“S1稳定流”是模型在S1继续训练的配对对照，不是固定不更新的模型。BA为各类别召回率的等权平均；不同数据集的任务和机会水平不同。

新Ψ使用手册固定参考、迹归一化、独立经验edge和全部超边缘相对质量，保留ref0和ref1000；ref1000从1001步起。旧Ξ为历史stage-keyed EMA/收缩白化/MP最大特征值绝对超额，使用已知stage路由。图像两者还使用不同残差空间：新信号为冻结ResNet18特征，旧信号为投影像素残差，因此不是单一公式消融，也不能按峰高直接判优。

全部18/18新增父条件、45/45冻结头回放、9/9预训练特征控制及60/60逐步补测完成，无任务运行失败。完整报告保留数值饱和和不符合预期的结果：固定旧任务并未在每次切换后普遍大跌；dSprites 4维有较晚的相对劣化；部分Shapes3D稳定分支有logvar裁剪饱和。当前为开发与探索诊断，未建立信号幅度普遍预测旧任务损失的结论，未完成手册Q1–Q3确认性验收。

## 代码、原始结果与校验

- [低维训练及冻结头回放代码](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/src) · [分析与绘图代码](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/tools)
- [逐步补测与绘图全部代码](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/src)
- [新增模型与探针结果](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/results) · [逐步测量原始结果](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/results)
- [低维协议](../../results/generated/soma_dense_switch_20261121/workspace/soma_lowdim_switch_probe_20261117/docs/PROTOCOL.md) · [逐步补测协议](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/docs/PROTOCOL.md)
- [67,680行逐seed绘图数值](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/reports/plot_points_by_seed.csv) · [原图源文件哈希](../../results/generated/soma_dense_switch_20261121/workspace/soma_dense_switch_windows_20261119/reports/figure_manifest.json)
- [3,863个新增归档文件的SHA256](../../results/generated/soma_dense_switch_20261121/asset_manifest.json) · [发布清单](publication_summary.json) · [便携重画核验](plot_verification.json) · [交接说明](HANDOFF.md)

新增归档原文件10,243,920,790字节，与原件逐文件一致；另有10个便于浏览的英文文件名图表副本。模型、数值缓存、JSONL和压缩包使用Git LFS；PNG/SVG直接在仓库浏览。原始数据集是独立输入，本次不重复上传，来源与哈希见[原始输入登记](raw_inputs.json)。虚拟环境、凭据、Python缓存没有导出。原代码中的本机绝对路径按原字节保留，重画使用独立迁移入口。

## 获取与重画

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch experiment/lowdim-dense-results-20261121 https://github.com/r1293870729-dev/soma.git
cd soma
git lfs pull
python -m pip install -r experiments/dense_switch_20261121/requirements-plot.txt
python experiments/dense_switch_20261121/run_archive.py --verify
python experiments/dense_switch_20261121/run_archive.py --plot --output ../soma_dense_replot
```

完整LFS取回包含主分支已有的约20.9 GB归档及本次新增资产。只查看图表可直接点上方链接；GitHub Source ZIP不能替代`git lfs pull`。只核验绘图数值可使用`--points-only`代替`--plot`。中文显示需要系统中文字体；原图使用Mac的Arial Unicode MS。重画只写新目录，不更新模型，不回写归档。

已从归档副本重画32 PNG与32 SVG；67,680行数值CSV与原表SHA256完全一致。`--verify`核验新增全部资产，原42条件可另用上一批`run_archive.py --verify`核验。重新训练需要原始数据和历史依赖，不能将图表目录当独立训练包。
