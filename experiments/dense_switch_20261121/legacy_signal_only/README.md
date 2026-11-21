# 四数据集：仅旧信号Ξ

每条变化流均覆盖1–3000步，使用全部60条件、每种维数全部三个seed。每个数据集一张图：五行对应五种潜维数，三列依次是完整3000步、970–1030步、1970–2030步。两次切换后第一批分别为1001、2001，局部各61个真实测点。

红线为3seed均值，浅红为最小–最大范围；不是置信区间。各面板独立线性纵轴，按刻度比较；不平滑、不归一化、不画新信号。预热无效值留空并标灰区，不把日志占位0当有效观测。

旧Ξ=relu(最大白化协方差特征值−MP边缘)，使用历史EMA/收缩白化及已知stage路由。前20个新stage批次可用上一stage基线fallback，第21批转为当前stage基线；尖峰不等于盲检测证据。

controlled完整逐步数据来自已完成且已核验的确定性重放native/train_history.csv；图像来自原逐批legacy_every_batch.csv。全部7320个窗口重叠点逐项一致；controlled另核对原稀疏日志。没有重新训练、修改信号或改动历史结果。

- [四数据集叠加总览](figures/all_four_datasets_old_xi_overview.png)

- [controlled：全部维数，完整与局部](figures/controlled_old_xi_full_and_windows.png) · [SVG](figures/controlled_old_xi_full_and_windows.svg)
- [dSprites：全部维数，完整与局部](figures/dsprites_old_xi_full_and_windows.png) · [SVG](figures/dsprites_old_xi_full_and_windows.svg)
- [Shapes3D：全部维数，完整与局部](figures/shapes3d_old_xi_full_and_windows.png) · [SVG](figures/shapes3d_old_xi_full_and_windows.svg)
- [CIFAR / CIFAR-C：全部维数，完整与局部](figures/cifar_old_xi_full_and_windows.png) · [SVG](figures/cifar_old_xi_full_and_windows.svg)

每数据集CSV保留全部原始旧Ξ、有效标记和绘图值；plot_aggregates.csv记录绘图聚合值；validation.json保存源文件哈希和一致性检查。旧7月源码缺失部分仍按历史协议重建标签保留。

复画命令（仅读取日志，依赖numpy和matplotlib）：

```bash
python plot_legacy_xi.py --source-workspace /path/to/outputs --output /path/to/new_output
# GitHub分开存放两个归档时，另传 --prior-workspace /path/to/prior_archive/workspace
```
