# 信号、MSE与全部下游探针合图

全部42/42个父条件已经完成，无失败。本次仅重新排图，未训练或改变原始结果。

每个数据集一组：每列一个维数；第一排新旧信号，第二排原始MSE，后续每排一个原探针任务。四数据集总合图包含同样的全部任务，便于在一张图片中查看。

三seed版为所有seed的均值与最小—最大范围；原seed版保留controlled seed0、其他seed10的实际曲线。数据点未经平滑；阴影不是置信区间。MSE是线性坐标，因此视觉形状会与此前对数坐标图不同，数值保持一致。

探针颜色：蓝=S1固定池，橙=S2固定池，绿=S3固定池；实线μ，虚线固定潜噪声z。自定义方向探针也按其原阶段S3完整纳入。controlled仍为原当前阶段θ普通准确率，不能等同独立旧任务保持率。

信号：橙=新Ψ参考0（左轴），紫=新Ψ参考1000（左轴），红=旧Ξ（右轴）。参考1000从1001批才可使用；旧Ξ预热无效区不补零。灰虚线为探针机会水平。

本次只展示阶段变化主轨迹。已运行稳定分支对照仍保留在原figures/稳定流对照_*，不混为独立seed。

## 图像

- [三seed_controlled_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/三seed_controlled_信号_MSE_全部探针_各维度.png)
- [原seed_controlled_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/原seed_controlled_信号_MSE_全部探针_各维度.png)
- [三seed_dsprites_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/三seed_dsprites_信号_MSE_全部探针_各维度.png)
- [原seed_dsprites_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/原seed_dsprites_信号_MSE_全部探针_各维度.png)
- [三seed_shapes3d_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/三seed_shapes3d_信号_MSE_全部探针_各维度.png)
- [原seed_shapes3d_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/原seed_shapes3d_信号_MSE_全部探针_各维度.png)
- [三seed_cifar_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/三seed_cifar_信号_MSE_全部探针_各维度.png)
- [原seed_cifar_信号_MSE_全部探针_各维度](../../figures/合图_信号_MSE_全部探针_20261117/原seed_cifar_信号_MSE_全部探针_各维度.png)
- [00_四数据集_全部维度_信号_MSE_全部探针_总合图](../../figures/合图_信号_MSE_全部探针_20261117/00_四数据集_全部维度_信号_MSE_全部探针_总合图.png)

代码重画需要本工程的原始结果与历史依赖：`.venv/bin/python tools/plot_combined_all.py`。打包图片可独立查看。

