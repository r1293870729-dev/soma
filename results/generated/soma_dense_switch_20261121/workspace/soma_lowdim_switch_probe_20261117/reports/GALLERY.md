# 结构切换聚焦图与2/4维补充

完整父条件60/60，新增低维18/18，冻结头重算45/45。

概览图将固定S1、当前阶段、900步冻结头分开；切换窗口图报告稳定对照校正后的百分点；全任务图保留S2/S3新增任务。所有原始轨迹仍为3000步，只将显示起点设为700。

当前阶段换了评估支持，分数跳变不等于旧任务遗忘；冻结头下降也可能是可逆坐标漂移。原重拟合S1探针和稳定对照共同用于解释。

- [全任务_cifar_信号_MSE_各维数](../figures/全任务_cifar_信号_MSE_各维数.png)
- [全任务_dsprites_信号_MSE_各维数](../figures/全任务_dsprites_信号_MSE_各维数.png)
- [全任务_shapes3d_信号_MSE_各维数](../figures/全任务_shapes3d_信号_MSE_各维数.png)
- [切换窗口_cifar_全部维数](../figures/切换窗口_cifar_全部维数.png)
- [切换窗口_dsprites_全部维数](../figures/切换窗口_dsprites_全部维数.png)
- [切换窗口_shapes3d_全部维数](../figures/切换窗口_shapes3d_全部维数.png)
- [总览_四数据集_全部维数_新旧信号_MSE_探针](../figures/总览_四数据集_全部维数_新旧信号_MSE_探针.png)
- [概览_cifar_全部维数_信号_MSE_三种探针](../figures/概览_cifar_全部维数_信号_MSE_三种探针.png)
- [概览_controlled_全部维数_信号_MSE_三种探针](../figures/概览_controlled_全部维数_信号_MSE_三种探针.png)
- [概览_dsprites_全部维数_信号_MSE_三种探针](../figures/概览_dsprites_全部维数_信号_MSE_三种探针.png)
- [概览_shapes3d_全部维数_信号_MSE_三种探针](../figures/概览_shapes3d_全部维数_信号_MSE_三种探针.png)
- [解释_平衡准确率_低分与采样噪声](../figures/解释_平衡准确率_低分与采样噪声.png)
