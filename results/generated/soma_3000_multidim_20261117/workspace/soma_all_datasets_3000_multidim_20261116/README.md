# SOMA：四个数据集、3000步、全部潜维数

已完成42/42个父条件，无失败；所有主轨迹均按1000/1000/1000步运行。controlled为2/4/8/16/32维；dSprites为8/16/32维；Shapes3D与CIFAR为16/32/64维，各三seed。

- [信号、MSE、全部探针：四数据集和各维度总合图](figures/合图_信号_MSE_全部探针_20261117/00_四数据集_全部维度_信号_MSE_全部探针_总合图.png)
- [合图说明与清晰分图](reports/combined_20261117/README.md)
- [实时进度](reports/LIVE_PROGRESS.md)
- [结果与图像入口](reports/RESULTS.md)
- [controlled原seed0五维度图](figures/单seed_controlled_0_多维度.png)
- [四数据集总览（完成后自动更新）](figures/00_四数据集_全部维度总览.png)
- [本轮协议](docs/PROTOCOL.md)
- [信号、探针与时间定义](docs/SIGNALS_AND_PROBES.md)

终端查看进度条：

```bash
cd /Users/rosyun/Documents/Codex/2026-11-10/ze/outputs/soma_all_datasets_3000_multidim_20261116
.venv/bin/python tools/status.py --watch
```

Ctrl+C只退出查看器，不停止训练。后台四个训练进程写入logs/train_*.log；监视与出图进程写入logs/monitor_v2.log。图像训练每个父条件含同一1000步模型分叉的稳定S1对照；额外计算不会把主曲线改成5000步。

图表随完整条件完成更新，n不足3或空列表示尚未完成。最终42个父条件全部完成后，自动保存PNG、SVG、原始测点CSV与`3000步多维度_图表与代码.zip`。压缩包包含本轮代码、依赖路径/哈希、报告、图和日志表格；完整本地数据及历史源码依赖仍按configs/frozen.json中的路径引用。
