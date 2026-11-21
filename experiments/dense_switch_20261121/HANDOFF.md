# 发布交接：2026-11-21

任务：上传本次完整实验结果和新图表至原GitHub仓库。目标仓库`r1293870729-dev/soma`，分支`experiment/lowdim-dense-results-20261121`，基线`2d012be74ecdcf35b4f8ae99bc34e987bbe1bba9`。此前42条件已由PR #6合并；本次追加18条件及60条件逐步补测，不重写历史。

## 文件和来源

新增目录`results/generated/soma_dense_switch_20261121/workspace`完整保留两个本地项目：低维/冻结探针项目2,920文件，逐步补测项目940文件，另3个controlled下载别名文件，共3,863文件、10,243,920,790字节。原模型、原图ID、角色、配置、校准、日志、快照、数值异常和原分析报告按原字节保存。上批源依赖和原42结果由同提交树内的旧归档提供，不重复复制20.9 GB。

`asset_manifest.json`登记每文件来源、字节数和SHA256；`figure_alias_manifest.json`登记10个英文图名副本。`publication_summary.json`登记条件数、更新数和发布状态。`export_assets.py`是本次使用的字节保留导出工具；不导出虚拟环境、Git目录、凭据或Python缓存，不跟随环境符号链接。

## 验证

1. 原件与发布副本3,863文件SHA256逐项一致，10个图表别名与来源一致；扫描导出文本中的私钥及GitHub token模式通过。
2. 从迁移后的归档源文件重画32 PNG和32 SVG；60/60条件均来自完成的精确重放。67,680行绘图数值CSV逐字节一致，SHA256为`599690923b6a0ab2e0e2e38343193b2d52c5d681e56deccb1e5ffc2ca237e354`。
3. 发布操作新增模型更新0次；原协议、权重、数据、模型、参考和checkpoint未修改。迁移器仅在独立绘图进程内解析绝对路径，原文件字节不变。
4. 上传过程使用Git LFS，完成后另检查远端对象存在并下载代表性二进制样本核验。最终远端提交和传输回执见`upload_verification.json`（完成传输后生成），不能仅凭准备清单声称已经上传完整。

本地绘图环境为既有Python 3.12环境，numpy 2.0.2、matplotlib 3.9.4；训练环境版本见requirements-train.txt。未执行依赖安装或新增训练。原运行资源和完成状态分别见两个归档项目的reports/LIVE_PROGRESS.json、低维reports/monitor_complete.json和逐步reports/completion_receipt.json。

## 实验状态和限制

20个数据集×潜维数组合，每组3 seed，共60条件；低维新增18/18、图像冻结头45/45、预训练特征控制9/9、逐步补测60/60完成。新增训练90,000次更新；确定性重放227,700次更新不能算新增独立试验。每条变化流3000步；图像同父稳定流从1000后续2000步。

报告与所有seed完整保留。controlled为历史当前池θ普通准确率，图像为独立角色的固定S1/当前阶段BA和冻结头诊断。数值饱和、当前池难度变化、冻结头坐标漂移、旧信号已知stage路由均已说明。支持容量与可读性的描述性比较；不支持普遍的即时遗忘、盲检测能力优越性、因果结论或手册Q1–Q3正式验收。没有独立审稿人accepted，也未解锁最终测试。

## 可重复命令

```bash
python experiments/dense_switch_20261121/run_archive.py --verify
python experiments/dense_switch_20261121/run_archive.py --points-only --output ../soma_dense_values
python experiments/dense_switch_20261121/run_archive.py --plot --output ../soma_dense_figures
```

上述命令需要完整LFS对象；便携入口只核验和重画，没有训练模式。原始训练、回放和监督器源码全部在归档中，启动新训练需另设输出目录和真实数据输入。

## 仅旧信号图补充

`legacy_signal_only/`保存五组PNG/SVG、四个逐seed CSV、一个绘图聚合CSV、脚本、源文件哈希、一致性验证和下载包。四个数据集各五种维数、各三个seed，全部1–3000步，共180,000行。全部7,320个切换窗口重叠点、controlled原4,545条稀疏记录与逐步重放一致。每图展示完整轨迹及两次±30步窗口，不画新信号、不改公式、不平滑，预热无效值留空。图表保留历史已知stage路由标签。新训练更新0次，模型和原归档文件无修改。复画脚本可用`--source-workspace`与`--prior-workspace`指定前后两个归档工作区。
