# SOMA 实验资产与协议审计

审计日期：2026-09-29。范围：6个上传原件、嵌套内容和当时为空的私有仓库。未访问用户其他机器；“未发现”仅表示本次资产中没有足以验证的证据，不断言用户从未执行过。

## 1. 总判断

已有实质性的探索性代码与实验结果，但不能直接作为当前手册Q1–Q3的正式证据。缺口不只是多跑seed，而是信号、时间/标签权限、目标、终点与状态追踪的版本错位。

本次状态：A01 submitted待复核；17项needs-revision；20项blocked；不擅自把任何任务标accepted。这不是“项目完成比例0%”。程序结束不等于手册任务验收，必须有冻结配置、原始/失败/资源日志、派生数据和独立复核。38项见TASK_STATUS；详细任务卡在全量本地包。

soma.pdf第4、5、7页分别把现象称“部分初步证据”，把独立诊断最终测试和自然流最终运行标待完成。旧论文过去时、n=10图注与[X]占位，不是相应实验资产已经完成的证明。

## 2. 原件与重复

三PDF：61页手册v0.1（2026-08-09）、60页Nature旧稿、18页讲解材料。三个ZIP：6文件spectral代码、123文件handoff、298文件四数据集检测器包。完整原件原名原字节保存在本地archive/uploads；SHA256见registries/uploads.json，远端仍待补传。

按SHA256核对123个handoff重复、6个spectral重复、158份结果文本重复，全部相同；287个重复文件不增加实验数量。独立spectral与handoff03源码并非全部相同，不能按名称覆盖。

四数据集包内嵌套结果.zip包含188文件：30个.pt、30个DONE及JSON/CSV。不能误称所有checkpoint缺失。安全抽查一个checkpoint（weights_only=True）键只有model/dataset/latent_dim，没有完整optimizer/RNG、固定白化状态和线上残差历史；终态模型不能重建全程谱。

## 3. 已做的被动检测：30次总计

|数据流|运行数|latent维度|每维seed数|
|---|---:|---|---:|
|controlled|3|4|3|
|dSprites|9|8、16、32|3|
|Shapes3D|9|16、32、64|3|
|CIFAR相关流（日志cifar）|9|16、32、64|3|

这30次都有DONE、历史summary的completed、3000步终点、train/probe历史和终态checkpoint。每次约303行信号日志，不是所有逐样本残差。3×1000阶段、batch64、特征维256、窗口512。阶段样本循环访问，不是CLEAR/CLOC自然单遍协议。

### 描述性复核

沿用旧标签：在已记录行中，stage_index>=2且local_step<=200为正类；每次42/303，比例0.138614。仅复核历史报告，不作新协议事件级推断、matched-FAR或显著性结论。不同latent维一起平均仅资产概览，9run不等于9独立生成器。

|数据流|MP/EMA-Ψ AUROC|MSE AUROC|−谱熵 AUROC（事后方向核对）|MP/EMA-Ψ AP|MSE AP|
|---|---:|---:|---:|---:|---:|
|controlled|0.824971|0.882382|0.838351|0.404668|0.429604|
|dSprites|0.724442|0.899086|0.835067|0.263502|0.440914|
|Shapes3D|0.646567|0.878368|0.643161|0.265760|0.464858|
|CIFAR|0.884470|0.362241|0.857771|0.551110|0.104097|

AP按同分数分组的average precision，不混同所有PR积分定义。旧_compute_auprc.py:5–17逐样本排序累加，未合并ties；对大量0分数Ψ会依赖同分数行次序。重算使用记录环境中的average_precision_score，原结果不覆盖。

低谱熵表示集中，旧报告把原始高熵方向用于“越大越风险”的比较，方向反了。原始熵AUROC约0.162/0.165/0.357/0.142；−熵只是已查看数据上的事后诊断，不能据此预选确认性基线。正式符号、阈值与最强比较器需独立hval冻结。

结论限于：旧信号响应随数据流变化，非普遍优于MSE；Shapes3D 0.646567也不满足历史文字“全部>=0.65”。AUROC排序差异不证明统计正交、结构必要性或固定seed过拟合。

### 不能作为B09验收的原因

passive_xib/signals_xib.py:122–151按真实stage_name建立/选择EMA白化；train_xib_unified.py:469–486把阶段传入信号，违反隐藏真值权限。signals_xib.py:184–186是MP解析edge，不是独立经验顺序统计。controlled也随机投影而非h=id，缩减协方差也不同于手册ridge。

train_xib_unified.py:342–393只有probe训练/验证，验证早停后又用同一验证准确率作为test；:410–418按global_step改变拆分seed，评估样本跨时间变化。两者需要修复，终态模型补不回历史。

已有代码forward发生在optimizer.step前，之后从保存的forward取残差；预测本身是pre-update，不能只看日志语句位置误判。真正问题是参考、阶段权限、数据角色与重放。

## 4. I1–I4已有结果：12父组合，214128行

z4/8/16/32四CSV各53532行，每维3父训练seed、每父2974窗口×6条件。条件：I0 parent、I1 directional、I2 isotropic、I2 equispectral、I3 norm-preserving random direction、I4 scaling。214128行不是独立n，12父训练组合也不是已证明的12独立生成器。

模式记录empirical:mass，不能把此组也笼统称MP。I4 2倍缩放ΔΨ=0，I3逐样本范数误差约5.1e−17，I1平均ΔΨ约7.12–7.53，I2等谱ΔΨ仅约1e−14数值误差，属于有价值的离线探索。

缺父残差chunk、完整谱/白化资产，manifest仍指本机Windows路径。parent_readout_abs_error最大值按维度约0.0305–0.0422，冻结容差缺失，不能说精确重放通过。现run_i1_i4_residual_interventions.py已只生成5条件，末尾说明删掉等谱I2；历史manifest是i2_mode=both且CSV6条件，生成版本不匹配。

现I2把新各向同性窗口中心能量匹配到parent，不是I1增加后的总能量；匹配对象与旧稿注入合同须作者冻结。I3保持逐样本范数并不保证去均值协方差能量逐窗不变，已见相对变化最大0.385，需报告。优先追回父资产/代码版本，不直接删旧等谱条件；干预不证明谱组织导致模型退化。

## 5. 主动SOMA已有9次pilot

seed 854007、891047466、995002589各跑signal/fixed/random，共9次3000步。代码和accepted日志均见split/merge/grow/prune四算子，不能因过时README说缺grow。

|条件|runs|实际inspection总数|accepted总数|final eval MSE均值|旧TAEF均值|终态dc|
|---|---:|---:|---:|---:|---:|---|
|signal|3|33|21|0.007024|0.289683|全1|
|fixed|3|36|15|0.011446|0.055556|全1|
|random|3|31|19|0.009899|0.150794|全1|

total_edit_events=300是每运行检查点条目数，不是300次inspection/accepted。终态K为1或2，未复现旧稿dc=2、K=4。signal在这批pilot MSE较低是描述事实，但协议问题未排除不能作为正式Ψ时机收益。旧BWT是base MSE−final MSE，不是probe准确率BWT；9次forgetting均0，不能声称降低遗忘。magnitude/oracle、独立audit/shadow和正式C07扫描没有找到运行包。

### P0：未来候选池

active_soma/scripts/run_controlled_stream_soma_psi.py:455–480预先materialize S1/T2/local-split的warmup/eval全部数据；soma/training/trainer.py:774–825去掉metadata后按cursor循环，不检查到达时间。

按现代码、配置与inspection日志可重放：candidate-size1536/batch128，每阶段12批；fixed/seed_854007前三检查90、290、620步依次消耗S1/T2/local-split池，主流边界却1001/2001。代码路径允许290步用T2、620步用local-split未来分布信息。历史缺逐样本arrival/hash，因此这是代码与日志推导，不是假称已核每次实际样本哈希。

warmup/eval互斥不等于因果合法。必须只用合法已到达样本，或作者明确批准且不包含未来分布的辅助协议；修复后从头重跑主动主比较。

### 其他关键问题

|问题|定位（相对active_soma，除说明外）|处理|
|---|---|---|
|MP/持续EMA，当前batch参与基线更新|soma/runtime/psi_signal.py:99–139；trainer.py:489–502|改新冻结white/经验edge，旧轨迹不能只换曲线|
|null公式存在但null目标未完整日志化|trainer.py:661–749|保留candidate<min(pre,null)正确思路，补三目标与组件|
|warmup/接受后新建optimizer|trainer.py:764–771、865–871|缺明确语义槽/年龄继承合同，不能宣称C01合规|
|候选索引推进training_generator|trainer.py:827–831|用隔离命名空间；证明父RNG/cursor回滚|
|复杂度仅K/边/dc加权数|soma/models/structure.py:51–56|不等于手册LK+Ld+LA+Pvar/2 logNref|
|B中心化后逐行截范数|soma/models/structure.py:93–101|不同缩放可破坏零均值，不是统一Frobenius恢复|
|TAEF按距离贪心|soma/runtime/operational_metrics.py:_match_steps_within_tolerance|edits[0,4]、boundaries[3,7]、tol3：贪心1对，最优2对；用精确匹配|
|RRB实际是MSE面积|scripts/make_controlled_soma_mse_bwt_report.py:96–139|mean(excess)×post_window单位MSE×step；手册RRB是截断恢复时长step；必须分名|
|resume新时间戳、checkpoint状态不全|passive_xib/train_xib_unified.py:441–449、524–526|不能保证断点恢复；持久ID和全状态重放|

未动态验证的部分标未证明/需验证，不断言每次运行都触发所有错误。时间权限和主终点等问题修复前不进入大规模Q2/Q3。

## 6. 阶段缺口和遗漏

A：需要角色/作者冻结、原环境、状态/资源与独立复核。B：需要新Ψ全链路、强多变量比较器hval预选、负对照、controlled4×/compute/参考漂移、独立最终测试和CLEAR/CLOC被动。C：需要完整继承/回滚、未来池修复、5触发条件、audit/shadow、正确RRB/TAEF和自限。D：没有CLEAR/CLOC相应最终实验资产。E：描述性重算不等于最高单位正式统计、Fig1–4/SF1–7/ST1–7及逐点重建闭环。

旧稿易漏的X01–X08：S1/T2/S2相图/跨流形塌缩；Shapes3D突变/渐变/循环；dSprites/Shapes3D/MPI3D的MIG/DCI/FactorVAE和c/r分工；few-shot/AUCshot；辅助项/复杂度/排序完整消融；补图额外基线名单；读出/噪声/batch/窗口/搜索网格；具体故障与因果权限回归。部分已经在B02/B11/C07，不重复增加任务；旧稿独有部分由作者决定保留/探索/删除。新Ψ无[0,1]界，旧Ξ的logit相图不能平移。

## 7. 本次实际验证

新增独立CPU工程参考：float64固定white/ridge、非重叠经验edge顺序统计、全部超边缘质量、低能量与covariance/Gram；精确TAEF与RRB时长；角色和未来到达检查。31个unittest通过，包括30个随机小匹配与穷举一致。不是完整模型集成、GPU或事务验收。

两个独立CPU进程各跑旧被动入口36步，train_history SHA256相同：67ebfc3c4ba9e9a2fcb5ec605ad5814aae5604faa7c8e28ab927c9d755a8faa5。描述性重算两次全部输出字节一致；正式冻结草案检查按预期退出2。以上是工程/审计证据，不是新协议科学结果。

全量本地包中的scripts/recompute_imported.py从历史CSV重算，保存输入哈希；scripts/verify_assets.py --full核验文件。详细原始路径、逐run表、完整任务卡与测试输出在本地reports/audit_20260929、registries和docs/tasks；远端未补传前不要假定这些历史资产已经存在。
