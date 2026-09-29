# 论文、讲解材料、手册与代码的版本冲突

记录差异，不代替作者决定最终论文版本。暂以手册v0.1作为工程对齐目标；历史论文/程序全部保留。未提供手册引用的2026-08-09完整正文/LaTeX工作区，不能确认当前Nature PDF就是最终正文。

|主题|旧稿/历史代码|手册v0.1目标|处置|
|---|---|---|---|
|信号|旧稿p14：Ξ=1−H/log r；现主代码为MP边缘质量和|固定white+独立经验edge的全部超边缘相对质量Ψ|三者单独命名；不能只换图中变量名|
|白化|被动stage-keyed EMA，主动持续EMA|独立white固定模型快照，均值和矩阵冻结|adaptive reference仅敏感性|
|协方差正则|(1−α)Σ+α trΣ/d I及epsilon|Σ+α trΣ/d I|不能直接复用α默认值|
|读出|controlled也随机投影|controlled h=id；自然冻结视觉读出|登记读出及权重版本|
|edge|MP解析边缘；干预empirical:mass缺完整校准资产|非重叠edge窗第ceil((1−β)B)顺序统计，独立edgecov|不以理论边缘或插值分位数替代|
|模型目标|旧稿A连续非负、dc×K的B、Ldis/Lorth/Lsym、计数复杂度|二值对称零对角A、K×dc的B、Rsep/Ruse、完整Lstr/Nref|作者先定最终版本；不直接平移旧消融|
|接受式|旧稿仅pre；当前trainer已有min(pre,null)|candidate<min(pre,null)−δacc，隔离、同预算与冻结eval|保留正确实现思路；补null日志和状态/RNG继承|
|独立候选块|warmup/eval预生成整个未来流|既角色互斥，也必须已合法到达|“独立”不能合法化未来分布暴露|
|触发对照|旧稿按test候选数匹配|cal预期inspection率，各自eligible日程|test accepted-matching只作事后敏感性|
|统计单位|常按seed或混合维度平均|受控独立生成器，seed嵌套；自然独立训练run|冻结父键与配对，不伪增n|
|lead time|旧稿变化点前报警，0/删失口径冲突|瞬时跳变检测延迟非负；仅对独立后续probe退化谈提前|跳变前报警是误报|
|恢复/BWT|MSE面积被称RRB，loss版BWT|RRB是I_drop×min(R,τpost)时长，联报no-drop|旧指标保留legacy标签，正式终点另算|
|自然事件|旧稿把时间桶当变化事件|可信共同外部事件才事件分支；否则共同时间面积+BWT|不从曲线造事件，自然不用TAEF|
|等效性|差异不显著据此称相当|事前等效界限、TOST、90%CI落入界限|P>0.05不是等效|
|补充范围|logit(Ξ)相图、解耦/fewshot和不一致基线名单|38任务未逐项列所有旧图面板|X01–X08逐项保留/降级/删除；Ψ无[0,1]界，旧logit不能直接用|

来源：Nature_SOMA_legacy.pdf p14–23、24–34、38–45、49–60；experiment_manual.pdf第3–9章和附录B；soma_overview.pdf p10、14–17。PDF原件在本地全量包，远端补传前不假定存在。

协议变更必须保留旧配置/结果，记录原因、父版本/hash、影响任务与图。已查看测试集的修改应标exploratory或使用新最终资产重新冻结；agent不得替作者冻结AUTHOR INPUT。
