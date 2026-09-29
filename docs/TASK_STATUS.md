# A01–E06状态、现有证据与待补工作

审计范围：本次上传资产。A01 submitted、17 needs-revision、20 blocked；0项被本次擅自accepted。不是“项目0%完成”。完整本地包的registries/tasks.csv/json与docs/tasks含详细步骤、输入、依赖、交付、验收和原文页码；下表是远端便览。

|ID|任务|状态|已有与下一步|
|---|---|---|---|
|A01|资产盘点|submitted|6原件、287重复核对、30checkpoint；补许可证/父残差/生成版本，独立复核|
|A02|环境与最小运行|needs-revision|31CPU测试+两36步旧入口一致；缺正式环境锁、新Ψ全链路重放/GPU|
|A03|角色/标签权限|blocked|缺不可变role ID与交集证明；修stage基线、probe和未来candidate|
|A04|统计/停止冻结|blocked|作者定生成器/run数量、嵌套seed、终点、family、停止/重跑与等效界|
|A05|ID/日志/资源|needs-revision|已有CSV/JSON但事件混名、资源不全；统一schema与失败分母|
|B01|Ψ参考|needs-revision|新CPU参考可用；接入pre-update残差、固定white/经验edge并独立重算|
|B02|数值后端|needs-revision|CPU cov/Gram通过；补厚尾/近奇异/edge活动集与冻结GPU审计|
|B03|edge与在线校准|blocked|主历史MP/EMA；补white/edge/edgecov/cal/cover与两种覆盖|
|B04|生成器/probe冻结|needs-revision|已有S1/T2/split；补独立渲染、moment/separability与三角色probe|
|B05|固定拓扑现象|needs-revision|controlled旧pilot3次z4；新Ψ多生成器从头跑，完整残差/谱/probe|
|B06|负对照|blocked|缺均值/尺度/噪声/平滑/参考污染/充分结构/难度匹配矩阵|
|B07|容量/计算/参考|needs-revision|若干维度扫描不等于controlled4×和CM-aligned/total；需实测资源|
|B08|强比较器预选|needs-revision|已有标量/谱；缺强多变量/子空间/顺序池、独立hval与matched-FAR|
|B09|Q1最终受控|blocked|旧30run已看；需新测试资产、冻结比较器与生成器级AUROC增量|
|B10|I1–I4|needs-revision|12父组合/214128行；补父残差、6条件生成版本、readout误差和I2能量定义|
|B11|自然被动/读出稳健|blocked|CIFAR/dSprites/Shapes3D不等于CLEAR/CLOC；补读出/非同构/敏感性|
|B12|Q1数据包|blocked|旧图表不等于Fig1–2/SF1–3/ST1–4正式Source Data|
|C01|算子与继承|needs-revision|四算子均有；补optimizer/矩/语义轴/RNG/cursor、复杂度/B规范和回滚|
|C02|双门/状态机|needs-revision|已有门控；补参数门校准、六状态边界、失败回滚和完整hash|
|C03|主动SOMA主运行|needs-revision|3signal pilot；Gate A未过，未来池必须修，新Ψ从头跑全事务|
|C04|触发对照|needs-revision|signal/fixed/random各3；缺magnitude/oracle、cal日程及逐字段公平|
|C05|audit/shadow|blocked|null接受代码有但null日志缺；未见独立audit和双方禁检shadow|
|C06|RRB/TAEF/机会|needs-revision|旧RRB是MSE面积、TAEF贪心有反例；补共同事件时长和机会重算|
|C07|自限/路径|blocked|单3000步不够；补长度前缀、硬上界、复杂度/排序/pool/warmup敏感性|
|C08|Q2数据包|blocked|需正确RRB主终点、完整资源/失败和Fig3/SF4–5/ST4–5|
|D01|CLEAR/CLOC登记|blocked|缺真实快照、许可、时间ID和共同事件资格|
|D02|自然基线资格|blocked|缺权威版本/标签回放教师资源/hval搜索与预选|
|D03|自然最终运行|blocked|未见CLEAR/CLOC主动结果；等Gate B后独立active从头跑|
|D04|自然终点|blocked|先冻事件/无事件分支，提RRB或共同时间面积+BWT/forgetting/final|
|D05|资源公平|blocked|缺初均峰末参数、候选峰值、访问/更新/FLOPs/墙钟/显存/存储|
|D06|自然机制/CAC|blocked|补语义映射、连续/结构原分量、μ敏感性；不可映射不补0|
|D07|Q3数据包|blocked|缺独立训练run配对与Fig4/SF6/ST1,5,6|
|E01|统一统计|needs-revision|描述性重算可用；正式需父键/配对、符号翻转、cluster bootstrap、BH/TOST|
|E02|主图/正文宏|blocked|缺正式Fig1–4与自动numbers.tex；旧占位不可填假数|
|E03|补图/完整敏感性|blocked|补SF1–7全网格/失败；旧稿额外面板需scope决定|
|E04|补表/登记|needs-revision|本次任务/资产管理可用；补ST1–7科学外键与唯一映射|
|E05|Source Data追踪|blocked|正式claim/task/endpoint/comparison/panel/run/hash闭环未齐|
|E06|重建/归档|needs-revision|本次包/脚本可用；缺完整论文干净环境重建、正式SHA/许可/持久归档|

## 旧稿遗漏/范围决策X01–X08

X01 S1/T2/S2相图和跨流形塌缩（新Ψ不能沿用旧logit）；X02 Shapes3D突变/渐变/循环；X03 dSprites/Shapes3D/MPI3D的MIG/DCI/FactorVAE及c/r分工；X04 few-shot、support/query和AUCshot；X05旧辅助项/当前辅助项的完整消融及排序/复杂度；X06补图额外基线名单冲突；X07读出/噪声/batch/窗口/候选预算网格；X08针对本次反例的故障/因果权限回归。

部分已归B02/B11/C07，不重复计数；旧稿独有内容需作者明确保留、降级探索性或删除主张。不得静默补默认值或把过时结果当成新协议证明。
