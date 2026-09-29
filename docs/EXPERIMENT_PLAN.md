# 待做实验执行计划

依据手册v0.1，保留旧稿范围冲突，不用已查看数据伪装新的最终测试。完整本地包另有38张任务卡和机器登记。本计划中的天数仅工程安排建议，非正式训练时长或完成承诺；正式GPU/样本预算由作者冻结。

## P0先行，不先堆seed

P0：角色与未来访问、固定参考/经验edge、probe隔离、完整结构状态/接受日志、正确RRB/TAEF、统计合同。P1：Q1–Q3主证据。P2：全敏感性、制图、追踪与归档。

A01–A05冻结/复核完成后才进入相关正式运行。Gate A前可并行数值fixture、生成器、比较器实现、自然数据登记、算子单元测试；C03大规模主动必须等Gate A，D03最终必须等Gate B。AUTHOR INPUT未冻只做不依赖缺项的工程步骤。

## W0 资产与重放入口

工程建议1–2工作日，下载/传输和作者等待另计。校验6原件和全量清单，区分GitHub管理提交与完整本地包，使用本地Git LFS补传，保持私有与秘密/许可审查。

追回I1–I4父残差chunk、完整谱/白化参考、原manifest和生成6条件的代码版本；追回历史训练环境与commit。30终态checkpoint不等于全程在线状态。旧Windows路径在新adapter解析，不覆盖原manifest。提交A01缺失/迁移/用途/hash清单；31CPU测试和两次36步smoke只证明工程入口，不是新Ψ正式结果。

## W1 协议冻结与致命缺陷修复

工程建议3–7工作日，复杂状态继承可更长。先或并行取得作者关于正文/信号/数据/生成器/比较器/模型目标/算子/门控/候选的定义。

### W1a 信号、probe与数据权限

新实现单独命名，不改legacy。observe→pre-update predict→residual/diagnostic→update；保存预测状态ID。controlled h=id，自然冻结视觉读出权重、许可、层和维度。

white一次估计均值与迹缩放ridge；edge使用非重叠窗第k顺序统计；edgecov只审覆盖；cal定报警/持续/冷却，cover分早中晚盲检在线FAR。移除stage_name选择基线；不要拼接不同参考白化的η。

每角色不可变sample ID和交集报告；probe-train/val/test三分，跨时间共同评估集合，val早停不再充当test。候选warm/eval必须互斥且arrival<=当前步；禁止预生成未来T2/local-split池进入早期inspection。

### W1b 结构事务和终点

登记所有K/dc相关语义轴、optimizer一二阶槽/年龄、在线矩、RNG、游标；未登记状态硬失败，不按shape猜映射。父快照深拷贝无alias；候选/null同父、同warmup数据/顺序/更新数和公共随机前缀。记录pre/null/candidate三完整目标及组件，first-accept严格比较；reject不推进父RNG/cursor；accept原子提交语义映射。

依据作者最终目标修B全局规范、Lstr/Nref和KL/辅助项，不能为降低loss私改权重。逐六状态回归：拒绝仅清持续计数，接受清双buffer，冷却后重新积满。

旧MSE面积保留legacy_excess_mse_area；正式RRB=I_drop×min(R,horizon)，联报no-drop。精确一对一最大基数/最小时延TAEF，同时重算eligible机会零模型而不保留旧CE。真正resume使用持久ID和完整optimizer/RNG/cursor。

每个缺陷写失败回归：阶段污染、未来访问、probe重叠、AP ties、TAEF反例、null日志缺项、NaN/OOM/进程中断、原子提交失败。未过就停工程，不进入大规模主动。

## W2 B01–B08：先校准和反证，再最终比较

|任务|运行条件|交付/验收|
|---|---|---|
|B01/B02|W<d与W>=d、秩亏/近奇异/低能量/尺度/edge附近，float64 CPU/GPU|全谱、活动集、误差/失败、资源；冻结容差内一致，不删坏窗|
|B03|white/edge/edgecov/cal/cover互斥；快照覆盖与在线覆盖分离|经验edge、覆盖CI、早中晚FAR和不可校准条件；cover不能回调|
|B04|S1→T2→局部四分支；多种实质不同固定非线性渲染|moment/separability与区域概率合同、隐藏真值权限、probe清单；可重放|
|B05|固定拓扑从头运行，禁inspection|raw residual或可重生快照、谱、Ψ/能量/MSE/ELBO/KL、probe轨迹，全部计划单位入分母|
|B06|均值、统一尺度、各向同性噪声、平滑可逆、参考污染、充分结构、难度匹配|每条件匹配误差和误报；失败条件不删|
|B07|标准fixed、4×、CM-fix-aligned、CM-fix-total、adaptive-reference敏感性|新样本/访问/更新/FLOPs/墙钟/容量账本，不能只按参数量配平|
|B08|标量、谱/子空间、cov-CUSUM/EWMA、MMD或energy、C2ST、因果顺序法|权威实现/许可、各自hval搜索、方向/平局规则、cal工作点、失败率；冻结最强多变量比较器|

现有30run可排错和估成本，不能成为新最终比较的未查看测试。I1–I4父资产追回后B10可并行迁移，不回写训练。

## W3 Q1最终与Gate A

作者批准最高单位计划、停止规则、冻结哈希及新测试资产。受控方法按生成器内配对，seed/事件/窗口先聚合，主估计量是生成器内Ψ减预选最强多变量检测器AUROC。

联报事件AUROC、AUPRC定义与无信息基线、固定FAR检测率、误报/1000合格步、非负检测延迟。仅独立且更晚的probe持续退化可算probe-relative lead，同时报告未检测；跳变前报警算误报。

B10固定白化/窗口/edge/阈值，对同父做I1方向注入、I2明确能量参照、I3范数保持随机方向、I4缩放，保存匹配误差。B11在CLEAR/CLOC独立被动manifest运行，冻结视觉/随机/内部读出和至少一种非同构模型，按各自cal稳健标准化，不使用自然TAEF。

Gate A需B12 Fig1–2/SF1–3/ST1–4/Source Data、数值/校准/强基线/负对照/容量计算/参考漂移与最高单位主结果可解释。增量不足可接受负结果/缩小适用范围，不能看test调Ψ或借Q2性能回选Q1。

## W4 Q2主矩阵、审计和Gate B

每独立生成器×冻结嵌套seed配对5条件：Psi/fixed/magnitude/rate-matched-random/oracle参照。共享连续模型、参数门、算子/继承、候选池/排序、null、warmup/eval、first-accept和预算，仅第二结构门不同。当前3条件×3seed pilot不等于该正式矩阵。

fixed/random按自身eligible序号及cal冻结inspection率；magnitude为trace(C)/d并匹配联合虚检查率；oracle真实边界后首个eligible仍过参数门。test accepted精确配平只能事后敏感性。

C03全事务日志；C05每accepted独立audit与同父null-warmed shadow，双方禁检，同后续数据/更新/随机前缀，明确是选择后配对诊断而非随机因果。

C06共同事件RRB主终点+no-drop，另报inspection/acceptance、精确TAEF、边界命中、无编辑、时延、机会CE和资源余差。未恢复封顶，E0=1未定义。C07同最大流前缀、复杂度有无、放宽硬上界、排序/池/warmup预算变体；先共同快照重放，再完整策略重跑。按生成器整曲线比较线性/幂律/饱和、留一预测；多T不增加n；满足手册条件才称次线性。

Gate B审查“除第二门外一致”、因果访问/回滚、主要时机收益或准确失败判断、完整资源差异。不能用高TAEF掩盖RRB没有收益。

## W5 自然流D01–D07

D01/D02可先准备；D03等Gate B/Q3批准。使用CLEAR/CLOC真实版本与时间索引，不是改名CIFAR人工流。冻结权威SimCLR/DINO、ER/A-GEM/DER++、合格fixed/fixed editor及动态方法角色；标签、教师、回放、额外视图等差异透明，对称hval调参。不能对齐者只作性能背景。

测试前共同可信外部事件与完整随访存在才事件分支RRB/no-drop+BWT，否则共同时间面积+BWT；两分支均forgetting/final。不得从模型/Ψ曲线造事件，自然不用TAEF。active/passive模型、阈值、残差、probe事件独立登记。

逐run记录参数初均峰末、候选瞬时容量、K/dc、新样本/重复访问/更新/FLOPs/墙钟/显存/持久临时存储，包括失败成本。CAC仅次级，连续继承与新增/删除/不可映射自由度不重复计费，报告两原分量和μ敏感性；映射缺失标不可用而非0。D07按独立训练run配对，Fig4/SF6/ST1,5,6；等效需事前界限、TOST90%CI。

## W6 E01–E06统计和归档

接口可先实现，正式输入等阶段验收。先分母、单位内聚合、配对与family，后配对均值差、双侧符号翻转（MC加plus-one）、最高层cluster bootstrap、曲线共同网格积分、BH与必要TOST。不得退化为窗口显著性。

建立claim→task→endpoint→comparison→family→panel→run/config/hash。Fig1–4/SF1–7/ST1–7和正文numbers.tex从Source Data自动生成，原值/显示值并存。抽成功/失败各一例，再从干净环境全量重建。缺面板保持planned/failed，不填模拟值，不编造归档DOI。

## 资源估算与工作分配

未冻结GPU型号、最终数据规模和单run成本，不能给出可信总GPU小时。工程建议W0 1–2天、W1 3–7天、各阶段fixture/登记/分析1–3天；正式训练、传输和作者等待另算，不据此承诺总工期。

对任务j记录生成器G、嵌套seed S、条件C、配置M、实测单次GPU小时h、存储b：N=G×S×C×M；GPU小时=N×h+校准/候选/null/探针/失败成本；存储=N×b+父残差/快照/派生资产。N不是独立统计n。用不接触test的成本pilot估峰值，动态编辑成本另报上下界，预算超限升级作者，不临时删条件/单位。

建议分工：集成agent负责A/资产，信号agent负责B01–B03，实验agent负责B04–B12，事务agent负责C01–C02，主动agent负责C03–C08，自然agent负责D，统计agent负责E；每个任务另有独立复核者。每轮交付命令、环境/代码/config/父hash、planned/complete/failed/incomplete、raw/derived/failures/resources、支持/不支持结论；执行agent不能自行accepted。每周看独立单位、阻塞、实际资源、下一Gate和偏离，不看显著性决定续跑。
