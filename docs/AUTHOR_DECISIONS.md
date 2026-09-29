# 作者冻结清单

全部为待决策，不是推荐默认值；工程fixture参数不能自动成为正式参数。每项需作者、独立复核者、时间、依据、配置hash与测试重新解锁条件。

|ID|需冻结内容|影响|
|---|---|---|
|F01|最终正文版本；旧Ξ主张与X01–X08保留/降级/删除|全部/E02–E05|
|F02|独立生成器与Q3训练数量、嵌套seed、停止/失败重跑、配对缺失|A04/正式推断|
|F03|数据版本/许可/ID/角色交集和final访问权限|A03/D01|
|F04|h、W、alpha、beta、floor、edge/cover窗数、步幅、数值与CPU/GPU容差|B01–B03|
|F05|自然读出来源/许可/权重hash/预处理/层/维度/归一化|B11/D03|
|F06|S1/T2/split长度、采样、固定渲染、噪声、局部区域/概率、moment/separability|B04–B07|
|F07|比较器权威实现、方向、hval搜索/主估计量/平局、cal FAR与持续|B08/B09|
|F08|I1强度与方向、I2能量匹配对象/等谱地位、I3空间、I4缩放、误差容差|B10|
|F09|结构先验/观测模型、KL权重、辅助项、Lstr编码/Nref、B规范与硬边界|max C01以后|
|F10|四算子证据Eop、尺度/平局、Kcand、初始化、语义轴和optimizer/矩继承|C01/C03/C07|
|F11|Gtheta/Gphi、M/H、检查/冷却、RMS梯度/异常和日程率|C02/C04|
|F12|已到达warm/eval/audit块、访问预算、随机前缀、warmup、delta_acc、shadow|C03–C05|
|F13|共同事件/无事件、probe下降/恢复/持续、随访上限、TAEF容差/机会零模型|B09/C06/D04|
|F14|endpoint/comparison/family/panel、符号翻转/cluster bootstrap/block、等效界、CAC映射和mu|A04/D06/E01|
|F15|硬件/驱动/软件锁、算力预算、超限审批、发布许可、正式commit与持久归档|A02/D05/E06|

先F01–F04/F06–F08支撑Q1；并行定义F09–F12做纯工程，但不跳Gate A。已有曲线已看，不能为了好结果反推决定。

记录模板：decision_id、state=author-input-needed、selected_value=null、alternatives、source/rationale、affected_tasks、approved_by、reviewed_by、frozen_at、config_sha256、test_assets_previously_viewed、new_test_asset_policy。不得编造签字、归档DOI或默认10seed正式计划。
