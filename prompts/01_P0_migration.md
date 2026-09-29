# P0迁移agent

遵守AGENTS和总控提示词。认领A02/A03/B01/B02/C01/C02/C06纯工程子步骤，不解锁final。

先写失败回归：stage标签不得进signal；white冻结；edge非重叠第k顺序统计且低能量留分母；pre-update预测；probe三角色/共同时间ID；future candidate拒绝；四算子拒绝保持父model/optimizer/moments/RNG/cursor；接受清buffer再冷却；三目标完整日志；TAEF[0,4]/[3,7]/tol3；RRB时长与MSE面积；持久ID的跨进程resume。

复用src/soma_protocol CPU参考但检查边界，不能自动认证生产实现。必要时加float64充分统计量ring buffer，再从raw residual独立重算。GPU不可用标not-measured。复杂度/语义继承未定义先等作者，不能按shape猜。

不改archive/legacy/imported。新版本adapter与配置、最小fixture、diff、测试、历史影响和AUTHOR INPUT清单齐交；没有通过因果权限和回滚，不跑主动大实验。
