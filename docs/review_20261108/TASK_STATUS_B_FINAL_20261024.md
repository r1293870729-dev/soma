# 38项任务状态（2026-10-24 B定稿版）

状态原则：B家族选择已完成，但科研任务不会自动升级为 accepted。下面是当时的最新任务状态与需要补的证据。

| ID | 状态 | 主要缺口 |
|---|---|---|
| A01 | needs-revision | 历史资产按信号家族和测量链登记；旧MP/EMA不自动升级 |
| A02 | needs-revision | CPU B参考已开发；补正式环境与GPU审计 |
| A03 | blocked | 全局source_id角色互斥与未来到达权限 |
| A04 | blocked | 冻结独立单位、终点、停止规则和比较族 |
| A05 | needs-revision | 日志需包含B版本、reference_id、活动模态、原始值和资源 |
| B01 | needs-revision | B质量求和与解析小例、B-A关系 |
| B02 | needs-revision | covariance/Gram、CPU/GPU、临界edge活动集 |
| B03 | blocked | 固定经验edge；edgecov与在线cover分离 |
| B04 | blocked | 生成器moment matching、独立可分性和probe冻结 |
| B05 | blocked | 新B固定拓扑完整轨迹 |
| B06 | blocked | 均值/尺度/噪声/平滑/充分结构等负对照 |
| B07 | blocked | 容量/计算/参考敏感性 |
| B08 | blocked | 强检测器hval预选、B-A增量和活动模态审计 |
| B09 | blocked | 新最终测试流上的B增量验证 |
| B10 | blocked | I1-I4父残差/参考/条件版本和能量匹配误差 |
| B11 | blocked | CLEAR/CLOC被动监测与读出稳健性 |
| B12 | blocked | Q1图表和Source Data |
| C01 | blocked | 四算子完整状态继承和回滚 |
| C02 | blocked | 双门、六状态机和阈值校准 |
| C03 | blocked | 用最终B从合法初态重跑主动SOMA |
| C04 | blocked | B/fixed/magnitude/random/oracle五条件 |
| C05 | blocked | 独立audit与matched no-edit shadow |
| C06 | blocked | 正确RRB、精确TAEF和机会校正 |
| C07 | blocked | 长度前缀、硬上界、路径依赖和循环测试 |
| C08 | blocked | Q2图表和Source Data |
| D01 | blocked | CLEAR/CLOC数据快照、许可、时间与事件登记 |
| D02 | blocked | 基线标签权限、回放/教师资源与hval预选 |
| D03 | blocked | Gate B后自然流独立最终运行 |
| D04 | blocked | 事件/无事件终点与BWT |
| D05 | blocked | 参数、访问、FLOPs、墙钟、显存和存储账本 |
| D06 | blocked | B/检查/接受/结构/CAC机制轨迹 |
| D07 | blocked | Q3图表和Source Data |
| E01 | blocked | 最高独立单位配对统计、bootstrap、FDR、等效性 |
| E02 | blocked | 自动主图与正文数字 |
| E03 | blocked | 完整敏感性补图 |
| E04 | blocked | 科学补表和机器可读登记 |
| E05 | blocked | claim到run/hash/图点的Source Data追踪 |
| E06 | blocked | 干净环境一键重建、SHA、许可和持久归档 |

## 通俗解释
- Q1：先证明报警器本身可靠，不让它控制模型。
- Q2：再证明报警后做结构编辑确实比固定时间/随机/幅度触发更有用。
- Q3：最后证明真实自然数据流中收益超过算力、内存和编辑成本。
