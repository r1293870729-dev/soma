# SOMA · 残差谱凝聚与双门控结构适应

**当前状态：已有探索性实验与部分工程验证，尚无按当前手册完整验收的 Q1–Q3 正式证据。**

本次历史盘点包含30次被动检测运行（2026-09-29审计账本）、12个父训练组合的I1–I4结果、9次主动SOMA pilot。注意：2026-08-26 Xi-B汇总报告按表格配置和每组3 seeds推算为39个配置-seed组合，与30-run审计账本不一致，需逐run对账；不得混用两个计数。

## 最新核对（2026-11-08）

- [实验手册逐项核对与通俗待办清单](docs/review_20261108/SOMA_实验手册逐项核对与通俗待办清单.md)：逐项对照A01–E06共38项，解释Q1/Q2/Q3、历史结果和协议缺口。
- [本轮上传清单](docs/review_20261108/UPLOAD_MANIFEST.md)：区分已写入GitHub的文本与尚未上传的原始PDF/ZIP/DOCX。
- [核对目录README](docs/review_20261108/README.md)。

## 原有项目导航

- [实验审计](docs/EXPERIMENT_AUDIT.md)：历史结果、数字复核、协议问题与结论边界。
- [待做实验计划](docs/EXPERIMENT_PLAN.md)：P0整改、Q1→Gate A→Q2→Gate B→Q3→统计归档。
- [38项状态表](docs/TASK_STATUS.md)：已有证据、缺口与下一步。
- [协议冲突](docs/PROTOCOL_CONFLICTS.md)：旧Ξ、历史MP/EMA-Ψ、手册固定white/经验edge-Ψ不可混用。
- [本地agent提示词](prompts/00_local_agent_master.md) 与 [AGENTS.md](AGENTS.md)。
- [原件SHA256清单](registries/uploads.json)：pending表示尚未由本次连接完成大文件传输。

## 研究边界

Q1固定拓扑研究诊断增量；Q2只更换第二结构门研究恢复负担；Q3在CLEAR/CLOC独立主动流审计长期表现、保留与资源。未经作者冻结的AUTHOR INPUT、最高单位数量、终点、阈值、基线和停止规则不得自动补值。工程测试通过不等于论文任务accepted。

仓库保持私有。大型原始包需在许可检查后通过Git LFS或外部数据存储迁移；本轮没有声称原始PDF、DOCX和大型ZIP已作为二进制文件上传。
