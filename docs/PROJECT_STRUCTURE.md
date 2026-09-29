# 分层结构与迁移边界

这是全量本地整理包的目标布局，远端实际传输以Git树和DELIVERY_STATUS为准。

```text
archive/uploads/             原6附件，原名原字节
legacy/                      handoff与spectral历史代码，只读
experiments/passive_xib/      spectral原样工作副本
experiments/active_soma/      handoff05原样工作副本
experiments/interventions/   handoff04原样工作副本
src/soma_protocol/           新手册CPU参考，不静默改历史训练
results/imported/            被动30、干预12父组合、主动9pilot
results/generated/           新运行，唯一ID，默认忽略
docs/references/             原论文/手册/讲解PDF及页码文本
docs/historical_reports/     旧报告不覆盖
docs/tasks/                  38详细卡
registries/                  原件/全量hash、任务、旧稿scope决定
reports/audit_20260929/       本次描述性重算与工程证据
reports/local/               新分析/上传回执，默认忽略
scripts/                     校验、重算、预检、smoke、LFS补传
prompts/                     总控/P0/Q1/Q2/Q3/统计
```

不把多个项目的data.py/models.py/utils.py全部混进顶层。现active soma包保持嵌套，新参考叫soma_protocol；后续按任务迁移，不把批量挪目录当作算法已重构合规。

只去重完全相同提取副本，原ZIP始终完整。历史source/result、工程smoke、正式新测试三类分开。外部完整数据集、干预父残差和历史全线上状态未提供。公开再分发许可未确定，不擅自添加MIT/Apache或改公开。
