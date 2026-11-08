# SOMA 上传清单（2026-11-08）

本目录记录本轮整理资料。注意：本轮通过 GitHub API 创建的是 UTF-8 文档；PDF/ZIP/DOCX 原始二进制没有被伪装成已上传。reports/source_reports 下的 .txt 是从用户上传 PDF 提取并整理的可检索文本摘要，不是原 PDF 完整二进制。

## 本轮实际写入仓库
- docs/review_20261108/SOMA_实验手册逐项核对与通俗待办清单.md：手册逐项核对、历史数据矛盾、通俗解释和下一步顺序。
- docs/review_20261108/FINAL_B_SIGNAL_SPEC_20261024.md：最终 B/mass 信号定义与历史版本边界。
- docs/review_20261108/TASK_STATUS_B_FINAL_20261024.md：2026-10-24 最新 38 项任务状态快照。
- docs/review_20261108/README.md：目录入口。
- docs/review_20261108/UPLOAD_MANIFEST.md：本清单。
- reports/source_reports/ 下的五份历史报告文本摘要：Xi-B、detector comparison、hyperparameter validation、I1-I4、controlled SOMA pilot。

## 当前对话中存在、但尚未上传为原始二进制的文件
- experiment_manual.pdf（手册 v0.1）
- Nature_SOMA(1).pdf（旧稿，含占位符，不能视为最终论文）
- soma.pdf（18 页讲解材料）
- Xi-B_signal_report_20260826_final.pdf
- detector_comparison_report_20260901_v4.pdf
- hyperparam_validation_report_v3.pdf
- i1i4_clear_report.pdf
- controlled_soma_psi_report.pdf
- SOMA_阶段实验盘点与汇报讲稿_20261108.docx
- SOMA_实验手册逐项核对与通俗待办清单_20261108.docx

## 大型原始包暂缓
- soma已有资料.zip：约 475 MB（解压后约 578 MB）。
- 四个数据集检测器效果.zip：约 159 MB。
普通 GitHub Git 单文件限制为 100 MB；这两份包还需先检查数据许可、模型权重和第三方数据内容。应在确认许可后通过 Git LFS 或外部数据存储迁移，并附 SHA256 manifest。

## 重要版本说明
- 2026-09-29 的 30-run 审计账本与 2026-08-26 Xi-B PDF 的 39 个配置-seed 推算不一致，尚待逐 run 对账。
- 旧 MP/EMA-B 与最终 B_mass_fixed_white_empirical_v1 是同一信号家族的不同测量链版本，不能混为同一结果。
- 2026-10-24 的 61 项测试和两个 600-step CPU 开发运行是工程证据，不是 Q1/Q2/Q3 正式实验验收。
