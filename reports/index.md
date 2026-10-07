# CycleTCM 报告索引

报告按研究类别归档；质证实验使用 `validation/日期/`，表、图、配对区间与来源放在同一报告目录的实验子目录中。原始图像、逐图预测和模型权重保留在本地 outputs/ 与 data/，不随报告发布。

| 类别 | 入口 | 内容 |
| --- | --- | --- |
| 2026-10-07 质证 | [完整报告](validation/20261007/report.md) | MLLM-only 控制、效率、Qwen 提示词、多种子证据、MedGemma P0 |
| 质证方案 | [方案与归档提示词](validation/20261007/plan.md) | 问题、实验定义、P0 与 A0–A2 原文 |
| 机器可读汇总 | [results.json](validation/20261007/results.json) | 对应报告的聚合结果与配对区间 |
| 正式复现 | [论文式报告](reproduction/20261005_222333_753567_analysis/paper_report.md) | 14 个正式运行、多种子与逐类分析、定性图 |
| 正式队列归档 | [队列汇总](reproduction/20261005_042629_774918_suite/reproduction_report.md) | 固定配置、队列状态与原始汇总 |
| 维护记录 | [maintenance/](maintenance/) | 实际输出清理与路径迁移清单 |

2026-10-07 的实验数据目录：

- [feature_controls/](validation/20261007/feature_controls/)：常量、随机与 Qwen MLLM-only 的聚合表、逐类指标及来源。
- [prompt_ablation/](validation/20261007/prompt_ablation/)：Qwen A0–A3 的表、图、提示词原文、A3 核查和配对区间。
- [medgemma_p0/](validation/20261007/medgemma_p0/)：visual、Qwen P0、MedGemma P0 的同种子比较及权重／输入来源。
- [efficiency.json](validation/20261007/efficiency.json)：原始效率测量；累计显存与 warm-up 限制见完整报告。
- [report_checks.json](validation/20261007/report_checks.json)：报告生成时实际完成的校验范围。
