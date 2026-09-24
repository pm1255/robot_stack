# 尚无成功源证据的任务与修复优先级

2026-09-24 核对公开报告及服务器遗漏记录后，当前清单共 546 类任务，114 类至少保存过一个成功源，432 类尚无成功源证据。这不是失败率，也不表示未覆盖任务全部尝试过。

| 环境 | 清单任务类 | 至少一个成功源 | 尚无成功源证据 |
|---|---:|---:|---:|
| ai2thor | 1 | 1 | 0 |
| maniskill | 74 | 11 | 63 |
| metaworld | 50 | 50 | 0 |
| robocasa | 317 | 2 | 315 |
| robodojo | 54 | 0 | 54 |
| robotwin | 50 | 50 | 0 |

其中 431 类缺少接入本项目的任务专家。它们需要补齐任务策略、资产、机器人控制与原生成功判定，不能靠降低扰动幅度解决。应先按抓取、放置、抽屉、门、导航等技能族共享专家，再适配各任务的目标与约束。

## 已有成功源，但还没有达到本轮验收门槛

这些与“尚无成功源”是不同集合，不应相加：

| 任务 | 已测问题 | 下一步 |
|---|---|---|
| MetaWorld soccer-v3 | 干净基线 3/5 | 分析接触方向、踢球落点及剩余动作预算 |
| MetaWorld stick-pull-v3 | 干净基线 3/5 | 检查工具抓取、对齐和拉动阶段 |
| ManiSkill PlugCharger-v1 | 干净基线 1/2 | 检查抓取姿态、插接对齐和接触后的重新规划 |
| ManiSkill PlaceSphere-v1 | 干净基线 2/2；三档扰动均无合格纠错 | 改变干预阶段、持续时间及恢复抓取方式，不能只缩小幅度 |

另有 23 类 MetaWorld 的初始动力学状态哈希重合，需要更完整的静态场景／任务配置指纹核验。这不是 23 个任务都执行失败。本轮总共测过 55 类干净基线，28 类通过小样本独立验收；其余历史成功示范不能自动当成稳定成功率。

## 修正漏记的绘图成功源

DrawTriangle-v1 在 seed 820 的 205 步成功源此前未被汇总：同一次采集中错误对照失败，随后恢复阶段出现原生 mplib/pybind11 GIL 错误。我们校验了两个已保存 HDF5 的哈希、状态、动作对齐和原生成功记录；没有完整纠错判定，也没有把它计为合格纠错。

证据：[源轨迹元数据](evidence/maniskill-drawing-source-record.json)、[部分运行审计](evidence/maniskill-drawing-partial-audit.json)、[恢复失败记录](evidence/maniskill-drawing-partial-progress.json)。这是补录已有成功证据，不是新产生的一条成功轨迹。

## 本轮绘图成功源复测

原生专家、300 步预算、最多一次规划流程，分别测试 820、821、822 三个种子，仅测成功源。每例墙钟时间上限 120 秒（包含启动、资产加载和执行）。全部尝试保留；没有测试新扰动或纠错。超时只表示本轮未得到完整结果，不能视为原生任务成功判定为假，也不能用来估计专家的实际执行成功率。

| 任务 | 成功源／尝试 | 异常 |
|---|---:|---:|
| DrawTriangle-v1 | 0/3 | 3 |
| DrawSVG-v1 | 0/3 | 3 |

本次六例均在 120 秒上限超时，未保存新的完整 HDF5。审计文件的 `passed=true` 只代表零个现存轨迹文件未触发检查错误，不能当作六次任务通过；cctl 作业完成也只代表诊断流程结束。

[完整复测报告](evidence/maniskill-drawing-baseline-summary.json) · [轨迹审计](evidence/maniskill-drawing-baseline-audit.json)


[全部尚无成功源任务的机器可读列表](evidence/source-backlog.json) · [按阶段筛选覆盖网页](https://pm1255.github.io/robot_stack/gallery.html) · [独立验收状态页](https://pm1255.github.io/robot_stack/readiness.html)。
