# 轨迹数量：2026-09-24 校准实验前的盘点快照

本快照不含后续规划器对照和逐任务校准产生的新增记录。

**1,322 条保存轨迹（排除字节完全相同的文件副本）**，分布在 1,321 个 HDF5 文件中。原始为 1,624 个文件、1,625 条容器内轨迹；MimicGen 的一个源容器包含两条示范。这里包含旧版本、调试和失败记录，不能直接称为可训练的成功纠错数据。不同文件仍可能共享源动作或初始场景。

| 环境／工具 | 排除完全相同副本后的轨迹数 |
|---|---:|
| ai2thor | 18 |
| metaworld | 969 |
| robosuite_lift | 64 |
| maniskill | 92 |
| mimicgen | 22 |
| robocasa | 37 |
| robotwin | 120 |

## 已发布纠错实验的统一口径

选择下方明确列出的 17 个实验，不重复累计 progress、子目录 summary 或缓存副本。共 **340 次尝试、327 条成功源记录、248 组合格纠错、989 条保存轨迹**。合格纠错对应源、错误对照、恢复三条；保存总数还包含不合格和不完整尝试。不同调度可使用同一初始场景，合格组数不是独立任务实例数。

| 环境 | 尝试 | 成功源记录 | 合格纠错组 | 保存轨迹 |
|---|---:|---:|---:|---:|
| ai2thor | 6 | 6 | 6 | 18 |
| metaworld | 286 | 278 | 221 | 842 |
| maniskill | 28 | 23 | 9 | 70 |
| robocasa | 9 | 9 | 8 | 27 |
| robotwin | 11 | 11 | 4 | 32 |

RoboTwin 本轮 6 次尝试全部得到成功源；1 个恢复规划器异常导致该回合只有源和对照两条，共保存 17 条。异常不计为纠错成功。RoboCasa 移动操作前轮为 2/3，修复放置控制后为 3/3；前后两轮和三个调度共享种子 1100，须按 `split_group` 放在同一数据划分中。

## 实验清单

- [ai2thor_long_scheduled_v1](evidence/ai2thor-long-summary.json)：3 次尝试，3 组合格纠错。
- [ai2thor_nav_20260924_v1](evidence/ai2thor-navigation-summary.json)：3 次尝试，3 组合格纠错。
- [corrections_cached_50_pick-place-v3_20260924](evidence/corrections_cached_50_pick-place-v3_20260924-summary.json)：50 次尝试，50 组合格纠错。
- [corrections_cached_50_push-v3_20260924](evidence/corrections_cached_50_push-v3_20260924-summary.json)：50 次尝试，50 组合格纠错。
- [corrections_pickplace_20260924_v2](evidence/corrections_pickplace_20260924_v2-summary.json)：5 次尝试，5 组合格纠错。
- [corrections_push_20260924_v2](evidence/corrections_push_20260924_v2-summary.json)：5 次尝试，5 组合格纠错。
- [metaworld_diversity_v2](evidence/metaworld-diversity-summary.json)：36 次尝试，24 组合格纠错。
- [metaworld_scheduled50_v1](evidence/metaworld50-scheduled-summary.json)：100 次尝试，69 组合格纠错。
- [metaworld_tools_v1](evidence/metaworld-tools-summary.json)：40 次尝试，18 组合格纠错。
- [maniskill_all_planners_v2](evidence/maniskill-all-summary.json)：24 次尝试，8 组合格纠错。
- [maniskill_corrections_v1](evidence/maniskill-smoke-summary.json)：4 次尝试，1 组合格纠错。
- [robocasa_nav_20260924_v4](evidence/robocasa-nav-summary.json)：3 次尝试，3 组合格纠错。
- [robotwin_correction_v1](evidence/robotwin-smoke-summary.json)：1 次尝试，0 组合格纠错。
- [robotwin_carry_v4](evidence/robotwin-carry-summary.json)：4 次尝试，3 组合格纠错。
- [robotwin_more_corrections_v1](evidence/robotwin-more-summary.json)：6 次尝试，1 组合格纠错。
- [robocasa_mobile_corrections_v2](evidence/robocasa-mobile-initial-summary.json)：3 次尝试，2 组合格纠错。
- [robocasa_mobile_corrections_v3](evidence/robocasa-mobile-summary.json)：3 次尝试，3 组合格纠错。

## 自己复算

```bash
python scripts/count_trajectories.py outputs --output outputs/trajectory-inventory.json
```

脚本逐文件读取 HDF5、计算 SHA256、统计容器中的轨迹，并保留无法识别／读取的错误。它统计保存记录，不替代 `robot_stack.audit` 的协议检查和原生动作回放。

[机器可读汇总](evidence/trajectory-counts.json) · [相对路径与校验和清单](evidence/trajectory-file-manifest.json) · [逐任务覆盖](task-coverage.md)
