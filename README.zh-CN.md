# Robot Stack

**在真实仿真动力学中采集成功示范、注入执行错误，并验证恢复。**

[在线可视化](https://pm1255.github.io/robot_stack/) · [English](README.md) · [纠错协议与数据结构](docs/correction-protocol.md) · [多环境实测](docs/benchmarks.md) · [贡献指南](CONTRIBUTING.md)

项目的核心是可复查的纠错数据，而不是仅让函数返回 `success=True`。环境接口负责动作与任务判定，策略负责选择动作，共用采集引擎负责预算、分叉、记录和配对。

## 现在代码还绑定具体任务吗？

**采集引擎不绑定任务，控制策略仍与任务有关。** 接入一个新环境不会自动获得解决其所有任务的专家。需要提供环境适配器、策略、原生成功判据和适合该任务的扰动。当前使用仿真真值与官方脚本专家，还不是视觉大模型或学习策略的标准 benchmark 评测。

原有 Isaac/A2D 管线以及三个 benchmark 的原生专家采集器保留；新的统一入口是 `python -m robot_stack.collect`，通用接口见 `robot_stack/core.py`。

![Paired correction in native simulation](docs/media/correction-triplet.gif)

真实 MetaWorld 动作回放：正常成功 / 错误对照失败 / 闭环恢复成功。按查看帧率播放，不代表物理时间。

## 已有真正的错误—纠错成功数据

每个源示范保存三条轨迹：

1. `source`：真实动作执行得到的成功轨迹。
2. `perturbed`：重执行源轨迹前缀，注入错误动作，再继续原轨迹剩余动作，作为没有反馈修正的对照。
3. `recovery`：从同种子重执行相同前缀和扰动，确认错误状态一致，再从当前物理状态闭环恢复。

只有源轨迹成功、物理偏差成立、错误对照失败、恢复分支成功，才标为 **有效纠错对**。恢复中不能 reset、瞬移、挂接物体或写入成功快照。所有扰动和恢复步数都计入总预算；失败和不合格样本也保留。

| 已实测内容 | 结果 | 边界 |
| --- | --- | --- |
| MetaWorld 推物纠错 | 50/50 有效纠错对 | 原生任务实例 0–49，seed 300–349 |
| MetaWorld 抓取放置纠错 | 50/50 有效纠错对 | 原生任务实例 0–49，seed 300–349 |
| RoboCasa 厨房导航纠错 | 3/3 有效纠错对 | NavigateKitchen，layout/style 1，seed 220–222，500 步预算 |
| 之前的 robosuite Lift | 正常 20/20，抓偏对照 0/20，恢复 20/20 | 有界重试实验，与新分叉协议分别统计 |
| MetaWorld 原生专家采集 | 8 类任务，80/80 | 小规模采集验证 |
| ManiSkill 原生专家采集 | 4 类任务，18/20 | 尚未接入通用纠错 |
| RoboTwin 原生专家采集 | 3 类任务，6/9 | 尚未接入通用纠错 |

新三分支实验使用 15 步笛卡尔动作扰动、500 步总预算。不同 reset seed 可能产生相同 MetaWorld 任务实例，因此默认同时更换原生 task_index；相同任务初态仍通过 `split_group` 分到同一数据划分。

MetaWorld 共 100 组有效纠错对、300 条轨迹，不重复计算之前 10 组试采样和性能对照运行。仅缓存任务定义后，两项任务的采集耗时由约 155/150 秒降至 43/45 秒；优化前后 300 个轨迹文件的状态、动作、阶段和成功标签完全一致。此为共享服务器并行运行的观测，非独占资源性能基准。[审计与效率证据](docs/evidence/metaworld-cache-comparison.json)。

## 快速复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[metaworld]'
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m robot_stack.collect --backend metaworld --task pick-place-v3 \
  --output outputs/pick-place --episodes 50 --seed-start 300 --perturb-steps 15
python -m robot_stack.audit outputs/pick-place --output outputs/audit.json
python -m robot_stack.replay outputs/pick-place/ep_0300/recovery.hdf5 \
  --output outputs/replay.json --video outputs/recovery.mp4
```

采集目录必须是新目录。视频需要 EGL 和 FFmpeg。HDF5 保存 T 个动作、T+1 个状态、逐步成功判定、动作阶段、源轨迹 ID、扰动/恢复区间。以 `split_group` 划分训练/验证集，不能随机打散同源分支。

## 导航与更多仿真器

已实现 RoboCasa 原生 `NavigateKitchen` 移动底座适配、AI2-THOR 原生场景的栅格导航适配，以及 RoboDojo 官方 `EvalEnv` 桥接。它们的真实运行状态、安装条件和能力限制在 [导航接入说明](docs/navigation.md) 单独列出；写好接口不等于通过物理验证。

RoboCasa 当前控制器不是完整避障/移动操作规划器。AI2-THOR 当前使用可达点真值地图，是自定义 PointNav 采集协议，不是官方 ObjectNav 分数。RoboDojo 需要完整 Isaac 环境、资产和真实策略，不能将其初始 `success=True` 当成任务完成。

仓库附有一个可直接检查的[真实纠错三分支样本](examples/correction/)，无需仿真器即可运行 `python -m robot_stack.audit examples/correction --output /tmp/robot-stack-example-audit.json` 检查文件。

## 网页、复查和协作

```bash
python scripts/build_collection_report.py --input outputs/pick-place --output docs/report.json
python -m http.server 8000 --directory docs
python -m unittest discover -s tests -v
```

网页展示真实结果和视频，CI 检查轻量契约。后续优先完善碰撞感知导航、任务策略、自然失败的状态反馈恢复，以及 MimicGen 的环境/子任务标注。**当前还没有跑通 MimicGen 扩增**，不能仅凭 HDF5 文件声称兼容。

上游项目、完整边界与贡献方式见 [英文首页](README.md)。
