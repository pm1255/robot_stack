# Robot Stack

函数调用式机器人操作与数据采集。原有代码是 **Isaac Sim / IsaacLab + A2D + GraspNet + cuRobo**；现新增 **robosuite Lift / Panda、MetaWorld、ManiSkill、RoboTwin** 采集后端。尚未包含可运行的 RoboCasa 场景后端。静态采集看板位于 `docs/index.html`。

## 本次修改：规划与执行衔接

- 抓取候选必须通过抓取规划和后续放置的预检查，才进入执行；默认最多检查 20 个候选，可用 `MAX_GRASP_CANDIDATES` 限制。
- 执行模式下，抓取、放置都从仿真读取当前关节位置。仅规划模式从上一段轨迹末态继续。
- 加速抽样保留轨迹末点；回放根据关节名重新排列列顺序，拒绝无效轨迹。
- 原子入口区分 `planning_success`、`execution_success`、`task_success`。尚无抓取/释放任务判定器，因此返回 `task_unverified`，不会将规划成功冒充成功示范；`success` 仅代表经过任务判定的成功。
- 仅规划、mock、物体挂接和目标强制覆盖属于调试；旧日志不追溯认定为真实成功。
- episode 开始即保存记录；异常保存阶段与耗时后停止，避免在未验证的损坏状态上继续采集。同一 episode 结果不覆盖，请每次使用新输出目录。此次尚未实现断点续采或多进程调度。

Isaac 衔接修改通过单元测试；尚未在 Isaac/RoboCasa 上测量成功率。robosuite Lift 已在 eval 开发容器进行 CPU 实测，见下文。候选预检查增加规划成本，是否提高成功示范/小时需用相同种子对照实验确定。cuRobo 当前服务也没有消费 `attached_object` 来建立被抓物体的碰撞模型，放置预检查仅代表现有规划模型下可达。

## 回归测试

Python 3.10+，只需 NumPy、h5py（测试不启动仿真）：

```bash
python -m pip install numpy h5py
python -m unittest discover -s tests -v
```

完整仿真仍需原有 IsaacLab、GraspNet、cuRobo 环境与资产路径。不要把旧启动脚本的 `--attach-target-during-place` 作为物理成功的数据采集方式。原子入口目前也没有完整的接近→闭爪→抬升→释放控制链；必须先补齐；新增的 robosuite 后端已提供独立物理控制链。

## 采集看板

```bash
python scripts/build_collection_report.py --input /path/to/run --output /path/to/report.json
python -m http.server 8000 --directory docs
```

打开 `http://localhost:8000`，默认载入已附报告，也可选择其他 JSON 文件。导入和本地视频播放都在浏览器进行，不上传文件。页面支持状态分布、episode 筛选、规划/执行标记及视频预览。它是静态报告查看器，尚不实时连接 train-eval。

`docs/report.json` 已包含新增三种环境的 109 回合真实结果（104 成功、5 失败），可按实验分组查看；可用导出脚本替换。脚本只导出必要字段，不导出原始错误、绝对路径或资产内容，但 episode 相对目录名仍会出现，公开前需确认可分享。

GitHub Pages：推送代码后，在仓库 Settings → Pages 选择 GitHub Actions，再手动运行 **Collection dashboard** workflow。当前修改尚未推送或发布。GitHub 原生 issue 必须登录，显示提交账号；化名不等于匿名。

## RoboCasa：下一阶段只做一种提效方法

首先在一个适配版本支持的 counter-to-sink 原子任务上实现**状态反馈 + 有限次数重试**。保留 `pick/place` 函数接口，具体执行改用 RoboCasa / robosuite 的控制器。任务类名须按服务器版本确认，例如旧版本 `PnPCounterToSink`，新版 `PickPlaceCounterToSink`；不直接复用 Isaac 的坐标、四元数约定或动作维度。

1. 固定任务、机器人、布局、物体分布、种子与步数预算，运行不带恢复的基线。
2. 每个子任务检查接近误差、接触/抬升、物体掉落、放置与释放；以环境任务判定器为最终依据。
3. 可恢复失败从**当前真实仿真状态**重新定位、选择下一抓取候选；重试次数与总步数均受限。不能通过恢复成功快照来伪装“纠错成功”。
4. 用相同种子进行对照，报告首次成功率、含重试最终成功率、成功轨迹/墙钟小时、每条成功轨迹耗时和失败类型。重试增加的步数必须计入预算。
5. 稳定后增加独立进程 worker，每个进程单独持有环境与输出分片；仅抽样渲染检查视频，其余先存状态/动作、之后再提取图像。

尚未实现的扩展：成功示范→保存完整 simulator state 与元数据→从中间状态施加参数化扰动→真实闭环恢复→重新检查最终成功。数据需记录 `source_demo_id`、`branch_step`、扰动类型/幅度/随机种子、错误起止、恢复起止及成功判定。按源轨迹分组划分训练/验证集，避免同源扰动泄漏。

MimicGen 扩增需要环境接口、物体位姿、末端位姿、夹爪动作、子任务终止信号及可回放 HDF5。原始 JSON/视频不能直接替代这些信息。先验证源轨迹可回放，再对匹配版本的接口做小规模生成；纠错分支是否能扩增需要单独验证，不能默认套用正常成功示范的子任务划分。

参考：[RoboCasa MimicGen](https://robocasa.ai/docs/build/html/use_cases/mimicgen.html)、[源数据检查](https://mimicgen.github.io/docs/tutorials/debugging_datagen.html)。

## 真实 RoboCasa 环境预检

`robocasa_preflight.py` 执行实际环境创建、reset、10 次零动作物理步进和有限值检查，并记录版本、动作维度、观测形状与环境任务判定。它不衡量策略成功率，不输出训练示范。无渲染检查可在 CPU 上进行；不需要 OSMesa 或可见 GPU。

```bash
export ROBOCASA_PYTHON=/path/to/simulator-env/bin/python
export ROBOCASA_SOURCE=/path/to/robocasa-source
export ROBOSUITE_SOURCE=/path/to/robosuite-source
bash scripts/run_robocasa_preflight.sh --output /path/to/new-preflight.json --steps 10
```

场景资产必须与 RoboCasa 版本一致。模块导入成功不代表 reset 成功；缺失 XML、网格或纹理会在加载场景时才暴露。恢复采集前先通过真实 reset 与步进检查。CPU 物理检查也不代表 GPU 渲染已通过。


## 先跑通：robosuite Lift（真实物理、CPU 采集）

已验证环境：Python 3.10.12、robosuite 1.5.2、MuJoCo 3.3.1、NumPy 2.2.5、h5py 3.16.0。Panda 标准控制器不需要 mink 或 robosuite_models；Lift 资产包含在 robosuite 中。先在干净环境安装兼容版本，再运行：

```bash
python -m pip install 'robosuite==1.5.2' 'mujoco==3.3.1' 'numpy==2.2.5' 'h5py==3.16.0'
MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m robosuite_collector.collect --output outputs/clean --seed-start 100 --episodes 20 --max-attempts 1 --label clean
MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m robosuite_collector.collect --output outputs/perturbed --seed-start 100 --episodes 20 --max-attempts 1 --first-grasp-offset .06 0 0 --label perturbed
MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m robosuite_collector.collect --output outputs/recovery --seed-start 100 --episodes 20 --max-attempts 3 --first-grasp-offset .06 0 0 --label recovery
python -m robosuite_collector.replay outputs/recovery/ep_0100.hdf5 --output outputs/replay.json
```

输出目录必须是新目录。默认总预算每回合 600 个控制步、20 Hz；重试所有步骤也计入预算。函数顺序为 `pick → lift → verify_hold`；失败时执行 `recover → pick → lift`，恢复在当前物理状态进行，不 reset、不瞬移、不挂接物体。首次抓取目标可按世界坐标 XYZ 注入固定偏移；这是可控错误注入，不是视觉预测误差。正常与扰动实验必须分组报告，不能把人为失败率当成自然基线。

成功条件：环境 Lift 判定为真、两侧夹爪接触成立、方块比 reset 时升高至少 10 cm，连续保持 10 个控制步。策略直接读取仿真真值，机械臂初始姿态固定；随机种子改变方块采样，不能代表真实机器人或复杂厨房任务泛化。

每条 HDF5 保存 MuJoCo XML、环境配置、seed、T 个动作及各自动作前状态、额外最终状态、动作后物体/末端轨迹、夹爪接触、逐步成功判定、函数阶段和失败原因。`trace/*[i]` 对应第 i 个动作之后，`states[i]` 对应该动作之前。结果 JSON 包含轨迹 SHA256。成功与失败都保存，异常记录独立标记，不能将失败 HDF5 当成成功示范。

动作回放工具重新创建同版本环境并以同一 seed reset，逐步重执行动作，与保存状态比较，阈值 1e-8；不是播放成功状态快照。可选 `--video /path/to/replay.mp4` 使用 EGL，并需系统 FFmpeg（或 `imageio-ffmpeg` 包）；渲染与采集分离，采集不渲染图像。后续策略训练仍需离线提取所需观测。

当前 HDF5 具有 robomimic 风格的状态/动作结构，但尚未包含 MimicGen 所需的 datagen_info 和子任务标注，也未通过 MimicGen 生成验证。第一阶段纠错是在同种子、同初始状态下重新运行并注入第一次抓偏；尚未实现任意成功轨迹中间步骤的状态分叉。


### 服务器实测（2026-09-24，种子 100–119）

| 条件 | 成功回合 | 组内采集墙钟时间 | 短程吞吐折算 |
| --- | --- | --- | --- |
| 正常抓取，1 次尝试 | 20/20 | 29.88 秒 | 2,410 成功条/小时 |
| 首次抓偏 +X 6 cm，1 次尝试 | 0/20 | 24.05 秒 | 0 成功条/小时 |
| 相同抓偏，最多 3 次尝试 | 20/20 | 45.70 秒 | 1,575 成功条/小时 |

恢复组实际全部在第 2 次抓取成功。三组统一每回合最多 600 步；使用独立种子 0、1 调试，正式种子不参与调参。计时包含环境建立、reset、物理步进、数据写入和清理；不含导入、排队、传输和离线渲染。吞吐是约半分钟至一分钟实测的折算，并非长时间稳定产能承诺。没有测量与旧 Isaac 流程的速度对比，也不能将 0→100% 解释成自然任务的普遍提升。

60 份 HDF5 校验通过；20 个种子的三组初始状态相同；20 个扰动失败轨迹与恢复轨迹的动作、状态前缀及失败末态完全一致。生成了 20 组正常成功→故意失败→恢复成功的关联记录。`scripts/audit_lift_comparison.py` 可复核这些条件；回归测试 13 项通过。

一条命令重跑对照（从仓库根目录执行，SIM_PYTHON 指向仿真 Python）：

```bash
SIM_PYTHON=/path/to/python bash scripts/run_lift_comparison.sh outputs/new_comparison 100 20
```

上述结果属于简单 Lift 任务、真值状态反馈和单一人为偏移；还不是 RoboCasa、视觉策略或 MimicGen 扩增结果。


## 多环境采集实测（2026-09-24）

| 环境 | 任务数 | 固定种子采集 | 结果 |
| --- | --- | --- | --- |
| MetaWorld 3.1.1 | 8 | 每任务 100–109 | 80/80 |
| ManiSkill 3.0.1 | 4 | 每任务 100–104 | 18/20 |
| RoboTwin | 3 | 每任务 100–102 | 6/9 |

这是官方脚本专家/运动规划器的小规模采集验证，不是已训练策略的基准分数，也不是完整 MT10/MT50 或 RoboTwin 全任务评测。没有成功种子筛选；失败记录和轨迹也保留。新后端尚未接入扰动恢复和 MimicGen；先前 Lift 的恢复结果不能直接推广到其他环境。详细环境、任务、回放和复现命令见 [多环境说明](docs/benchmarks.md)。
