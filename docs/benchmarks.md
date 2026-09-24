# 多环境采集验证

> 本页保留早期原生专家基线。最新全任务扫描、三分支纠错、独立回放和启动命令见 [逐任务覆盖](task-coverage.md)。
2026-09-24，在 eval 服务器完成 15 类任务、109 次固定种子物理执行：104 成功，5 失败，0 采集异常。成功来自各环境原生任务判定器，并要求实际执行完成。没有用成功状态替换物理执行，也没有先搜索容易成功的种子。

| 环境 | 任务 | 成功 / 总数 |
| --- | --- | --- |
| MetaWorld | reach-v3、push-v3、pick-place-v3、door-open-v3、drawer-open-v3、button-press-v3、window-open-v3、faucet-open-v3 | 各 10/10 |
| ManiSkill | PickCube-v1、PushCube-v1、StackCube-v1 | 各 5/5 |
| ManiSkill | PlaceSphere-v1 | 3/5 |
| RoboTwin | stack_blocks_two | 3/3 |
| RoboTwin | turn_switch | 2/3 |
| RoboTwin | place_empty_cup | 1/3 |

本轮以运行可靠采集管线为目标；使用官方专家，尚未训练学习策略。MetaWorld 是分别构建的 MT1 任务子集，不是完整 MT10/MT50；ManiSkill 明确设置每回合 500 步预算；RoboTwin 使用 demo_clean 和 aloha-agilex。因此不能将这些小样本比例当成标准排行榜分数。

## 独立回放结果

- MetaWorld：80 个 HDF5 校验；每类抽取 seed=100，共 8 条真实动作回放通过。逐步状态与观测最大绝对误差均为 0，成功标记序列一致。
- ManiSkill：20 个原生 HDF5 校验；每类抽取第一条成功示范，共 4 条动作回放通过。逐步环境状态最大绝对误差为 0，成功标记序列一致；保存了四段回放视频。
- RoboTwin：9 个原生 HDF5 校验；堆叠 seed=100、开关 seed=100、空杯 seed=101 的原生规划路径回放全部再次通过最终任务判定。此项未检验逐步数值状态等价。
- 13 项回归测试通过；网页实际加载 109 条记录、104 条成功，筛选得到全部 5 条失败，三段内置视频均成功加载。

抽检回放不等于所有轨迹均已回放；全部 109 个轨迹文件的 SHA256 都已复核。

## 环境隔离与复现

以下命令均从仓库根目录运行。三个 Python 环境必须独立，不能将不同版本的 SAPIEN/NumPy 依赖装进同一个环境。运行前准备对应上游机器人与场景资产。输出目录必须不存在。

MetaWorld：Python 3.10、metaworld 3.1.1、MuJoCo 3.3.0、NumPy 2.2.5、Gymnasium 1.3.0、h5py 3.16.0。

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m benchmark_collector.metaworld --output outputs/mw-run \
  --episodes 10 --seed-start 100 --benchmark-seed 20260924 --max-steps 500
python -m benchmark_collector.replay_metaworld outputs/mw-run/pick-place-v3/ep_0100.hdf5 \
  --output outputs/mw-replay.json --video outputs/mw-replay.mp4
```

每个任务使用 MT1(task, seed=20260924) 的 train_tasks[0:10]，reset seed 为 100–109。专家原始动作和裁剪至环境边界后的执行动作均保存；首次原生成功即结束。状态与观测为 T+1，动作为 T，保存完整 MuJoCo integration state。20 FPS 回放视频用于查看动作，不代表物理时间。

ManiSkill：Python 3.10、mani_skill 3.0.1、SAPIEN 3.0.3、mplib 0.1.1、PyTorch 2.4.1+cu121、NumPy 1.26.4、Gymnasium 0.29.1。使用单张 4090 容器，物理后端为 physx_cpu；这套版本即使设置 render_backend=none，资产材质初始化仍需要可用的渲染设备，不能直接在没有 Vulkan 渲染设备的开发容器运行。

```bash
python -m benchmark_collector.maniskill --output outputs/ms-run \
  --episodes 5 --seed-start 100 --max-steps 500
PYTHONPATH=. python scripts/verify_maniskill.py \
  --input outputs/ms-run --output outputs/ms-replays --video
```

调用原生 Panda MP_SOLUTIONS 与 RecordEpisode，记录 state 观测、pd_joint_pos 动作、T+1 环境状态、成功标记和原生 JSON 元数据。采集不保存图像，之后通过真实动作回放抽样渲染视频。动作预算包括专家全部执行步骤。

RoboTwin：已有原生代码和资产，Python 3.10、SAPIEN 3.0.0b1、mplib 0.2.1、PyTorch 2.4.1+cu121、NumPy 1.26.4。使用单张 4090；运行时读取已配置的 RoboTwin 根目录，不修改上游代码和共享依赖。上游代码文件 SHA256 存在 run_config.json，避免仅依赖浮动版本名。

```bash
python -m benchmark_collector.robotwin --root /path/to/RoboTwin \
  --output outputs/rt-run --episodes 3 --seed-start 100 --replay-first
python -m benchmark_collector.replay_robotwin \
  outputs/rt-run/place_empty_cup/ep_0101_episode_result.json \
  --root /path/to/RoboTwin --output outputs/rt-cup-replay.json
```

设置 head RGB 320×240、关闭腕部图像、save_freq=30，保留原生 HDF5、视频和左右臂规划路径。成功要求 play_once 实际执行后的 check_success 与 plan_success 均为真。`--replay-first` 仅回放每个任务的首个种子且要求该回合成功；其他成功回合可使用独立回放命令。原生视频为 30 FPS 查看采样帧，不保证对应真实物理时间。回放工具读取本管线生成的可信路径 pickle，不要加载来源不明的 pickle。

## 数据质量与边界

所有回合保存 SHA256，成功/失败标记独立。MetaWorld 与 ManiSkill 检查有限值和状态/动作长度对齐；RoboTwin 检查原生关节和末端数据。失败数据必须通过 episode_result.json 过滤，不能将目录里全部轨迹视为成功示范。MetaWorld 抽样动作回放与保存状态逐步比较；RoboTwin 原生规划路径回放检查最终任务成功，未做逐步数值状态等价判定。

本轮墙钟：MetaWorld 25.96 秒，ManiSkill 59.40 秒，RoboTwin 457.42 秒。前两项包含环境创建、规划或控制、物理执行和保存，排除导入、排队、渲染；RoboTwin 还包含原生图像/视频保存及两次抽样回放。这些配置不同，不能直接比较吞吐或外推长期成功条数/小时。

后续优先处理 RoboTwin 空杯放置和 ManiSkill 球体放置的自然失败，加入有总步数限制的状态反馈恢复，再做同种子对照。本页早期实验尚未实现中段扰动和恢复分支；后续版本已为 MetaWorld、ManiSkill 和实验性 RoboTwin 接入统一三分支协议，最新结果单独报告。跨任务 MimicGen datagen_info 仍需适配。现有 HDF5 不能直接宣称已支持 MimicGen 扩增。

参考：[MetaWorld 官方专家](https://metaworld.farama.org/benchmark/expert_trajectories/)、[ManiSkill 运动规划](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html)、[RoboTwin 配置](https://robotwin-platform.github.io/doc/usage/configurations.html)。
