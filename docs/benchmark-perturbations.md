# 每个环境的扰动类型与验证边界

[交互式类型说明](https://pm1255.github.io/robot_stack/perturbations.html) · [启动配置器](https://pm1255.github.io/robot_stack/playground.html)

一次扰动必须真实执行原生动作。`steps`、`repeat`、`gap` 与总预算同时约束执行；`at.step`、`at.fraction` 或 `at.event` 指向成功源轨迹中的位置，后者不是扰动分支上的实时语义触发。有效纠错还要求源成功、扰动造成偏差、错误对照失败、恢复成功及配对检查通过。

## MetaWorld · 单臂操作

50 类任务已有合格纠错样本；不表示每种扰动都在每类任务上有效。

恢复方式：原生专家读取当前仿真状态重新选择动作。

| 类型 | 实际动作 | 参数 | 单位 | 适合插入的位置 |
|---|---|---|---|---|
| cartesian_offset | 沿指定方向推离末端，夹爪命令可同时配置 | direction:[x,y,z]；gripper∈[-1,1] | 归一化动作，乘以 strength | 接近、抓取、搬运或放置 |
| random_cartesian | 逐步采样末端平移噪声，种子固定 | 使用 episode seed 派生随机序列 | 每步各轴∈[-strength,strength] | 任何仍有恢复预算的阶段 |
| gripper_open | 停住末端平移并张开夹爪 | 无 | 负夹爪动作，幅度 strength | 已接触或抓持后 |
| gripper_close | 停住末端平移并闭合夹爪 | 无 | 正夹爪动作，幅度 strength | 尚未到位、需要张爪时 |
| action_reverse | 反向执行当前专家建议的平移动作 | 无 | 归一化平移动作取反并缩放 | 推、拉、接近阶段 |
| action_hold | 平移命令置零，保留当前专家的夹爪命令 | 无 | 控制动作步 | 作为停顿对照；未必产生错误 |

## RoboTwin · 双臂操作

50 类成功源视频；纠错任务覆盖另行报告。当前内置只有 joint_offset，不能把不同参数包装成多种已经验证的故障。

恢复方式：撤销原生规划调用栈，从当前物体姿态重新执行 play_once；脚本可能假定初始场景，恢复能力仍是实验性的。

| 类型 | 实际动作 | 参数 | 单位 | 适合插入的位置 |
|---|---|---|---|---|
| joint_offset | 双臂指定关节的电机目标偏移，可同时张开双夹爪；遵守原生关节限位 | joint；offset；open_grippers（默认 true） | offset 为关节弧度；steps 是物理步 | 第一次或后续持物、双臂交接、堆叠阶段 |

## ManiSkill · 原生规划专家

10 类成功源、7 类合格纠错任务；12 个注册专家不等于 74 个环境 ID 都能解决。

恢复方式：停止原生动作流，从当前状态重新调用运动规划；max_replans 限制重规划次数。

| 类型 | 实际动作 | 参数 | 单位 | 适合插入的位置 |
|---|---|---|---|---|
| joint_offset | 指定关节目标偏移，同时打开夹爪 | joint；offset | offset 为关节弧度；steps 是控制步 | 持物或接近目标时 |
| gripper_open | 保持机械臂目标，张开夹爪 | 无；棍式机器人不支持夹爪 | 归一化夹爪命令 | 抓持后 |
| gripper_close | 保持机械臂目标，闭合夹爪 | 无；棍式机器人不支持夹爪 | 归一化夹爪命令 | 抓取前 |
| joint_hold | 保持机械臂关节目标，夹爪维持打开 | 无 | 控制步 | 停顿或失去抓持的对照 |

## RoboCasa · 底盘与移动操作

NavigateKitchen 的底盘纠错已有证据。移动操作适配器使用原生 PickPlaceCounterToSink；实验结果与导航任务分开报告。arm_* 与 gripper_* 仅适用于移动操作适配器。

恢复方式：底盘根据当前位置与朝向闭环导航；移动操作根据原生接触判定选择重新抓取、抬升、搬运和放置。

| 类型 | 实际动作 | 参数 | 单位 | 适合插入的位置 |
|---|---|---|---|---|
| base_yaw | 错误转向；保持当前夹爪命令 | 无 | 归一化底盘角速度 × strength | 导航或持物移动 |
| base_translation | 沿底盘控制坐标系施加平移速度 | direction:[x,y] | 归一化底盘速度 × strength | 导航或持物移动 |
| base_reverse | 沿默认底盘控制轴的负方向移动，不是逐步倒放轨迹 | 无 | 归一化底盘速度 | 导航 |
| base_hold | 底盘速度置零 | 无 | 控制步 | 停顿对照，可能无有效错误 |
| arm_offset | 机械臂平移命令偏移，保持夹爪 | direction:[x,y,z] | 机械臂基座参考系中的归一化 OSC 增量 | 接近、持物、放置 |
| gripper_open | 末端平移命令置零并张开夹爪 | 无 | 归一化夹爪命令 | 抓取完成或刚刚抬起 |
| gripper_close | 末端平移命令置零并闭合夹爪 | 无 | 归一化夹爪命令 | 抓取前 |
| arm_hold | 机械臂增量置零，保持夹爪命令 | 无 | 控制步 | 停顿对照 |

## AI2-THOR · PointNav

FloorPlan1 自定义目标点导航，含六个连续目的地；未声称全场景 ObjectNav 或导航加操作已验证。

恢复方式：使用原生可达网格，从当前朝向与位置重新规划路径。

| 类型 | 实际动作 | 参数 | 单位 | 适合插入的位置 |
|---|---|---|---|---|
| wrong_heading | 原生 RotateRight，每次转 90° | strength 必须为 1 | 离散旋转次数 | 路口或抵达某个源里程碑时 |
| backtrack | 原生 MoveBack，按当前朝向后退 | moveMagnitude 使用适配器 grid | 网格移动次数 | 移动阶段 |
| lateral_drift | 原生 MoveRight，向机器人右侧移动 | moveMagnitude 使用适配器 grid | 网格移动次数 | 移动阶段 |
| navigation_hold | 原生 Pass，不移动 | 无 | 离散动作次数 | 停顿对照，通常不构成位姿偏差 |

## RoboDojo · 接口桥接

只完成原生 reset 与物理步验证。尚无内置、已验证的通用扰动类型或恢复策略。

恢复方式：调用方必须提供原生 policy、state_reader、feature_reader 和实际执行动作的 perturbation 回调。

## MimicGen · 轨迹扩增工具

Lift 原生扩增 10 条、7 条成功。对象坐标变换及轨迹迁移是数据增强，不等同于故意注入错误。

恢复方式：先标注任务与恢复子段、对象坐标系和源轨迹关系；扩增结果仍需经过独立的扰动／恢复采集与审计才能算纠错对。

## 不把这些概念混在一起

- 调度接口支持某类型，不代表该类型已在全部任务上制造可恢复的错误。
- hold 等对照可能没有达到误差阈值，应保留为未合格记录。
- RoboTwin 的物理步不能与 ManiSkill、MetaWorld 的控制步直接比较。
- 当前没有通用的物体瞬移、强制吸附、相机遮挡、传感器噪声或指令语义扰动实现；不能把这些写成已支持的效果。
- 错误特征中的位置、关节角和夹爪开度可能混合，误差阈值不是统一的厘米数。

## 本轮移动操作与 RoboTwin 验证

RoboCasa 在固定 layout/style=1、apple、seed=1100 下，错误转向、机械臂偏移和持物张爪最终为 3/3 合格纠错；不能外推到所有场景或八种类型。RoboTwin 新增放杯子合格纠错，交接恢复失败和三层堆叠对照仍成功同样公开。 [移动操作视频](https://pm1255.github.io/robot_stack/robocasa-mobile.html) · [RoboTwin 视频与完整结果](https://pm1255.github.io/robot_stack/robotwin.html)。
