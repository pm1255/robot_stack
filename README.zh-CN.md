# Robot Stack

**新增 ManiSkill 原生专家接入、RoboTwin 实验性逐步驱动、逐任务覆盖报告，以及下方直接展示的 132 个真实动图。** [覆盖边界与启动命令](docs/task-coverage.md)。

**统一多扰动调度。** 支持插入步数、比例、源轨迹阶段事件，以及类型、强度、持续步数、次数和间隔。[配置器与真实案例](https://pm1255.github.io/robot_stack/playground.html) · [启动命令与 Python 接口](docs/perturbations.md)。MetaWorld 首轮 100 次尝试得到 69 组有效纠错，覆盖 48 类任务；独立工具任务补测 18/40，补齐其余两类，累计 50 类任务均有有效样本。这是任务类别覆盖，不代表所有任务实例、所有错误都能恢复，也不代表所有 benchmark 已全面适配。

**在真实仿真动力学中采集成功示范、注入执行错误，并验证恢复。**

[在线可视化](https://pm1255.github.io/robot_stack/) · [English](README.md) · [纠错协议与数据结构](docs/correction-protocol.md) · [多环境实测](docs/benchmarks.md) · [贡献指南](CONTRIBUTING.md)

项目的核心是可复查的纠错数据，而不是仅让函数返回 `success=True`。环境接口负责动作与任务判定，策略负责选择动作，共用采集引擎负责预算、分叉、记录和配对。

**RoboTwin 成功源覆盖已达 50/50 类。** 首轮固定种子 37/50；独立补测 11/26；最后限次补采 5/13，保留全部失败。纠错覆盖仍为双臂堆叠一类，不能从源覆盖外推。 [Evidence](docs/task-coverage.md).

## 现在代码还绑定具体任务吗？

**采集引擎不绑定任务，控制策略仍与任务有关。** 接入一个新环境不会自动获得解决其所有任务的专家。需要提供环境适配器、策略、原生成功判据和适合该任务的扰动。当前使用仿真真值与官方脚本专家，还不是视觉大模型或学习策略的标准 benchmark 评测。

原有 Isaac/A2D 管线以及三个 benchmark 的原生专家采集器保留；新的统一入口是 `python -m robot_stack.collect`，通用接口见 `robot_stack/core.py`。

![Paired correction in native simulation](docs/media/correction-triplet.gif)

真实 MetaWorld 动作回放：正常成功 / 错误对照失败 / 闭环恢复成功。按查看帧率播放，不代表物理时间。


## 直接在仓库主页观看真实结果

下方动图均来自真实仿真记录，分别标明成功源轨迹、纠错成功、MimicGen 扩增和失败案例。播放速度不代表物理时间。[查看可搜索案例库与任务覆盖](https://pm1255.github.io/robot_stack/gallery.html)。

<table>
<tr><td align="center"><b>MetaWorld · repeated errors</b><br><img src="docs/media/gallery/metaworld-multi-recovery.gif" width="280" alt="MetaWorld · repeated errors"><br>Recovery / 3 interventions</td><td align="center"><b>MetaWorld · tool use</b><br><img src="docs/media/gallery/stick-pull-recovery.gif" width="280" alt="MetaWorld · tool use"><br>Recovery / stick pulling</td></tr>
<tr><td align="center"><b>AI2-THOR · six destinations</b><br><img src="docs/media/gallery/ai2thor-long-recovery.gif" width="280" alt="AI2-THOR · six destinations"><br>Recovery / 3 navigation errors</td><td align="center"><b>RoboCasa · kitchen navigation</b><br><img src="docs/media/gallery/robocasa-recovery.gif" width="280" alt="RoboCasa · kitchen navigation"><br>Recovery / mobile base</td></tr>
<tr><td align="center"><b>ManiSkill · cube stacking</b><br><img src="docs/media/gallery/maniskill-stack-cube.gif" width="280" alt="ManiSkill · cube stacking"><br>Successful source demonstration</td><td align="center"><b>RoboTwin · dual-arm stacking</b><br><img src="docs/media/gallery/robotwin-stack-blocks.gif" width="280" alt="RoboTwin · dual-arm stacking"><br>Successful source demonstration</td></tr>
<tr><td align="center"><b>MimicGen · generated success</b><br><img src="docs/media/gallery/mimicgen-success.gif" width="280" alt="MimicGen · generated success"><br>Augmented demonstration</td><td align="center"><b>MimicGen · retained failure</b><br><img src="docs/media/gallery/mimicgen-failure.gif" width="280" alt="MimicGen · retained failure"><br>Failure / not a recovery sample</td></tr>
</table>



### RoboTwin · 双臂堆叠的真实故意错误与纠错

seed 101；正常成功 3,935 个物理步，错误对照 4,175 步后失败，恢复分支 7,748 步后成功。三条轨迹独立回放的状态误差均为 0。四次固定配置尝试中有 3 组有效纠错，失败样本保留。 [Replay evidence](docs/evidence/robotwin-carry-replay.json).

<table><tr><th>Source · success</th><th>Perturbed · failure</th><th>Recovery · success</th></tr><tr><td><img width="280" src="docs/media/gallery/robotwin-carry-source.gif" alt="RoboTwin source"></td><td><img width="280" src="docs/media/gallery/robotwin-carry-perturbed.gif" alt="RoboTwin perturbed"></td><td><img width="280" src="docs/media/gallery/robotwin-carry-recovery.gif" alt="RoboTwin recovery"></td></tr></table>

### ManiSkill · 7 类任务的成功／错误／恢复对照

每行是同一种子与源示范的三条原生轨迹：正常成功、故意扰动后沿原动作执行失败、从扰动状态重新规划后成功。21 条轨迹全部独立回放通过，状态误差为 0。 [Replay evidence](docs/evidence/maniskill-seven-task-replay.json).

<table>
<tr><th>Source · success</th><th>Perturbed · failure</th><th>Recovery · success</th></tr>
<tr><th colspan="3">LiftPegUpright-v1 · seed 810</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/LiftPegUpright-v1-source.json"><img src="docs/media/gallery/maniskill/LiftPegUpright-v1-source.gif" width="250" alt="LiftPegUpright-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/LiftPegUpright-v1-perturbed.json"><img src="docs/media/gallery/maniskill/LiftPegUpright-v1-perturbed.gif" width="250" alt="LiftPegUpright-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/LiftPegUpright-v1-recovery.json"><img src="docs/media/gallery/maniskill/LiftPegUpright-v1-recovery.gif" width="250" alt="LiftPegUpright-v1 recovery"></a></td></tr>
<tr><th colspan="3">PegInsertionSide-v1 · seed 810</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/PegInsertionSide-v1-source.json"><img src="docs/media/gallery/maniskill/PegInsertionSide-v1-source.gif" width="250" alt="PegInsertionSide-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/PegInsertionSide-v1-perturbed.json"><img src="docs/media/gallery/maniskill/PegInsertionSide-v1-perturbed.gif" width="250" alt="PegInsertionSide-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/PegInsertionSide-v1-recovery.json"><img src="docs/media/gallery/maniskill/PegInsertionSide-v1-recovery.gif" width="250" alt="PegInsertionSide-v1 recovery"></a></td></tr>
<tr><th colspan="3">PickCube-v1 · seed 810</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/PickCube-v1-source.json"><img src="docs/media/gallery/maniskill/PickCube-v1-source.gif" width="250" alt="PickCube-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/PickCube-v1-perturbed.json"><img src="docs/media/gallery/maniskill/PickCube-v1-perturbed.gif" width="250" alt="PickCube-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/PickCube-v1-recovery.json"><img src="docs/media/gallery/maniskill/PickCube-v1-recovery.gif" width="250" alt="PickCube-v1 recovery"></a></td></tr>
<tr><th colspan="3">PlaceSphere-v1 · seed 800</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/PlaceSphere-v1-source.json"><img src="docs/media/gallery/maniskill/PlaceSphere-v1-source.gif" width="250" alt="PlaceSphere-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/PlaceSphere-v1-perturbed.json"><img src="docs/media/gallery/maniskill/PlaceSphere-v1-perturbed.gif" width="250" alt="PlaceSphere-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/PlaceSphere-v1-recovery.json"><img src="docs/media/gallery/maniskill/PlaceSphere-v1-recovery.gif" width="250" alt="PlaceSphere-v1 recovery"></a></td></tr>
<tr><th colspan="3">PullCube-v1 · seed 811</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/PullCube-v1-source.json"><img src="docs/media/gallery/maniskill/PullCube-v1-source.gif" width="250" alt="PullCube-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/PullCube-v1-perturbed.json"><img src="docs/media/gallery/maniskill/PullCube-v1-perturbed.gif" width="250" alt="PullCube-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/PullCube-v1-recovery.json"><img src="docs/media/gallery/maniskill/PullCube-v1-recovery.gif" width="250" alt="PullCube-v1 recovery"></a></td></tr>
<tr><th colspan="3">PushCube-v1 · seed 811</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/PushCube-v1-source.json"><img src="docs/media/gallery/maniskill/PushCube-v1-source.gif" width="250" alt="PushCube-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/PushCube-v1-perturbed.json"><img src="docs/media/gallery/maniskill/PushCube-v1-perturbed.gif" width="250" alt="PushCube-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/PushCube-v1-recovery.json"><img src="docs/media/gallery/maniskill/PushCube-v1-recovery.gif" width="250" alt="PushCube-v1 recovery"></a></td></tr>
<tr><th colspan="3">StackCube-v1 · seed 810</th></tr>
<tr><td><a href="docs/evidence/maniskill-gallery/StackCube-v1-source.json"><img src="docs/media/gallery/maniskill/StackCube-v1-source.gif" width="250" alt="StackCube-v1 source"></a></td><td><a href="docs/evidence/maniskill-gallery/StackCube-v1-perturbed.json"><img src="docs/media/gallery/maniskill/StackCube-v1-perturbed.gif" width="250" alt="StackCube-v1 perturbed"></a></td><td><a href="docs/evidence/maniskill-gallery/StackCube-v1-recovery.json"><img src="docs/media/gallery/maniskill/StackCube-v1-recovery.gif" width="250" alt="StackCube-v1 recovery"></a></td></tr>
</table>


<details>
<summary>展开 RoboTwin 首轮 37 类成功源轨迹动图</summary>

这是首轮 50 个任务、固定 seed 900 的 37 个成功源任务，直接由原生采集过程录制。这里的源示范不计为纠错对；其余 13 次失败／错误仍在完整报告中。 [All 50 attempts](docs/evidence/robotwin-all-summary.json).

<table>
<tr><td align="center"><b>adjust_bottle</b><br><img width="240" src="docs/media/gallery/robotwin-sources/adjust_bottle.gif" alt="adjust_bottle successful source"><br>82 recorded frames · source</td><td align="center"><b>beat_block_hammer</b><br><img width="240" src="docs/media/gallery/robotwin-sources/beat_block_hammer.gif" alt="beat_block_hammer successful source"><br>65 recorded frames · source</td><td align="center"><b>blocks_ranking_rgb</b><br><img width="240" src="docs/media/gallery/robotwin-sources/blocks_ranking_rgb.gif" alt="blocks_ranking_rgb successful source"><br>254 recorded frames · source</td></tr>
<tr><td align="center"><b>blocks_ranking_size</b><br><img width="240" src="docs/media/gallery/robotwin-sources/blocks_ranking_size.gif" alt="blocks_ranking_size successful source"><br>264 recorded frames · source</td><td align="center"><b>click_alarmclock</b><br><img width="240" src="docs/media/gallery/robotwin-sources/click_alarmclock.gif" alt="click_alarmclock successful source"><br>52 recorded frames · source</td><td align="center"><b>click_bell</b><br><img width="240" src="docs/media/gallery/robotwin-sources/click_bell.gif" alt="click_bell successful source"><br>44 recorded frames · source</td></tr>
<tr><td align="center"><b>grab_roller</b><br><img width="240" src="docs/media/gallery/robotwin-sources/grab_roller.gif" alt="grab_roller successful source"><br>52 recorded frames · source</td><td align="center"><b>handover_block</b><br><img width="240" src="docs/media/gallery/robotwin-sources/handover_block.gif" alt="handover_block successful source"><br>169 recorded frames · source</td><td align="center"><b>handover_mic</b><br><img width="240" src="docs/media/gallery/robotwin-sources/handover_mic.gif" alt="handover_mic successful source"><br>124 recorded frames · source</td></tr>
<tr><td align="center"><b>hanging_mug</b><br><img width="240" src="docs/media/gallery/robotwin-sources/hanging_mug.gif" alt="hanging_mug successful source"><br>196 recorded frames · source</td><td align="center"><b>lift_pot</b><br><img width="240" src="docs/media/gallery/robotwin-sources/lift_pot.gif" alt="lift_pot successful source"><br>61 recorded frames · source</td><td align="center"><b>move_can_pot</b><br><img width="240" src="docs/media/gallery/robotwin-sources/move_can_pot.gif" alt="move_can_pot successful source"><br>83 recorded frames · source</td></tr>
<tr><td align="center"><b>move_playingcard_away</b><br><img width="240" src="docs/media/gallery/robotwin-sources/move_playingcard_away.gif" alt="move_playingcard_away successful source"><br>63 recorded frames · source</td><td align="center"><b>move_stapler_pad</b><br><img width="240" src="docs/media/gallery/robotwin-sources/move_stapler_pad.gif" alt="move_stapler_pad successful source"><br>80 recorded frames · source</td><td align="center"><b>pick_dual_bottles</b><br><img width="240" src="docs/media/gallery/robotwin-sources/pick_dual_bottles.gif" alt="pick_dual_bottles successful source"><br>70 recorded frames · source</td></tr>
<tr><td align="center"><b>place_a2b_right</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_a2b_right.gif" alt="place_a2b_right successful source"><br>81 recorded frames · source</td><td align="center"><b>place_burger_fries</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_burger_fries.gif" alt="place_burger_fries successful source"><br>132 recorded frames · source</td><td align="center"><b>place_cans_plasticbox</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_cans_plasticbox.gif" alt="place_cans_plasticbox successful source"><br>159 recorded frames · source</td></tr>
<tr><td align="center"><b>place_container_plate</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_container_plate.gif" alt="place_container_plate successful source"><br>89 recorded frames · source</td><td align="center"><b>place_dual_shoes</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_dual_shoes.gif" alt="place_dual_shoes successful source"><br>128 recorded frames · source</td><td align="center"><b>place_fan</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_fan.gif" alt="place_fan successful source"><br>87 recorded frames · source</td></tr>
<tr><td align="center"><b>place_mouse_pad</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_mouse_pad.gif" alt="place_mouse_pad successful source"><br>86 recorded frames · source</td><td align="center"><b>place_object_basket</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_object_basket.gif" alt="place_object_basket successful source"><br>139 recorded frames · source</td><td align="center"><b>place_object_scale</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_object_scale.gif" alt="place_object_scale successful source"><br>80 recorded frames · source</td></tr>
<tr><td align="center"><b>place_object_stand</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_object_stand.gif" alt="place_object_stand successful source"><br>77 recorded frames · source</td><td align="center"><b>place_phone_stand</b><br><img width="240" src="docs/media/gallery/robotwin-sources/place_phone_stand.gif" alt="place_phone_stand successful source"><br>71 recorded frames · source</td><td align="center"><b>press_stapler</b><br><img width="240" src="docs/media/gallery/robotwin-sources/press_stapler.gif" alt="press_stapler successful source"><br>65 recorded frames · source</td></tr>
<tr><td align="center"><b>put_bottles_dustbin</b><br><img width="240" src="docs/media/gallery/robotwin-sources/put_bottles_dustbin.gif" alt="put_bottles_dustbin successful source"><br>364 recorded frames · source</td><td align="center"><b>rotate_qrcode</b><br><img width="240" src="docs/media/gallery/robotwin-sources/rotate_qrcode.gif" alt="rotate_qrcode successful source"><br>85 recorded frames · source</td><td align="center"><b>shake_bottle</b><br><img width="240" src="docs/media/gallery/robotwin-sources/shake_bottle.gif" alt="shake_bottle successful source"><br>136 recorded frames · source</td></tr>
<tr><td align="center"><b>shake_bottle_horizontally</b><br><img width="240" src="docs/media/gallery/robotwin-sources/shake_bottle_horizontally.gif" alt="shake_bottle_horizontally successful source"><br>154 recorded frames · source</td><td align="center"><b>stack_blocks_three</b><br><img width="240" src="docs/media/gallery/robotwin-sources/stack_blocks_three.gif" alt="stack_blocks_three successful source"><br>254 recorded frames · source</td><td align="center"><b>stack_blocks_two</b><br><img width="240" src="docs/media/gallery/robotwin-sources/stack_blocks_two.gif" alt="stack_blocks_two successful source"><br>173 recorded frames · source</td></tr>
<tr><td align="center"><b>stack_bowls_three</b><br><img width="240" src="docs/media/gallery/robotwin-sources/stack_bowls_three.gif" alt="stack_bowls_three successful source"><br>270 recorded frames · source</td><td align="center"><b>stack_bowls_two</b><br><img width="240" src="docs/media/gallery/robotwin-sources/stack_bowls_two.gif" alt="stack_bowls_two successful source"><br>183 recorded frames · source</td><td align="center"><b>stamp_seal</b><br><img width="240" src="docs/media/gallery/robotwin-sources/stamp_seal.gif" alt="stamp_seal successful source"><br>77 recorded frames · source</td></tr>
<tr><td align="center"><b>turn_switch</b><br><img width="240" src="docs/media/gallery/robotwin-sources/turn_switch.gif" alt="turn_switch successful source"><br>53 recorded frames · source</td></tr>
</table>
</details>


<details>
<summary>展开补采新增的 13 类成功源轨迹</summary>

来自另外记录的固定种子补测与限次成功补采，不混入首轮成功率，也不计为纠错对。每张动图链接到对应实验报告。

<table>
<tr><td align="center"><b>dump_bin_bigbin</b><br><a href="docs/evidence/robotwin-followup-0-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/dump_bin_bigbin.gif" alt="dump_bin_bigbin successful source"></a><br>seed 902 · 190 frames · source</td><td align="center"><b>move_pillbottle_pad</b><br><a href="docs/evidence/robotwin-followup-1-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/move_pillbottle_pad.gif" alt="move_pillbottle_pad successful source"></a><br>seed 901 · 85 frames · source</td><td align="center"><b>open_laptop</b><br><a href="docs/evidence/robotwin-followup-0-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/open_laptop.gif" alt="open_laptop successful source"></a><br>seed 901 · 84 frames · source</td></tr>
<tr><td align="center"><b>open_microwave</b><br><a href="docs/evidence/robotwin-bootstrap-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/open_microwave.gif" alt="open_microwave successful source"></a><br>seed 903 · 293 frames · source</td><td align="center"><b>pick_diverse_bottles</b><br><a href="docs/evidence/robotwin-bootstrap-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/pick_diverse_bottles.gif" alt="pick_diverse_bottles successful source"></a><br>seed 905 · 68 frames · source</td><td align="center"><b>place_a2b_left</b><br><a href="docs/evidence/robotwin-followup-1-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_a2b_left.gif" alt="place_a2b_left successful source"></a><br>seed 901 · 82 frames · source</td></tr>
<tr><td align="center"><b>place_bread_basket</b><br><a href="docs/evidence/robotwin-followup-0-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_bread_basket.gif" alt="place_bread_basket successful source"></a><br>seed 901 · 128 frames · source</td><td align="center"><b>place_bread_skillet</b><br><a href="docs/evidence/robotwin-bootstrap-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_bread_skillet.gif" alt="place_bread_skillet successful source"></a><br>seed 903 · 96 frames · source</td><td align="center"><b>place_can_basket</b><br><a href="docs/evidence/robotwin-bootstrap-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_can_basket.gif" alt="place_can_basket successful source"></a><br>seed 903 · 136 frames · source</td></tr>
<tr><td align="center"><b>place_empty_cup</b><br><a href="docs/evidence/robotwin-followup-1-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_empty_cup.gif" alt="place_empty_cup successful source"></a><br>seed 902 · 102 frames · source</td><td align="center"><b>place_shoe</b><br><a href="docs/evidence/robotwin-followup-0-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/place_shoe.gif" alt="place_shoe successful source"></a><br>seed 901 · 102 frames · source</td><td align="center"><b>scan_object</b><br><a href="docs/evidence/robotwin-followup-0-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/scan_object.gif" alt="scan_object successful source"></a><br>seed 902 · 92 frames · source</td></tr>
<tr><td align="center"><b>put_object_cabinet</b><br><a href="docs/evidence/robotwin-bootstrap-summary.json"><img width="240" src="docs/media/gallery/robotwin-sources/put_object_cabinet.gif" alt="put_object_cabinet successful source"></a><br>seed 909 · 149 frames · source</td></tr>
</table>
</details>

### MetaWorld 全部 50 类任务：纠错成功回放

每类展示一个实测有效纠错样本。下方 50 条恢复轨迹再次独立回放，记录状态误差均为 0；点击动图可查看证据。对应源轨迹、错误对照、恢复轨迹共 [150 条独立回放](docs/evidence/metaworld50-independent-replay.json)全部通过。这代表任务类别已有样本，不代表所有种子、扰动或 MT50 评测配置均成功。

<table>
<tr><td align="center"><b>assembly-v3</b><br><a href="docs/evidence/metaworld50-gallery/assembly-v3.json"><img src="docs/media/gallery/metaworld50/assembly-v3.gif" width="240" alt="assembly-v3 verified recovery"></a><br>seed 600 · 219 actions</td><td align="center"><b>basketball-v3</b><br><a href="docs/evidence/metaworld50-gallery/basketball-v3.json"><img src="docs/media/gallery/metaworld50/basketball-v3.gif" width="240" alt="basketball-v3 verified recovery"></a><br>seed 600 · 174 actions</td><td align="center"><b>bin-picking-v3</b><br><a href="docs/evidence/metaworld50-gallery/bin-picking-v3.json"><img src="docs/media/gallery/metaworld50/bin-picking-v3.gif" width="240" alt="bin-picking-v3 verified recovery"></a><br>seed 600 · 220 actions</td></tr>
<tr><td align="center"><b>box-close-v3</b><br><a href="docs/evidence/metaworld50-gallery/box-close-v3.json"><img src="docs/media/gallery/metaworld50/box-close-v3.gif" width="240" alt="box-close-v3 verified recovery"></a><br>seed 600 · 158 actions</td><td align="center"><b>button-press-topdown-v3</b><br><a href="docs/evidence/metaworld50-gallery/button-press-topdown-v3.json"><img src="docs/media/gallery/metaworld50/button-press-topdown-v3.gif" width="240" alt="button-press-topdown-v3 verified recovery"></a><br>seed 600 · 104 actions</td><td align="center"><b>button-press-topdown-wall-v3</b><br><a href="docs/evidence/metaworld50-gallery/button-press-topdown-wall-v3.json"><img src="docs/media/gallery/metaworld50/button-press-topdown-wall-v3.gif" width="240" alt="button-press-topdown-wall-v3 verified recovery"></a><br>seed 600 · 94 actions</td></tr>
<tr><td align="center"><b>button-press-v3</b><br><a href="docs/evidence/metaworld50-gallery/button-press-v3.json"><img src="docs/media/gallery/metaworld50/button-press-v3.gif" width="240" alt="button-press-v3 verified recovery"></a><br>seed 600 · 97 actions</td><td align="center"><b>button-press-wall-v3</b><br><a href="docs/evidence/metaworld50-gallery/button-press-wall-v3.json"><img src="docs/media/gallery/metaworld50/button-press-wall-v3.gif" width="240" alt="button-press-wall-v3 verified recovery"></a><br>seed 600 · 110 actions</td><td align="center"><b>coffee-button-v3</b><br><a href="docs/evidence/metaworld50-gallery/coffee-button-v3.json"><img src="docs/media/gallery/metaworld50/coffee-button-v3.gif" width="240" alt="coffee-button-v3 verified recovery"></a><br>seed 600 · 79 actions</td></tr>
<tr><td align="center"><b>coffee-pull-v3</b><br><a href="docs/evidence/metaworld50-gallery/coffee-pull-v3.json"><img src="docs/media/gallery/metaworld50/coffee-pull-v3.gif" width="240" alt="coffee-pull-v3 verified recovery"></a><br>seed 600 · 103 actions</td><td align="center"><b>coffee-push-v3</b><br><a href="docs/evidence/metaworld50-gallery/coffee-push-v3.json"><img src="docs/media/gallery/metaworld50/coffee-push-v3.gif" width="240" alt="coffee-push-v3 verified recovery"></a><br>seed 600 · 91 actions</td><td align="center"><b>dial-turn-v3</b><br><a href="docs/evidence/metaworld50-gallery/dial-turn-v3.json"><img src="docs/media/gallery/metaworld50/dial-turn-v3.gif" width="240" alt="dial-turn-v3 verified recovery"></a><br>seed 600 · 117 actions</td></tr>
<tr><td align="center"><b>disassemble-v3</b><br><a href="docs/evidence/metaworld50-gallery/disassemble-v3.json"><img src="docs/media/gallery/metaworld50/disassemble-v3.gif" width="240" alt="disassemble-v3 verified recovery"></a><br>seed 600 · 203 actions</td><td align="center"><b>door-close-v3</b><br><a href="docs/evidence/metaworld50-gallery/door-close-v3.json"><img src="docs/media/gallery/metaworld50/door-close-v3.gif" width="240" alt="door-close-v3 verified recovery"></a><br>seed 600 · 73 actions</td><td align="center"><b>door-lock-v3</b><br><a href="docs/evidence/metaworld50-gallery/door-lock-v3.json"><img src="docs/media/gallery/metaworld50/door-lock-v3.gif" width="240" alt="door-lock-v3 verified recovery"></a><br>seed 600 · 115 actions</td></tr>
<tr><td align="center"><b>door-open-v3</b><br><a href="docs/evidence/metaworld50-gallery/door-open-v3.json"><img src="docs/media/gallery/metaworld50/door-open-v3.gif" width="240" alt="door-open-v3 verified recovery"></a><br>seed 600 · 121 actions</td><td align="center"><b>door-unlock-v3</b><br><a href="docs/evidence/metaworld50-gallery/door-unlock-v3.json"><img src="docs/media/gallery/metaworld50/door-unlock-v3.gif" width="240" alt="door-unlock-v3 verified recovery"></a><br>seed 600 · 88 actions</td><td align="center"><b>drawer-close-v3</b><br><a href="docs/evidence/metaworld50-gallery/drawer-close-v3.json"><img src="docs/media/gallery/metaworld50/drawer-close-v3.gif" width="240" alt="drawer-close-v3 verified recovery"></a><br>seed 600 · 108 actions</td></tr>
<tr><td align="center"><b>drawer-open-v3</b><br><a href="docs/evidence/metaworld50-gallery/drawer-open-v3.json"><img src="docs/media/gallery/metaworld50/drawer-open-v3.gif" width="240" alt="drawer-open-v3 verified recovery"></a><br>seed 600 · 149 actions</td><td align="center"><b>faucet-close-v3</b><br><a href="docs/evidence/metaworld50-gallery/faucet-close-v3.json"><img src="docs/media/gallery/metaworld50/faucet-close-v3.gif" width="240" alt="faucet-close-v3 verified recovery"></a><br>seed 600 · 95 actions</td><td align="center"><b>faucet-open-v3</b><br><a href="docs/evidence/metaworld50-gallery/faucet-open-v3.json"><img src="docs/media/gallery/metaworld50/faucet-open-v3.gif" width="240" alt="faucet-open-v3 verified recovery"></a><br>seed 600 · 92 actions</td></tr>
<tr><td align="center"><b>hammer-v3</b><br><a href="docs/evidence/metaworld50-gallery/hammer-v3.json"><img src="docs/media/gallery/metaworld50/hammer-v3.gif" width="240" alt="hammer-v3 verified recovery"></a><br>seed 600 · 103 actions</td><td align="center"><b>hand-insert-v3</b><br><a href="docs/evidence/metaworld50-gallery/hand-insert-v3.json"><img src="docs/media/gallery/metaworld50/hand-insert-v3.gif" width="240" alt="hand-insert-v3 verified recovery"></a><br>seed 600 · 95 actions</td><td align="center"><b>handle-press-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/handle-press-side-v3.json"><img src="docs/media/gallery/metaworld50/handle-press-side-v3.gif" width="240" alt="handle-press-side-v3 verified recovery"></a><br>seed 600 · 80 actions</td></tr>
<tr><td align="center"><b>handle-press-v3</b><br><a href="docs/evidence/metaworld50-gallery/handle-press-v3.json"><img src="docs/media/gallery/metaworld50/handle-press-v3.gif" width="240" alt="handle-press-v3 verified recovery"></a><br>seed 600 · 79 actions</td><td align="center"><b>handle-pull-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/handle-pull-side-v3.json"><img src="docs/media/gallery/metaworld50/handle-pull-side-v3.gif" width="240" alt="handle-pull-side-v3 verified recovery"></a><br>seed 600 · 115 actions</td><td align="center"><b>handle-pull-v3</b><br><a href="docs/evidence/metaworld50-gallery/handle-pull-v3.json"><img src="docs/media/gallery/metaworld50/handle-pull-v3.gif" width="240" alt="handle-pull-v3 verified recovery"></a><br>seed 600 · 156 actions</td></tr>
<tr><td align="center"><b>lever-pull-v3</b><br><a href="docs/evidence/metaworld50-gallery/lever-pull-v3.json"><img src="docs/media/gallery/metaworld50/lever-pull-v3.gif" width="240" alt="lever-pull-v3 verified recovery"></a><br>seed 600 · 105 actions</td><td align="center"><b>peg-insert-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/peg-insert-side-v3.json"><img src="docs/media/gallery/metaworld50/peg-insert-side-v3.gif" width="240" alt="peg-insert-side-v3 verified recovery"></a><br>seed 600 · 125 actions</td><td align="center"><b>peg-unplug-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/peg-unplug-side-v3.json"><img src="docs/media/gallery/metaworld50/peg-unplug-side-v3.gif" width="240" alt="peg-unplug-side-v3 verified recovery"></a><br>seed 600 · 151 actions</td></tr>
<tr><td align="center"><b>pick-out-of-hole-v3</b><br><a href="docs/evidence/metaworld50-gallery/pick-out-of-hole-v3.json"><img src="docs/media/gallery/metaworld50/pick-out-of-hole-v3.gif" width="240" alt="pick-out-of-hole-v3 verified recovery"></a><br>seed 600 · 178 actions</td><td align="center"><b>pick-place-v3</b><br><a href="docs/evidence/metaworld50-gallery/pick-place-v3.json"><img src="docs/media/gallery/metaworld50/pick-place-v3.gif" width="240" alt="pick-place-v3 verified recovery"></a><br>seed 600 · 88 actions</td><td align="center"><b>pick-place-wall-v3</b><br><a href="docs/evidence/metaworld50-gallery/pick-place-wall-v3.json"><img src="docs/media/gallery/metaworld50/pick-place-wall-v3.gif" width="240" alt="pick-place-wall-v3 verified recovery"></a><br>seed 600 · 183 actions</td></tr>
<tr><td align="center"><b>plate-slide-back-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/plate-slide-back-side-v3.json"><img src="docs/media/gallery/metaworld50/plate-slide-back-side-v3.gif" width="240" alt="plate-slide-back-side-v3 verified recovery"></a><br>seed 600 · 95 actions</td><td align="center"><b>plate-slide-back-v3</b><br><a href="docs/evidence/metaworld50-gallery/plate-slide-back-v3.json"><img src="docs/media/gallery/metaworld50/plate-slide-back-v3.gif" width="240" alt="plate-slide-back-v3 verified recovery"></a><br>seed 600 · 81 actions</td><td align="center"><b>plate-slide-side-v3</b><br><a href="docs/evidence/metaworld50-gallery/plate-slide-side-v3.json"><img src="docs/media/gallery/metaworld50/plate-slide-side-v3.gif" width="240" alt="plate-slide-side-v3 verified recovery"></a><br>seed 600 · 213 actions</td></tr>
<tr><td align="center"><b>plate-slide-v3</b><br><a href="docs/evidence/metaworld50-gallery/plate-slide-v3.json"><img src="docs/media/gallery/metaworld50/plate-slide-v3.gif" width="240" alt="plate-slide-v3 verified recovery"></a><br>seed 600 · 88 actions</td><td align="center"><b>push-back-v3</b><br><a href="docs/evidence/metaworld50-gallery/push-back-v3.json"><img src="docs/media/gallery/metaworld50/push-back-v3.gif" width="240" alt="push-back-v3 verified recovery"></a><br>seed 600 · 116 actions</td><td align="center"><b>push-v3</b><br><a href="docs/evidence/metaworld50-gallery/push-v3.json"><img src="docs/media/gallery/metaworld50/push-v3.gif" width="240" alt="push-v3 verified recovery"></a><br>seed 600 · 96 actions</td></tr>
<tr><td align="center"><b>push-wall-v3</b><br><a href="docs/evidence/metaworld50-gallery/push-wall-v3.json"><img src="docs/media/gallery/metaworld50/push-wall-v3.gif" width="240" alt="push-wall-v3 verified recovery"></a><br>seed 600 · 151 actions</td><td align="center"><b>reach-v3</b><br><a href="docs/evidence/metaworld50-gallery/reach-v3.json"><img src="docs/media/gallery/metaworld50/reach-v3.gif" width="240" alt="reach-v3 verified recovery"></a><br>seed 600 · 65 actions</td><td align="center"><b>reach-wall-v3</b><br><a href="docs/evidence/metaworld50-gallery/reach-wall-v3.json"><img src="docs/media/gallery/metaworld50/reach-wall-v3.gif" width="240" alt="reach-wall-v3 verified recovery"></a><br>seed 600 · 68 actions</td></tr>
<tr><td align="center"><b>shelf-place-v3</b><br><a href="docs/evidence/metaworld50-gallery/shelf-place-v3.json"><img src="docs/media/gallery/metaworld50/shelf-place-v3.gif" width="240" alt="shelf-place-v3 verified recovery"></a><br>seed 600 · 150 actions</td><td align="center"><b>soccer-v3</b><br><a href="docs/evidence/metaworld50-gallery/soccer-v3.json"><img src="docs/media/gallery/metaworld50/soccer-v3.gif" width="240" alt="soccer-v3 verified recovery"></a><br>seed 600 · 143 actions</td><td align="center"><b>stick-pull-v3</b><br><a href="docs/evidence/metaworld50-gallery/stick-pull-v3.json"><img src="docs/media/gallery/metaworld50/stick-pull-v3.gif" width="240" alt="stick-pull-v3 verified recovery"></a><br>seed 621 · 186 actions</td></tr>
<tr><td align="center"><b>stick-push-v3</b><br><a href="docs/evidence/metaworld50-gallery/stick-push-v3.json"><img src="docs/media/gallery/metaworld50/stick-push-v3.gif" width="240" alt="stick-push-v3 verified recovery"></a><br>seed 622 · 105 actions</td><td align="center"><b>sweep-into-v3</b><br><a href="docs/evidence/metaworld50-gallery/sweep-into-v3.json"><img src="docs/media/gallery/metaworld50/sweep-into-v3.gif" width="240" alt="sweep-into-v3 verified recovery"></a><br>seed 600 · 114 actions</td><td align="center"><b>sweep-v3</b><br><a href="docs/evidence/metaworld50-gallery/sweep-v3.json"><img src="docs/media/gallery/metaworld50/sweep-v3.gif" width="240" alt="sweep-v3 verified recovery"></a><br>seed 600 · 144 actions</td></tr>
<tr><td align="center"><b>window-close-v3</b><br><a href="docs/evidence/metaworld50-gallery/window-close-v3.json"><img src="docs/media/gallery/metaworld50/window-close-v3.gif" width="240" alt="window-close-v3 verified recovery"></a><br>seed 600 · 114 actions</td><td align="center"><b>window-open-v3</b><br><a href="docs/evidence/metaworld50-gallery/window-open-v3.json"><img src="docs/media/gallery/metaworld50/window-open-v3.gif" width="240" alt="window-open-v3 verified recovery"></a><br>seed 600 · 122 actions</td></tr>
</table>

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
| ManiSkill 统一纠错 | 首轮全专家扫描 19/24 成功源轨迹、8/24 有效纠错对 | 已有成功源轨迹覆盖 10 类，结合试采样，有效纠错覆盖 7 类；保留 4 次配置错误 |
| RoboTwin 原生采集与逐步驱动 | 50 类任务全部尝试，seed 900 成功源轨迹 **37/50** | stack_blocks_two 修复后 **3/4 有效纠错对**、12 条轨迹审计通过；跨任务恢复仍属实验性 |

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

RoboCasa 当前控制器不是完整避障/移动操作规划器。AI2-THOR 当前使用可达点真值地图，是自定义 PointNav 采集协议，不是官方 ObjectNav 分数。RoboDojo 已复用完整共享 Isaac 环境和资产，完成 `stack_bowls` 原生 reset 与 10 个物理步；尚未验证真实策略完成任务，不能将其初始 `success=True` 当成任务完成。

仓库附有一个可直接检查的[真实纠错三分支样本](examples/correction/)，无需仿真器即可运行 `python -m robot_stack.audit examples/correction --output /tmp/robot-stack-example-audit.json` 检查文件。

## 网页、复查和协作

```bash
python scripts/build_collection_report.py --input outputs/pick-place --output docs/report.json
python -m http.server 8000 --directory docs
python -m unittest discover -s tests -v
```

网页展示真实结果和视频，CI 检查轻量契约。后续优先完善碰撞感知导航、任务策略、自然失败的状态反馈恢复，以及 MimicGen 的环境/子任务标注。**MimicGen 已通过原生 Lift 验证：2 条源示范生成 10 条新轨迹，7 条成功，3 条失败保留，全部 10 条通过独立动作重放。** 详见 [复现命令与适用边界](docs/mimicgen.md)。这些扩增示范尚不属于新的纠错对。AI2-THOR FloorPlan1 也已完成 3/3 组原生导航纠错验证。

上游项目、完整边界与贡献方式见 [英文首页](README.md)。
