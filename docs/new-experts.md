# 新增任务专家：成功源优先

本批补上 5 个此前没有注册专家的任务，均已在原生模拟器中保存过成功源。专家注册、至少一个成功源、稳定成功率、合格纠错是四个不同层级。

| 环境 / 任务 | 共享技能 | 开发种子成功源 | 冻结代码后的新种子成功源 | 当前限制 |
|---|---|---:|---:|---|
| ManiSkill / PickSingleYCB-v1 | OBB 抓取、抬升、保持 TCP 偏移搬运 | 2/2 | 4/5 | 原生物体采样；一次抓取/规划失败，未验证纠错 |
| ManiSkill / PokeCube-v1 | 抓工具、对齐工具尖端、推方块 | 2/2 | 未测 | 三档扰动均未得到合格错误对照 |
| RoboCasa / PickPlaceSinkToCounter | 底盘导航、倾斜抓取、搬运放置 | v4 调试 1/2；旧成功场景回归 1/2 | 2/5 | v4 在 seed 1102 退化；仍受水龙头/水槽遮挡影响 |
| RoboCasa / CheesyBread | 移动抓取、对齐面包、放下奶酪 | 1/2 | 0/5 | 已有成功例，但新场景抓取及接触放置不稳定 |
| RoboCasa / PackDessert | 移动抓取、避开容器中食物后放置 | 1/2 | 1/5 | 放置目标形状与占用变化仍会导致失败 |

ManiSkill 新种子为 4105–4109；RoboCasa CheesyBread / PackDessert 为 1105–1109。水槽 v4 用已知失败种子 1101、1105 调试，再用 1110–1114 验证；1102、1107 仅用于回归，不算新种子。v3 水槽的 1105–1109 结果为 1/5，后续已经用于调试，不能再次当作独立验证。每组只有 5 个新场景，不能据此宣称全任务成功率。

RoboCasa 限定 PandaOmron、layout/style=1；水槽任务使用 apple，两个复合任务保留原生物体采样。ManiSkill 使用原生 Panda 或 Panda wristcam、PhysX CPU 动力学。控制器使用特权状态、原生成功判定与真实 `env.step`，不改对象位姿或放宽成功条件。MPlib 使用 screw 规划，失败后尝试 RRTConnect；这不代表已建立所有家具的完整碰撞地图。

## 用户入口

先在对应模拟器的独立 Python 环境安装本仓库。ManiSkill 可使用 `python -m pip install -e '.[maniskill]'`。RoboCasa 使用 [现有原生环境与依赖说明](robocasa-mobile.md)，不建议将所有模拟器依赖混装。

```bash
# 注册表只说明实现存在；实际结果看报告。
python -m robot_stack.experts.catalog --backend maniskill
python -m robot_stack.experts.catalog --backend robocasa
# pip 安装后等价入口：robot-stack-experts --backend maniskill

# YCB 需要资产。可写的资产目录由用户选择。
export MS_ASSET_DIR="$PWD/runtime/maniskill-assets"
python -m mani_skill.utils.download_asset ycb --non-interactive

# 从仓库根目录运行；每个 output 必须是新目录。
bash examples/experts/sources.sh PickSingleYCB-v1 outputs/ycb-sources 4105 5
bash examples/experts/sources.sh PokeCube-v1 outputs/poke-sources 4205 5

# 切换至 RoboCasa Python 环境后：
bash examples/experts/sources.sh PickPlaceSinkToCounter outputs/sink-sources 1110 5
bash examples/experts/sources.sh CheesyBread outputs/cheese-sources 1105 5
bash examples/experts/sources.sh PackDessert outputs/dessert-sources 1105 5
```

脚本调用统一的 `python -m robot_stack.collect --source-only`；第四个参数控制尝试数，第三个参数控制起始种子。审计验证保存文件的一致性，不把失败轨迹变成成功，也不把没有文件的异常当作成功。

服务器无法出网时，可从官方地址下载 YCB 压缩包再传入服务器：`https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip`。本批使用的 26,229,037 字节文件及 SHA256 见 [资产清单](evidence/new-experts/ycb-asset-manifest.json)。解压后的模型位置为 `$MS_ASSET_DIR/data/assets/mani_skill2_ycb`，不需要改代码硬编码服务器路径。

## 扰动与纠错入口仍然统一

去掉 `--source-only`，传入 `--schedule` 即可进入源 / 错误对照 / 恢复采集。使用 [逐环境扰动文档](benchmark-perturbations.md) 设置步骤、比例或里程碑、类型、幅度、时长和次数。能调用不等于该任务已验证成功纠错。

```bash
# 本批真实执行过的 PokeCube 校准，可复现实验失败结论。
bash examples/experts/poke-calibration.sh outputs/poke-calibration
```

该实验在源轨迹 60% 处施加 15 步 joint_offset，三档强度 0.125 / 0.25 / 0.5。2 个种子 × 3 档共 6 次：物理状态有偏离，原动作对照仍 6/6 成功，恢复也 6/6 成功，因此 **合格纠错为 0/6**。校准器没有选择有效档位，也没有启动后续独立纠错验收。接下来需要改变干预阶段/类型或恢复方案，不能把正常仍能成功的轨迹标为“纠错”。

## 证据与复用边界

[网页及原生成功回放](https://pm1255.github.io/robot_stack/experts.html) · [各次实验代码哈希](evidence/new-experts/experiment-code-manifest.json) · [64 项合约测试](evidence/new-experts/contract-tests.txt)。历史 v1/v2/v3/v4 报告全部保留，包括缺资产异常、调试失败和退化。重复种子跨版本运行不是独立场景，不汇总成扩大后的成功率。

共享技能位于 `robot_stack/experts/maniskill.py` 和 `robot_stack/adapters/robocasa_mobile.py`。ManiSkill 注册表将新增策略与上游策略合并；RoboCasa 通过任务名映射被操作物体与目标容器。新增任务不必重写数据记录、扰动、回放和审计，但仍需适配机器人、可执行策略及任务约束。当前尚未为抽屉、门、液体、烹饪等其余技能族补齐专家。

当前覆盖清单有 546 类任务，119 类有至少一个成功源；427 类尚无成功源，其中 426 类缺注册专家。这个清单是已安装版本的任务登记，并非所有主流 benchmark 的全集。[完整缺口与后续顺序](source-backlog.md)。
