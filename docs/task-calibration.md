# 每个任务先测基线，再校准可恢复扰动

统一验收要求是：任务有可复现的成功源生成能力，并有真实造成失败且能够恢复的扰动配置。注册环境、一个成功视频或某条恢复轨迹，不能替代逐任务的成功率测量。

新的 `robot-stack-calibrate` 入口按三个阶段工作：

1. **无扰动基线**：独立执行专家，保存成功和失败的 `source.hdf5`，测量成功数／尝试数。默认基线门槛为 80%，可配置。达不到门槛先标记 `baseline_below_target`，不靠降低扰动掩盖专家的问题。
2. **扰动校准**：只在校准种子上运行全部指定档位。默认尝试原配置幅度的 1/8、1/4、1/2、1 倍，保留每次结果。选择能产生合格纠错且达到恢复率门槛的最小幅度；减小后对照仍成功的配置不能算合格纠错。
3. **独立验证**：在查看验证结果前冻结选中的配置，再换一组种子执行。检查初始场景的 `split_group`，相同初始场景不能伪装成独立验证。MetaWorld 还使用不重叠的 task_index。默认至少五个不同验证场景、验证源成功率 ≥80%、失败对照条件下恢复率 ≥50%，且至少一个合格纠错，才通过配置门槛。

这些是小样本经验门槛，并非统计置信保证。恢复率的分母是有效错误且原动作对照失败的完整尝试；界面同时显示分子、分母和总尝试数。校准样本和验证样本分开，不能用调参样本宣称泛化成功率。验收标准可以提高。

## 安装和启动

在对应仿真器环境中安装项目。不同仿真器的依赖与资产仍需分别安装；本命令不会自动下载所有仿真器。

```bash
pip install -e .

# 单独测一个任务的成功源，不注入任何扰动
robot-stack-collect --backend metaworld --task pick-place-v3 \
  --source-only --episodes 10 --seed-start 2000 --max-steps 500 \
  --output outputs/pick-place-baseline

# MetaWorld 全部 50 类任务：基线 + 分档扰动 + 独立验证
robot-stack-calibrate --backend metaworld --tasks all \
  --episodes 5 --seed-start 2100 \
  --validation-episodes 5 --validation-seed-start 3100 \
  --scales 0.25 0.5 1 --max-steps 500 --workers 4 \
  --schedule examples/perturbations/simple.json \
  --output outputs/metaworld-calibrated

# ManiSkill 对指定任务降低关节扰动，保持持续时间不变
robot-stack-calibrate --backend maniskill \
  --tasks PickCube-v1 PlaceSphere-v1 LiftPegUpright-v1 PullCubeTool-v1 PlugCharger-v1 \
  --episodes 5 --seed-start 810 \
  --validation-episodes 5 --validation-seed-start 1810 \
  --scales 0.125 0.25 0.5 --max-steps 1000 --workers 4 \
  --adapter-options '{"native_horizon":1200,"max_replans":4}' \
  --schedule examples/perturbations/maniskill-late.json \
  --output outputs/maniskill-calibrated
```

MetaWorld 的原生训练 split 最多 50 个 task_index，校准和验证实例总数不能超过 50。单独用 `robot-stack-collect` 换 seed 并不等于换 task_index；如需与本轮前十个场景分离，可逐次传入 `--adapter-options '{"task_index":10}'` 等未使用的索引，并检查 `split_group`。本轮 ManiSkill 实测为节省诊断预算使用两个校准种子，验证仍用五个新种子；上述五个校准种子的命令用于更充分的后续采集。

`--scales` 默认缩放每个事件的 `strength`，不会悄悄改变插入位置、持续时间、错误阈值或夹爪布尔参数。示例关节 offset 为 0.8 弧度，因此三个档位对应 0.1、0.2、0.4 弧度；原来的 15 步持续时间和张爪行为不变。AI2-THOR 使用离散动作，应设 `--axis steps` 缩短动作次数，不能把 90°动作假装连续缩小。时长缩放向上取整，至少一个原生动作；多个比例可能落在同一时长。

`--min-source-rate`、`--min-recovery-rate` 和 `--min-validation-episodes` 可配置验收门槛。`--workers` 只控制隔离子进程数量；不会把所有任务塞入同一个非线程安全的仿真实例。资产加载失败、超时和专家异常均保留。

## 如何读取结果

- `run_config.json`：任务、种子、档位、预算和门槛，运行前固定。
- `baseline/`：无扰动源轨迹；不会重复累计各扰动档位重新生成的源作为基线样本。
- `calibration/`：全部档位的三分支尝试，包含无有效错误及恢复失败。
- `selected_profiles.json`：验证开始前固定的候选配置。
- `selected/<task>.json`：可直接传给 `robot-stack-collect --schedule` 的调度文件；使用前检查对应任务的验证状态。
- `validation/`：冻结配置在独立种子上的结果。
- `summary.json`：每个任务的基线、各档指标、验证指标和最终状态。

每个任务的上述目录都位于 `<output>/<task>/` 下；选中配置目录和汇总文件位于输出根目录。

任务状态明确区分：缺少专家、基线未达标、尚无合格扰动、验证场景不足、场景重叠、验证恢复率不足，以及通过当前配置的验证。不能把零分母写成 0% 或 100%。没有内置专家的任务会保留在 `--tasks all` 的报告中，不会被悄悄排除；对于提供自定义 `adapter_factory` 的任务仍可尝试。

```bash
python -m robot_stack.audit outputs/maniskill-calibrated \
  --output outputs/maniskill-calibrated/audit.json
python -m robot_stack.readiness \
  --coverage docs/evidence/task-coverage.json \
  --calibrations outputs/maniskill-calibrated/summary.json \
  --output outputs/task-readiness.json
```

`readiness` 保留全任务清单，并叠加本次测量。已有成功示范但没有完成分档与独立验证的任务，仍显示“待校准”；不能因此宣称所有 benchmark 已完成。

## 2026-09-24 服务器实测结果

本轮独立测量 55 类任务。MetaWorld 每类使用五个校准场景和最多五个独立验证场景；ManiSkill 每类使用两个校准种子和最多五个独立验证种子。低基线任务不进入扰动校准，没有合格候选的任务不进入验证。全部失败均保留。

| 环境 | 任务类 | 无扰动基线达到本轮 80% 门槛 | 通过独立验证 | 实际采集尝试 | 审计 HDF5 文件 | 合格纠错组（校准＋验证） |
|---|---:|---:|---:|---:|---:|---:|
| metaworld | 50 | 48 | 25 | 1210 | 3122 | 849 |
| maniskill | 5 | 4 | 3 | 49 | 127 | 21 |

“采集尝试”包括干净基线，以及各档校准和独立验证的一次三分支尝试；不能与 HDF5 文件数混用。合格纠错总数包含调参集，不应用于宣称独立验证成功率。

| ManiSkill 任务 | 干净基线 | 选中 offset | 验证成功源 | 验证失败对照后恢复 | 合格纠错／全部验证尝试 |
|---|---:|---:|---:|---:|---:|
| PickCube-v1 | 2/2 | 0.2 rad | 5/5 | 3/3 | 3/5 |
| PlaceSphere-v1 | 2/2 | 未选中 | — | — | — |
| LiftPegUpright-v1 | 2/2 | 0.2 rad | 5/5 | 5/5 | 5/5 |
| PullCubeTool-v1 | 2/2 | 0.1 rad | 5/5 | 3/5 | 3/5 |
| PlugCharger-v1 | 1/2 | 未选中 | — | — | — |

PlaceSphere 的三个幅度均未得到合格纠错；PlugCharger 基线 1/2，先改进成功源专家。MetaWorld 的 `soccer-v3` 和 `stick-pull-v3` 均为 3/5，低于本轮门槛。其余逐任务结果见[验收网页](https://pm1255.github.io/robot_stack/readiness.html)。

MetaWorld 另有 23 类触发初始动力学状态哈希重合，虽 task_index 不重叠，当前指纹仍不足以证明场景独立。这可能与 MuJoCo 动力学状态未包含静态模型配置有关，尚需完整场景指纹核验；本轮保守地不计通过，不把它们说成恢复失败，也不直接认定原生场景重复。

这次清单包含 546 类任务，只有本轮实际测量的任务有新基线，不能宣称全部 benchmark 已达标。

已通过的调度文件保存在 [`examples/calibrated`](../examples/calibrated)，对应证据与分母见 `manifest.json`。例如：

```bash
robot-stack-collect --backend maniskill --task PickCube-v1 \
  --episodes 10 --seed-start 4000 --max-steps 1000 \
  --adapter-options '{"native_horizon":1200,"max_replans":4}' \
  --schedule examples/calibrated/maniskill/PickCube-v1.json \
  --output outputs/pickcube-new-scenes
```

新种子的结果仍需重新审计；这条命令不是十条必然成功的承诺。完整证据：[MetaWorld 报告](evidence/calibration/metaworld-summary.json)、[MetaWorld 审计](evidence/calibration/metaworld-audit.json)、[ManiSkill 报告](evidence/calibration/maniskill-summary.json)、[ManiSkill 审计](evidence/calibration/maniskill-audit.json)。
