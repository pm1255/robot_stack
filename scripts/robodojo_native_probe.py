"""Native RoboDojo asset/reset/physics probe; does not claim policy success."""
import argparse
import importlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app
result = {'native_reset': False, 'physics_steps': 0, 'task_success': None, 'task': 'stack_bowls',
          'upstream_commit': '726e9aabfaa642203722eb126f5eaf0f37f3e1ad'}
env = None
try:
    from omegaconf import OmegaConf
    from env.global_configs import ROOT_DIR
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization
    registry = importlib.import_module('task.RoboDojo.task_registry')
    root = Path(ROOT_DIR)
    eval_cfg = load_yaml(str(root / 'env_cfg/arx_x5.yml'))
    eval_cfg.update(task_name='stack_bowls', num_envs=1, device_id=0, eval_batch=False,
                    policy_name='robot_stack_probe', additional_info='native_smoke', seed=0)
    config = {key: load_yaml(str(root / 'env_cfg' / key / (value + '.yml')))
              for key, value in eval_cfg['config'].items()}
    config.update(task_env=load_yaml(registry.task_config_path(str(root / 'task/RoboDojo/config'), 'stack_bowls')),
                  eval_cfg=eval_cfg)
    cfg = process_randomization(OmegaConf.create(config))
    cfg, _ = process_config(cfg, task_name='stack_bowls')
    cfg.sim.scene.num_envs = 1
    cfg.sim.seed = [0]
    cfg.camera.default_frequency = 25
    _, task_class = registry.load_task_class('stack_bowls')
    env = task_class(cfg, app)
    env.reset(seed=[0])
    result['native_reset'] = True
    for i in range(10):
        env.sim_step(render=False)
        result['physics_steps'] = i + 1
except Exception as exc:
    import traceback
    result.update(error=repr(exc), traceback=traceback.format_exc())
finally:
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('ROBOT_STACK_PROBE', json.dumps(result), flush=True)
    if env is not None:
        env.close()
    app.close()
if 'error' in result:
    raise SystemExit(1)
