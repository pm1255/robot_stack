"""Check dataset quota accounting without requiring a native simulator."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_collector import robotwin


class BootstrapContracts(unittest.TestCase):
    def collect(self, outcomes, target):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'native'
            output = Path(temporary) / 'collected'
            for name in ('envs/_base_task.py', 'envs/example.py', 'task_config/demo_clean.yml'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture\n')

            class Env:
                plan_success = True
                left_joint_path = right_joint_path = []
                FRAME_IDX = 1

                def setup_demo(self, seed, save_path, **kwargs):
                    self.success = outcomes[seed]
                    self.output = Path(save_path)

                def play_once(self):
                    return {}

                def check_success(self):
                    return self.success

                def save_traj_data(self, index):
                    pass

                def merge_pkl_to_hdf5_video(self):
                    path = self.output / 'data/episode0.hdf5'
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b'fixture: native data inspection is mocked')

                def close_env(self, **kwargs):
                    pass

            argv = ['collector', '--root', str(root), '--output', str(output),
                    '--tasks', 'example', '--episodes', str(len(outcomes)),
                    '--seed-start', '0', '--success-target', str(target)]
            cwd, paths = Path.cwd(), list(sys.path)
            native_import = robotwin.importlib.import_module
            try:
                with patch.object(sys, 'argv', argv), \
                     patch.dict(sys.modules, {'torch': SimpleNamespace(set_num_threads=lambda n: None)}), \
                     patch('importlib.metadata.version', return_value='fixture'), \
                     patch.object(robotwin, 'config', side_effect=lambda r, out, task: {'save_path': str(out)}), \
                     patch.object(robotwin, 'inspect_hdf5', return_value={'joint': [1], 'endpose': [1]}), \
                     patch.object(robotwin.importlib, 'import_module', side_effect=lambda name, *a, **kw:
                                  SimpleNamespace(example=Env) if name == 'envs.example'
                                  else native_import(name, *a, **kw)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    status = robotwin.main()
                summary = json.loads((output / 'summary.json').read_text())
                files = list((output / 'example').glob('*_episode_result.json'))
                return status, summary, len(files)
            finally:
                os.chdir(cwd)
                sys.path[:] = paths

    def test_stops_at_target_and_keeps_failed_attempt(self):
        status, summary, records = self.collect([False, True, True], 1)
        self.assertEqual((status, summary['successes'], records), (0, 1, 2))
        self.assertEqual([r['seed'] for r in summary['results']], [0, 1])
        self.assertTrue(summary['seed_search'])
        self.assertTrue(summary['success_targets_reached'])

    def test_unmet_quota_exhausts_cap_and_returns_failure(self):
        status, summary, records = self.collect([False, True, False], 2)
        self.assertEqual((status, summary['successes'], records), (1, 1, 3))
        self.assertFalse(summary['success_targets_reached'])
