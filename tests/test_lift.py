import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from robosuite_collector.lift import LiftSkills, StepBudgetExceeded
from robosuite_collector.collect import save_trajectory
from scripts.build_collection_report import build_report


class FakeEnv:
    action_dim = 7
    action_spec = (-np.ones(7), np.ones(7))
    cube = object()
    robots = [SimpleNamespace(gripper=object())]

    def __init__(self):
        self.tick = 0
        self.obs = {'robot0_eef_pos': np.zeros(3), 'cube_pos': np.array([0., 0., .8])}
        self.sim = SimpleNamespace(get_state=lambda: np.array([float(self.tick)]))
        self.grasp = False
        self.oracle = True

    def step(self, action):
        self.tick += 1
        return self.obs, 0, False, {}

    def _check_grasp(self, **kwargs):
        return self.grasp

    def _check_success(self):
        return self.oracle


class LiftContracts(unittest.TestCase):
    def test_oracle_alone_or_transient_lift_cannot_certify_success(self):
        env = FakeEnv()
        skills = LiftSkills(env, env.obs)
        self.assertFalse(skills.success())
        env.grasp = True
        self.assertFalse(skills.success())
        env.obs['cube_pos'][2] = .95
        self.assertTrue(skills.success())
        # Losing contact, despite the oracle and height, fails the physical gate.
        env.grasp = False
        self.assertFalse(skills.success())

    def test_action_budget_and_invalid_action_never_step_past_limits(self):
        env = FakeEnv()
        skills = LiftSkills(env, env.obs, max_steps=1)
        with self.assertRaises(ValueError):
            skills.step([0, 0, 0], 2)
        self.assertEqual(env.tick, 0)
        skills.step([0, 0, 0], -1)
        with self.assertRaises(StepBudgetExceeded):
            skills.step([0, 0, 0], -1)
        self.assertEqual(env.tick, 1)
        self.assertEqual(len(skills.states), len(skills.actions) + 1)

    def test_recovery_steps_are_in_same_episode_and_budget(self):
        env = FakeEnv()  # No reset method: recovery must use this physical state.
        skills = LiftSkills(env, env.obs, max_steps=4)
        skills.pick = lambda **kwargs: 'grasp_not_established'
        result = skills.run(max_attempts=3)
        self.assertFalse(result['success'])
        self.assertEqual(env.tick, 4)
        self.assertEqual(result['failures'][-1]['reason'], 'Episode action budget exhausted')
        self.assertTrue(all(s['phase'] == 'open_for_retry' for s in skills.samples))

    def test_saved_pre_action_states_and_final_state_align(self):
        env = FakeEnv()
        skills = LiftSkills(env, env.obs)
        skills.step([0, 0, 0], -1)
        skills.step([0, 0, 0], 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            save_trajectory(path, skills, '<mujoco/>', {}, {'seed': 0, 'success': False})
            with h5py.File(path) as f:
                demo = f['data/demo_0']
                np.testing.assert_array_equal(demo['states'][:].ravel(), [0, 1])
                np.testing.assert_array_equal(demo['final_state'][:], [2])
                self.assertEqual(demo['actions'].shape, (2, 7))

    def test_controller_success_requires_task_and_execution_evidence(self):
        base = dict(validation_version=1, success=True, status='success', backend='robosuite',
                    control_success=True, execution_success=True, task_success=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'ep_episode_result.json'
            for backend, field in ((b, f) for b in ('robosuite', 'metaworld', 'maniskill', 'robotwin')
                                   for f in (None, 'task_success', 'execution_success', 'control_success')):
                record = dict(base, backend=backend)
                if field:
                    record[field] = False
                path.write_text(json.dumps(record))
                report = build_report(directory)
                self.assertEqual(report['summary']['verified_successes'], int(field is None))


if __name__ == '__main__':
    unittest.main()
