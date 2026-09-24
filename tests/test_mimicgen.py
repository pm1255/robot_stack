from types import SimpleNamespace
import unittest
import numpy as np
from robot_stack.mimicgen.environment import LiftEnvironment
from robot_stack.mimicgen.__main__ import check_state, write_info


class NativeStub:
    action_spec = (-np.ones(7), np.ones(7))
    robots = [SimpleNamespace(gripper=None)]
    cube = None

    def __init__(self):
        self.tick, self.good, self.grasp = 0, True, True
        self.sim = SimpleNamespace(get_state=lambda: np.array([self.tick]))

    def step(self, action):
        self.tick += 1
        action[:] = 0  # A simulator may mutate its input buffer.
        return {'cube_pos': np.array([0, 0, 1.])}, 0, False, {}

    def _check_success(self):
        return self.good

    def _check_grasp(self, **kwargs):
        return self.grasp


def make_stub(budget=20):
    env = LiftEnvironment(1, budget)
    env.native = NativeStub()
    env.initial_z = .8
    env.states, env.actions, env.flags, env.stable_flags = [env.state()], [], [], []
    env.stable_count = 0
    return env


class MimicGenContracts(unittest.TestCase):
    def test_final_success_is_not_ever_success(self):
        env = make_stub()
        for _ in range(9):
            env.step(np.ones(7))
        self.assertFalse(env.is_success()['task'])
        env.step(np.ones(7))
        self.assertTrue(env.is_success()['task'])
        env.native.grasp = False
        env.step(np.ones(7))
        self.assertFalse(env.is_success()['task'])
        self.assertTrue(any(env.stable_flags))
        np.testing.assert_array_equal(env.actions[0], np.ones(7))
        self.assertEqual(len(env.states), len(env.actions) + 1)

    def test_budget_and_malformed_actions_cannot_advance_simulator(self):
        env = make_stub(1)
        for action in (np.zeros(6), np.full(7, np.nan), np.full(7, 1.1)):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.native.tick, 0)
        env.step(np.zeros(7))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))
        self.assertEqual(env.native.tick, 1)

    def test_replay_rejects_nonfinite_divergent_and_wrong_shape_states(self):
        for other in (np.array([np.nan]), np.array([1.001]), np.array([1., 1.])):
            with self.assertRaises(ValueError):
                check_state(np.array([1.]), other)
        with self.assertRaises(ValueError):
            write_info(None, [])
