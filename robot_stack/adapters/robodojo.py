"""Bridge for a caller-created official RoboDojo EvalEnv (single environment).

RoboDojo's public release is eval-only. Supply a real policy, a numeric state
reader, and a task-appropriate perturbation generator. Construct the native env
inside its Isaac Sim launcher; this bridge does not pretend to bootstrap Isaac.
"""
import numpy as np


class RoboDojoAdapter:
    backend = 'robodojo'

    def __init__(self, task, *, env_factory, policy, state_reader, feature_reader,
                 perturbation, upstream_commit):
        self.task, self.env_factory, self.policy = task, env_factory, policy
        self.state_reader, self.feature_reader = state_reader, feature_reader
        self.perturbation, self.env = perturbation, None
        self.metadata = {'upstream_commit':upstream_commit, 'state_kind':'caller_state_reader',
                         'perturbation_type':'caller_physical_action_generator',
                         'runtime_validation':'requires_native_Isaac_run',
                         'success_criterion':'native end_flag and native success, with physical actions'}

    def reset(self, seed):
        self.close()
        self.env = self.env_factory()
        if self.env.num_envs != 1:
            raise ValueError('Branch replay currently requires one native environment')
        self.env.reset(seed=[seed])
        if hasattr(self.policy,'reset'):
            self.policy.reset()
        self.steps = 0

    def state(self):
        return np.asarray(self.state_reader(self.env), dtype=float)

    def error_features(self):
        return np.asarray(self.feature_reader(self.env), dtype=float)

    def step(self, action):
        if self.terminal():
            raise RuntimeError('Native episode ended; refusing silent no-op action')
        before = self.env.take_action_cnt[0]
        self.env.take_action(action)
        if self.env.take_action_cnt[0] != before + 1:
            raise RuntimeError('Native RoboDojo did not execute the requested action')
        self.steps += 1

    def expert_action(self):
        return self.policy(self.env.get_obs())

    def perturbation_actions(self, steps):
        return self.perturbation(self.env, steps)

    def success(self):
        # Native EvalEnv initializes success=True BEFORE any task completion.
        return bool(self.steps > 0 and self.env.end_flag[0] and self.env.success[0])

    def terminal(self):
        return bool(self.env.end_flag[0])

    def close(self):
        if self.env is not None:
            self.env.close(); self.env = None
