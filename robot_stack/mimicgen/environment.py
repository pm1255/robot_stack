"""Native Lift execution and strict final success for MimicGen."""
import numpy as np
from robosuite_collector.collect import make_env


class LiftEnvironment:
    """Minimal public MimicGen environment contract, backed by native dynamics.

    No reset_to or simulator-state write operation is exposed. Every generated
    episode starts with a fresh seeded reset and executes physical OSC actions.
    """
    def __init__(self, seed, budget=600, render=False):
        self.seed, self.budget, self.native = seed, budget, None
        self.render = render

    def reset(self):
        if self.native is not None:
            self.native.close()
        self.native, self.config = make_env(self.seed, self.budget, render=self.render)
        self.obs = self.native.reset()
        self.initial_z = float(self.obs['cube_pos'][2])
        self.initial_xml = self.native.sim.model.get_xml()
        self.states, self.actions, self.flags, self.stable_flags = [self.state()], [], [], []
        self.stable_count = 0
        return self.obs

    def state(self):
        return np.asarray(self.native.sim.get_state().flatten()).copy()

    def get_state(self):
        return {'states': self.state(), 'model': self.initial_xml}

    def get_observation(self):
        return {k: np.asarray(v).copy() for k, v in self.obs.items()}

    def step(self, action):
        if len(self.actions) >= self.budget:
            raise RuntimeError('MimicGen generation exceeded the action budget')
        action = np.asarray(action, dtype=float).copy()
        lo, hi = self.native.action_spec
        if action.shape != lo.shape or not np.isfinite(action).all() or (action < lo).any() or (action > hi).any():
            raise ValueError('Invalid generated action')
        self.obs, reward, done, info = self.native.step(action.copy())
        self.actions.append(action)
        self.states.append(self.state())
        good = bool(self.native._check_success() and self.native._check_grasp(
            gripper=self.native.robots[0].gripper, object_geoms=self.native.cube)
            and self.obs['cube_pos'][2] - self.initial_z >= .10)
        self.flags.append(good)
        self.stable_count = self.stable_count + 1 if good else 0
        self.stable_flags.append(self.stable_count >= 10)
        return self.obs, reward, done, info

    def is_success(self):
        return {'task': bool(self.stable_flags and self.stable_flags[-1])}

    def serialize(self):
        return {'env_name': 'Lift', 'type': 1, 'env_kwargs': self.config}

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
