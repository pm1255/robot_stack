import numpy as np
from importlib.metadata import version
from benchmark_collector.metaworld import create, state, discover_tasks


class MetaWorldAdapter:
    backend = 'metaworld'

    def __init__(self, task, *, task_index=0, benchmark_seed=20260924, render=False, error_features='eef'):
        if task not in discover_tasks():
            raise ValueError(f'No bundled expert for {task}; register your own adapter/policy')
        self.task, self.task_index, self.benchmark_seed = task, task_index, benchmark_seed
        self.render_enabled, self.env = render, None
        if error_features not in {'eef','manipulation'}:
            raise ValueError('error_features must be eef or manipulation')
        self.feature_mode = error_features
        self.metadata = {'task_index': task_index, 'benchmark_seed': benchmark_seed,
                         'versions': {n: version(n) for n in ('metaworld', 'mujoco', 'numpy')},
                         'state_kind': 'mujoco.mjSTATE_INTEGRATION',
                         'error_features': 'end_effector_xyz_metres',
                         'perturbation_type': 'bounded_cartesian_action_burst_open_gripper'}
        if error_features == 'manipulation':
            self.metadata['error_features'] = 'eef_xyz_metres; gripper_opening_normalized_times_0.05; object_xyz_metres'

    def reset(self, seed):
        from metaworld import policies
        self.close()
        self.env, self.obs, _ = create(self.task, self.benchmark_seed, self.task_index, seed, self.render_enabled)
        self.policy = getattr(policies, discover_tasks()[self.task])()
        self.flag, self.done = False, False
        self.initial_object_height = float(self.obs[6])

    def state(self):
        return state(self.env)

    def error_features(self):
        if self.feature_mode == 'manipulation':
            return np.r_[self.obs[:3], .05*self.obs[3], self.obs[4:7]]
        return self.obs[:3]

    def events(self):
        labels = []
        if self.obs[3] < .25:
            labels.append('gripper_closed')
        if self.obs[6] - self.initial_object_height > .03:
            labels.append('object_raised_3cm')
        return labels

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != self.env.action_space.shape or not np.isfinite(a).all():
            raise ValueError('Invalid MetaWorld action')
        self.obs, _, terminated, truncated, info = self.env.step(np.clip(a, -1, 1))
        self.flag, self.done = bool(info['success']), bool(terminated or truncated)

    def expert_action(self):
        return np.clip(self.policy.get_action(self.obs), -1, 1)

    def perturbation_actions(self, steps):
        return [[1., -1., 1., -1.] for _ in range(steps)]

    def success(self):
        return self.flag

    def terminal(self):
        return self.done

    def render(self):
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
