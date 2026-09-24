"""Native ManiSkill planners, streamed actions and current-state replanning."""
from importlib.metadata import version
import hashlib
import inspect
import numpy as np
from robot_stack.action_stream import ActionStream


def discover_tasks():
    from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
    return dict(MP_SOLUTIONS)



def compatible_policy(solve):
    """Fix only the 3.0.1 stick solver's obsolete constructor argument.

    Clone the expert's globals so neither the installed library nor other
    planners are patched. Planning and physics remain upstream implementations.
    """
    if version('mani_skill') != '3.0.1' or 'PandaStickMotionPlanningSolver' not in solve.__globals__:
        return solve, None
    from types import FunctionType
    from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
    class StickSolver(BaseMotionPlanningSolver):
        MOVE_GROUP = 'panda_hand_tcp'
        def __init__(self, env, debug=False, vis=True, base_pose=None,
                     visualize_target_grasp_pose=True, print_env_info=True,
                     joint_vel_limits=.9, joint_acc_limits=.9):
            super().__init__(env, debug=debug, vis=vis, base_pose=base_pose,
                print_env_info=print_env_info, joint_vel_limits=joint_vel_limits,
                joint_acc_limits=joint_acc_limits)
    namespace = dict(solve.__globals__, PandaStickMotionPlanningSolver=StickSolver)
    clone = FunctionType(solve.__code__, namespace, solve.__name__, solve.__defaults__, solve.__closure__)
    clone.__kwdefaults__ = solve.__kwdefaults__
    return clone, 'ManiSkill 3.0.1 stick constructor: omit obsolete visualization argument'


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def flatten(value):
    if isinstance(value, dict):
        parts = [flatten(value[k]) for k in sorted(value)]
        return np.concatenate(parts) if parts else np.empty(0)
    return array(value).astype(np.float64).reshape(-1)


class ManiSkillAdapter:
    backend = 'maniskill'

    def __init__(self, task, *, render=False, native_horizon=1000, max_replans=3, env_options=None):
        policies = discover_tasks()
        if task not in policies:
            raise ValueError(f'No installed native planner for {task}')
        self.task = task
        self.solve, compatibility = compatible_policy(policies[task])
        self.render_enabled, self.native_horizon = render, native_horizon
        self.max_replans, self.env_options = max_replans, dict(env_options or {})
        if max_replans < 1 or native_horizon < 1:
            raise ValueError('Positive planning and native action budgets required')
        self.env, self.stream = None, None
        self.metadata = {
            'versions': {n:version(n) for n in ('mani_skill','sapien','mplib','torch','numpy','greenlet')},
            'state_kind':'native_get_state_dict_sorted_numeric_leaves',
            'error_features':'tcp_xyz_metres_and_robot_joint_positions_radians',
            'perturbation_type':'bounded_joint_target_offset',
            'policy':'upstream synchronous planner streamed; current-state restart after intervention',
            'planner_source_sha256':hashlib.sha256(inspect.getsource(self.solve).encode()).hexdigest(),
            'native_horizon':native_horizon, 'max_replans':max_replans,
            'compatibility_adjustment':compatibility,
            'setup_reset_behavior':'planner reset returns current observation; never resets physics',
        }

    def reset(self, seed):
        import gymnasium as gym
        import mani_skill.envs
        import torch
        torch.set_num_threads(1)
        self.close()
        self.env = gym.make(self.task, obs_mode='state', control_mode='pd_joint_pos',
            sim_backend='physx_cpu', render_backend='sapien_cuda' if self.render_enabled else 'none',
            render_mode='rgb_array', max_episode_steps=self.native_horizon, **self.env_options)
        self.seed, self.replans, self.steps = seed, 0, 0
        self.obs, self.info = self.env.reset(seed=seed)
        self.last_result, self.pending = None, None
        self.flag, self.done = False, False
        self.policy_exhausted = False
        self.metadata['initial_state_dimension'] = len(self.state())

    def state(self):
        return flatten(self.env.unwrapped.get_state_dict())

    def error_features(self):
        native = self.env.unwrapped
        return np.r_[array(native.agent.tcp.pose.p).reshape(-1),
                     array(native.agent.robot.get_qpos()).reshape(-1)]

    def events(self):
        return []

    def hold_action(self):
        qpos = array(self.env.unwrapped.agent.robot.get_qpos()).reshape(-1)
        n = self.env.action_space.shape[-1]
        if n == len(qpos):  # Panda stick (no gripper)
            return qpos.copy()
        return np.r_[qpos[:n-1], 1.]

    def expert_action(self):
        if self.pending is not None:
            return self.pending.copy()
        if self.stream is None:
            if self.replans >= self.max_replans:
                self.policy_exhausted = True
                return self.hold_action()
            self.stream = ActionStream(self.env, self.solve, self.seed,
                                       lambda: (self.obs,self.info))
            self.replans += 1
        action = self.stream.next_action(self.last_result)
        if action is None:
            self.stream.close(); self.stream = None
            # Replanning is bounded; no reset or recursion over empty planners.
            return self.hold_action()
        self.pending = np.asarray(action,dtype=float)
        return self.pending.copy()

    def step(self, action):
        action = np.asarray(action,dtype=float)
        if action.shape != self.env.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid ManiSkill action')
        low, high = self.env.action_space.low, self.env.action_space.high
        if np.any(action < low-1e-6) or np.any(action > high+1e-6):
            raise ValueError('ManiSkill action outside native bounds')
        self.last_result = self.env.step(action)
        self.obs, _, terminated, truncated, self.info = self.last_result
        self.flag = bool(array(self.env.unwrapped.evaluate()['success']).item())
        self.done = bool(array(terminated).item() or array(truncated).item())
        self.steps += 1
        self.pending = None

    def after_intervention(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        self.pending = None
        self.replans = 0
        self.policy_exhausted = False
        self.last_result = None

    def make_perturbation(self, event, rng):
        kind, steps = event['type'], event['steps']
        strength, params = event.get('strength',1.), event.get('parameters',{})
        allowed = {'joint','offset'} if kind == 'joint_offset' else set()
        if set(params)-allowed:
            raise ValueError(f'Unsupported parameters for {kind}')
        action = self.hold_action()
        low, high = self.env.action_space.low, self.env.action_space.high
        if kind == 'joint_offset':
            index, offset = params.get('joint',0), params.get('offset',.6)
            if type(index) is not int or not 0 <= index < min(7,len(action)) or not np.isfinite(offset):
                raise ValueError('Invalid joint or offset')
            action[index] += strength*offset
        elif kind in {'gripper_open','gripper_close'}:
            if len(action) != 8:
                raise ValueError('Selected embodiment has no Panda gripper')
            action[-1] = strength*(1 if kind=='gripper_open' else -1)
        elif kind != 'joint_hold':
            raise ValueError(f'Unsupported ManiSkill perturbation: {kind}')
        action = np.clip(action,low,high)
        return [action.tolist() for _ in range(steps)]

    def perturbation_actions(self, steps):
        return self.make_perturbation({'type':'joint_offset','steps':steps},np.random.default_rng(0))

    def success(self):
        return self.flag

    def terminal(self):
        return self.done or self.policy_exhausted

    def render(self):
        frame = array(self.env.render())
        return frame[0] if frame.ndim == 4 else frame

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        if self.env is not None:
            self.env.close(); self.env = None
