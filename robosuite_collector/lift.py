"""Closed-loop pick/lift primitives for Panda + robosuite Lift (world OSC).

Simulator state is used as an oracle observation. No object attachment,
teleportation, or successful-state restore is used to solve the task.
"""
from __future__ import annotations

import numpy as np


class StepBudgetExceeded(RuntimeError):
    pass


class LiftSkills:
    def __init__(self, env, observation, *, max_steps=600, stable_steps=10):
        self.env, self.obs = env, observation
        self.max_steps, self.stable_steps = max_steps, stable_steps
        keys = [k for k in observation if k.endswith('eef_pos')]
        if len(keys) != 1 or env.action_dim != 7:
            raise ValueError('This controller supports one Panda arm with 7D world OSC_POSE actions')
        if max_steps < 1 or stable_steps < 1:
            raise ValueError('Step budgets must be positive')
        self.eef_key = keys[0]
        self.initial_cube_z = float(observation['cube_pos'][2])
        self.states = [self.state()]
        self.actions, self.samples, self.events = [], [], []
        self.phase, self.attempt = 'init', 0

    def state(self):
        return np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64).copy()

    @property
    def eef(self):
        return np.asarray(self.obs[self.eef_key]).copy()

    @property
    def cube(self):
        return np.asarray(self.obs['cube_pos']).copy()

    def grasped(self):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper,
                                         object_geoms=self.env.cube))

    def success(self):
        return bool(self.env._check_success() and self.grasped()
                    and self.cube[2] - self.initial_cube_z >= .10)

    def step(self, target, grip):
        if len(self.actions) >= self.max_steps:
            raise StepBudgetExceeded('Episode action budget exhausted')
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Target must be a finite XYZ position')
        action = np.zeros(7)
        action[:3] = np.clip((target - self.eef) / .05, -.6, .6)
        action[6] = float(grip)
        low, high = self.env.action_spec
        if not np.isfinite(action).all() or np.any(action < low) or np.any(action > high):
            raise ValueError('Action is outside the environment limits')
        self.obs, reward, done, _ = self.env.step(action)
        state = self.state()
        if not np.isfinite(state).all():
            raise RuntimeError('Non-finite simulator state')
        self.actions.append(action.copy())
        self.states.append(state)
        self.samples.append({'step': len(self.actions), 'phase': self.phase,
                             'attempt': self.attempt, 'eef_xyz': self.eef.tolist(),
                             'cube_xyz': self.cube.tolist(), 'grasped': self.grasped(),
                             'environment_success': bool(self.env._check_success()),
                             'verified_success': self.success(), 'reward': float(reward)})
        if done:
            raise StepBudgetExceeded('Environment horizon reached')

    def move_to(self, target, *, grip, phase, max_steps=70, tolerance=.008, hold_steps=3):
        self.phase = phase
        start = len(self.actions)
        stable = 0
        for _ in range(max_steps):
            self.step(target, grip)
            stable = stable + 1 if np.linalg.norm(self.eef - target) <= tolerance else 0
            if stable >= hold_steps:
                self.events.append({'function': phase, 'start_step': start,
                                    'end_step': len(self.actions), 'attempt': self.attempt,
                                    'reached': True})
                return True
        self.events.append({'function': phase, 'start_step': start,
                            'end_step': len(self.actions), 'attempt': self.attempt,
                            'reached': False})
        return False

    def close_gripper(self, target):
        self.phase = 'close_gripper'
        start = len(self.actions)
        for _ in range(25):
            self.step(target, 1)
        result = self.grasped()
        self.events.append({'function': 'close_gripper', 'start_step': start,
                            'end_step': len(self.actions), 'attempt': self.attempt,
                            'grasped': result})
        return result

    def pick(self, *, offset=(0., 0., 0.)):
        """Return explicit failure reason; every move uses current observations."""
        goal = self.cube + np.asarray(offset)
        if not self.move_to(goal + [0, 0, .13], grip=-1, phase='approach'):
            return 'approach_not_reached'
        # Refresh the cube position after approach; retain only declared perturbation.
        goal = self.cube + np.asarray(offset) + [0, 0, .005]
        if not self.move_to(goal, grip=-1, phase='descend', max_steps=60):
            return 'grasp_pose_not_reached'
        if not self.close_gripper(goal):
            return 'grasp_not_established'
        return None

    def lift(self):
        target = self.eef + [0, 0, .18]
        self.move_to(target, grip=1, phase='lift', max_steps=70)
        self.phase = 'verify_hold'
        stable = 0
        for _ in range(self.stable_steps):
            self.step(target, 1)
            stable = stable + 1 if self.success() else 0
        return stable == self.stable_steps

    def recover(self):
        """Release, retreat and reobserve in the same physical episode."""
        self.phase = 'open_for_retry'
        current = self.eef
        for _ in range(15):
            self.step(current, -1)
        retreat = self.eef.copy()
        retreat[2] = max(retreat[2], self.cube[2] + .15)
        return self.move_to(retreat, grip=-1, phase='retreat_for_retry')

    def run(self, *, max_attempts=1, first_grasp_offset=(0., 0., 0.)):
        if max_attempts < 1:
            raise ValueError('max_attempts must be positive')
        failures = []
        try:
            for attempt in range(max_attempts):
                self.attempt = attempt
                offset = first_grasp_offset if attempt == 0 else (0., 0., 0.)
                reason = self.pick(offset=offset)
                if reason is None:
                    if self.lift():
                        return {'success': True, 'attempts': attempt + 1,
                                'first_attempt_success': attempt == 0, 'failures': failures}
                    reason = 'lift_or_stable_hold_failed'
                failures.append({'attempt': attempt, 'reason': reason, 'step': len(self.actions)})
                if attempt + 1 < max_attempts and not self.recover():
                    failures.append({'attempt': attempt, 'reason': 'retreat_failed',
                                     'step': len(self.actions)})
                    break
        except StepBudgetExceeded as exc:
            failures.append({'attempt': self.attempt, 'reason': str(exc), 'step': len(self.actions)})
        return {'success': False, 'attempts': self.attempt + 1,
                'first_attempt_success': False, 'failures': failures}
