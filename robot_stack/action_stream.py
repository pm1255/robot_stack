"""Turn a synchronous native planner into a same-thread action stream.

The planner cannot execute physics through the proxy. A setup reset returns the
current observation; a second reset or a reset after requesting actions fails.
Only the owning adapter may apply actions. Cancelling unwinds the planner stack.
"""
import numpy as np
from copy import deepcopy


class ActionStream:
    def __init__(self, env, solve, seed, current_observation):
        from greenlet import greenlet, getcurrent
        self.env, self.solve, self.seed = env, solve, seed
        self.current_observation = current_observation
        self.parent = getcurrent()
        self.requests, self.reset_calls = 0, 0
        self.started, self.finished = False, False
        self.worker = greenlet(self._run, parent=self.parent)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        if self.reset_calls or self.requests:
            raise RuntimeError('Native planner attempted a non-setup reset')
        self.reset_calls += 1
        return self.current_observation()

    def step(self, action):
        self.requests += 1
        saved = deepcopy(action) if isinstance(action, dict) else np.asarray(action, dtype=float).copy()
        return self.parent.switch(('action', saved))

    def _run(self):
        result = self.solve(self, seed=self.seed, debug=False, vis=False)
        return ('done', result)

    def next_action(self, previous_result=None):
        if self.finished:
            return None
        value = self.worker.switch(previous_result) if self.started else self.worker.switch()
        self.started = True
        if value[0] == 'done':
            self.finished = True
            return None
        if value[0] != 'action':
            raise RuntimeError('Invalid native planner yield')
        return value[1]

    def close(self):
        from greenlet import GreenletExit
        if self.started and not self.worker.dead:
            self.worker.throw(GreenletExit)
        self.finished = True
