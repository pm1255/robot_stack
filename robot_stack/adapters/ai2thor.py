"""Oracle PointNav in native iTHOR scenes; not an official ObjectNav benchmark.

GetReachablePositions supplies a privileged map. Only MoveAhead and rotations
execute trajectories. No teleportation or scene reset occurs during recovery.
"""
from collections import deque
from importlib.metadata import version
import math
import os
from pathlib import Path
import numpy as np


class AI2ThorNavigationAdapter:
    backend = 'ai2thor'

    def __init__(self, task, *, cache_dir=None, render=False, grid_size=.25):
        if not task.startswith('FloorPlan'):
            raise ValueError('task must be a native FloorPlan scene name')
        self.task, self.grid, self.controller = task, grid_size, None
        self.cache_dir = Path(cache_dir or os.environ.get('AI2THOR_CACHE_DIR', '.runtime/ai2thor')).resolve()
        self.metadata = {'versions': {'ai2thor': version('ai2thor')},
                         'task_protocol': 'oracle PointNav on reachable grid, stop within 0.10m',
                         'map_source': 'native GetReachablePositions', 'grid_size': self.grid,
                         'state_kind': 'agent_pose_only_static_scene_not_full_Unity_snapshot',
                         'error_features': 'xz_metres_and_yaw_cos_sin',
                         'perturbation_type': 'wrong_heading_RotateRight_90_degrees',
                         'rendering': True}

    def reset(self, seed):
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
        cache = self.cache_dir
        class LocalCacheController(Controller):
            @property
            def base_dir(self):
                return str(cache)
        if self.controller is None:
            self.controller = LocalCacheController(platform=CloudRendering, scene=self.task,
                width=320, height=240, gridSize=self.grid, rotateStepDegrees=90,
                snapToGrid=True, visibilityDistance=1.5, renderDepthImage=False,
                renderInstanceSegmentation=False)
        event = self.controller.reset(scene=self.task)
        if not event.metadata['lastActionSuccess']:
            raise RuntimeError(event.metadata['errorMessage'])
        self.event = self.controller.step(action='GetReachablePositions')
        if not self.event.metadata['lastActionSuccess']:
            raise RuntimeError('Native reachable-map query failed')
        self.origin = self.xy()
        self.nodes = {self.key(p['x'], p['z']) for p in self.event.metadata['actionReturn']}
        start = self.key(*self.xy())
        candidates = sorted(n for n in self.nodes if abs(n[0]-start[0])+abs(n[1]-start[1]) >= 8)
        if not candidates:
            raise RuntimeError('No sufficiently distant navigation target')
        self.goal = candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
        self.metadata.update(goal_grid=list(self.goal), origin_xz=self.origin.tolist(), scene=self.task)
        self.blocked, self.last_planned_edge = set(), None
        self.initial_pose = self.state().tolist()

    def xy(self):
        pos = self.event.metadata['agent']['position']
        return np.array([pos['x'], pos['z']], dtype=float)

    def key(self, x, z):
        return (round((x-self.origin[0])/self.grid), round((z-self.origin[1])/self.grid))

    def state(self):
        agent = self.event.metadata['agent']
        p, r = agent['position'], agent['rotation']
        return np.array([p['x'],p['y'],p['z'],r['x'],r['y'],r['z'],agent['cameraHorizon']])

    def error_features(self):
        yaw = math.radians(self.event.metadata['agent']['rotation']['y'])
        return np.r_[self.xy(), math.cos(yaw), math.sin(yaw)]

    def path(self):
        start = self.key(*self.xy())
        parents, queue = {start:None}, deque([start])
        while queue:
            here = queue.popleft()
            if here == self.goal:
                route = []
                while here != start:
                    route.append(here); here = parents[here]
                return list(reversed(route))
            for dx,dz in [(1,0),(-1,0),(0,1),(0,-1)]:
                nxt = here[0]+dx, here[1]+dz
                if nxt in self.nodes and nxt not in parents and (here,nxt) not in self.blocked:
                    parents[nxt] = here; queue.append(nxt)
        raise RuntimeError('No collision-free grid route to selected goal')

    def expert_action(self):
        route = self.path()
        if not route:
            return {'action':'Done'}
        here, nxt = self.key(*self.xy()), route[0]
        desired = math.degrees(math.atan2(nxt[0]-here[0], nxt[1]-here[1])) % 360
        yaw = self.event.metadata['agent']['rotation']['y']
        delta = (desired-yaw+180) % 360-180
        if abs(delta)>1:
            self.last_planned_edge = None
            return {'action':'RotateRight' if delta>0 else 'RotateLeft','degrees':min(abs(delta),90)}
        self.last_planned_edge = (here,nxt)
        return {'action':'MoveAhead','moveMagnitude':self.grid}

    def step(self, action):
        if action.get('action') not in {'MoveAhead','RotateRight','RotateLeft','Done'}:
            raise ValueError('Only physical navigation actions are allowed')
        self.event = self.controller.step(**action)
        if not self.event.metadata['lastActionSuccess'] and action['action']=='MoveAhead' and self.last_planned_edge:
            self.blocked.add(self.last_planned_edge)
        self.last_planned_edge = None

    def perturbation_actions(self, steps):
        return [{'action':'RotateRight','degrees':90} for _ in range(steps)]

    def success(self):
        target = self.origin + np.array(self.goal)*self.grid
        return bool(np.linalg.norm(self.xy()-target) <= .10)

    def terminal(self):
        return False

    def render(self):
        return self.event.frame

    def close(self):
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
