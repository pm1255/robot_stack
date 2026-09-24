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

    def __init__(self, task, *, cache_dir=None, executable_path=None, render=False, grid_size=.25,
                 waypoints=1):
        if not task.startswith('FloorPlan'):
            raise ValueError('task must be a native FloorPlan scene name')
        self.task, self.grid, self.controller = task, grid_size, None
        if isinstance(waypoints, bool) or not isinstance(waypoints, int) or waypoints < 1:
            raise ValueError('waypoints must be a positive integer')
        self.waypoint_count = waypoints
        self.executable_path = executable_path
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
        import ai2thor.build
        # ai2thor 5.0.0 ships an HTTP URL; use the same official public bucket over TLS.
        if ai2thor.build.base_url == 'http://s3-us-west-2.amazonaws.com/ai2-thor-public/':
            ai2thor.build.base_url = 'https://ai2-thor-public.s3-us-west-2.amazonaws.com/'
        cache = self.cache_dir
        class LocalCacheController(Controller):
            @property
            def base_dir(self):
                return str(cache)
        if self.controller is None:
            self.controller = LocalCacheController(platform=CloudRendering, scene=self.task,
                local_executable_path=self.executable_path,
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
        self.goals, self.stage = [self.goal], 0
        rng = np.random.default_rng(seed+1)
        for _ in range(1, self.waypoint_count):
            prev = self.goals[-1]
            next_nodes = sorted(n for n in self.nodes if abs(n[0]-prev[0])+abs(n[1]-prev[1]) >= 8)
            if not next_nodes:
                raise RuntimeError('No sufficiently distant waypoint')
            self.goals.append(next_nodes[int(rng.integers(len(next_nodes)))])
        self.metadata.update(goal_grid=list(self.goal), origin_xz=self.origin.tolist(), scene=self.task)
        if self.waypoint_count > 1:
            self.metadata.update(task_protocol='custom sequential multi-goal PointNav; not mobile manipulation',
                                 waypoint_grids=[list(g) for g in self.goals], waypoints=self.waypoint_count,
                                 state_kind='agent_pose_and_waypoint_progress_not_full_Unity_snapshot')
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
        values = [p['x'],p['y'],p['z'],r['x'],r['y'],r['z'],agent['cameraHorizon']]
        return np.array(values + ([self.stage] if self.waypoint_count > 1 else []))

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
        if action.get('action') not in {'MoveAhead','MoveBack','MoveRight','MoveLeft','RotateRight','RotateLeft','Done','Pass'}:
            raise ValueError('Only physical navigation actions are allowed')
        self.event = self.controller.step(**action)
        if not self.event.metadata['lastActionSuccess'] and action['action']=='MoveAhead' and self.last_planned_edge:
            self.blocked.add(self.last_planned_edge)
        self.last_planned_edge = None
        if self.waypoint_count > 1 and self.stage < len(self.goals) and self.at_goal():
            self.stage += 1
            if self.stage < len(self.goals):
                self.goal = self.goals[self.stage]

    def perturbation_actions(self, steps):
        return [{'action':'RotateRight','degrees':90} for _ in range(steps)]

    def at_goal(self):
        target = self.origin + np.array(self.goal)*self.grid
        return bool(np.linalg.norm(self.xy()-target) <= .10)

    def success(self):
        return self.stage == len(self.goals) if self.waypoint_count > 1 else self.at_goal()

    def events(self):
        return [f'waypoint:{i+1}:complete' for i in range(self.stage)]

    def terminal(self):
        return False

    def render(self):
        return self.event.frame

    def close(self):
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
