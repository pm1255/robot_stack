import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from isaac_collector.runtime.execution_checks import (
    sample_trajectory, read_robot_state, terminal_robot_state, classify_outcome,
)
from isaac_collector.run_atomic_manipulation import _select_feasible_grasp_with_curobo
from scripts.build_collection_report import build_report


class ExecutionChecks(unittest.TestCase):
    def test_stride_preserves_goal(self):
        q = np.arange(24).reshape(8, 3)
        np.testing.assert_array_equal(sample_trajectory(q, 3), q[[0, 3, 6, 7]])
        np.testing.assert_array_equal(sample_trajectory(q, 100), q[[0, 7]])
        self.assertEqual(len(sample_trajectory(q[:1], 4)), 1)

    def test_invalid_trajectories(self):
        for q in ([], [[float('nan')]], [[float('inf')]], [[]]):
            with self.assertRaises(ValueError):
                sample_trajectory(q, 1)

    def test_reads_measured_state_in_joint_order(self):
        dc = SimpleNamespace(get_dof_position=lambda handle: {7: .4, 9: -.2}[handle])
        state = read_robot_state({'dc': dc, 'curobo_joint_names': ['b', 'a'],
                                  'joint_mapping': {'a': 7, 'b': 9}})
        self.assertEqual(state, {'joint_names': ['b', 'a'], 'positions': [-.2, .4]})

    def test_terminal_reorders_joints(self):
        state = terminal_robot_state({'success': True, 'joint_names': ['b', 'a'],
                                      'positions': [[0, 0], [1, 2]]}, ['a', 'b'])
        self.assertEqual(state['positions'], [2, 1])

    def test_planner_cannot_certify_task(self):
        result = classify_outcome(planning_success=True, execution_success=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'task_unverified')
        for opts in ({'plan_only': True}, {'debug': True}, {'execution_success': False},
                     {'planning_success': False}):
            values = dict(planning_success=True, execution_success=True, task_success=True)
            values.update(opts)
            self.assertFalse(classify_outcome(**values)['success'])
        self.assertTrue(classify_outcome(planning_success=True, execution_success=True,
                                         task_success=True)['success'])

    def test_candidate_rejects_unreachable_place_and_chains_state(self):
        calls = []
        def plan(service, *, task, robot_state, target_robot):
            calls.append((task, robot_state['positions']))
            return {'success': len(calls) != 2, 'positions': [[2.0]], 'joint_names': ['a']}
        args = SimpleNamespace(graspnet_target_key='grasp_pose_world', pickup_target_z=None,
                               place_target_z=None)
        with patch('isaac_collector.run_atomic_manipulation._plan_curobo', side_effect=plan), \
             patch.dict(os.environ, {'MAX_GRASP_CANDIDATES': '2', 'DEBUG_FORCE_REACHABLE_PICK': '0'}):
            out = _select_feasible_grasp_with_curobo(
                grasp_result={'candidates': [{'score': 1}, {'score': .5}]}, curobo_service=None,
                robot_state={'joint_names': ['a'], 'positions': [0.]}, robot_world=np.eye(4),
                place_offset_robot=[0, .1, 0], args=args,
                make_targets_from_graspnet_result=lambda **kw: (np.eye(4), np.eye(4)),
                summarize_plan=lambda p: {'success': p['success']},
                place_action=SimpleNamespace(target_robot=None))
        self.assertEqual(out[0], 1)
        self.assertEqual(calls, [('pickup', [0.]), ('putdown', [2.]),
                                 ('pickup', [0.]), ('putdown', [2.])])
        self.assertEqual([r['success'] for r in out[-1]], [False, True])


class Reports(unittest.TestCase):
    def test_report_does_not_promote_legacy_or_debug_success(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {'success': True, 'secret': '/server/private/assets'},
                classify_outcome(planning_success=True, execution_success=True, debug=True),
                classify_outcome(planning_success=True, execution_success=True, task_success=True),
                {'validation_version': 1, 'success': True, 'status': 'success', 'task_success': True},
            ]
            for i, record in enumerate(records):
                (root / f'ep_{i}_episode_result.json').write_text(json.dumps(record))
            (root / 'broken_episode_result.json').write_text('{')
            report = build_report(root)
            self.assertEqual(report['summary']['attempts'], 5)
            self.assertEqual(report['summary']['verified_successes'], 1)
            self.assertEqual(report['summary']['confirmed_fraction'], .2)
            self.assertEqual(report['summary']['status_counts']['legacy_unverified'], 2)
            self.assertNotIn('/server/private', json.dumps(report))

    def test_empty_report_has_no_rate(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(build_report(Path(tmp))['summary']['confirmed_fraction'])


if __name__ == '__main__':
    unittest.main()
