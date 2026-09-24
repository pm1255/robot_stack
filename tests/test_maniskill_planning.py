import unittest
from types import SimpleNamespace
import numpy as np

from robot_stack.adapters.maniskill import ManiSkillAdapter
from robot_stack.adapters.maniskill_planning import (
    PlanningFailure, new_diagnostics, with_planner_fallback,
)


class PlannerPortfolioContracts(unittest.TestCase):
    def solver(self, screw, rrt):
        calls = []

        class Native:
            def move_to_pose_with_screw(self, pose, dry_run=False):
                calls.append(('screw', dry_run))
                return screw

            def move_to_pose_with_RRTConnect(self, pose, dry_run=False):
                calls.append(('rrt', dry_run))
                return rrt

            def follow_path(self, result, refine_steps=0):
                calls.append(('execute', result, refine_steps))
                return 'actions'

        namespace = {'PandaArmMotionPlanningSolver': Native}
        exec('def solve(): return PandaArmMotionPlanningSolver()', namespace)
        original = namespace['solve']
        diagnostics = new_diagnostics()
        wrapped = with_planner_fallback(original, diagnostics)
        self.assertIs(type(original()), Native)
        return wrapped(), calls, diagnostics

    def test_rrt_is_only_used_after_screw_failure_and_execution_is_once(self):
        plan = {'position': [1, 2]}
        solver, calls, stats = self.solver(-1, plan)
        self.assertEqual(solver.move_to_pose_with_screw('target', refine_steps=3), 'actions')
        self.assertEqual(calls, [('screw', True), ('rrt', True), ('execute', plan, 3)])
        self.assertEqual(stats['rrt_successes'], 1)
        solver, calls, _ = self.solver(plan, -1)
        solver.move_to_pose_with_screw('target')
        self.assertEqual([c[0] for c in calls], ['screw', 'execute'])

    def test_dry_run_does_not_execute_and_total_failure_aborts_script(self):
        solver, calls, _ = self.solver(-1, {'position': [1]})
        solver.move_to_pose_with_screw('target', dry_run=True)
        self.assertEqual(calls, [('screw', True), ('rrt', True)])
        solver, calls, _ = self.solver(-1, -1)
        self.assertEqual(solver.move_to_pose_with_screw('target', dry_run=True), -1)
        with self.assertRaises(PlanningFailure):
            solver.move_to_pose_with_screw('target')
        self.assertNotIn('execute', [c[0] for c in calls])

    def test_policy_hold_preserves_grasp_but_joint_intervention_still_opens(self):
        adapter = ManiSkillAdapter.__new__(ManiSkillAdapter)
        adapter.last_gripper = -.8
        native = SimpleNamespace(agent=SimpleNamespace(robot=SimpleNamespace(
            get_qpos=lambda: np.arange(9, dtype=float))))
        adapter.env = SimpleNamespace(unwrapped=native,
                                      action_space=SimpleNamespace(shape=(8,)))
        self.assertEqual(adapter.hold_action(preserve_gripper=True)[-1], -.8)
        self.assertEqual(adapter.hold_action()[-1], 1.)
