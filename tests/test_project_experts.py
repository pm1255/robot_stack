import unittest
from types import SimpleNamespace
import numpy as np
from robot_stack.experts.maniskill import _move
from robot_stack.adapters.maniskill_planning import PlanningFailure
from robot_stack.adapters.robocasa_mobile import RoboCasaMobileManipulationAdapter


class ProjectExpertContracts(unittest.TestCase):
    def test_failed_waypoint_aborts_before_physics_or_later_stage(self):
        calls=[]
        planner=SimpleNamespace(
            move_to_pose_with_screw=lambda pose,dry_run:calls.append(('screw',dry_run)) or np.int64(-1),
            move_to_pose_with_RRTConnect=lambda pose,dry_run:calls.append(('rrt',dry_run)) or -1,
            follow_path=lambda *a,**kw:calls.append(('physics',)))
        with self.assertRaises(PlanningFailure):_move(planner,'unreachable')
        self.assertEqual(calls,[('screw',True),('rrt',True)])

    def test_fallback_executes_only_selected_plan(self):
        plan={'position':np.zeros((2,7))};executed=[]
        planner=SimpleNamespace(move_to_pose_with_screw=lambda *a,**kw:-1,
            move_to_pose_with_RRTConnect=lambda *a,**kw:plan,
            follow_path=lambda result,refine_steps:executed.append((result,refine_steps)))
        _move(planner,'reachable',refine=12)
        self.assertEqual(len(executed),1)
        self.assertIs(executed[0][0],plan)
        self.assertEqual(executed[0][1],12)

    def test_container_target_tracks_live_destination_without_moving_it(self):
        a=RoboCasaMobileManipulationAdapter.__new__(RoboCasaMobileManipulationAdapter)
        poses=np.array([[1.,2.,.8],[3.,4.,.9]])
        a.destination_name='bread';a.task='CheesyBread'
        a.env=SimpleNamespace(obj_body_id={'bread':1},sim=SimpleNamespace(data=SimpleNamespace(body_xpos=poses)))
        np.testing.assert_allclose(a.placement_target(),[3.,4.,1.])
        poses[1,0]=3.4
        np.testing.assert_allclose(a.placement_target(),[3.4,4.,1.])
        np.testing.assert_allclose(poses[1],[3.4,4.,.9])

    def test_grasp_query_uses_task_object_instead_of_assuming_obj(self):
        a=RoboCasaMobileManipulationAdapter.__new__(RoboCasaMobileManipulationAdapter)
        cheese=object();a.object_name='cheese'
        a.robot=SimpleNamespace(gripper={'right':'right'})
        a.env=SimpleNamespace(objects={'cheese':cheese},_check_grasp=lambda grip,obj:grip=='right' and obj is cheese)
        self.assertTrue(a.grasped())
