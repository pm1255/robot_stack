import unittest
from types import SimpleNamespace
import numpy as np
from robot_stack.adapters.robocasa_mobile import RoboCasaMobileManipulationAdapter


class MobileCommandContracts(unittest.TestCase):
    def adapter(self):
        class Composite:
            _action_split_indexes={'right':(0,6),'right_gripper':(6,7),'base':(7,10),'base_mode':(10,11)}
            def create_action_vector(self,parts):
                command=np.zeros(11)
                for name,value in parts.items():
                    start,end=self._action_split_indexes[name];command[start:end]=value
                return command
        a=RoboCasaMobileManipulationAdapter.__new__(RoboCasaMobileManipulationAdapter)
        a.robot=SimpleNamespace(composite_controller=Composite())
        a.env=SimpleNamespace(action_dim=11,step=lambda action:None)
        a.gripper=-1.;a.stage_steps=0;a.total_steps=0
        return a

    def test_replayed_gripper_command_survives_navigation_intervention(self):
        a=self.adapter();recorded=np.zeros(11);recorded[6]=1
        a.step(recorded)  # Physical prefix does not call expert_action.
        commands=a.make_perturbation(dict(type='base_yaw',steps=3,strength=.8),None)
        self.assertEqual(len(commands),3)
        self.assertEqual(commands[0][6],1)
        self.assertEqual(commands[0][9],.8)
        self.assertEqual(a.total_steps,1)

    def test_bad_direction_rejected_before_environment_step(self):
        a=self.adapter()
        for direction in ([1,2],[0,0,float('nan')],[1.1,0,0]):
            with self.assertRaises(ValueError):
                a.make_perturbation(dict(type='arm_offset',steps=2,parameters={'direction':direction}),None)
        self.assertEqual(a.total_steps,0)

    def test_recovery_restarts_pick_for_lost_object_without_reset(self):
        a=self.adapter();a.grasped=lambda:False;a.success=lambda:False
        a.stage='navigate_sink';a.stage_steps=50
        a.after_intervention()
        self.assertEqual(a.stage,'navigate_pick')
        self.assertEqual(a.stage_steps,0)
        self.assertEqual(a.total_steps,0)
