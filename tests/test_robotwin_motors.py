"""Native ALOHA shares one articulation: both arm edits must survive."""
from types import SimpleNamespace
import unittest
import numpy as np
from robot_stack.adapters.robotwin import RoboTwinAdapter


class Joint:
    def __init__(self):self.position=0.;self.velocity=0.
    def get_drive_target(self):return [self.position]
    def get_drive_velocity_target(self):return [self.velocity]
    def get_limits(self):return [[-2.,2.]]
    def set_drive_target(self,value):self.position=value
    def set_drive_velocity_target(self,value):self.velocity=value


class Entity:
    def __init__(self):self.joints=[Joint() for _ in range(4)]
    def get_active_joints(self):return self.joints
    def get_qf(self):return np.zeros(4)
    def set_qf(self,value):pass


class MotorContracts(unittest.TestCase):
    def adapter(self):
        adapter=RoboTwinAdapter.__new__(RoboTwinAdapter)
        entity=Entity();adapter.entities=[entity]
        robot=SimpleNamespace(left_entity=entity,right_entity=entity,
            left_gripper_val=0.,right_gripper_val=0.,
            left_arm_joints=[entity.joints[0]],right_arm_joints=[entity.joints[1]],
            left_gripper=[(entity.joints[2],1.,0.)],right_gripper=[(entity.joints[3],1.,0.)],
            left_gripper_scale=[0.,1.],right_gripper_scale=[0.,1.])
        adapter.env=SimpleNamespace(robot=robot)
        adapter.native_scene=SimpleNamespace(step=lambda:None)
        adapter.steps=0;adapter.pending=None
        return adapter

    def test_shared_motor_vector_keeps_both_arm_and_gripper_edits(self):
        adapter=self.adapter()
        actions=adapter.make_perturbation(dict(type='joint_offset',steps=2,parameters=dict(offset=.5)),None)
        self.assertEqual(len(actions[0]['motors']),1)
        adapter.step(actions[0])
        self.assertEqual([j.position for j in adapter.entities[0].joints],[.5,.5,1.,1.])
        self.assertEqual(actions[0]['grippers'],[1.,1.])
        actions[0]['motors'][0]['positions'][0]=99
        self.assertEqual(actions[1]['motors'][0]['positions'][0],.5)

    def test_legacy_duplicate_vectors_replay_last_applied_command(self):
        adapter=self.adapter();command=adapter.motor_command()
        command['motors'].insert(0,dict(positions=[.5]*4,velocities=[0.]*4,forces=[0.]*4))
        adapter.step(command)
        self.assertEqual([j.position for j in adapter.entities[0].joints],[0.]*4)
