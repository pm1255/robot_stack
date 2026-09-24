"""MimicGen interface for the project's robosuite 1.5 world-frame Panda Lift."""
import numpy as np
from mimicgen.env_interfaces.base import MG_EnvInterface
from robosuite.utils import transform_utils as T
from .environment import LiftEnvironment


def pose(position, rotation):
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = rotation, position
    return value


class LiftInterface(MG_EnvInterface):
    INTERFACE_TYPE = 'robot_stack_robosuite15'

    @property
    def native(self):
        return self.env.native

    @property
    def controller(self):
        return self.native.robots[0].part_controllers['right']

    def get_robot_eef_pose(self):
        site = self.native.robots[0].eef_site_id['right']
        return pose(self.native.sim.data.site_xpos[site], self.native.sim.data.site_xmat[site].reshape(3, 3))

    def target_pose_to_action(self, target_pose, relative=True):
        if not relative:
            raise ValueError('Only world-frame relative OSC actions are supported')
        current = self.get_robot_eef_pose()
        dp = target_pose[:3, 3] - current[:3, 3]
        dr = T.quat2axisangle(T.mat2quat(target_pose[:3, :3] @ current[:3, :3].T))
        return np.clip(np.r_[dp, dr] / self.controller.output_max, -1, 1)

    def action_to_target_pose(self, action, relative=True):
        if not relative:
            raise ValueError('Only world-frame relative OSC actions are supported')
        current = self.get_robot_eef_pose()
        delta = np.asarray(action[:6]) * self.controller.output_max
        return pose(current[:3, 3] + delta[:3], T.quat2mat(T.axisangle2quat(delta[3:])) @ current[:3, :3])

    def action_to_gripper_action(self, action):
        return np.asarray(action[-1:])

    def get_object_poses(self):
        body = self.native.sim.model.body_name2id(self.native.cube.root_body)
        return {'cube': pose(self.native.sim.data.body_xpos[body], self.native.sim.data.body_xmat[body].reshape(3, 3))}

    def get_subtask_term_signals(self):
        return {'grasp': int(self.native._check_grasp(gripper=self.native.robots[0].gripper, object_geoms=self.native.cube))}
