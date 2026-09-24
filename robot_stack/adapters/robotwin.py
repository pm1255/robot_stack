"""Stream native RoboTwin motor commands at physical scene-step boundaries.

Experimental: original experts can be restarted from current object poses, but
their assumptions may not hold after arbitrary errors. Every action is exactly
one native physics step. No object pose, grasp attachment or physics is edited.
"""
from copy import deepcopy
import importlib
from importlib.metadata import version
import hashlib
from pathlib import Path
import os
import random
import sys
import tempfile
import numpy as np
from robot_stack.action_stream import ActionStream


class SceneProxy:
    def __init__(self, adapter, native):
        self.adapter, self.native = adapter, native
    def __getattr__(self,name):
        return getattr(self.native,name)
    def step(self):
        return self.adapter.stream.step(self.adapter.motor_command())


class RoboTwinAdapter:
    backend = 'robotwin'

    def __init__(self, task, *, root, render=False):
        from robot_stack.inventory import file_tasks
        self.root=Path(root).resolve()
        if task not in file_tasks(self.root/'envs'):
            raise ValueError(f'No native task class {task}')
        self.task, self.render_enabled = task, render
        self.env, self.stream, self.scratch = None, None, None
        self.metadata = dict(state_kind='actor_and_articulation_poses_velocities_and_joints',
            error_features='robot_joint_positions_radians',
            perturbation_type='native_motor_joint_offset',
            action_unit='one_native_physics_step',
            policy='upstream play_once streamed; restart from current state after intervention',
            experimental=True,
            versions={name:version(name) for name in ('sapien','torch','numpy','mplib','greenlet')},
            source_hashes={name:hashlib.sha256((self.root/name).read_bytes()).hexdigest()
                          for name in (f'envs/{task}.py','envs/_base_task.py','envs/robot/robot.py')},
            motor_layout='one_vector_per_unique_articulation')

    def reset(self,seed):
        from benchmark_collector.robotwin import config
        import torch
        torch.set_num_threads(2)
        self.close()
        self.previous_cwd=Path.cwd()
        os.chdir(self.root)
        for path in [self.root,self.root/'description/utils']:
            if str(path) not in sys.path:sys.path.insert(0,str(path))
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.scratch=tempfile.TemporaryDirectory(prefix='robot-stack-native-')
        cls=getattr(importlib.import_module('envs.'+self.task),self.task)
        self.env=cls()
        self.native_scene=None
        cfg=config(self.root,Path(self.scratch.name),self.task)
        cfg.update(save_data=False,collect_data=False,save_freq=None,render_freq=0)
        self.env.setup_demo(now_ep_num=0,seed=seed,**cfg)
        self.native_scene=self.env.scene
        left, right = self.env.robot.left_entity, self.env.robot.right_entity
        self.entities = [left] if left is right else [left, right]
        self.metadata['shared_articulation'] = left is right
        import sapien
        self.actors=[(actor,actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent))
                     for actor in self.native_scene.get_all_actors()]
        self.articulations=list(self.native_scene.get_all_articulations())
        self.env.scene=SceneProxy(self,self.native_scene)
        self.steps,self.seed,self.exhausted=0,seed,False
        self.pending=None
        self.metadata['physics_timestep']=float(self.native_scene.get_timestep())

    def state(self):
        values=[]
        for actor,body in self.actors:
            pose=actor.get_pose()
            values.extend([pose.p,pose.q,
                           body.get_linear_velocity() if body else np.zeros(3),
                           body.get_angular_velocity() if body else np.zeros(3)])
        for articulation in self.articulations:
            pose=articulation.get_root_pose()
            values.extend([pose.p,pose.q,articulation.get_root_linear_velocity(),
                           articulation.get_root_angular_velocity(),articulation.get_qpos(),articulation.get_qvel()])
        return np.concatenate([np.asarray(x).reshape(-1) for x in values])

    def error_features(self):
        return np.concatenate([e.get_qpos() for e in self.entities])

    def motor_command(self):
        command={'motors':[], 'grippers':[float(self.env.robot.left_gripper_val),float(self.env.robot.right_gripper_val)]}
        for entity in self.entities:
            joints=entity.get_active_joints()
            command['motors'].append(dict(
                positions=[float(j.get_drive_target()[0]) for j in joints],
                velocities=[float(j.get_drive_velocity_target()[0]) for j in joints],
                forces=entity.get_qf().tolist()))
        return command

    def expert_action(self):
        if self.pending is not None:return deepcopy(self.pending)
        if self.stream is None:
            self.stream=ActionStream(self.env,lambda proxy,**kw:self.env.play_once(),self.seed,lambda:(None,{}))
        command=self.stream.next_action(None)
        if command is None:
            self.exhausted=True
            return self.motor_command()
        self.pending=command
        return deepcopy(command)

    def step(self,action):
        parts = action['motors']
        # Earlier experimental files duplicated the shared articulation; the
        # second complete vector was the one actually applied by that version.
        if len(self.entities) == 1 and len(parts) == 2:
            parts = parts[-1:]
        if len(parts) != len(self.entities) or len(action['grippers']) != 2:
            raise ValueError('Invalid native motor command')
        if not np.isfinite(action['grippers']).all() or not all(0 <= v <= 1 for v in action['grippers']):
            raise ValueError('Invalid gripper commands')
        for entity,part in zip(self.entities,parts):
            joints=entity.get_active_joints()
            for key in ('positions','velocities','forces'):
                if len(part[key])!=len(joints) or not np.isfinite(part[key]).all():
                    raise ValueError('Invalid native motor vector')
            for joint,p,v in zip(joints,part['positions'],part['velocities']):
                joint.set_drive_target(p);joint.set_drive_velocity_target(v)
            entity.set_qf(np.asarray(part['forces']))
        self.env.robot.left_gripper_val,self.env.robot.right_gripper_val=action['grippers']
        self.native_scene.step()
        self.steps+=1;self.pending=None

    def after_intervention(self):
        if self.stream:self.stream.close()
        self.stream=None;self.pending=None;self.exhausted=False
        self.env.plan_success=True

    def make_perturbation(self,event,rng):
        if event['type']!='joint_offset':raise ValueError('RoboTwin supports joint_offset only')
        params=event.get('parameters',{})
        if set(params)-{'joint','offset','open_grippers'}:raise ValueError('Unknown RoboTwin intervention parameter')
        index,offset=params.get('joint',0),params.get('offset',.5)*event.get('strength',1.)
        if type(index) is not int or not 0<=index<len(self.env.robot.left_arm_joints) or not np.isfinite(offset):
            raise ValueError('Invalid arm joint offset')
        command=self.motor_command()
        robot=self.env.robot
        for side,entity,arm,grip,scale in zip(range(2),[robot.left_entity,robot.right_entity],
                [robot.left_arm_joints,robot.right_arm_joints],
                [robot.left_gripper,robot.right_gripper],
                [robot.left_gripper_scale,robot.right_gripper_scale]):
            joints=entity.get_active_joints()
            part=command['motors'][next(i for i,e in enumerate(self.entities) if e is entity)]
            target_index=joints.index(arm[index])
            limits=np.asarray(arm[index].get_limits()).reshape(-1)
            part['positions'][target_index]=float(np.clip(part['positions'][target_index]+offset,*limits[:2]))
            part['velocities']=[0.]*len(joints)
            if params.get('open_grippers',True):
                command['grippers'][side]=1.
                for joint,multiplier,bias in grip:
                    part['positions'][joints.index(joint)]=float(scale[1]*multiplier+bias)
        return [deepcopy(command) for _ in range(event['steps'])]

    def perturbation_actions(self,steps):
        return self.make_perturbation({'type':'joint_offset','steps':steps},np.random.default_rng(0))

    def success(self):
        return bool(self.steps>0 and self.env.check_success())
    def terminal(self):
        return self.exhausted
    def render(self):
        self.env._update_render()
        return self.env.get_obs()['observation']['head_camera']['rgb']
    def close(self):
        if self.stream:self.stream.close();self.stream=None
        if self.env:
            if getattr(self,'native_scene',None) is not None:self.env.scene=self.native_scene
            self.env.close_env(clear_cache=True);self.env=None
        if self.scratch:self.scratch.cleanup();self.scratch=None
        if hasattr(self,'previous_cwd'):os.chdir(self.previous_cwd)
