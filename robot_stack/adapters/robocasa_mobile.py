"""Privileged physical mobile manipulation in native RoboCasa pick/place tasks."""
import numpy as np
from .robocasa import RoboCasaNavigationAdapter


MOBILE_TASKS = {
    'PickPlaceCounterToSink': ('obj', None),
    'PickPlaceSinkToCounter': ('obj', 'container'),
    'CheesyBread': ('cheese', 'bread'),
    'PackDessert': ('dessert', 'cooked_food_container'),
}


class RoboCasaMobileManipulationAdapter(RoboCasaNavigationAdapter):
    def __init__(self, task='PickPlaceCounterToSink', *, layout=1, style=1,
                 obj_groups='apple', render=False):
        super().__init__('NavigateKitchen',layout=layout,style=style,render=render)
        if task not in MOBILE_TASKS:
            raise ValueError('No bundled mobile manipulation expert for '+task)
        self.object_name,self.destination_name=MOBILE_TASKS[task]
        self.task,self.obj_groups=task,obj_groups
        self.metadata.update(policy='privileged base-feedback and Cartesian manipulation state machine',
                             error_features='base_xy_eef_xyz_object_xyz_gripper_aperture',
                             perturbation_type='scheduled_native_base_arm_and_gripper_commands',
                             obj_groups=(obj_groups if task.startswith('PickPlace') else 'native_task_default'),experimental=True,
                             expert_provider='robot_stack',skill_family='mobile_grasp_transport_place',
                             manipulated_object=self.object_name,destination_object=self.destination_name,
                             reset_compatibility="stable_counter_region_geometry_order_and_scoped_fixture_rng_v1")

    def reset(self,seed):
        import robocasa,robosuite
        from robosuite.controllers import load_composite_controller_config
        self.close()
        from .robocasa_rng import seeded_scene_initialization
        from robocasa.models.fixtures.counter import Counter
        cfg=load_composite_controller_config(robot='PandaOmron')
        with seeded_scene_initialization(seed, Counter):
            self.env=robosuite.make(self.task,robots='PandaOmron',controller_configs=cfg,
                has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,
                use_object_obs=True,seed=seed,layout_ids=self.layout,style_ids=self.style,
                generative_textures=None,randomize_cameras=False,control_freq=20,ignore_done=True,
                obj_registries=('objaverse','lightwheel'),
                **({'obj_groups':self.obj_groups} if self.task.startswith('PickPlace') else {}))
            self.env.reset()
        self.robot=self.env.robots[0]
        self.body=self.env.sim.model.body_name2id('mobilebase0_base')
        self.eef_id=self.robot.eef_site_id['right'];self.obj_id=self.env.obj_body_id[self.object_name]
        self.arm=self.robot.composite_controller.part_controllers['right']
        self.yaw=float(self.env.counter.rot+np.pi/2)
        self.place=self.placement_target()
        self.metadata['placement_target']=self.place.tolist()
        self.initial_object=self.object_pos().copy();self.initial_base=self.pose().copy()
        self.transit_height=max(1.10,self.initial_object[2]+.15)
        self.stage='navigate_pick';self.stage_steps=0;self.gripper=-1.;self.total_steps=0
        self.metadata.update(robocasa_version=robocasa.__version__,robosuite_version=robosuite.__version__,
                             language=self.env.get_ep_meta()['lang'],episode_meta=self.env.get_ep_meta(),
                             initial_object=self.initial_object.tolist(),initial_base=self.initial_base.tolist())
        # Native offscreen initialization calls sim.forward() before reset and
        # changes contact warm-start state. Render the existing simulation data
        # with MuJoCo's visualizer, which does not advance/forward the dynamics.
        self._renderer = None
        if self.render_enabled:
            import mujoco
            self._renderer = mujoco.Renderer(self.env.sim.model._model, height=480, width=640)
            self._render_options = mujoco.MjvOption()
            self._render_options.geomgroup[0] = 0
            self._render_options.geomgroup[1] = 1
            self._overview = mujoco.MjvCamera()
            self._overview.lookat[:] = [1.45, -.4, .8]
            self._overview.distance = 2.8
            self._overview.azimuth = 90
            self._overview.elevation = -25

    def placement_target(self):
        if self.destination_name is None:
            return np.asarray(self.env.sink.pos).copy()+[.20,0,.17]
        target_id=self.env.obj_body_id[self.destination_name]
        target=self.env.sim.data.body_xpos[target_id].copy()
        if self.task=='PackDessert':
            food=self.env.sim.data.body_xpos[self.env.obj_body_id['cooked_food']]
            away=target[:2]-food[:2]
            if np.linalg.norm(away)<.01:away=np.array([1.,0.])
            target[:2]+=.045*away/np.linalg.norm(away)
        return target+[0,0,.10]

    def object_pos(self):return self.env.sim.data.body_xpos[self.obj_id].copy()
    def eef_pos(self):return self.env.sim.data.site_xpos[self.eef_id].copy()
    def grasped(self):return bool(self.env._check_grasp(self.robot.gripper['right'],self.env.objects[self.object_name]))
    def error_features(self):
        return np.r_[self.pose()[:2],self.eef_pos(),self.object_pos(),self.env._get_observations()['robot0_gripper_qpos']]
    def action(self,velocity):
        return self.robot.composite_controller.create_action_vector({'base':np.asarray(velocity),
                     'right_gripper':[self.gripper],'base_mode':1})
    def arm_action(self,target,gripper):
        from scipy.spatial.transform import Rotation
        self.gripper=gripper
        desired=np.diag([1.,-1.,-1.])
        if self.task=='PickPlaceSinkToCounter':
            # Approach below the faucet from the front instead of driving the
            # wrist vertically through the spout over deeper sink objects.
            left=np.array([-np.sin(self.yaw),np.cos(self.yaw),0.])
            desired=Rotation.from_rotvec(-.6*left).as_matrix()@desired
        current=self.env.sim.data.site_xmat[self.eef_id].reshape(3,3)
        rot=Rotation.from_matrix(desired@current.T).as_rotvec()
        origin=self.arm.origin_ori
        delta=np.r_[np.clip(origin.T@(target-self.eef_pos())*12,-1,1),
                    np.clip(origin.T@rot*1.5,-1,1)]
        return self.robot.composite_controller.create_action_vector({'right':delta,'right_gripper':[gripper],'base_mode':-1})
    def navigate(self,target):
        self.env.target_pos=np.r_[target[:2],0.]
        self.env.target_ori=np.array([0.,0.,self.yaw])
        return super().expert_action()
    def transition(self,stage):self.stage=stage;self.stage_steps=0
    def expert_action(self):
        self.place=self.placement_target()
        obj=self.object_pos();eef=self.eef_pos()
        forward=np.array([np.cos(self.yaw),np.sin(self.yaw)])
        if self.stage=='navigate_pick':
            self.gripper=-1.
            goal=obj[:2]-forward*.55
            distance=np.linalg.norm(self.pose()[:2]-goal)
            reachable=(self.task!='PickPlaceCounterToSink' and self.stage_steps>70 and distance<.18)
            if distance<.035 or reachable:self.transition('approach' if self.task=='PickPlaceCounterToSink' else 'clear_pick')
            else:return self.navigate(goal)
        if self.stage=='clear_pick':
            if eef[2]>=self.transit_height-.025:self.transition('approach')
            else:return self.arm_action(np.r_[eef[:2],self.transit_height],-1)
        if self.stage=='approach':
            target=obj+[0,0,.15]
            if self.task!='PickPlaceCounterToSink':target[2]=self.transit_height
            if self.task=='PickPlaceSinkToCounter':
                target=obj+np.r_[-forward*.12,.16]
            if np.linalg.norm(eef-target)<.025:self.transition('descend')
            else:return self.arm_action(target,-1)
        if self.stage=='descend':
            target=obj+[0,0,.005]
            tolerance=.013 if self.task=='PickPlaceCounterToSink' else .035
            if np.linalg.norm(eef-target)<tolerance:self.transition('grasp')
            else:return self.arm_action(target,-1)
        if self.stage=='grasp':
            if self.stage_steps>=20:self.transition('lift')
            else:return self.arm_action(obj+[0,0,.005],1)
        if self.stage=='lift':
            # Clear the counter without lifting into the overhead cabinetry.
            target=np.r_[eef[:2],max(1.08,self.initial_object[2]+.14)]
            if self.grasped() and obj[2]>self.initial_object[2]+.07 and eef[2]>target[2]-.025:self.transition('navigate_sink')
            elif self.stage_steps>100 and not self.grasped():self.transition('navigate_pick')
            return self.arm_action(target,1)
        if self.stage=='navigate_sink':
            self.gripper=1.
            goal=self.place[:2]-forward*.55
            distance=np.linalg.norm(self.pose()[:2]-goal)
            reachable=(self.task!='PickPlaceCounterToSink' and self.stage_steps>70 and distance<.18)
            if distance<.035 or reachable:self.transition('place')
            else:return self.navigate(goal)
        if self.stage=='place':
            # Placement is above the bowl; a 5 cm waypoint tolerance avoids
            # waiting indefinitely at an OSC/contact equilibrium. Native
            # containment + gripper-distance still decide episode success.
            if np.linalg.norm(eef-self.place)<.05:self.transition('release')
            else:return self.arm_action(self.place,1)
        if self.stage=='release':
            if self.stage_steps>=20:self.transition('retreat')
            else:return self.arm_action(self.place,-1)
        if self.stage=='retreat':
            if self.task!='PickPlaceCounterToSink' and self.stage_steps>60 and not self.success():
                self.transition('navigate_pick')
            return self.arm_action(self.place+[0,0,.25],-1)
        return self.action([0,0,0])
    def step(self,action):
        super().step(action);self.stage_steps+=1;self.total_steps+=1
        start,_=self.robot.composite_controller._action_split_indexes['right_gripper']
        self.gripper=float(np.asarray(action)[start])
    def after_intervention(self):
        self.gripper=1. if self.grasped() else -1.
        if self.grasped():self.transition('lift')
        elif self.success():self.transition('retreat')
        else:self.transition('navigate_pick')
    def make_perturbation(self,event,rng):
        kind=event['type'];params=event.get('parameters',{});strength=event.get('strength',1.)
        if kind in {'base_yaw','base_translation','base_reverse','base_hold'}:
            allowed={'direction'} if kind=='base_translation' else set()
            if set(params)-allowed:raise ValueError('Unsupported base intervention parameters')
            velocity=np.zeros(3)
            if kind=='base_yaw':velocity[2]=strength
            elif kind in {'base_translation','base_reverse'}:
                direction=np.asarray(params.get('direction',[1,0]),dtype=float)
                if direction.shape!=(2,) or not np.isfinite(direction).all() or np.max(np.abs(direction))>1:
                    raise ValueError('direction must contain two normalized finite values')
                velocity[:2]=direction*strength*(-1 if kind=='base_reverse' else 1)
            command=self.action(velocity)
        elif kind in {'arm_offset','gripper_open','gripper_close','arm_hold'}:
            allowed={'direction'} if kind=='arm_offset' else set()
            if set(params)-allowed:raise ValueError('Unsupported arm intervention parameters')
            delta=np.zeros(6)
            if kind=='arm_offset':
                direction=np.asarray(params.get('direction',[1,0,0]),dtype=float)
                if direction.shape!=(3,) or not np.isfinite(direction).all() or np.max(np.abs(direction))>1:
                    raise ValueError('direction must contain three normalized finite values')
                delta[:3]=strength*direction
            grip=(-strength if kind=='gripper_open' else strength if kind=='gripper_close' else self.gripper)
            command=self.robot.composite_controller.create_action_vector(
                {'right':delta,'right_gripper':[grip],'base_mode':-1})
        else:raise ValueError('Unsupported mobile manipulation intervention: '+kind)
        return [command.copy().tolist() for _ in range(event['steps'])]
    def events(self):
        labels=[self.stage]
        if self.grasped():labels.append('object_grasped')
        if self.object_pos()[2]>self.initial_object[2]+.10:labels.append('object_lifted')
        return labels
    def render(self):
        if self._renderer is None:
            raise RuntimeError('Construct the adapter with render=True')
        self._renderer.update_scene(self.env.sim.data._data, camera=self._overview,
                                    scene_option=self._render_options)
        overview = self._renderer.render().copy()
        self._renderer.update_scene(self.env.sim.data._data, camera='robot0_agentview_left',
                                    scene_option=self._render_options)
        return np.concatenate((overview, self._renderer.render()), axis=1)
    def close(self):
        renderer=getattr(self,'_renderer',None)
        if renderer is not None:
            renderer.close();self._renderer=None
        super().close()
