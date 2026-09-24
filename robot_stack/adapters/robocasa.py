"""Native RoboCasa NavigateKitchen with feedback control of the Omron base."""
from importlib.metadata import version
import numpy as np


class RoboCasaNavigationAdapter:
    backend = 'robocasa'

    def __init__(self, task='NavigateKitchen', *, layout=1, style=1, render=False):
        if task != 'NavigateKitchen':
            raise ValueError('Bundled RoboCasa controller currently supports NavigateKitchen only')
        self.task, self.layout, self.style = task, layout, style
        self.render_enabled, self.env = render, None
        self.metadata = {'layout':layout, 'style':style,
                         'versions':{n:version(n) for n in ('mujoco','numpy')},
                         'state_kind':'mujoco.mjSTATE_INTEGRATION',
                         'error_features':'base_xy_metres_and_yaw_cos_sin',
                         'policy':'privileged target-pose base velocity feedback',
                         'perturbation_type':'wrong_base_yaw_velocity_burst'}

    def reset(self, seed):
        import robocasa
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        self.close()
        self.env = robosuite.make(self.task, robots='PandaOmron',
            controller_configs=load_composite_controller_config(robot='PandaOmron'),
            has_renderer=False,has_offscreen_renderer=self.render_enabled,use_camera_obs=False,
            use_object_obs=True,seed=seed,layout_ids=self.layout,style_ids=self.style,
            generative_textures=None,randomize_cameras=False,control_freq=20,ignore_done=True,
            obj_registries=('lightwheel',))
        self.env.reset()
        self.body = self.env.sim.model.body_name2id('mobilebase0_base')
        self.metadata.update(robocasa_version=robocasa.__version__, robosuite_version=robosuite.__version__,
                             target_pos=self.env.target_pos.tolist(),target_ori=self.env.target_ori.tolist(),
                             language=self.env.get_ep_meta()['lang'])
        self.initial_yaw = self.pose()[2]

    def pose(self):
        from robosuite.utils.transform_utils import mat2euler
        return np.r_[self.env.sim.data.body_xpos[self.body][:2],
                     mat2euler(self.env.sim.data.body_xmat[self.body].reshape(3,3))[2]]

    def state(self):
        import mujoco
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        sim=self.env.sim
        result=np.empty(mujoco.mj_stateSize(sim.model._model,spec))
        mujoco.mj_getState(sim.model._model,sim.data._data,result,spec)
        return result

    def error_features(self):
        x,y,yaw=self.pose()
        return np.array([x,y,np.cos(yaw),np.sin(yaw)])

    def action(self, velocity):
        return self.env.robots[0].composite_controller.create_action_vector(
            {'base':np.asarray(velocity), 'base_mode':1})

    def expert_action(self):
        x,y,yaw=self.pose()
        delta=self.env.target_pos[:2]-[x,y]
        desired=(self.env.target_ori[2]-yaw+np.pi)%(2*np.pi)-np.pi
        # Convert desired world translation into the current base frame.
        c,s=np.cos(yaw),np.sin(yaw)
        local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        return self.action(np.r_[np.clip(1.5*local,-.35,.35),np.clip(1.5*desired,-.7,.7)])

    def step(self, action):
        a=np.asarray(action,dtype=float).copy()
        if a.shape != (self.env.action_dim,) or not np.isfinite(a).all():
            raise ValueError('Invalid RoboCasa action')
        self.env.step(a)

    def perturbation_actions(self, steps):
        return [self.action([0,0,1.]).tolist() for _ in range(steps)]

    def success(self):
        return bool(self.env._check_success())

    def terminal(self):
        return False

    def render(self):
        return self.env.sim.render(width=640,height=480,camera_name='robot0_agentview_left')[::-1]

    def close(self):
        if self.env is not None:
            self.env.close(); self.env=None
