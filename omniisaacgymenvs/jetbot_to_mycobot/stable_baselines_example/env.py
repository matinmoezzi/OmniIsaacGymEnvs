from symbol import parameters
import gym
from gym import spaces

import numpy as np
import os
import math
import carb
import torch
import pathlib


class MyCobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1024,  # 256
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()

        usd_file_name = "mycobot_gripper_simplified_hands_deleted.usd"
        mycobot_asset_path = os.path.join(pathlib.Path(__file__).parent.resolve(), usd_file_name)

        from mycobot import MyCobot

        self.mycobot = self._my_world.scene.add(
            MyCobot(
                prim_path="/mycobot",
                usd_path=mycobot_asset_path,
                name="MyCobot", 
                translation=torch.tensor([0.0, 0.0, 14.7]),
                scaling_factor=0.05)
        )

        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )

        from omni.isaac.core.prims.xform_prim import XFormPrim

        self._tip = XFormPrim(prim_path="/mycobot/gripper_shell/collisions", name="gripper_collisions")

        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdLux

        stage = get_current_stage()
        light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
        light.GetPrim().GetAttribute("intensity").Set(500)

        self.seed(seed)
        self.set_world_window()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)

        self.num_dof = 6

        self.action_space = spaces.Box(low=-1, high=1.0, shape=(6,), dtype=np.float32)
        # Vision Based observation_space, which need to match the method get_observation return
        self.observation_space = spaces.Box(low=int(0), high=int(255), shape=(64, 64, 3), dtype=np.intc)

        self.max_velocity = 1
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):  # action is produced by policy (embeded)
        previous_mycobot_tip_position, _ = self._tip.get_world_pose()
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            self.mycobot.apply_action(ArticulationAction(action))
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_mycobot_tip_position, _ = self._tip.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_mycobot_tip_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_mycobot_tip_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        if current_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        mycobot_angular_velocity = self.mycobot.get_angular_velocity()
        mycobot_angle_postion = self.mycobot.get_world_pose()

        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return gt["rgb"][:, :, :3]

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def set_world_window(self):
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
        new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(viewport_handle)
        viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
        viewport_window.set_active_camera("/mycobot/camera_flange/rgb_camera")
        viewport_window.set_texture_resolution(64, 64)
        # viewport_window.set_camera_position("/OmniverseKit_Persp",100,100,60,True)
        # viewport_window.set_camera_target("/OmniverseKit_Persp",0,0,0,True)
        viewport_window.set_window_pos(1000, 400)
        viewport_window.set_window_size(420, 420)
        self.viewport_window = viewport_window
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        # self.my_world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return
