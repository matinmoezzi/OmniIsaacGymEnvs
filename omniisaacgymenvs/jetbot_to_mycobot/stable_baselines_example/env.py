# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from symbol import parameters
import gym
from gym import spaces
import numpy as np
import math
import carb
import torch


class MyCobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1024, # 256
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
        # from omni.isaac.wheeled_robots.robots import WheeledRobot
        # from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid
        # from omni.isaac.core.utils.nucleus import get_assets_root_path

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        # self._my_world.scene.add_default_ground_plane()
        # assets_root_path = get_assets_root_path()
        # if assets_root_path is None:
        #     carb.log_error("Could not find Isaac Sim assets folder")
        #     return
        # jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        # absolute path to the mycobot usd file
        # mycobot_asset_path = "/home/chay/Desktop/Sim2Real_22Fall/mycobot_gripper_simplified_hands_deleted.usd"
        mycobot_asset_path = "mycobot_gripper_simplified_hands_deleted.usd"
        # self.jetbot = self._my_world.scene.add(
        #     WheeledRobot(
        #         prim_path="/jetbot",
        #         name="my_jetbot",
        #         wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        #         create_robot=True,
        #         usd_path=jetbot_asset_path,
        #         position=np.array([0, 0.0, 0.03]),
        #         orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        #     )
        # )
        from omni.isaac.core.robots.robot import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(usd_path=mycobot_asset_path, prim_path='/mycobot')
        mycobot = Robot(prim_path='/mycobot', name='MyCobot',scale=torch.tensor([1, 1, 1])*0.05)

        # self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )

        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.prims import GeometryPrim
        self._mycobot = ArticulationView(prim_paths_expr="/mycobot",name='mycobot_view')
        self.tip = ArticulationView()

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
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.max_velocity = 1
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.jetbot.get_world_pose()
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward = action[0]
        raw_angular = action[1]

        # we want to force the jetbot to always drive forward
        # so we transform to [0,1].  we also scale by our max velocity
        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.jetbot.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
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
        # jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        # jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        # jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        # goal_world_position, _ = self.goal.get_world_pose()
        # return np.concatenate(
        #     [
        #         jetbot_world_position,
        #         jetbot_world_orientation,
        #         jetbot_linear_velocity,
        #         jetbot_angular_velocity,
        #         goal_world_position,
        #     ]
        # )
        mycobot_angular_velocity = self._mycobot.get_angular_velocities()
        mycobot_angle_postion = self._mycobot.get_world_poses()

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
        new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(
            viewport_handle
        )
        viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
        viewport_window.set_active_camera("/mycobot/camera_flange/rgb_camera")
        viewport_window.set_texture_resolution(128, 128)
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

    def get_joint_velociies(self):
        full_dofs_velocities = self._mycobot.get_joint_velocities()
        joint_dof_velocities = [full_dofs_velocities[i] for i in self._mycobot_dof_indicies]
        return joint_dof_velocities

    def set_joint_velocity(self, velocities) -> None:
        full_dofs_velocities = [None] * self.num_dof
        for i in range(self.num_dof):
            full_dofs_velocities[self.num_dof[i]] = velocities[i]
        self._mycobot.set_joint_velocities(velocities=np.array(full_dofs_velocities))
        return
