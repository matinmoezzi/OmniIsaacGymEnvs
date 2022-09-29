from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.mycobot import MyCobot
from gym import spaces

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
from omni.kit import viewport_widgets_manager
import torch
import math


class MyCobotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._mycobot_positions = torch.tensor([0.0, 0.0, 0.2])
        self.scaling_factor = self._task_cfg["sim"]["MyCobot"]["scaling_factor"]
        self.prismatic_joint_limit = self._task_cfg["sim"]["MyCobot"]["prismatic_joint_limit"]
        self.revolute_joint_limit = self._task_cfg["sim"]["MyCobot"]["revolute_joint_limit"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 1
        self._num_actions = 8

        # [joint1,joint2,joint3,joint4,joint5,joint6,right_hand,left_hand]
        self.action_space = spaces.Box(
            np.concatenate((np.ones(6) * -np.radians(self.revolute_joint_limit), np.zeros(2))),
            np.concatenate(
                (
                    np.ones(6) * np.radians(self.revolute_joint_limit),
                    np.ones(2) * self.prismatic_joint_limit * self.scaling_factor,
                )
            ),
        )

        # Initializing the camera
        self.sd_helper = None
        self.viewport_window = None
        # self._set_camera()

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        mycobot = MyCobot(
            prim_path=self.default_zero_env_path + "/MyCobot",
            name="MyCobot",
            translation=self._mycobot_positions,
            scaling_factor=self.scaling_factor,
        )
        # applies articulation settings from the task configuration yaml file
        from omni.isaac.core.prims import RigidPrimView
        from omni.isaac.core.objects import DynamicCuboid

        scene.add(
            DynamicCuboid(
                prim_path="/World/envs/env_0/cube",
                name="visual_cube",
                position=np.array([0.00, -0.20, 0.025]),
                size=0.03,
                color=np.array([1.0, 0, 0]),
            )
        )
        self._sim_config.apply_articulation_settings(
            "MyCobot", get_prim_at_path(mycobot.prim_path), self._sim_config.parse_actor_config("MyCobot")
        )
        super().set_up_scene(scene)

        # self._set_camera()
        self._set_camera()
        self._mycobots = ArticulationView(prim_paths_expr="/World/envs/.*/MyCobot", name="mycobot_view")
        scene.add(self._mycobots)
        self._objects = RigidPrimView(prim_paths_expr="/World/envs/env_.*/cube", name="cube_view")
        scene.add(self._objects)
        return

    def get_observations(self) -> dict:

        self._env._world.render()
        camera_matrix = []
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        for i in range(self._num_envs):
            gt = self.sd_helper.get_groundtruth(
                ["rgb"], self.viewport_window[i], verify_sensor_init=False, wait_for_sensor_data=0
            )["rgb"][:, :, :3]
            camera_matrix.append(gt)
        dof_pos = self._mycobots.get_joint_positions()
        dof_vol = self._mycobots.get_joint_velocities()

        observations = {self._mycobots.name: {"obs_buf": camera_matrix}}
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._mycobots.count, self._mycobots.num_dof), dtype=torch.float32, device=self._device)

        indices = torch.arange(self._mycobots.count, dtype=torch.int32, device=self._device)
        self._mycobots.set_joint_positions(actions, indices=indices)

    def reset_idx(self, env_ids):
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.joint1_dof_idx = self._mycobots.get_dof_index("joint1_Rev")
        self.joint2_dof_idx = self._mycobots.get_dof_index("joint2_Rev")
        self.joint3_dof_idx = self._mycobots.get_dof_index("joint3_Rev")
        self.joint4_dof_idx = self._mycobots.get_dof_index("joint4_Rev")
        self.joint5_dof_idx = self._mycobots.get_dof_index("joint5_Rev")
        self.joint6_dof_idx = self._mycobots.get_dof_index("joint6_Rev")
        self.joint9_dof_idx = self._mycobots.get_dof_index("joint9_Pris")
        self.joint10_dof_idx = self._mycobots.get_dof_index("joint10_Pris")
        # randomize all envs
        indices = torch.arange(self._mycobots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def _set_camera(self):
        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        # if set env_# to .*, "Accessed schema on invalid prim" will pop up.
        # Adding the camera under camera_flange group after converting
        camera_path_default = "/World/envs/env_{}/MyCobot/camera_flange/Camera"
        camera_path_set = []
        self.viewport_window = []
        for i in range(self._num_envs):
            camera_path = camera_path_default.format(i)
            camera_path_set.append(camera_path)
            camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path))
            camera.GetClippingRangeAttr().Set((0.01, 10000))

        FLAG = True  # Whether initialize the viewport for each camera.
        if not FLAG:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface()
            for i in range(self._num_envs):
                viewport_handle.get_viewport_window().set_active_camera(str(camera_path_set[i]))
                viewport_window = viewport_handle.get_viewport_window()
                viewport_window.set_texture_resolution(64, 64)
                self.viewport_window.append(viewport_window)

        else:
            for i in range(self._num_envs):
                viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
                # Upper line of code linked with how many viewpoint_window get created.
                new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(
                    viewport_handle
                )
                viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
                viewport_window.set_active_camera(camera_path_set[i])
                viewport_window.set_texture_resolution(128, 128)
                viewport_window.set_window_pos(1000, 400)
                viewport_window.set_window_size(420, 420)
                # viewport_window.show_hide_window(False)
                self.viewport_window.append(viewport_window)

        self.sd_helper = SyntheticDataHelper()
        for i in range(self._num_envs):
            self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window[i])
        self._env._world.render()
        for i in range(self._num_envs):
            self.sd_helper.get_groundtruth(["rgb"], self.viewport_window[i])
        return

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = torch.zeros(1)

    def is_done(self) -> None:
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        self.reset_buf[:] = resets
