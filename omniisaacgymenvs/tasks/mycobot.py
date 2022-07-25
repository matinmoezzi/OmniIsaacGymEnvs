from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.mycobot import MyCobot

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class MyCobotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._mycobot_positions = torch.tensor([0.0, 0.0, 0.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 7

        # Initializing the camera
        self.sd_helper = None
        self.viewport_window = None
        # self._set_camera()

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        mycobot = MyCobot(
            prim_path=self.default_zero_env_path + "/MyCobot", name="MyCobot", translation=self._mycobot_positions
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "MyCobot", get_prim_at_path(mycobot.prim_path), self._sim_config.parse_actor_config("MyCobot")
        )
        super().set_up_scene(scene)
        self._mycobots = ArticulationView(prim_paths_expr="/World/envs/.*/MyCobot", name="mycobot_view")
        scene.add(self._mycobots)
        return

    def get_observations(self) -> dict:

        # self._env._world.render()
        # # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        # gt = self.sd_helper.get_groundtruth(
        #     ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        # )
        # self.obs_buf = gt["rgb"][:, :, :3]

        # observations = {self._mycobots.name: {"obs_buf": self.obs_buf}}
        observations = {self._mycobots.name: {"obs_buf": []}}
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._mycobots.count, self._mycobots.num_dof), dtype=torch.float32, device=self._device)

        indices = torch.arange(self._mycobots.count, dtype=torch.int32, device=self._device)
        self._mycobots.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        # randomize all envs
        indices = torch.arange(self._mycobots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def _set_camera(self):
        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        camera_path = "/jetbot/chassis/rgb_camera/jetbot_camera"
        camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path))
        camera.GetClippingRangeAttr().Set((0.01, 10000))
        if not self._env._render:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(128, 128)
        else:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(
                viewport_handle
            )
            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(128, 128)
            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            self.viewport_window = viewport_window
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        self._env._world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = torch.zeros(1)

    def is_done(self) -> None:
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        self.reset_buf[:] = resets
