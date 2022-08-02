from typing import Optional
import numpy as np
from pathlib import Path
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage


class MyCobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "MyCobot",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable

        intanceable_asset_usd = Path(__file__).parent / "../assets/mycobot_urdf/mycobot_urdf.usd"

        if self._usd_path is None:
            self._usd_path = str(intanceable_asset_usd)

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )
