import numpy as np
import torch

from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage

class Real_Room:
    def __init__(
        self,
        scene, 
        position=[0.0, 0.0, 0.0], 
        orientation=[0.0, 0.0, 0.0], 
        scale=[1.0, 1.0, 1.0], 
        usd_path:str=None, 
        prim_path:str="/World/Room"
    ):
        self._room_position = position
        self._room_orientation = orientation
        self._room_scale = scale
        self._room_prim_path = find_unique_string_name(prim_path,is_unique_fn=lambda x: not is_prim_path_valid(x))
        self._room_usd_path = usd_path

        # add room to stage
        add_reference_to_stage(self._room_usd_path, self._room_prim_path)

        self._room_prim = XFormPrim(
            self._room_prim_path, 
            name="Room", 
            scale=self._room_scale, 
            position=self._room_position, 
            orientation=euler_angles_to_quat(self._room_orientation, degrees=True)
        )