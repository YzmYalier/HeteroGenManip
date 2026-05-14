import numpy as np
import omni.kit.commands
import omni.physxdemos as demo
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.rotations import euler_angles_to_quat


class Human():
    def __init__(self,path,position=None,orientation=None, scale=np.array([0.7, 0.7, 0.7])):
        self.path=path
        self.prim_path="/World/Human"
        add_reference_to_stage(usd_path=path,prim_path=self.prim_path)

        if position is None:
            position=np.array([0,0,0])
        if orientation is None:
            orientation=euler_angles_to_quat([90.0,0.,0.],degrees=True)
        else:
            orientation=euler_angles_to_quat(orientation,degrees=True)
        self.rigid_form=SingleXFormPrim(
            prim_path=self.prim_path,
            name="human",
            position=position,
            orientation=orientation,
            scale=scale,
        )
        
        self.geom_prim=SingleGeometryPrim(
            prim_path=self.prim_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("meshSimplification")
