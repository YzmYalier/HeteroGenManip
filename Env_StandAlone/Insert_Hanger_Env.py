from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from scipy.spatial.transform import Rotation as R
# from transform_utils import Rotation
import omni
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema

import os
import sys
import time
import numpy as np
from termcolor import cprint
import threading
import torch

# IsaacSim 依赖
import omni.replicator.core as rep
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, Usd, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import RigidPrim,XFormPrim,SingleXFormPrim,GeometryPrim,SingleRigidPrim,SingleArticulation
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api.materials import PhysicsMaterial
from omni.physx.scripts import physicsUtils
import omni
# 自定义依赖
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment  # 复用衣物创建逻辑
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud,compute_similarity
from Env_Config.Robot.Franka import Franka
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.utils.string import find_unique_string_name
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models
from FoundationModels.Uni3D.Uni3D_Encapsulation import Uni3D_Encapsulation

import json

class Hanger:
    def __init__(self, position=torch.tensor, orientation=[0.0, 0.0, 0.0], scale=[1, 1, 1], usd_path=str, prim_path:str="/World/Hanger"):
        self._hanger_position = position
        self._hanger_orientation = orientation
        self._hanger_scale = scale
        self._hanger_prim_path = find_unique_string_name(prim_path,is_unique_fn=lambda x: not is_prim_path_valid(x))
        self._hanger_usd_path = usd_path

        # add wash_machine to stage
        add_reference_to_stage(self._hanger_usd_path, self._hanger_prim_path)

        self._hanger_prim = SingleXFormPrim(
            self._hanger_prim_path, 
            name="hanger", 
            scale=self._hanger_scale, 
            position=self._hanger_position, 
            orientation=euler_angles_to_quat(self._hanger_orientation, degrees=True)
        )
        self.usd_prim = get_prim_at_path(self._hanger_prim_path)
        angulardamp_attr = self.usd_prim.CreateAttribute(
            "physxRigidBody:angularDamping",
            Sdf.ValueTypeNames.Float,
            True
        )
        lineardamp_attr = self.usd_prim.CreateAttribute(
            "physxRigidBody:linearDamping",
            Sdf.ValueTypeNames.Float,
            True
        )
        contactimpulse_attr = self.usd_prim.CreateAttribute(
            "physxRigidBody:maxContactImpulse",
            Sdf.ValueTypeNames.Float,
            True
        )
        stable_attr = self.usd_prim.CreateAttribute(
            "physxRigidBody:stabilizationThreshold",
            Sdf.ValueTypeNames.Float,
            True
        )
        # Set the default values for the attributes
        self.usd_prim.GetAttribute("physxRigidBody:angularDamping").Set(50.0)
        self.usd_prim.GetAttribute("physxRigidBody:linearDamping").Set(50.0)
        self.usd_prim.GetAttribute("physxRigidBody:maxContactImpulse").Set(0.00005)
        self.usd_prim.GetAttribute("physxRigidBody:stabilizationThreshold").Set(0.01)

        # self._hanger_rigid_prim=RigidPrim(
        #     prim_path=self._hanger_prim_path,
        #     scale=self._hanger_scale, 
        #     position=self._hanger_position, 
        #     orientation=euler_angles_to_quat(self._hanger_orientation, degrees=True),
        #     name="ClothHanger",
        #     mass=0.1
        # )
        # self._hanger_rigid_prim_view=RigidPrimView(
        #     prim_paths_expr=self._hanger_prim_path,
        #     name="ClothHanger",
        #     densities=[10],
        #     masses=[100]
        # )
        # self._hanger_rigid_prim_view.disable_gravities()
    # def add_gravity(self):
    #     self._hanger_rigid_prim_view.enable_gravities()
        
class HangEnv(BaseEnv):
    def __init__(    
        self,    
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        env_dx:float=0.0,
        env_dy:float=0.0,
        env_dz:float=0.0,
        theta:float=0.0,
        ground_material_usd:str=None,#"Assets/Material/Floor/Fabric001.usd"
        record_video_flag:bool=False, 
    ):
        super().__init__()

        # self.scene.add_default_ground_plane(z_position=-0.3)

        # if garment_config is None:
        #     self.garmentconfig=GarmentConfig()
        # else:
        #     self.garmentconfig=garment_config
        # if franka_config is None:
        #     self.robotConfig=FrankaConfig()
        # else:
        #     self.robotConfig=franka_config
        # self.robots=self.import_franka(self.robotConfig)
        # self.garment = list()


        # garment = Garment(self.world, self.garmentconfig)
        # garment.set_mass(30)
        # self.garment.append(garment)
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )

        # load garment
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path="Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top603/TNLC_Top603_obj.usd" if usd_path is None else usd_path,
        )
        # pos=np.array([0.8,0.8,0.3]),ori=np.array([0,0,np.pi]),scale=np.array([0.01,0.01,0.01])
        # pos=[np.array([-0.08,-0.25,-0.2]),np.array([1.5,-0.53587,-0.25])],ori=[np.array([0,0,0]),np.array([0,0,np.pi])]
        # Here are some example garments you can try:
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket152/TCLC_Jacket152_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top566/TCLC_Top566_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top584/TCLC_Top584_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top118/TCLC_top118_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top476/TCLC_Top476_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top030/TCLC_Top030_obj.usd",  
        self.franka = Franka(
            self.world, 
            position=np.array([-0.08,-0.25,0.05]),
            orientation=np.array([0.0, 0.0, 0.0]),
            robot_name="Franka")
        
        self.franka2 = Franka(
            self.world, 
            position=np.array([1.5,-0.53587,0.05]),
            orientation=np.array([0,0,180]),
            robot_name="Franka2")
        
        self.env_camera=Recording_Camera(camera_position=np.array([0.7, -4.44385, 2.4]),camera_orientation=np.array([0.0, 26.0, 90.0]))

        self.hanger = Hanger(position=np.array([0.66683,-0.13,0.78557]),orientation=[90.,0.,180.],scale=[0.025,0.025,0.033],usd_path="Assets/Ziyu_Assets/hanger.usd",prim_path="/World/Hanger")

        # self.control=Control(self.world,self.robots,self.garment)
        
        self.scene.add(
            FixedCuboid(
                name="cylinder",
                position=[0.66973, 0.0, 0.76577],
                prim_path="/World/cylinder",
                scale=np.array([0.02,0.02,2.0]),
                orientation=euler_angles_to_quat([90.0,0.0,0.0],degrees=True),

                color=np.array([180,180,180]),
                visible=True,
            )
        )


        self.scene.add(
            FixedCuboid(
                name="platform",
                position=[0.85,0,0.15],
                prim_path="/World/platform",
                scale=np.array([1.0,1.0,0.3]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),

                color=np.array([180,180,180]),
                visible=True,
            )
        )
        
        self.reset()

        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(
            depth_enable=True,
        )
        
        # self.object_camera.initialize(
        #     segment_pc_enable=True, 
        #     segment_prim_path_list=[
        #         "/World/hanger1",
        #         "/World/hanger2",
        #         "/World/hanger3",
        #     ]
        # )
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
            
        # move garment to the target position
        self.garment.set_pose(pos=pos, ori=ori)
        self.position = pos
        self.orientation = ori
                
        # open hand to be initial state
        # self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(200):
            self.step()
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint(f"env_dx: {env_dx}", "magenta")
        cprint(f"env_dy: {env_dy}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])

        cprint("World Ready!", "green", "on_green")
        # self.scene_grav = self.world.get_physics_context()._physics_scene
        # self.scene_grav.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0.0, -1))
        # self.scene_grav.CreateGravityMagnitudeAttr().Set(5)

        # rigid_group_path=[]
        # rigid_group_path.append("/World/platform")


        # self.collision_group=self.control.collision_group()


        # self.control.add_rigid_target(rigid_group_path)
        # self.control.add_custom_group(group_path=[self.hanger._hanger_prim_path+"hangertry/ClothHanger"],robot_filter=False)
        # self.stage = omni.usd.get_context().get_stage()
        # self.draw=_debug_draw.acquire_debug_draw_interface()





if __name__ == "__main__":
    # garment_config = GarmentConfig(pos=np.array([0.8,0.8,0.3]),ori=np.array([0,0,np.pi]),scale=np.array([0.01,0.01,0.01]))
    # garment_config = GarmentConfig(pos=np.array([0.91366,-1.04,0.15]),ori=np.array([0,0,0]),scale=np.array([0.007,0.007,0.007]))
    
    # franka_config = FrankaConfig(franka_num=2,pos=[np.array([-0.08,-0.25,-0.2]),np.array([1.5,-0.53587,-0.25])],ori=[np.array([0,0,0]),np.array([0,0,np.pi])])
    env = HangEnv(pos=np.array([0.8, 0.0, 0.35]), ori=np.array([0, 0, np.pi]))

    # print("now start to attach!")
    # #env.control.make_attachment(position=[np.array([0.74102,-0.60048,0.06]),np.array([0.85297,-0.60048,0.055])],flag=[True,True])
    # env.control.make_attachment(position=[np.array([0.76967,-0.67977,0.036]),np.array([0.82,-0.63,0.036])],flag=[True,True])
    # env.control.control_gravities([False,False])
    # env.control.attach(object_list=env.garment,flag=[True,True])
    # print("attach finished!")

    # env.control.robot_open([False,False])
    # print("robot open finished!")
    # env.control.robot_goto_position(pos=[np.array([0.5,-0.13156,0.6]),env.control.attachlist[1].get_position()],ori=[None,None],flag=[True,True])

    # env.control.robot_close([True,True])
    # print("robot close finished!")

    # delete_prim("/World/cylinder")
    # # if you want to grab the attachment block, you must use "move"
    # env.control.move(pos=[np.array([0.64636,-0.13156,0.7]),np.array([0.89,-0.67,0.23])],ori=[None,None],flag=[False,True])


    while True:
        env.step()

simulation_app.close()

