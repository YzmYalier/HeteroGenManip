from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# load external package
import os
import sys
import time
import numpy as np
import torch
import open3d as o3d
from termcolor import cprint
import threading
import omni

# load isaac-relevant package
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility, delete_prim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils
from isaacsim.core.api.materials import PhysicsMaterial
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, Usd, PhysxSchema
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Room.Object_Tools import hanger_load, set_prim_visible_group, delete_prim_group, dynamic_hanger_load
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from MFMDP.deploy_policy import MFMDP_Encapsulation

def Collision_Group_Setup(world:World):
    stage = world.stage
    
    bag_group_path = "/World/Collision_Group/bag_group"
    robot_group_path = "/World/Collision_Group/robot_group"

    bag_group = UsdPhysics.CollisionGroup.Define(stage, bag_group_path)
    robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)

    filter_bag = bag_group.CreateFilteredGroupsRel()
    filter_robot = robot_group.CreateFilteredGroupsRel()

    collectionAPI_bag = Usd.CollectionAPI.Apply(bag_group.GetPrim(), "colliders")
    collectionAPI_robot = Usd.CollectionAPI.Apply(robot_group.GetPrim(), "colliders")

    collectionAPI_bag.CreateIncludesRel().AddTarget("/World/Bag")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexLeft")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexRight")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/Garment")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/garment_helper0")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/garment_helper1")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/garment_helper2")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/garment_helper3")
    collectionAPI_robot.CreateIncludesRel().AddTarget("/World/garment_helper4")
    
    filter_bag.AddTarget(robot_group_path)

class StoreTops_Env(BaseEnv):
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
        training_data_num:int=100,
        checkpoint_num:int=1500,
        stage_str:str="all",
        policy_name:str="MFMDP",
    ):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
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
            usd_path="Assets/Garment/Tops/Collar_noSleeve_FrontClose/TCNC_Top184/TCNC_Top184_obj.usd" if usd_path is None else usd_path,
        )
        # Here are some example garments you can try:
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket152/TCLC_Jacket152_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top566/TCLC_Top566_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top584/TCLC_Top584_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top118/TCLC_top118_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top476/TCLC_Top476_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top030/TCLC_Top030_obj.usd",  

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.3, 0.55]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.3, 0.55]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, -3.0, 6.75]), 
            camera_orientation=np.array([0, 60.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, -4.0, 3.5]),
            camera_orientation=np.array([0, 33.0, 90.0]),
            prim_path="/World/env_camera",
        )

        self.env_right_camera = Recording_Camera(
            camera_position=np.array([-5, 0.6, 0.7]),
            camera_orientation=np.array([0, 0.0, 0.0]),
            prim_path="/World/env_right_camera",
        )
        
        self.env_left_camera = Recording_Camera(
            camera_position=np.array([5, 0.6, 0.7]),
            camera_orientation=np.array([0, 0.0, -180.0]),
            prim_path="/World/env_left_camera",
        )
        
        self.env_back_camera = Recording_Camera(
            camera_position=np.array([0.0, -5.65, 4.0]),
            camera_orientation=np.array([0, 30.0, 90.0]),
            prim_path="/World/env_back_camera",
        )
                
        self.object_camera = Recording_Camera(
            camera_position=np.array([0.0, -6.6, 4.9]),
            camera_orientation=np.array([0, 30.0, 90.0]),
            prim_path="/World/object_camera",
        )
        
        self.garment_pcd = None
        self.object_pcd = None
        self.points_affordance_feature = None
        self.object_feature = None
        
        input_dict = {
            "task_name": "Store_Tops",
            "stage_str": stage_str,
            "data_num": training_data_num,
            "checkpoint_num": checkpoint_num,
            "active_object_n_component": 0,
            "passive_object_n_component": 5,
            "category": "Tops_NoSleeve",
            "demo_pcd_path": None,
        }
        self.policy = getattr(sys.modules[__name__], f"{policy_name}_Encapsulation")(init_dict=input_dict)

        # load bag
        self.bag_prim_path = "/World/Bag"
        add_reference_to_stage(
            usd_path="Assets/bag.usd",  
            prim_path=self.bag_prim_path
        )
        self.env_dx = env_dx
        self.env_dy = env_dy
        bag_pose = np.array([0.0, 1.5, 0.0]) + np.array([self.env_dx, self.env_dy, 0.05])
        self.bag = SingleXFormPrim(
            prim_path=self.bag_prim_path,
            position=bag_pose,
            orientation=euler_angles_to_quat(np.array([0.0,0.0,0.0]), degrees=True),
            # scale=np.array([0.015, 0.025, 0.005]),
            scale=np.array([0.025, 0.035, 0.005]),
        )
        # self.physics_material = PhysicsMaterial(
        #     prim_path="/World/Physics/Material",
        #     name="physics_material",
        #     static_friction=0.001,    # 静摩擦系数
        #     dynamic_friction=0.0005,  # 动摩擦系数
        #     restitution=0.0       # 弹性系数
        # )
        # self.stage = self.bag.prim.GetStage()  
        # bag_prim = self.stage.GetPrimAtPath(self.bag_prim_path)
        # physicsUtils.add_physics_material_to_prim(self.stage, bag_prim, "/World/Physics/Material")
        # collision_api = UsdPhysics.CollisionAPI.Apply(bag_prim)
        # mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(bag_prim)
        # mesh_collision_api.CreateApproximationAttr().Set("sdf")
        # collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(bag_prim)
        # collision_api.CreateSdfResolutionAttr().Set(1024)
        # self.collisiongroup = CollisionGroup(
        #     self.world,
        #     helper_path=["/World/Bag"],
        #     garment=False,
        #     collide_with_garment=True,
        #     collide_with_robot=True,
        # )
        # self.set_mass(stage=self.stage, prim=bag_prim, mass=50)
        self.bag_center = bag_pose
        
        self.debug_tool = VisualCuboid(
            prim_path="/World/Debug_cube",
            name="debug_cube",
            scale=np.array([0.01, 0.01, 0.01]),
            color=np.array([1.0, 0.0, 0.0]),
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            visible=True,
        )

        bag_max_x, bag_max_y, bag_max_z, bag_min_x, bag_min_y, bag_min_z = self.get_prim_dimensions(self.bag_prim_path)
        
        self.scene.add(
            FixedCuboid(
                name="garment_helper0",
                position=[(bag_max_x + bag_min_x) / 2, (bag_max_y + bag_min_y) / 2, bag_min_z],
                prim_path="/World/garment_helper0",
                scale=np.array([bag_max_x - bag_min_x, bag_max_y - bag_min_y, 0.01]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        self.scene.add(
            FixedCuboid(
                name="garment_helper1",
                position=[bag_max_x - 0.05, (bag_max_y + bag_min_y) / 2, (bag_max_z + bag_min_z) / 2 - 0.1],
                prim_path="/World/garment_helper1",
                scale=np.array([0.01, bag_max_y - bag_min_y, bag_max_z - bag_min_z]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        self.scene.add(
            FixedCuboid(
                name="garment_helper2",
                position=[bag_min_x + 0.05, (bag_max_y + bag_min_y) / 2, (bag_max_z + bag_min_z) / 2 - 0.1],
                prim_path="/World/garment_helper2",
                scale=np.array([0.01, bag_max_y - bag_min_y, bag_max_z - bag_min_z]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        self.scene.add(
            FixedCuboid(
                name="garment_helper3",
                position=[(bag_max_x + bag_min_x) / 2, bag_max_y - 0.05, (bag_max_z + bag_min_z) / 2 - 0.1],
                prim_path="/World/garment_helper3",
                scale=np.array([bag_max_x - bag_min_x, 0.01, bag_max_z - bag_min_z]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        self.scene.add(
            FixedCuboid(
                name="garment_helper4",
                position=[(bag_max_x + bag_min_x) / 2, bag_min_y + 0.05, (bag_max_z + bag_min_z) / 2 - 0.1],
                prim_path="/World/garment_helper4",
                scale=np.array([bag_max_x - bag_min_x, 0.01, bag_max_z - bag_min_z]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        
        Collision_Group_Setup(self.world)
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(
            depth_enable=True,
        )

        self.env_right_camera.initialize(
            depth_enable=True,
        )
        
        self.env_left_camera.initialize(
            depth_enable=True,
        )
        
        self.env_back_camera.initialize(
            depth_enable=True,
        )
        
        self.object_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Bag"
            ]
        )

        self.env_camera_intrinsics = self.env_camera.camera.get_intrinsics_matrix()
        self.env_camera_extrinsics = self.env_camera.camera.get_view_matrix_ros()
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
            
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
                
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

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
    
    def get_prim_dimensions(self,prim_path):

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Prim {prim_path} 不存在！")
            return None
    
        included_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy]
        
        # 创建BBoxCache实例，传入所有必要参数
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            included_purposes,
            useExtentsHint=True,  # 使用范围提示来加速计算
            ignoreVisibility=True # 忽略可见性，计算所有几何体
        )
    
        # 通过实例调用ComputeWorldBound方法
        bbox = bbox_cache.ComputeWorldBound(prim)
        range = bbox.GetRange()
        
        # 获取包围盒最小、最大坐标
        min_vec = range.GetMin()
        max_vec = range.GetMax()
        
        return max_vec[0], max_vec[1], max_vec[2], min_vec[0], min_vec[1], min_vec[2]
    
    def get_obs(self):
        joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
        joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
        
        joint_state = np.array([*joint_pos_L, *joint_pos_R])

        env_rgb = self.env_camera.get_rgb_graph(save_or_not=False)
        env_right_rgb = self.env_right_camera.get_rgb_graph(save_or_not=False)
        env_left_rgb = self.env_left_camera.get_rgb_graph(save_or_not=False)
        env_back_rgb = self.env_back_camera.get_rgb_graph(save_or_not=False)

        env_depth = self.env_camera.get_depth_graph()
        env_right_depth = self.env_right_camera.get_depth_graph()
        env_left_depth = self.env_left_camera.get_depth_graph()
        env_back_depth = self.env_back_camera.get_depth_graph()

        env_point_cloud = self.env_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        env_right_point_cloud = self.env_right_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        env_left_point_cloud = self.env_left_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        env_back_point_cloud = self.env_back_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        
        end_pose_left, end_ori_left = self.bimanual_dex.dexleft.get_cur_ee_pos()
        end_pose_right, end_ori_right = self.bimanual_dex.dexright.get_cur_ee_pos()
        end_state_left = np.array([self.bimanual_dex.dexleft_gripper_state])
        end_state_right = np.array([self.bimanual_dex.dexright_gripper_state])

        obs = { 
            "joint_state": joint_state,
            "end_state": np.concatenate([end_pose_left, end_ori_left, end_state_left, end_pose_right, end_ori_right, end_state_right]),
            "camera_intrinsics": self.env_camera_intrinsics,
            "camera_extrinsics": self.env_camera_extrinsics,

            "env_image": env_rgb,
            "env_right_image": env_right_rgb,
            "env_left_image": env_left_rgb,
            "env_back_image": env_back_rgb,

            "env_depth": env_depth,
            "env_right_depth": env_right_depth,
            "env_left_depth": env_left_depth,
            "env_back_depth": env_back_depth,

            "env_point_cloud": env_point_cloud,
            "env_right_point_cloud": env_right_point_cloud,
            "env_left_point_cloud": env_left_point_cloud,
            "env_back_point_cloud": env_back_point_cloud,

            "active_object_point_cloud": self.garment_pcd,
            "passive_object_point_cloud": self.object_pcd,
            "active_object_point_feature": self.points_affordance_feature,
            "passive_object_point_feature": self.object_feature,
        }
        
        return obs
        
    def pre_grasp(self, flag):
        # get manipulation points from GAM Model
        manipulation_points, indices, points_similarity = self.policy.model_GAM.get_manipulation_points(input_pcd=self.garment_pcd, index_list=[1021, 1819])
            
        self.points_affordance_feature = normalize_columns(points_similarity.T)

        self.object_feature, _ = self.policy.model_Uni3D.get_manipulation_points(input_pcd=self.object_pcd)

        # get lift height
        y_min = self.garment_pcd[np.argmin(self.garment_pcd[:, 1])]
        y_max = self.garment_pcd[np.argmax(self.garment_pcd[:, 1])]
        garment_length=y_max[1]-y_min[1]
        lift_height=garment_length*1.1+0.45

        cprint(f"lift height: {lift_height}", "blue")
        
        manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment
        
        if flag:
            # move both dexhand to the manipulation points
            self.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))


        
        
# if __name__=="__main__":
def StoreTops_Validation(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, ground_material_usd, validation_flag, record_video_flag, training_data_num, checkpoint_num, stage_str, policy_name):

    env = StoreTops_Env(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, ground_material_usd, record_video_flag, training_data_num, checkpoint_num, stage_str, policy_name)

    env.garment.particle_material.set_gravity_scale(0.07)

    # env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/img",".png"))
    # point_cloud = env.env_camera.get_pointcloud_from_depth()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # save_path = get_unique_filename("Data/Store_Tops/pcd",".ply")
    # o3d.io.write_point_cloud(save_path, pcd)
    # env.env_camera.get_depth_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/depth",".png"))
    # intrinsics_matrix = env.env_camera.camera.get_intrinsics_matrix()
    # view_matrix = env.env_camera.camera.get_view_matrix_ros()
    # pcd_image = env.env_camera.camera.get_image_coords_from_world_points(point_cloud[:, :3])
    # matrix_path = get_unique_filename("Data/Store_Tops/intrinsics_matrix", ".npy")
    # np.save(matrix_path, intrinsics_matrix)
    # view_matrix_path = get_unique_filename("Data/Store_Tops/view_matrix", ".npy")
    # np.save(view_matrix_path, view_matrix)
    # pcd_image_path = get_unique_filename("Data/Store_Tops/pcd_image", ".npy")
    # np.save(pcd_image_path, pcd_image)
    # print("全部保存完毕！")
    # exit()
    
    # env.env_right_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/img_right",".png"))
    # env.env_left_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/img_left",".png"))
    # env.env_back_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/img_back",".png"))
    # hide prim to get object point cloud
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    env.object_pcd, color = env.object_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        sampled_point_num=2048,
        real_time_watch=False,
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Bag"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        sampled_point_num=2048,
        real_time_watch=False,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Bag"],
        visible=True,
    )

    for i in range(50):
        env.step()
        
    if record_video_flag:
        env.thread_record.start()
        
    env.pre_grasp(flag=(stage_str == "remain"))

    if policy_name != "TriDFA":
        step_lim = 16 + (6 if stage_str == "all" else 0)

        for i in range(step_lim):
            
            print(f"Stage_1_Step: {i}")
            
            obs = env.get_obs()

            action=env.policy.get_action(obs)
            
            # print("action_shape:",action.shape)
            
            for j in range(4):
                
                action_L = ArticulationAction(joint_positions=action[j][:30])
                action_R = ArticulationAction(joint_positions=action[j][30:])

                env.bimanual_dex.dexleft.apply_action(action_L)
                env.bimanual_dex.dexright.apply_action(action_R)
                
                for _ in range(5):    
                    env.step()
                
                obs = env.get_obs()

                env.policy.update_obs(obs)
    else:
        step_lim = 22

        for i in range(step_lim):
            
            print(f"Stage_1_Step: {i}")
            
            obs = env.get_obs()

            action=env.policy.get_action(obs)
            
            end_state_L = action[:8]
            end_state_R = action[8:]
            print(env.bimanual_dex.dexleft.get_cur_ee_pos(), env.bimanual_dex.dexright.get_cur_ee_pos())
            print("end_state_L:", end_state_L)
            print("end_state_R:", end_state_R)
            if env.bimanual_dex.dexleft_gripper_state == 1 or env.bimanual_dex.dexright_gripper_state == 1:
                # end_state_L[0] = end_state_L[0] - 0.01
                end_state_L[1] = end_state_L[1] - 0.01
                end_state_L[2] = end_state_L[2] - 0.03
                # end_state_R[0] = end_state_R[0] + 0.01
                end_state_R[1] = end_state_R[1] - 0.01
                end_state_R[2] = end_state_R[2] - 0.03
            env.bimanual_dex.dense_move_both_ik_val(
                left_pos=end_state_L[:3],
                left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                right_pos=end_state_R[:3],
                right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
            )
            gripper_state_L = round((end_state_L[7] + end_state_R[7]) / 2)
            gripper_state_R = round((end_state_L[7] + end_state_R[7]) / 2)
            cur_gripper_state_L = env.bimanual_dex.dexleft_gripper_state
            cur_gripper_state_R = env.bimanual_dex.dexright_gripper_state
            if gripper_state_L != cur_gripper_state_L or gripper_state_R != cur_gripper_state_R:
                print("change gripper state!", gripper_state_L, gripper_state_R)
                env.bimanual_dex.set_both_hand_state(
                    "close" if gripper_state_L == 0 else "open",
                    "close" if gripper_state_R == 0 else "open",
                )
    
    for i in range(100):
        env.step()

    env.garment.particle_material.set_gravity_scale(0.1)

    garment_pose = env.garment.get_garment_center_pos()
    
    bag_max_x, bag_max_y, bag_max_z, bag_min_x, bag_min_y, bag_min_z = env.get_prim_dimensions(env.bag_prim_path)
    
    success = bag_max_x > garment_pose[0] and bag_min_x < garment_pose[0] \
                and bag_max_y > garment_pose[1] and bag_min_y < garment_pose[1] \
                and bag_max_z > garment_pose[2] and bag_min_z < garment_pose[2]

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"garment_center_pos: {env.garment.get_garment_center_pos()}", "blue")
    cprint(f"bag_box: x[{bag_min_x}, {bag_max_x}] y[{bag_min_y}, {bag_max_y}] z[{bag_min_z}, {bag_max_z}]", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")

    if record_video_flag:
        if not os.path.exists(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video"):
            os.makedirs(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video")
        env.env_camera.create_mp4(get_unique_filename(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video/video", ".mp4"))
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if validation_flag:
        with open(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/validation_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")
        if not os.path.exists(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic"):
            os.makedirs(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename(f"Data/Store_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic/img",".png"))


if __name__=="__main__":
    
    args = parse_args_val()
    
    # initial setting
    pos = np.array([0, 0.7, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.1
    env_dy = 0.0 
    env_dz = 0.0
    theta = 20.0

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(args.seed)
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.15, 0.15) # changeable
            env_dy = np.random.uniform(-0.05, 0.05) # changeable
            env_dz = np.random.uniform(-0.05, 0.05) # changeable
            theta = np.random.uniform(0.0, 0.0) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.5, 0.8) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"FoundationModels/GAM/checkpoints/Tops_NoSleeve/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    StoreTops_Validation(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, args.ground_material_usd, args.validation_flag, args.record_video_flag, args.training_data_num, args.checkpoint_num, args.stage_str, args.policy_name)

    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()