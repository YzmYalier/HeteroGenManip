from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

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
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud,compute_similarity
from Env_Config.Robot.Franka import Franka
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models
from FoundationModels.Uni3D.Uni3D_Encapsulation import Uni3D_Encapsulation

import json

class StoreMug_Env(BaseEnv):
    def __init__(
        self,
        closet_usd_path: str,  
        ground_material_usd: str = None,  # 地面材质路径 "Assets/Material/Floor/Fabric001.usd"
        closet_pos: np.array = None,
        rigid_pos: np.array = None,
        rigid_usd: str = None,
        rigid_ori: np.array = None,
        rigid_scale: np.array = np.array([0.15, 0.15, 0.15]),
        record_video_flag: bool = False
    ):
        super().__init__()
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        # ---------------------- 场景资产添加 ---------------------- #
        self.franka = Franka(
            self.world, 
            position=np.array([0.2, 1, 0]),
            orientation=np.array([0.0, 0.0, -90.0]),
            robot_name="Franka")

        # link0,1,2,3  "/World/Franka/panda_link0/panda_joint1"
        self.closet_prim_path = "/World/Closet"
        add_reference_to_stage(
            usd_path=closet_usd_path,  
            prim_path=self.closet_prim_path
        )
        self.closet = SingleXFormPrim(
            prim_path=self.closet_prim_path,
            position=closet_pos,
            orientation=euler_angles_to_quat(np.array([0.0,0.0,-90]), degrees=True),
            scale=np.array([0.5, 0.5, 0.5])
            )

        self.physics_material = PhysicsMaterial(
            prim_path="/World/Physics/Material",
            name="physics_material",
            static_friction=2.5,    # 静摩擦系数
            dynamic_friction=2.5,  # 动摩擦系数
            restitution=0.0       # 弹性系数
        )
        
        # self.closet.apply_visya_material(physics_material)
        # self.franka.apply_physics_material(physics_material)
        self.stage = self.closet.prim.GetStage()  
        closet_prim = self.stage.GetPrimAtPath(self.closet_prim_path)
        physicsUtils.add_physics_material_to_prim(self.stage, closet_prim, "/World/Physics/Material")
        # link_2_prim = stage.GetPrimAtPath("/World/Closet/link_2")
        # self.setup_friction_properties(link_2_prim)
        self.set_mass(stage=self.stage, prim=closet_prim, mass=15.0)  # 设置橱柜质量为10kg
        self.set_revolute_joint_damping(stage=self.stage, joint_path="/World/Closet/link_0/joint_1", damping_value=3, stiffness_value=0.5)  # 修复关节驱动
        self.set_revolute_joint_damping(stage=self.stage, joint_path="/World/Closet/link_0/joint_2", damping_value=3, stiffness_value=0.5)
        # self.set_revolute_joint_damping(stage=stage, joint_path="/World/Closet/link_3/joint_2", damping_value=3)
        # joint_prim = self.stage.GetPrimAtPath("/World/Closet/link_0/joint_1")
       
        # # 验证关节存在性
        # if joint_prim.IsValid():       
        #     # 创建或修改阻尼属性
        #     pos_attr = joint_prim.GetAttribute("drive:linear:physics:targetPosition")
        #     v_attr = joint_prim.GetAttribute("drive:linear:physics:targetVelocity")
        #     pos_attr.Set(0.15)
        #     v_attr.Set(0.1)
            
            
        # self.cube = self.scene.add(
        #                 DynamicCuboid(
        #                     prim_path="/World/Cube",
        #                     name="target_cube",
        #                     position=rigid_pos,
        #                     size=0.05,
        #                     color=np.array([1.0, 0.0, 0.0]),
        #                     scale=np.array([1, 3, 1])
        #                 )
        #             )
        add_reference_to_stage(
            usd_path=rigid_usd,  
            prim_path="/World/rigid"
        )
        self.rigid = SingleRigidPrim(
            prim_path="/World/rigid",
            position=rigid_pos,
            orientation=euler_angles_to_quat(rigid_ori, degrees=True),
            scale=rigid_scale,
            mass=0.001,
            # density=0.1
            )
        rigid_prim = self.stage.GetPrimAtPath("/World/rigid")

        # 为资产添加碰撞体
        collision_api = UsdPhysics.CollisionAPI.Apply(rigid_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(rigid_prim)
        # mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")
        # collision_api.GetCollisionEnabledAttr().Set(True)
        # from pxr import PhysxSchema 
        # collision_schema = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(rigid_prim) 
        # collision_schema.CreateHullVertexLimitAttr().Set(64) 
        # collision_schema.CreateMaxConvexHullsAttr().Set(64) 
        # collision_schema.CreateMinThicknessAttr().Set(0.001) 
        # collision_schema.CreateShrinkWrapAttr().Set(True) 
        # collision_schema.CreateErrorPercentageAttr().Set(0.1) 
        mesh_collision_api.CreateApproximationAttr().Set("sdf")
        collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(rigid_prim)
        collision_api.CreateSdfResolutionAttr().Set(1024)
        
        self.physics_material2 = PhysicsMaterial(
            prim_path="/World/Physics/Material2",
            name="physics_material2",
            static_friction=0.05,    # 静摩擦系数
            dynamic_friction=0.05,   # 动摩擦系数
            restitution=0.05        # 弹性系数
        )
        # self.stage = self.rigid.prim.GetStage()  
        # rigid_prim = self.stage.GetPrimAtPath("/World/rigid")
        physicsUtils.add_physics_material_to_prim(self.stage, rigid_prim, "/World/Physics/Material2")
        
        self.scene.add(
            FixedCuboid(
                name="rigid_helper",
                position=[-0.37, 0.8, 0.2],
                prim_path="/World/rigid_helper",
                scale=np.array([0.4,0.4,0.001]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        self.rigid_helper_path=["/World/rigid_helper"]
        self.helper_path=['/World/defaultGroundPlane/GroundPlane', '/World/Closet']
        self.collisiongroup = CollisionGroup(
            self.world,
            helper_path=self.helper_path,
            garment=False,
            collide_with_garment=True,
            collide_with_robot=True,
        )
        self.collisiongroup.add_collision(group_path="rigid_helper", target=self.rigid_helper_path, garment=True, helper=False, robot=False)
        # self.path=['/World/Franka']
        # self.collisiongroup.add_collision(group_path="hanger", target=self.hanger_path, garment=True, helper=False, robot=False)
        # # 4. 相机（用于记录过程）
        self.env_camera = Recording_Camera(
            camera_position=np.array([-2.8, 3.05, 3.05]),
            camera_orientation=np.array([-10, 37, -45.0]),
            prim_path="/World/SceneCamera"
        )
        
        # self.garment_camera = Recording_Camera(
        #     camera_position=np.array([hat_pos[0], hat_pos[1], 5]),
        #     camera_orientation=np.array([0, 90.0, 90.0]),
        #     prim_path="/World/garment_camera",
        # )

        self.rigid_camera = Recording_Camera(
            camera_position=np.array([-2.8, 3.05, 3.05]),
            camera_orientation=np.array([-10, 37, -45.0]),
            prim_path="/World/RigidCamera"
        )
        
        self.articulated_camera = Recording_Camera(
            camera_position=np.array([-2.8, 3.05, 3.05]),
            camera_orientation=np.array([-10, 37, -45.0]),
            prim_path="/World/ArticulatedCamera"
        )
        
        self.garment_pcd = None
        self.rigid_pcd = None
        self.articulated_pcd = None
        self.points_affordance_feature = None
        self.rigid_point_feature = None
        self.articulated_point_feature = None

        self.rigid_model = Uni3D_Encapsulation(demo_pcd_path="FoundationModels/Uni3D/demo_pcds/mug.ply")

        # 初始化场景
        self.reset()
        # self.garment.set_pose(pos=np.array([hat_pos[0], hat_pos[1], 0.1]),ori=np.array([0.0, 0.0, 0.0]))

        scenePrim = self.stage.GetPrimAtPath("/physicsScene")
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scenePrim)
        physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10*1024)
        
        self.env_camera.initialize(
            depth_enable=True)  # 启用深度和RGB
        # self.franka._robot.gripper.set_joint_positions(self.franka._robot.gripper.joint_opened_positions)
        # self.garment_camera.initialize(
        #     segment_pc_enable=True, 
        #     segment_prim_path_list=[
        #         "/World/Garment/garment",
        #     ]
        # )
        self.rigid_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/rigid",
            ]
        )
        self.articulated_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Closet",
            ]
        )

        self.env_camera_intrinsics = self.env_camera.camera.get_intrinsics_matrix()
        self.env_camera_extrinsics = self.env_camera.camera.get_view_matrix_ros()

        self.record_video_flag = record_video_flag
        if self.record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
            
        for i in range(100):
            self.step()
        cprint("Closet Environment Ready!", "green", "on_green")
        
    def setup_friction_properties(self, prim, static_friction=20, dynamic_friction=20, restitution=0.0):
        # 确保prim有物理场景
        stage = prim.GetStage()

        # 创建物理材质路径
        material_path = prim.GetPath().AppendChild("physicsMaterial")
        
        # 定义材质
        material = UsdShade.Material.Define(stage, material_path)
        
        # 应用物理材质API
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        
        # 设置摩擦和弹性属性
        material_api.CreateStaticFrictionAttr().Set(static_friction)
        material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
        material_api.CreateRestitutionAttr().Set(restitution)
        
        # 使用已有的工具函数将材质应用到基元
        
        physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
        
    def set_mass(self, stage, prim: Usd.Prim, mass:float):
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(mass)
        else:
            mass_api.GetMassAttr().Set(mass)

    def set_revolute_joint_damping(self, stage, joint_path: str, damping_value: float, stiffness_value = 0.5):
        """
        修改RevoluteJoint的阻尼参数
        :param joint_path: 关节在USD场景中的路径（例如"/World/Robot/rear_left_joint"）
        :param damping_value: 目标阻尼值（推荐范围：0.01 - 1e5）
        :return: 是否修改成功
        """
        # 获取当前场景上下文
        # stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(joint_path)
        
        # 验证关节存在性
        if not joint_prim.IsValid():
            print(f"❌ 错误：关节路径 {joint_path} 不存在")
            return False
            
        # 验证关节类型
        # if not joint_prim.GetTypeName() == "PhysicsRevoluteJoint":
        #     print(f"⚠️ 警告：{joint_path} 不是RevoluteJoint类型")
        #     return False
        
        # 创建或修改阻尼属性
        damping_attr = joint_prim.GetAttribute("drive:linear:physics:damping")
        stiffness_attr = joint_prim.GetAttribute("drive:linear:physics:stiffness")
        # 设置新阻尼值（带范围验证）
        valid_value = np.clip(damping_value, 0.001, 1e6)  # 限制合理范围
        damping_attr.Set(valid_value)
        stiffness_attr.Set(stiffness_value)
        
        return True
    
    def set_revolute_joint(self, joint_path: str, damping_value: float, stiffness_value: float):
        # 获取当前场景上下文
        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(joint_path)
        
        # 验证关节存在性
        if not joint_prim.IsValid():
            print(f"❌ 错误：关节路径 {joint_path} 不存在")
            return False
            
        # 验证关节类型
        # if not joint_prim.GetTypeName() == "PhysicsRevoluteJoint":
        #     print(f"⚠️ 警告：{joint_path} 不是RevoluteJoint类型")
        #     return False
        
        # 创建或修改阻尼属性
        damping_attr = joint_prim.GetAttribute("drive:angular:physics:damping")
        stiffness_attr = joint_prim.GetAttribute("drive:angular:physics:stiffness")
        
        # 设置新值
        damping_attr.Set(damping_value)
        stiffness_attr.Set(stiffness_value)
        
        return True
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
    
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos = self.franka.get_joint_positions()
            
            joint_state = np.array([*joint_pos])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            depth = self.env_camera.get_depth_graph()

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )

            end_pose, end_ori = self.franka.get_cur_ee_pos()
            end_state = np.array([self.franka.gripper_state])
            
            self.saving_data.append({ 
                "joint_state": joint_state,
                "end_state": np.concatenate([end_pose, end_ori, end_state]),
                "env_image": rgb,
                "env_depth": depth,
                "camera_intrinsics": self.env_camera_intrinsics,
                "camera_extrinsics": self.env_camera_extrinsics,
                "env_point_cloud": point_cloud,
                "active_object_point_cloud":self.rigid_pcd,
                "passive_object_point_cloud":self.articulated_pcd,
                "active_object_point_feature": self.rigid_point_feature,
                "passive_object_point_feature": self.articulated_point_feature,
            })
        
        self.step_num += 1

def StoreMug(rigid_pos, rigid_ori, rigid_usd, rigid_scale, closet_pos, closet_usd, data_collection_flag, record_video_flag):

    env = StoreMug_Env(
        closet_usd_path=closet_usd,
        # hat_pos=np.array([-0.2, 1.1, 0.1]),
        # rigid_pos=np.array([-0.5, 1.0, 0.5]),
        closet_pos=closet_pos,
        rigid_pos=rigid_pos,
        rigid_usd=rigid_usd,
        rigid_ori=rigid_ori,
        rigid_scale=rigid_scale,
        record_video_flag=record_video_flag
    )

    env.garment_pcd = np.zeros((2048, 3))
    env.points_affordance_feature = np.zeros((2048,2))
    
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/Closet"],
        visible=False,
    )
    for i in range(50):
        env.step()

    env.rigid_pcd, color = env.rigid_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        sampled_point_num=512,
        save_path=get_unique_filename("rigid", extension=".ply"),
        # real_time_watch=True,
    )

    env.rigid_point_feature, manipulation_points = env.rigid_model.get_manipulation_points(env.rigid_pcd, [344, 363, 257])
    
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/Closet"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/rigid"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.articulated_pcd, color = env.articulated_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        sampled_point_num=512,
        save_path=get_unique_filename("data1", extension=".ply"),
        # real_time_watch=True,
    )

    env.articulated_point_feature, _ = env.rigid_model.get_manipulation_points(env.articulated_pcd)

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/rigid"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # 开柜子  
    # -----------------------45746----------------------------
    # z=0.47

    joint_prim = env.stage.GetPrimAtPath("/World/Closet/link_0/joint_1")
    if joint_prim.IsValid():
        # 创建或修改阻尼属性
        type_attr = joint_prim.GetAttribute("drive:linear:physics:type")
        type_attr.Set("acceleration")
        # pos_attr = joint_prim.GetAttribute("state:linear:physics:position")
        v_attr = joint_prim.GetAttribute("drive:linear:physics:targetVelocity")
        # pos_attr.Set(0.45)
        v_attr.Set(0.1)
        
    env.franka.open_gripper()
    for i in range(180):
        env.step()
    v_attr.Set(0.0)
    env.set_revolute_joint_damping(stage=env.stage, joint_path="/World/Closet/link_0/joint_1", damping_value=1000000)  
    env.set_revolute_joint_damping(stage=env.stage, joint_path="/World/Closet/link_0/joint_2", damping_value=1000000)

    prim = env.stage.GetPrimAtPath("/World/Closet/link_1")
    
    xform = UsdGeom.Xformable(prim)
    
    # 获取translate操作
    translate_op = xform.GetLocalTransformation()
    pose = translate_op.ExtractTranslation()
    if pose[0] < -0.50 or pose[0] > -0.40:
        print("Articulation error")
        exit(0)

    # import open3d as o3d
    # # For DinoV2
    # env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Mug/img",".png"))
    # point_cloud = env.env_camera.get_pointcloud_from_depth()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # save_path = get_unique_filename("Data/Store_Mug/pcd",".ply")
    # o3d.io.write_point_cloud(save_path, pcd)
    # env.env_camera.get_depth_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Mug/depth",".png"))
    # intrinsics_matrix = env.env_camera.camera.get_intrinsics_matrix()
    # view_matrix = env.env_camera.camera.get_view_matrix_ros()
    # pcd_image = env.env_camera.camera.get_image_coords_from_world_points(point_cloud[:, :3])
    # matrix_path = get_unique_filename("Data/Store_Mug/intrinsics_matrix", ".npy")
    # np.save(matrix_path, intrinsics_matrix)
    # view_matrix_path = get_unique_filename("Data/Store_Mug/view_matrix", ".npy")
    # np.save(view_matrix_path, view_matrix)
    # pcd_image_path = get_unique_filename("Data/Store_Mug/pcd_image", ".npy")
    # np.save(pcd_image_path, pcd_image)
    # print("全部保存完毕！")
    # exit()

    # 方块
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/rigid"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.articulated_pcd, color = env.articulated_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        sampled_point_num=512,
        save_path=get_unique_filename("ppa", extension=".ply"),
        # real_time_watch=True,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/rigid"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    # rigid_x_max, rigid_y_max, rigid_z_max, rigid_x_min, rigid_y_min, rigid_z_min = env.get_prim_dimensions("/World/rigid")
    rigid_position, _ = env.rigid.get_world_pose()
    # print(rigid_z_max)   

    if record_video_flag:
        env.thread_record.start() 

    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Store_Mug", stage_str="all")
    length = 0.08
    rigid_x, rigid_y, rigid_z = rigid_position
    theta = rigid_ori[2]
    theta_old = 90-theta
    theta = np.radians(90 - theta)
    # env.franka.open_gripper()

    # print(manipulation_points)

    mug_handler_direction = np.append((manipulation_points[2] - (manipulation_points[0] + manipulation_points[1]) / 2.0)[:2], 0.0)
    # print("mug_handler_direction: ", mug_handler_direction)
    mug_handler_direction /= np.linalg.norm(mug_handler_direction, ord=2)
    theta_ = np.arctan2(mug_handler_direction[1], mug_handler_direction[0]) * 180.0 / np.pi
    # print(theta_, 90-theta_old)
    theta_old = 90 - theta_
    # print("mug_handler_direction: ", mug_handler_direction)
    step1_position = manipulation_points[2] + mug_handler_direction * 0.15
    # print("step1_position: ", step1_position)
    step2_position = np.append(((manipulation_points[0] + manipulation_points[1] + manipulation_points[2]) / 3.0)[:2], manipulation_points[2][2])

    env.franka.Rmpflow_Move(target_position=step1_position, target_orientation=np.array([90.0, 90.0, -theta_old]))
    env.franka.Dense_Rmpflow_Move(target_position=step2_position, target_orientation=np.array([90.0, 90.0, -theta_old]), dense_sample_scale=0.005)

    env.franka.close_gripper()

    env.data_split = int(env.step_num / 5)

    for i in range(20):
        env.step()
    delete_prim_group(env.rigid_helper_path)
    env.franka.Rmpflow_Move(target_position=closet_dpos + np.array([-0.05, 0.6, 0.7]), target_orientation=np.array([180.0, 0.0, 90.0]))
    env.franka.Rmpflow_Move(target_position=closet_dpos + np.array([-0.05, 0.44, 0.7]), target_orientation=np.array([180.0, 0.0,90.0]))#90.0, 90.0,0.0
    env.franka.open_gripper()

    if data_collection_flag:
        env.stop_record()
        
    for i in range(50):
        env.step()
    x_max, y_max, z_max, x_min, y_min, z_min = env.get_prim_dimensions("/World/Closet")
    rigid_x, rigid_y, rigid_z = env.rigid.get_world_pose()[0]
    rigid_flag = False
    if rigid_x < x_max and rigid_x > x_min and rigid_y < y_max and rigid_y > y_min:
        rigid_flag = True
    success = rigid_flag
    cprint(f"final result: {success}", color="green", on_color="on_green")
    if record_video_flag and success:
        if not os.path.exists("Data/Store_Mug/video"):
            os.makedirs("Data/Store_Mug/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Store_Mug/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Store_Mug/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  rigid_pos:{rigid_pos}  rigid_ori:{rigid_ori}  rigid_usd:{rigid_usd}  closet_pos:{closet_pos} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz()
            if not os.path.exists("Data/Store_Mug/final_state_pic"):
                os.makedirs("Data/Store_Mug/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Mug/final_state_pic/img",".png"))
    
if __name__ == "__main__":

    args = parse_args_record()

    with open("Assets/mug/mug_9/config.json", "r") as f:
        config = json.load(f)
    # 示例参数（需根据实际路径调整）
    
    CLOSET_URDF_PATH = "Assets/closet.usd"  # 已转换的URDF路径
    np.random.seed(int(time.time()))
    rigid_x = np.random.uniform(-0.45, -0.34)
    rigid_y = np.random.uniform(0.7, 0.9)
    # cup_type = np.random.choice([0, 1, 3])
    # cup_type = 3
    rigid_pos = np.array([rigid_x, rigid_y, 0.4])
    # theta = np.random.uniform(-45,0)
    rigid_ori_x = np.random.uniform(config["ori_x"][0], config["ori_x"][1])
    rigid_ori_y = np.random.uniform(config["ori_y"][0], config["ori_y"][1])
    rigid_ori_z = np.random.uniform(config["ori_z"][0], config["ori_z"][1])
    rigid_ori = np.array([rigid_ori_x, rigid_ori_y, rigid_ori_z])
    theta = rigid_ori_z
    # rigid_ori = np.array([0, 0, theta])
    rigid_scale = np.array(config["scale"])
    rigid_usd = config["usd_path"]

    closet_dx = np.random.uniform(-0.1, 0.05)
    closet_dz = np.random.uniform(-0.1, 0.05)
    closet_dpos = np.array([closet_dx, 0.0, closet_dz])
    closet_pos = closet_dpos + np.array([0.0, 0.0, 0.36])

    StoreMug(
        rigid_pos=rigid_pos,
        rigid_ori=rigid_ori,
        rigid_usd=rigid_usd,
        rigid_scale=rigid_scale,
        closet_pos=closet_pos,
        closet_usd=CLOSET_URDF_PATH,
        data_collection_flag=args.data_collection_flag,
        record_video_flag=args.record_video_flag
    )
        
    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()
