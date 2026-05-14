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
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
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
from Env_Config.Utils_Project.Parse import parse_args_val
# from PA3FF.pointcept.models.SAMPart3D import SAMPart3D
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
from MFMDP.deploy_policy import MFMDP_Encapsulation

import json

class StoreMug_Env(BaseEnv):
    def __init__(
        self,
        closet_usd_path: str,  
        ground_material_usd: str = None,  # 地面材质路径 "Assets/Material/Floor/Fabric001.usd"
        closet_pos: np.array = None,
        closet_ori: np.array = None,
        rigid_pos: np.array = None,
        rigid_usd: str = None,
        rigid_ori: np.array = None,
        rigid_scale: np.array = np.array([0.015, 0.015, 0.015]),
        record_video_flag: bool = False,
        training_data_num:int=100,
        checkpoint_num:int=1500,
        stage_str:str="all",
        policy_name:str="MFMDP",
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
            orientation=euler_angles_to_quat(closet_ori, degrees=True),
            scale=np.array([0.5, 0.25, 0.5])
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
        add_reference_to_stage(
            usd_path=rigid_usd,  
            prim_path="/World/rigid"
        )
        self.rigid_pose = rigid_pos
        self.rigid = SingleRigidPrim(
            prim_path="/World/rigid",
            position=rigid_pos,
            orientation=euler_angles_to_quat(rigid_ori, degrees=True),
            scale=rigid_scale,
            mass=0.0001,
            # density=0.1
            )
        rigid_prim = self.stage.GetPrimAtPath("/World/rigid")

        # 为资产添加碰撞体
        collision_api = UsdPhysics.CollisionAPI.Apply(rigid_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(rigid_prim)
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
        
        self.rigid_pcd = None
        self.articulated_pcd = None
        self.points_affordance_feature = None
        self.rigid_point_feature = None
        self.articulated_point_feature = None
        
        input_dict = {
            "task_name": "Store_Mug",
            "stage_str": stage_str,
            "data_num": training_data_num,
            "checkpoint_num": checkpoint_num,
            "active_object_n_component": 5,
            "passive_object_n_component": 5,
            "category": "Tops_FrontOpen",
            "demo_pcd_path": "FoundationModels/Uni3D/demo_pcds/mug.ply",
        }
        self.policy = getattr(sys.modules[__name__], f"{policy_name}_Encapsulation")(init_dict=input_dict)

        # 初始化场景
        self.reset()
        # self.garment.set_pose(pos=np.array([hat_pos[0], hat_pos[1], 0.1]),ori=np.array([0.0, 0.0, 0.0]))

        scenePrim = self.stage.GetPrimAtPath("/physicsScene")
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scenePrim)
        physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10*1024)
        
        self.env_camera.initialize(depth_enable=True)  # 启用深度和RGB
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

        self.record_video_flag = record_video_flag
        if self.record_video_flag:
            self.record_thread = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.record_thread.daemon = True
            
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

    def check_rigid_pushed(self):
        cur_pose = self.rigid.get_world_pose()[0]
        if abs(cur_pose[2] - self.rigid_pose[2]) < 0.05:
            # 在地面
            if np.linalg.norm(cur_pose[:2] - self.rigid_pose[:2]) > 0.05:
                # 在地面被推动
                return True
        return False

    def get_obs(self):
        
        joint_state = self.franka.get_joint_positions()

        env_rgb = self.env_camera.get_rgb_graph(save_or_not=False)

        env_depth = self.env_camera.get_depth_graph()

        env_point_cloud = self.env_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )

        obs = { 
            "joint_state": joint_state,

            "env_image": env_rgb,

            "env_depth": env_depth,

            "env_point_cloud": env_point_cloud,

            "active_object_point_cloud": self.rigid_pcd,
            "passive_object_point_cloud": self.articulated_pcd,
            "active_object_point_feature": self.rigid_point_feature,
            "passive_object_point_feature": self.articulated_point_feature,
        }
        
        return obs

    def pre_grasp(self, flag):
        self.rigid_point_feature, manipulation_points = self.policy.model_Uni3D.get_manipulation_points(self.rigid_pcd, [344, 363, 257])
        self.articulated_point_feature, _ = self.policy.model_Uni3D.get_manipulation_points(self.articulated_pcd)
        mug_handler_direction = np.append((manipulation_points[2] - (manipulation_points[0] + manipulation_points[1]) / 2.0)[:2], 0.0)
        mug_handler_direction /= np.linalg.norm(mug_handler_direction, ord=2)
        theta_ = np.arctan2(mug_handler_direction[1], mug_handler_direction[0]) * 180.0 / np.pi
        theta_old = 90 - theta_
        step1_position = manipulation_points[2] + mug_handler_direction * 0.15
        step2_position = np.append(((manipulation_points[0] + manipulation_points[1] + manipulation_points[2]) / 3.0)[:2], manipulation_points[2][2])
        print(manipulation_points)

        if flag:
            self.franka.Rmpflow_Move(target_position=step1_position, target_orientation=np.array([90.0, 90.0, -theta_old]))
            self.franka.Dense_Rmpflow_Move(target_position=step2_position, target_orientation=np.array([90.0, 90.0, -theta_old]), dense_sample_scale=0.005)

            self.franka.close_gripper()

def Store_Mug_Validation(active_pos, passive_pos, active_ori, passive_ori, active_usd_path, rigid_scale, ground_material_usd, validation_flag, record_video_flag, training_data_num, checkpoint_num, stage_str, policy_name):
    env = StoreMug_Env(
        closet_usd_path="Assets/closet.usd",
        # hat_pos=np.array([-0.2, 1.1, 0.1]),
        # rigid_pos=np.array([-0.5, 1.0, 0.5]),
        closet_pos=passive_pos + np.array([0.0, 0.0, 0.36]),
        closet_ori=passive_ori,
        rigid_pos=active_pos,
        rigid_usd=active_usd_path,
        rigid_ori=active_ori,
        rigid_scale=rigid_scale,
        ground_material_usd=ground_material_usd,
        record_video_flag=record_video_flag,
        training_data_num=training_data_num,
        checkpoint_num=checkpoint_num,
        stage_str=stage_str,
        policy_name=policy_name,
    )
    if record_video_flag:     
        env.record_thread.start()

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
        # return False
        exit(100)
    # return True

    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/Closet"],
        visible=False,
    )
    for i in range(50):
        env.step()

    env.rigid_pcd, color = env.rigid_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        sampled_point_num=512,
        save_path=get_unique_filename("data1", extension=".ply"),
        # real_time_watch=True,
    )

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
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/Franka", "/World/rigid"],
        visible=True,
    )
    for i in range(50):
        env.step()

    env.pre_grasp(flag=(stage_str == "remain"))

    step_lim = 40 + (30 if stage_str == "all" else 0)

    for i in range(step_lim):
        
        print(f"Stage_1_Step: {i}")
        
        obs = env.get_obs()
        actions = env.policy.get_action(obs)

        for j in range(4):
            
            action = ArticulationAction(joint_positions=actions[j])
            env.franka.apply_action(action)
            
            for _ in range(5):    
                env.step()
                
            obs = env.get_obs()
            
            env.policy.update_obs(obs)
    
    # 将rigid重力加大
    env.rigid.set_mass(0.01)

    for i in range(50):
        env.step()
    x_max, y_max, z_max, x_min, y_min, z_min = env.get_prim_dimensions("/World/Closet")
    rigid_x, rigid_y, rigid_z = env.rigid.get_world_pose()[0]
    rigid_flag = False
    if rigid_x < x_max and rigid_x > x_min and rigid_y < y_max and rigid_y > y_min:
        rigid_flag = True
    success = rigid_flag
    if record_video_flag:
        if not os.path.exists(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video"):
            os.makedirs(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video")
        env.env_camera.create_mp4(get_unique_filename(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video/video", ".mp4"))
    if validation_flag:
        if not os.path.exists(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic"):
            os.makedirs(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic/img",".png"))
        with open(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/validation_log.txt", "a") as f:
            f.write(f"result:{success} \n")
        with open(f"Data/Store_Mug_ood_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/pose_log.txt", "a") as f:
            f.write(f"{(rigid_pos[0] + 0.45) / 0.11} {(rigid_pos[1] - 0.7) / 0.2} {(rigid_ori_z - config['ori_z'][0]) / (config['ori_z'][1] - config['ori_z'][0])} {(closet_dx + 0.1) / 0.15} {(closet_dz + 0.1) / 0.15} {success}\n")
        simulation_app.close()
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()

if __name__=="__main__":
    args = parse_args_val()
    seed = int(time.time())
    mug_list = [0, 2, 3, 4, 5, 6, 9]
    if args.env_random_flag:
        np.random.seed(seed)
        mug_id = np.random.choice(mug_list, 1)[0]
        print("mug_id: ", mug_id)
        with open("Env_Validation/Mugs.txt", "a") as f:
            f.write(f"{mug_id}\n")
        exit()
        with open(f"Assets/mug/mug_{mug_id}/config.json", "r") as f:
            config = json.load(f)
        rigid_x = np.random.uniform(-0.45, -0.34)
        rigid_y = np.random.uniform(0.7, 0.9)
        rigid_pos = np.array([rigid_x, rigid_y, 0.3])
        # theta = np.random.uniform(-45,0)
        rigid_ori_x = np.random.uniform(config["ori_x"][0], config["ori_x"][1])
        rigid_ori_y = np.random.uniform(config["ori_y"][0], config["ori_y"][1])
        rigid_ori_z = np.random.uniform(config["ori_z"][0], config["ori_z"][1])
        rigid_ori = np.array([rigid_ori_x, rigid_ori_y, rigid_ori_z])
        theta = rigid_ori_z
        rigid_scale = np.array(config["scale"])
        rigid_usd = config["usd_path"]
        closet_dx = np.random.uniform(-0.1, 0.05)
        closet_dz = np.random.uniform(-0.1, 0.05)
        closet_dpos = np.array([closet_dx, 0.0, closet_dz])
        closet_dori = np.random.uniform(-0, 0)
        closet_ori = np.array([0.0, 0.0, closet_dori - 90.0])  # 橱柜的初始朝向

    Store_Mug_Validation(
        active_pos=rigid_pos,
        passive_pos=closet_dpos,
        active_ori=rigid_ori,
        passive_ori=closet_ori,
        active_usd_path=rigid_usd,
        rigid_scale=rigid_scale,
        ground_material_usd=args.ground_material_usd,
        validation_flag=args.validation_flag,
        record_video_flag=args.record_video_flag,
        training_data_num=args.training_data_num,
        checkpoint_num=args.checkpoint_num,
        stage_str=args.stage_str,
        policy_name=args.policy_name,
    )

    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()