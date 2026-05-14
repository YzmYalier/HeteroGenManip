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

# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Room.Object_Tools import hanger_load, set_prim_visible_group, delete_prim_group, dynamic_hanger_load
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models

from FoundationModels.Uni3D.Uni3D_Encapsulation import Uni3D_Encapsulation
class HangTops_Env(BaseEnv):
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
            usd_path="Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top603/TNLC_Top603_obj.usd" if usd_path is None else usd_path,
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
            dexleft_pos=np.array([-0.8, 0.0, 0.6]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.6]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, -3.0, 6.75]), 
            camera_orientation=np.array([0, 60.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 6.65, 4.0]),
            camera_orientation=np.array([0, 30.0, -90.0]),
            prim_path="/World/env_camera",
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
        
        # load GAM Model
        self.model_GAM = GAM_Encapsulation(catogory="Tops_LongSleeve")

        self.model_Uni3D = Uni3D_Encapsulation()

        # load hanger
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.hanger_center = hanger_load(self.scene, env_dx, env_dy, env_dz, theta)
        
        self.judge_camera = Recording_Camera(
            camera_position=np.array([self.hanger_center[0], 6.0, 0.5]),
            camera_orientation=np.array([0, 0.0, -90.0]),
            prim_path="/World/judge_camera",
        )    
        
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
        
        self.judge_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        self.object_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/hanger1",
                "/World/hanger2",
                "/World/hanger3",
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
    
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            env_rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            env_depth = self.env_camera.get_depth_graph()

            env_point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )

            end_pose_left, end_ori_left = self.bimanual_dex.dexleft.get_cur_ee_pos()
            end_pose_right, end_ori_right = self.bimanual_dex.dexright.get_cur_ee_pos()
            end_state_left = np.array([self.bimanual_dex.dexleft_gripper_state])
            end_state_right = np.array([self.bimanual_dex.dexright_gripper_state])

            self.saving_data.append({ 
                "joint_state": joint_state,
                "end_state": np.concatenate([end_pose_left, end_ori_left, end_state_left, end_pose_right, end_ori_right, end_state_right]),
                "camera_intrinsics": self.env_camera_intrinsics,
                "camera_extrinsics": self.env_camera_extrinsics,

                "env_image": env_rgb,

                "env_depth": env_depth,

                "env_point_cloud": env_point_cloud,

                "active_object_point_cloud": self.garment_pcd,
                "passive_object_point_cloud": self.object_pcd,
                "active_object_point_feature": self.points_affordance_feature,
                "passive_object_point_feature": self.object_feature,
            })
        
        self.step_num += 1
        
        
# if __name__=="__main__":
def HangTops(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, ground_material_usd, data_collection_flag, record_video_flag):
    
    env = HangTops_Env(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, ground_material_usd, record_video_flag)
    
    env.garment.particle_material.set_gravity_scale(0.7)
    
    # For DinoV2
    # env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Hang_Tops/img",".png"))
    # point_cloud = env.env_camera.get_pointcloud_from_depth()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # save_path = get_unique_filename("Data/Hang_Tops/pcd",".ply")
    # o3d.io.write_point_cloud(save_path, pcd)
    # env.env_camera.get_depth_graph(save_or_not=True,save_path=get_unique_filename("Data/Hang_Tops/depth",".png"))
    # intrinsics_matrix = env.env_camera.camera.get_intrinsics_matrix()
    # view_matrix = env.env_camera.camera.get_view_matrix_ros()
    # pcd_image = env.env_camera.camera.get_image_coords_from_world_points(point_cloud[:, :3])
    # matrix_path = get_unique_filename("Data/Hang_Tops/intrinsics_matrix", ".npy")
    # np.save(matrix_path, intrinsics_matrix)
    # view_matrix_path = get_unique_filename("Data/Hang_Tops/view_matrix", ".npy")
    # np.save(view_matrix_path, view_matrix)
    # pcd_image_path = get_unique_filename("Data/Hang_Tops/pcd_image", ".npy")
    # np.save(pcd_image_path, pcd_image)
    # print("全部保存完毕！")
    
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
        # real_time_watch=True,
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/hanger1", "/World/hanger2", "/World/hanger3"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        sampled_point_num=2048,
        # real_time_watch=True,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/hanger1", "/World/hanger2", "/World/hanger3"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    if record_video_flag:
        env.thread_record.start()

    # get manipulation points from GAM Model
    manipulation_points, indices, points_similarity = env.model_GAM.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[838,179])
    
    if abs(manipulation_points[0][0]-manipulation_points[1][0])<0.1:
        print("model false")
        simulation_app.close()
        
    env.points_affordance_feature = normalize_columns(points_similarity.T)

    env.object_feature, _ = env.model_Uni3D.get_manipulation_points(input_pcd=env.object_pcd)
        
    # get lift height
    y_min = env.garment_pcd[np.argmin(env.garment_pcd[:, 1])]
    y_max = env.garment_pcd[np.argmax(env.garment_pcd[:, 1])]
    garment_length=y_max[1]-y_min[1]
    lift_height=env.hanger_center[2]+garment_length/2

    cprint(f"lift height: {lift_height}", "blue")
    
    manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Hang_Tops", stage_str="all")

    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    env.data_split = int(env.step_num / 5)

    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

    dis=np.sqrt((manipulation_points[0][0]-manipulation_points[1][0])**2+(manipulation_points[0][1]-manipulation_points[1][1])**2)/2

    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))

    left_lift_points, right_lift_points = np.array([-0.25*sin_theta-dis*cos_theta+env.hanger_center[0], env.hanger_center[1]-0.25*cos_theta+dis*sin_theta, 0.5]), np.array([-0.25*sin_theta+dis*cos_theta+env.hanger_center[0], env.hanger_center[1]-0.25*cos_theta-dis*sin_theta, 0.5])    # set z-axis to 0.65 to make sure dexhand can lift the garment
    
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    left_lift_points, right_lift_points = np.array([-0.25*sin_theta-dis*cos_theta+env.hanger_center[0], env.hanger_center[1]-0.25*cos_theta+dis*sin_theta, 0.9]), np.array([-0.25*sin_theta+dis*cos_theta+env.hanger_center[0], env.hanger_center[1]-0.25*cos_theta-dis*sin_theta, 0.9])    # set z-axis to 0.65 to make sure dexhand can lift the garment

    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    
    if lift_height >1.3:
        lift_height=1.3
        
    dis = dis + 0.02
    left_lift_points, right_lift_points = np.array([0.10*sin_theta-dis*cos_theta+env.hanger_center[0], env.hanger_center[1]+0.10*cos_theta+dis*sin_theta, lift_height]), np.array([0.10*sin_theta+dis*cos_theta+env.hanger_center[0], env.hanger_center[1]+0.10*cos_theta-dis*sin_theta, lift_height])  
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    ylen = garment_length/2.5
    left_lift_points, right_lift_points = np.array([ylen*sin_theta-dis*cos_theta+env.hanger_center[0], env.hanger_center[1]+ylen*cos_theta+dis*sin_theta, 0.9]), np.array([ylen*sin_theta+dis*cos_theta+env.hanger_center[0], env.hanger_center[1]+ylen*cos_theta-dis*sin_theta, 0.9])  
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
        
    env.garment.particle_material.set_gravity_scale(2.0)
    
    for i in range(100):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(0.7)
    
    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
                
    success=True
    
    pcd_judge, _ = env.judge_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True
    )
    z_values = pcd_judge[:, 2]  # 假设 pcd_judge 的形状是 (N, 3)
    points_below_threshold = np.sum(z_values < 0.01)
    if points_below_threshold > 15:
        success=False
    elif np.sqrt((env.garment.get_garment_center_pos()[0] - env.hanger_center[0])**2 + (env.garment.get_garment_center_pos()[1] - env.hanger_center[1])**2) > 0.04:
        success=False
    else:
        success=True

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"points_below_threshold: {points_below_threshold}", "blue")
    cprint(f"garment_center_pos_x: {env.garment.get_garment_center_pos()[0]}", "blue")
    cprint(f"hanger_center_x: {env.hanger_center[0]}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag and success:
        if not os.path.exists("Data/Hang_Tops/video"):
            os.makedirs("Data/Hang_Tops/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Hang_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Hang_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz(env_change=True)
            if not os.path.exists("Data/Hang_Tops/final_state_pic"):
                os.makedirs("Data/Hang_Tops/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Hang_Tops/final_state_pic/img",".png"))




if __name__=="__main__":
    
    args = parse_args_record()
    
    # initial setting
    pos = np.array([0, 0.7, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.1
    env_dy = 0.0 
    env_dz = 0.0
    theta = 20.0

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.15, 0.15) # changeable
            env_dy = np.random.uniform(-0.05, 0.05) # changeable
            env_dz = np.random.uniform(-0.05, 0.05) # changeable
            theta = np.random.uniform(-20.0, 20.0) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.5, 0.8) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"FoundationModels/GAM/checkpoints/Tops_LongSleeve/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    HangTops(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()