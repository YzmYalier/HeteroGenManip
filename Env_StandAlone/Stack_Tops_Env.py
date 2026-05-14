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
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
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
from Env_Config.Room.Object_Tools import hanger_load, set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models
class Stack_Tops_Env(BaseEnv):
    def __init__(
        self, 
        active_pos:np.ndarray=None, 
        passive_pos:np.ndarray=None, 
        active_ori:np.ndarray=None, 
        passive_ori:np.ndarray=None, 
        active_usd_path:str=None, 
        passive_usd_path:str=None, 
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
        self.active_garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            # scale=np.array([0.008, 0.008, 0.008]),
            usd_path="Assets/Garment/Tops/Collar_noSleeve_FrontClose/TCNC_Top184/TCNC_Top184_obj.usd" if active_usd_path is None else active_usd_path,
        )
        self.passive_garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            # scale=np.array([0.008, 0.008, 0.008]),
            usd_path="Assets/Garment/Tops/Collar_noSleeve_FrontClose/TCNC_Top184/TCNC_Top184_obj.usd" if passive_usd_path is None else passive_usd_path,
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
        
        self.garment_camera2 = Recording_Camera(
            camera_position=np.array([0.0, -3.0, 6.75]), 
            camera_orientation=np.array([0, 60.0, 90.0]),
            prim_path="/World/garment_camera2",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 6.65, 4.0]),
            camera_orientation=np.array([0, 30.0, -90.0]),
            prim_path="/World/env_camera",
        )
        
        
        self.active_pcd = None
        self.passive_pcd = None
        self.active_feat = None
        self.passive_feat = None
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_NoSleeve")        
        
        self.judge_camera = Recording_Camera(
            camera_position=np.array([0.0, -3.0, 6.75]), 
            camera_orientation=np.array([0, 60.0, 90.0]),
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
        self.garment_camera2.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment_1",
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
        
        self.env_camera_intrinsics = self.env_camera.camera.get_intrinsics_matrix()
        self.env_camera_extrinsics = self.env_camera.camera.get_view_matrix_ros()
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
            
        # move garment to the target position
        self.active_garment.set_pose(pos=np.array([active_pos[0], active_pos[1], 0.2]), ori=active_ori)
        self.active_pos = [active_pos[0], active_pos[1], 0.2]
        self.passive_pos = [passive_pos[0], passive_pos[1], 0.2]
        self.passive_garment.set_pose(pos=np.array([passive_pos[0], passive_pos[1], 0.2]), ori=passive_ori)
        self.orientation = active_ori
                
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(200):
            self.step()
            
        # cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        # cprint(f"usd_path: {usd_path}", "magenta")
        # cprint(f"pos_x: {pos[0]}", "magenta")
        # cprint(f"pos_y: {pos[1]}", "magenta")
        # cprint(f"env_dx: {env_dx}", "magenta")
        # cprint(f"env_dy: {env_dy}", "magenta")
        # cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])

        cprint("World Ready!", "green", "on_green")
    
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            depth = self.env_camera.get_depth_graph()

            point_cloud = self.env_camera.get_pointcloud_from_depth(
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
                "env_image": rgb,
                "env_depth": depth,
                "env_point_cloud": point_cloud,
                "active_object_point_cloud":self.active_pcd,    
                "passive_object_point_cloud":self.passive_pcd,
                "active_object_point_feature": self.active_feat,
                "passive_object_point_feature": self.passive_feat,
            })
        
        self.step_num += 1
        
        
# if __name__=="__main__":
def StackTops(active_pos, passive_pos, active_ori, passive_ori, active_usd_path, passive_usd_path, ground_material_usd, data_collection_flag, record_video_flag):

    env = Stack_Tops_Env(active_pos=active_pos, passive_pos=passive_pos,active_ori=active_ori,passive_ori=passive_ori,active_usd_path=active_usd_path, passive_usd_path=passive_usd_path, ground_material_usd=ground_material_usd, record_video_flag=record_video_flag)

    env.active_garment.particle_material.set_gravity_scale(0.7)
    env.passive_garment.particle_material.set_gravity_scale(1.5)

    # For DinoV2
    # env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Stack_Tops/img",".png"))
    # point_cloud = env.env_camera.get_pointcloud_from_depth()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # save_path = get_unique_filename("Data/Stack_Tops/pcd",".ply")
    # o3d.io.write_point_cloud(save_path, pcd)
    # env.env_camera.get_depth_graph(save_or_not=True,save_path=get_unique_filename("Data/Stack_Tops/depth",".png"))
    # intrinsics_matrix = env.env_camera.camera.get_intrinsics_matrix()
    # view_matrix = env.env_camera.camera.get_view_matrix_ros()
    # pcd_image = env.env_camera.camera.get_image_coords_from_world_points(point_cloud[:, :3])
    # matrix_path = get_unique_filename("Data/Stack_Tops/intrinsics_matrix", ".npy")
    # np.save(matrix_path, intrinsics_matrix)
    # view_matrix_path = get_unique_filename("Data/Stack_Tops/view_matrix", ".npy")
    # np.save(view_matrix_path, view_matrix)
    # pcd_image_path = get_unique_filename("Data/Stack_Tops/pcd_image", ".npy")
    # np.save(pcd_image_path, pcd_image)
    # print("全部保存完毕！")
    # exit()
    
    # # hide prim to get object point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    env.passive_pcd, color = env.garment_camera2.get_point_cloud_data_from_segment(
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
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment_1"],
        visible=False,
    )
    for i in range(50):
        env.step()
            
    env.active_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        sampled_point_num=2048,
        # real_time_watch=True,
    )

    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment_1"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    if calculate_overlap(env.active_pcd, env.passive_pcd) > 0.05:
        print("overlap!!")
        simulation_app.close()
    
    if record_video_flag:
        env.thread_record.start()

    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Stack_Tops", stage_str="all")

    # get manipulation points from GAM Model
    # 1913 1756
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=env.active_pcd, index_list=[1913, 1756]) #1819,1021  838,179
    
    if abs(manipulation_points[0][0]-manipulation_points[1][0])<0.1:
        print("model false")
        simulation_app.close()
        
    env.active_feat = normalize_columns(points_similarity.T)
    # TODO: More points
    passive_manipulation_points, _ , points_similarity = env.model.get_manipulation_points(input_pcd=env.passive_pcd, index_list=[1913, 1756, 528, 587])
    # TODO: check if model prediction is correct
    env.passive_feat = normalize_columns(points_similarity.T)
        
    # get lift height
    y_min = env.active_pcd[np.argmin(env.active_pcd[:, 1])]
    y_max = env.active_pcd[np.argmax(env.active_pcd[:, 1])]
    garment_length=y_max[1]-y_min[1]
    lift_height = garment_length + 0.1
    left_dis = manipulation_points[0][1] - y_min[1]
    right_dis = manipulation_points[1][1] - y_min[1]
    cprint(f"lift height: {lift_height}", "blue")
    
    manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment

    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    env.data_split = int(env.step_num / 5)

    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

    left_lift_points, right_lift_points = np.array([manipulation_points[0][0], manipulation_points[0][1], lift_height]), np.array([manipulation_points[1][0], manipulation_points[1][1], lift_height])    # set z-axis to 0.65 to make sure dexhand can lift the garment
    
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    # print(passive_manipulation_points[0][0])
    # print(passive_manipulation_points[0][1] - left_dis)
    # print(lift_height)
    
    left_lift_points, right_lift_points = np.array([passive_manipulation_points[0][0], (passive_manipulation_points[0][1] - left_dis + 0.02), lift_height]), np.array([passive_manipulation_points[1][0], (passive_manipulation_points[1][1] - right_dis + 0.02), lift_height])

    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    lift_height =  lift_height - 0.15
    left_lift_points, right_lift_points = np.array([passive_manipulation_points[0][0], (passive_manipulation_points[0][1] - left_dis + 0.02), lift_height]), np.array([passive_manipulation_points[1][0], (passive_manipulation_points[1][1] - right_dis + 0.02), lift_height])  
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    left_lift_points, right_lift_points = np.array([passive_manipulation_points[0][0], passive_manipulation_points[0][1]+0.025, 0.15]), np.array([passive_manipulation_points[1][0], passive_manipulation_points[1][1]+0.025, 0.15])  
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
        
    env.active_garment.particle_material.set_gravity_scale(10.0)
    
    for i in range(100):
        env.step()

    env.active_garment.particle_material.set_gravity_scale(0.7)

    # # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment_1"],
        visible=False,
    )
    for i in range(50):
        env.step()
                
    # success=True
    
    pcd_judge, _ = env.judge_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True
    )
    set_prim_visible_group(
        prim_path_list=["/World/Garment/garment_1"],
        visible=True,
    )
    over_lap = calculate_overlap(pcd_judge, env.passive_pcd)
    success = over_lap > 0.7

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"Overlap: {over_lap}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag:
        if not os.path.exists("Data/Stack_Tops/video"):
            os.makedirs("Data/Stack_Tops/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Stack_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Stack_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.active_garment.usd_path}  active_pos:{env.active_pos} passive_pos:{env.passive_pos} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz(env_change=False)
            if not os.path.exists("Data/Stack_Tops/final_state_pic"):
                os.makedirs("Data/Stack_Tops/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Stack_Tops/final_state_pic/img",".png"))

def calculate_overlap(active_cloud, passive_cloud):
    active_cloud_xy = active_cloud[:, :2]
    passive_cloud_xy = passive_cloud[:, :2]
    hull1 = ConvexHull(active_cloud_xy)
    hull2 = ConvexHull(passive_cloud_xy)
    polygon1_points = active_cloud_xy[hull1.vertices]
    polygon2_points = passive_cloud_xy[hull2.vertices]
    
    # 创建多边形
    polygon1 = Polygon(polygon1_points)
    polygon2 = Polygon(polygon2_points)

    intersection = polygon1.intersection(polygon2).area
    over_lap = intersection / polygon2.area
    print("overlap:", over_lap)
    return over_lap
    
    
if __name__=="__main__":
    
    args = parse_args_record()
    # initial setting
    active_pos = np.array([-0.2, 0.3, 0.2])
    passive_pos = np.array([0.2, 0.9, 0.2])   
    active_ori = np.array([0.0, 0.0, 0.0])
    passive_ori = np.array([0.0, 0.0, 0.0])
    usd_path = None

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            passive_pos_x = np.random.uniform(-0.2, 0.2)
            passive_pos_y = np.random.uniform(0.8, 1.0)
            passive_pos = np.array([passive_pos_x, passive_pos_y, 0.2])
        if args.garment_random_flag:
            active_pos_x = np.random.uniform(-0.2, 0.2)
            active_pos_y = np.random.uniform(0.2, 0.4)
            active_pos = np.array([active_pos_x, active_pos_y, 0.2])

            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"FoundationModels/GAM/checkpoints/Tops_NoSleeve/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    StackTops(active_pos=active_pos, passive_pos=passive_pos, active_ori=active_ori, passive_ori=passive_ori, active_usd_path=usd_path, passive_usd_path=usd_path, ground_material_usd=args.ground_material_usd, data_collection_flag=args.data_collection_flag, record_video_flag=args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()