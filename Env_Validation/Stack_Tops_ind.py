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
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
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
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from MFMDP.deploy_policy import MFMDP_Encapsulation


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
        
        input_dict = {
            "task_name": "Stack_Tops",
            "stage_str": stage_str,
            "data_num": training_data_num,
            "checkpoint_num": checkpoint_num,
            "active_object_n_component": 0,
            "passive_object_n_component": 0,
            "category": "Tops_NoSleeve",
            "demo_pcd_path": None,
        }
        self.policy = getattr(sys.modules[__name__], f"{policy_name}_Encapsulation")(init_dict=input_dict)   
        
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
        
    def get_obs(self):
        joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
        joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
        
        joint_state = np.array([*joint_pos_L, *joint_pos_R])

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

            "active_object_point_cloud": self.active_pcd,
            "passive_object_point_cloud": self.passive_pcd,
            "active_object_point_feature": self.active_feat,
            "passive_object_point_feature": self.passive_feat,
        }
        
        return obs

    def pre_grasp(self, flag):
        manipulation_points, indices, points_similarity = self.policy.model_GAM.get_manipulation_points(input_pcd=self.active_pcd, index_list=[1913, 1756]) #1819,1021  838,179
    
        if abs(manipulation_points[0][0]-manipulation_points[1][0])<0.1:
            print("model false")
            simulation_app.close()
            
        self.active_feat = normalize_columns(points_similarity.T)
        # TODO: More points
        passive_manipulation_points, _ , points_similarity = self.policy.model_GAM.get_manipulation_points(input_pcd=self.passive_pcd, index_list=[1913, 1756, 528, 587])
        # TODO: check if model prediction is correct
        self.passive_feat = normalize_columns(points_similarity.T)
            
        # get lift height
        y_min = self.active_pcd[np.argmin(self.active_pcd[:, 1])]
        y_max = self.active_pcd[np.argmax(self.active_pcd[:, 1])]
        garment_length=y_max[1]-y_min[1]
        lift_height = garment_length + 0.1
        left_dis = manipulation_points[0][1] - y_min[1]
        right_dis = manipulation_points[1][1] - y_min[1]
        cprint(f"lift height: {lift_height}", "blue")
        
        manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment

        if flag:
            # move both dexhand to the manipulation points
            self.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

# if __name__=="__main__":
def StackTops_Validation(active_pos, passive_pos, active_ori, passive_ori, active_usd_path, passive_usd_path, ground_material_usd, validation_flag, record_video_flag, training_data_num, checkpoint_num, stage_str, policy_name):

    env = Stack_Tops_Env(active_pos=active_pos, passive_pos=passive_pos,active_ori=active_ori,passive_ori=passive_ori,active_usd_path=active_usd_path, passive_usd_path=passive_usd_path, ground_material_usd=ground_material_usd, record_video_flag=record_video_flag, training_data_num=training_data_num, checkpoint_num=checkpoint_num, stage_str=stage_str, policy_name=policy_name)

    env.active_garment.particle_material.set_gravity_scale(0.7)
    env.passive_garment.particle_material.set_gravity_scale(1.5)
    
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
        return False
    
    if record_video_flag:
        env.thread_record.start()

    env.pre_grasp(flag=(stage_str == "remain"))
    
    for i in range(20):
        env.step()

    step_lim = 9 + (4 if stage_str == "all" else 0)

    for i in range(step_lim):
        
        print(f"Stage_1_Step: {i}")
        
        obs = env.get_obs()
        action=env.policy.get_action(obs)
        
        for j in range(4):
            
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            obs = env.get_obs()
            
            env.policy.update_obs(obs)

    for i in range(100):
        env.step()
        
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
                
    if record_video_flag:
        if not os.path.exists(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video"):
            os.makedirs(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video")
        env.env_camera.create_mp4(get_unique_filename(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/video/video", ".mp4"))
    try:
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
        success = over_lap > 0.5
    except:
        print("no garment")
        over_lap = 0.0
        success = False

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"Overlap: {over_lap}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.

    if validation_flag:
        if not os.path.exists(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}"):
            os.makedirs(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}")
        # write into .log file
        with open(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/validation_log.txt", "a") as f:
            f.write(f"result:{success}  overlap:{over_lap}  active_pos:{active_pos}  passive_pos:{passive_pos}  active_ori:{active_ori}  passive_ori:{passive_ori}  active_usd_path:{active_usd_path}  passive_usd_path:{passive_usd_path}\n")
        if not os.path.exists(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic"):
            os.makedirs(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename(f"Data/Stack_Tops_ind_Validation_{policy_name}_{stage_str}_{checkpoint_num}_{training_data_num}/final_state_pic/img",".png"))

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
    
    args = parse_args_val()
    # initial setting
    active_pos = np.array([-0.2, 0.3, 0.2])
    passive_pos = np.array([0.2, 0.9, 0.2])   
    active_ori = np.array([0.0, 0.0, 0.0])
    passive_ori = np.array([0.0, 0.0, 0.0])
    usd_path = None

    # seed = int(time.time())
    seed = args.seed

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(seed)
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

    StackTops_Validation(active_pos=active_pos, passive_pos=passive_pos, active_ori=active_ori, passive_ori=passive_ori, active_usd_path=usd_path, passive_usd_path=usd_path, ground_material_usd=args.ground_material_usd, validation_flag=args.validation_flag, record_video_flag=args.record_video_flag, training_data_num=args.training_data_num, checkpoint_num=args.checkpoint_num, stage_str=args.stage_str, policy_name=args.policy_name)
    
    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()