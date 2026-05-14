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
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from MFMDP.MFMDP import MFMDP
from IL_Baselines.Diffusion_Policy_3D.DP3 import DP3
sys.path.append("/media/isaac/e2fa93f5-512e-4f9a-9023-40cf92b0ad74/DexGarmentLab/PA3FF")
sys.path.append("/media/isaac/e2fa93f5-512e-4f9a-9023-40cf92b0ad74/DexGarmentLab/Uni3D")
from PA3FF.pointcept.models.SAMPart3D import SAMPart3D
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models
from sonata.transform import Compose

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
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
        training_data_num:int=100,
        stage_1_checkpoint_num:int=1500, 
        stage_2_checkpoint_num:int=1500, 
        stage_3_checkpoint_num:int=1500, 
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
        self.articulated_pcd = None
        self.points_affordance_feature = None
        self.rigid_point_feature = None
        
        args,ds_init = parse_args([])
        args.model = "create_uni3d"
        args.pc_model = "eva02_base_patch14_448"
        args.pc_feat_dim = 768
        args.num_group = 128
        args.group_size = 32
        args.pc_encoder_dim = 512
        args.embed_dim = 1024
        
        self.extractor_ragid = getattr(models, args.model)(args=args).cuda()
        ckpt = torch.load("/media/isaac/e2fa93f5-512e-4f9a-9023-40cf92b0ad74/DexGarmentLab/FoundationModels/Uni3D/logs/model.pt") 
        self.extractor_ragid.load_state_dict(ckpt['module'])
        self.extractor_ragid.eval()
                
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")        
        
        # load hanger
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.hanger_center = hanger_load(self.scene, env_dx, env_dy, env_dz, theta)
        
        self.judge_camera = Recording_Camera(
            camera_position=np.array([self.hanger_center[0], 6.0, 0.5]),
            camera_orientation=np.array([0, 0.0, -90.0]),
            prim_path="/World/judge_camera",
        )    
        
        self.mfmdp = MFMDP(task_name="Hang_Tops_stage_1", data_num=training_data_num, checkpoint_num=stage_1_checkpoint_num)
        # self.dp3 = DP3(task_name="Hang_Tops_stage_1", data_num=training_data_num, checkpoint_num=stage_1_checkpoint_num)
        
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
        
        
# if __name__=="__main__":
def HangTops(pos, ori, usd_path, env_dx, env_dy,env_dz,theta, ground_material_usd, validation_flag, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num):
    
    env = HangTops_Env(pos, ori, usd_path, env_dx, env_dy, env_dz,theta, ground_material_usd, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num)
    
    env.garment.particle_material.set_gravity_scale(0.45)
    
    env.articulated_pcd = np.zeros((2048, 3))
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
        sampled_point_num=512
        # real_time_watch=True,
    )
    batch_pcd = torch.from_numpy(env.object_pcd).unsqueeze(0).cuda()
    batch_color = torch.full_like(batch_pcd, 0.5)
    with torch.no_grad():
        feat, center, group_feat = env.extractor_ragid.point_encoder(batch_pcd, batch_color)

    # 扩展维度以便广播计算 (B N G)
    point_distances = torch.cdist(batch_pcd, center)

    # 获取最近组索引 (B N)
    closest_group = point_distances.argmin(dim=-1)  

    # 分配最近组的特征给点 (B N C)
    point_feats = torch.gather(group_feat, 1, closest_group.unsqueeze(-1).expand(-1, -1, group_feat.shape[-1]))
    env.rigid_point_feature = point_feats.squeeze(0).cpu().detach().numpy()
    
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
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[838,179])
    
    env.points_affordance_feature = normalize_columns(points_similarity.T)
        
    manipulation_points[:, 2] = 0.00  # set z-axis to 0.025 to make sure dexhand can grasp the garment

    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    for i in range(20):
        env.step()
    
    for i in range(9):
        
        print(f"Stage_1_Step: {i}")

        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        obs['agent_pos']=joint_state
        obs['environment_point_cloud']=env.env_camera.get_pointcloud_from_depth()
        obs['garment_point_cloud']=env.garment_pcd
        obs['rigid_point_cloud']=env.object_pcd
        obs['points_affordance_feature']=env.points_affordance_feature
        obs['articulated_point_cloud'] = env.articulated_pcd
        obs['rigid_point_feature'] = env.rigid_point_feature
        # obs['point_cloud']=env.env_camera.get_pointcloud_from_depth()
        action=env.mfmdp.get_action(obs)
        # action = env.dp3.get_action(obs)
        
        print("action_shape:",action.shape)
        
        for j in range(4):
            
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])
            
            obs = dict()
            obs['agent_pos']=joint_state
            obs['environment_point_cloud']=env.env_camera.get_pointcloud_from_depth()
            obs['garment_point_cloud']=env.garment_pcd
            obs['rigid_point_cloud']=env.object_pcd
            obs['points_affordance_feature']=env.points_affordance_feature
            obs['articulated_point_cloud'] = env.articulated_pcd
            obs['rigid_point_feature'] = env.rigid_point_feature
            # obs['point_cloud']=env.env_camera.get_pointcloud_from_depth()
            env.mfmdp.update_obs(obs)
            # env.dp3.update_obs(obs)
        
    env.garment.particle_material.set_gravity_scale(2.0)
    
    for i in range(100):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(0.45)
    
    # make prim visible
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag:
        if not os.path.exists("Data/Hang_Tops_Validation_HALO/video"):
            os.makedirs("Data/Hang_Tops_Validation_HALO/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Hang_Tops_Validation_HALO/video/video", ".mp4"))
   
        
        
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
    elif env.garment.get_garment_center_pos()[0] > env.hanger_center[0]+0.035 or env.garment.get_garment_center_pos()[0] < env.hanger_center[0]-0.035:
        success=False
    else:
        success=True

    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"points_below_threshold: {points_below_threshold}", "blue")
    cprint(f"garment_center_pos_x: {env.garment.get_garment_center_pos()[0]}", "blue")
    cprint(f"hanger_center_x: {env.hanger_center[0]}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")

    if validation_flag:
        if not os.path.exists("Data/Hang_Tops_Validation_HALO"):
            os.makedirs("Data/Hang_Tops_Validation_HALO")
        # write into .log file
        with open("Data/Hang_Tops_Validation_HALO/validation_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")
        if not os.path.exists("Data/Hang_Tops_Validation_HALO/final_state_pic"):
            os.makedirs("Data/Hang_Tops_Validation_HALO/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Hang_Tops_Validation_HALO/final_state_pic/img",".png"))




if __name__=="__main__":
    
    args = parse_args_val()
    
    # initial setting
    pos = np.array([0, 0.7, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.1
    env_dy = 0.0
    env_dz = 0.08
    theta = 0.0

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        # np.random.seed(42)
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.15, 0.15) # changeable
            env_dy = np.random.uniform(-0.1, 0.1) # changeable
            env_dz = np.random.uniform(-0.08, 0.08)
            theta = np.random.uniform(-30.0, 30.0)
        if args.garment_random_flag:
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.5, 0.8) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"FoundationModels/GAM/checkpoints/Tops_LongSleeve/assets_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    HangTops(pos, ori, usd_path, env_dx, env_dy, env_dz, theta, args.ground_material_usd, args.validation_flag, args.record_video_flag, args.training_data_num, args.stage_1_checkpoint_num, args.stage_2_checkpoint_num, args.stage_3_checkpoint_num)

    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()