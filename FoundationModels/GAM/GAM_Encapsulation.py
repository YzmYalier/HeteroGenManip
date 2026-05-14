import os
import sys
import random
import numpy as np
import torch
import open3d as o3d
from termcolor import cprint


sys.path.append(os.getcwd()) # change to your specific path
sys.path.append("FoundationModels/GAM")
from FoundationModels.GAM.model.pointnet2_GAM import GAM_Model
from Env_Config.Utils_Project.Point_Cloud_Manip import furthest_point_sampling, normalize_pcd_points_xy, visualize_pointcloud_with_colors, colormap

class GAM_Encapsulation:
    
    def __init__(self, catogory:str="Tops_LongSleeve"):
        '''
        load model
        '''
        self.catogory = catogory
        # set resume path
        resume_path = f"FoundationModels/GAM/checkpoints/{self.catogory}/checkpoint.pth"
        # set seed
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)
        random.seed(seed)
        # define model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = GAM_Model(normal_channel=False, feature_dim=512).cuda()
        self.model.load_state_dict(torch.load(resume_path, weights_only=False))
        self.model = self.model.to(self.device)
        self.model.eval()
        
    
    def get_feature(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get feature of input point cloud
        '''
        normalized_pcd, *_ = normalize_pcd_points_xy(input_pcd)
        normalize_pcd = np.expand_dims(normalized_pcd, axis=0)
        
        with torch.no_grad():
        
            pcd_features = self.model(
                torch.from_numpy(normalize_pcd).to(self.device).float(),
            ).squeeze(0)
            # print(pcd_features.shape)
        
        if index_list is not None:
            target_features_list = []
            for i in index_list:
                target_features_list.append(pcd_features[i])
            return torch.stack(target_features_list)
        else:
            return pcd_features
        
    def get_manipulation_points(self, input_pcd:np.ndarray, index_list:list=None):
        '''
        get manipulation points of input point cloud
        '''
        
        #get model output (feature)
        demo_pcd = o3d.io.read_point_cloud(f"FoundationModels/GAM/checkpoints/{self.catogory}/demo_garment.ply").points
        demo_feature = self.get_feature(demo_pcd, index_list)
        manipulate_feature = self.get_feature(input_pcd)
        
        # normalize feature
        demo_feature_normalized = torch.nn.functional.normalize(demo_feature, p=2, dim=1)
        manipulate_feature_normalized = torch.nn.functional.normalize(manipulate_feature, p=2, dim=1)
        result = torch.matmul(demo_feature_normalized, manipulate_feature_normalized.T)
        
        cprint("----------- GAM Inference Begin -----------", color="blue", attrs=["bold"])
        
        # get max similarity score and indices
        max_values, max_indices = torch.max(result, dim=1)
        cprint(f"similarity score: {max_values}", color="blue")
        cprint(f"relevant indices: {max_indices}", color="blue")
        cprint(f"similarity result shape: {result.shape}", color="blue")
        # get manipulation points
        manipulation_points = input_pcd[max_indices.detach().cpu().numpy()]
        cprint(f"manipulation points: \n {manipulation_points}", color="blue")
        
        cprint("----------- GAM Inference End -----------", color="blue", attrs=["bold"])
        
        return manipulation_points, max_indices.detach().cpu().numpy(), result.cpu().numpy()
    
    def get_colormap_points(self, input_pcd:np.ndarray):
        '''
        get colors for each corresponding point in input pcd.
        '''
        
        #get model output (feature)
        demo_pcd = o3d.io.read_point_cloud(f"FoundationModels/GAM/checkpoints/{self.catogory}/demo_garment.ply").points
        demo_feature = self.get_feature(demo_pcd)
        manipulate_feature = self.get_feature(input_pcd)
        
        # normalize feature
        demo_feature_normalized = torch.nn.functional.normalize(demo_feature, p=2, dim=1)
        manipulate_feature_normalized = torch.nn.functional.normalize(manipulate_feature, p=2, dim=1)

        result = torch.matmul(manipulate_feature_normalized, demo_feature_normalized.T)
        
        # get max similarity score and indices
        max_values, max_indices = torch.max(result, dim=1)
        print("similarity score: ", max_values)
        print("relevant indices: ", max_indices)
        
        # get manipulation points
        corresponding_demo_color = input_pcd[max_indices.detach().cpu().numpy()]
        print("manipulation points: \n", corresponding_demo_color)
        return corresponding_demo_color, max_indices.detach().cpu().numpy()
    
    def get_demo_garment_with_color(self):
        '''
        get demo garment with color
        '''
        demo_pcd = o3d.io.read_point_cloud(f"FoundationModels/GAM/checkpoints/{self.catogory}/demo_garment.ply")
        self.demo_points = np.asarray(demo_pcd.points)
        self.demo_points_color = colormap(self.demo_points)
        return self.demo_points, self.demo_points_color
    
    def visualize_pcd_corresponce(self, input_pcd:np.ndarray, save_or_not:bool=False, save_path:str=None):
        '''
        visualize exact point corresponce between input pcd and demo garment.
        '''
        # visualize demo garment with color
        self.get_demo_garment_with_color()
        visualize_pointcloud_with_colors(self.demo_points, self.demo_points_color)
        index_list = list(range(len(self.demo_points)))
        print("index_list: ", index_list)
        print("len(index_list): ", len(index_list))
        _, color_indices = self.get_colormap_points(input_pcd)
        # visualize input pcd with color
        color = np.ones_like(input_pcd)
        for i in range(len(color_indices)):
            color[i] = self.demo_points_color[color_indices[i]]
        print("color: ", color)
        visualize_pointcloud_with_colors(input_pcd, color, save_or_not=save_or_not, save_path=save_path)

