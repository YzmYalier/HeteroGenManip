from MFMDP.MFMDP import MFMDP
from sklearn.decomposition import PCA
import pickle
from FoundationModels.GAM.GAM_Encapsulation import GAM_Encapsulation
from FoundationModels.Uni3D.Uni3D_Encapsulation import Uni3D_Encapsulation

class MFMDP_Encapsulation:
    def __init__(self, init_dict):
        task_name = init_dict["task_name"] + "_stage_" + init_dict["stage_str"]
        data_num = init_dict["data_num"]
        checkpoint_num = init_dict["checkpoint_num"]
        active_object_n_component = init_dict["active_object_n_component"]
        passive_object_n_component = init_dict["passive_object_n_component"]
        catogory = init_dict["category"]
        demo_pcd_path = init_dict["demo_pcd_path"]

        # load GAM Model
        self.model_GAM = GAM_Encapsulation(catogory=catogory)   

        # load Uni3D Model
        self.model_Uni3D = Uni3D_Encapsulation(demo_pcd_path=demo_pcd_path)

        self.policy = MFMDP(task_name=task_name, data_num=data_num, checkpoint_num=checkpoint_num)
        self.active_object_pca = PCA(n_components=active_object_n_component) if active_object_n_component > 0 else None
        self.passive_object_pca = PCA(n_components=passive_object_n_component) if passive_object_n_component > 0 else None

        if self.active_object_pca:
            with open(f'MFMDP/data/{task_name}_{data_num}.zarr/active_object_pca.pkl', 'rb') as f:
                self.active_object_pca = pickle.load(f)

        if self.passive_object_pca:
            with open(f'MFMDP/data/{task_name}_{data_num}.zarr/passive_object_pca.pkl', 'rb') as f:
                self.passive_object_pca = pickle.load(f)

    def encode_obs(self, observation):
        obs = dict()
        obs["environment_point_cloud"] = observation["env_point_cloud"]
        obs["active_object_point_cloud"] = observation["active_object_point_cloud"]
        obs["passive_object_point_cloud"] = observation["passive_object_point_cloud"]
        if self.active_object_pca:
            obs['active_object_point_feature'] = self.active_object_pca.transform(observation['active_object_point_feature'])
        else:
            obs['active_object_point_feature'] = observation['active_object_point_feature']
        if self.passive_object_pca:
            obs['passive_object_point_feature'] = self.passive_object_pca.transform(observation['passive_object_point_feature'])
        else:
            obs['passive_object_point_feature'] = observation['passive_object_point_feature']
        obs["agent_pos"] = observation["joint_state"]
        return obs
    
    def get_action(self, observation):
        obs = self.encode_obs(observation)
        return self.policy.get_action(obs)
    
    def update_obs(self, observation):
        obs = self.encode_obs(observation)
        self.policy.update_obs(obs)