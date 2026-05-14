import sys
from FoundationModels.Uni3D.models.uni3d import create_uni3d
from FoundationModels.Uni3D.utils.params import parse_args
import FoundationModels.Uni3D.models.uni3d as models
import torch
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.cm as cm

class Uni3D_Encapsulation:
    def __init__(self, demo_pcd_path=None):
        args, ds_init = parse_args([])
        args.model = "create_uni3d"
        args.pc_model = "eva02_base_patch14_448"
        args.pc_feat_dim = 768
        args.num_group = 512
        args.group_size = 32
        args.pc_encoder_dim = 512
        args.embed_dim = 1024

        self.uni3d = getattr(models, args.model)(args=args).cuda()
        print(type(self.uni3d))
        ckpt = torch.load("FoundationModels/Uni3D/logs/model.pt") 
        self.uni3d.load_state_dict(ckpt['module'])
        self.uni3d.eval()
        
        if demo_pcd_path is not None:
            demo_pcd = o3d.io.read_point_cloud(demo_pcd_path)
            self.demo_feats = self.get_feature(np.asarray(demo_pcd.points, np.float32), None)

    def get_feature(self, input_pcd:np.ndarray, index_list:list=None):
        batch_pcd = torch.from_numpy(input_pcd).unsqueeze(0).cuda()
        batch_color = torch.full_like(batch_pcd, 0.5)

        with torch.no_grad():
            feat, center, group_feat = self.uni3d.point_encoder(batch_pcd, batch_color)

        B, N, _ = batch_pcd.shape
        B, G, _ = center.shape

        # 扩展维度以便广播计算 (B N G)
        point_distances = torch.cdist(batch_pcd, center)

        # 获取最近组索引 (B N)
        closest_group = point_distances.argmin(dim=-1)

        # 分配最近组的特征给点 (B N C)
        point_feats = torch.gather(group_feat, 1, closest_group.unsqueeze(-1).expand(-1, -1, group_feat.shape[-1]))

        feats = point_feats.squeeze(0)

        if index_list is not None:
            target_features_list = []
            for i in index_list:
                target_features_list.append(feats[i])
            return torch.stack(target_features_list)
        else:
            return feats

    def get_manipulation_points(self, input_pcd:np.ndarray, index_list=None):
        '''
        index_list: [0] for 把手上端; [1] for 把手下端; [2] for 把手外沿
        '''

        # only get feature
        if index_list is None:
            input_feats = self.get_feature(np.asarray(input_pcd, np.float32))
            return input_feats.detach().cpu().numpy(), None

        demo_feats = torch.stack([self.demo_feats[i] for i in index_list]) if index_list else self.demo_feats
        input_feats = self.get_feature(np.asarray(input_pcd, np.float32))

        # normalize feature
        demo_feats_normalized = torch.nn.functional.normalize(demo_feats, p=2, dim=1)
        input_feats_normalized = torch.nn.functional.normalize(input_feats, p=2, dim=1)

        # calculate cosine similarity
        similarity = torch.mm(demo_feats_normalized, input_feats_normalized.t())
        max_values, max_indices = torch.max(similarity, dim=1)

        return input_feats.detach().cpu().numpy(), input_pcd[max_indices.detach().cpu().numpy()]

    def visualize_features(self, input_pcd: o3d.geometry.PointCloud, index_list=None):
        feats = self.get_feature(input_pcd, index_list)

        # PCA降维到3维
        pca = PCA(n_components=3)
        feats_pca = pca.fit_transform(feats.cpu().numpy())

        for _ in range(3):
            feats_pca[:, _] = ((feats_pca[:, _] - feats_pca[:, _].min()) / (feats_pca[:, _].max() - feats_pca[:, _].min())) ** 0.5

        # 转换为RGB颜色
        rgb_colors = cm.jet(feats_pca)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(input_pcd.points))
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

        o3d.visualization.draw_geometries([pcd])

def get_feature(input_pcd, index_list=None):
    args,ds_init = parse_args([])
    args.model = "create_uni3d"
    args.pc_model = "eva02_base_patch14_448"
    args.pc_feat_dim = 768
    args.num_group = 512
    args.group_size = 32
    args.pc_encoder_dim = 512
    args.embed_dim = 1024

    uni3d = getattr(models, args.model)(args=args).cuda()
    print(type(uni3d))
    ckpt = torch.load("logs/model.pt") 
    uni3d.load_state_dict(ckpt['module'])
    uni3d.eval()

    batch_pcd = torch.from_numpy(np.asarray(input_pcd.points, dtype=np.float32)).unsqueeze(0).cuda()
    batch_color = torch.from_numpy(np.asarray(input_pcd.colors, dtype=np.float32)).unsqueeze(0).cuda()
    with torch.no_grad():
        feat, center, group_feat = uni3d.point_encoder(batch_pcd, batch_color)

    B, N, _ = batch_pcd.shape
    B, G, _ = center.shape

    # 扩展维度以便广播计算 (B N G)
    point_distances = torch.cdist(batch_pcd, center)

    # 获取最近组索引 (B N)
    closest_group = point_distances.argmin(dim=-1)

    # 分配最近组的特征给点 (B N C)
    point_feats = torch.gather(group_feat, 1, closest_group.unsqueeze(-1).expand(-1, -1, group_feat.shape[-1]))

    feats = point_feats[0].cpu()
    points = batch_pcd[0].cpu()

    if index_list is not None:
        target_features_list = []
        for i in index_list:
            target_features_list.append(feats[i])
        return torch.stack(target_features_list)
    else:
        return feats


if __name__ == "__main__":
    indices = [1192, 772, 456]

    demo_pcd = o3d.io.read_point_cloud(f"FoundationModels/Uni3D/demo_pcds/mug.ply")
    input_pcd = o3d.io.read_point_cloud(f"FoundationModels/Uni3D/demo_pcds/mug.ply")
    
    demo_feats = get_feature(demo_pcd, indices)
    input_feats = get_feature(input_pcd)
    
    # normalize feature
    demo_feats_normalized = torch.nn.functional.normalize(demo_feats, p=2, dim=1)
    input_feats_normalized = torch.nn.functional.normalize(input_feats, p=2, dim=1)

    # calculate cosine similarity
    similarity = torch.mm(demo_feats_normalized, input_feats_normalized.t())
    
    print("Cosine Similarity Matrix:")
    print(similarity.shape)
    max_values, max_indices = torch.max(similarity, dim=1)
    similarity[2][max_indices[2]] = 0.0
    print(max_values)
    print(max_indices)
    max_values, max_indices = torch.max(similarity, dim=1)
    print(max_values)
    print(max_indices)

    colors = np.zeros((2048, 3))
    colors[max_indices[0]] = [1, 0, 0]  # Red for matched points
    colors[max_indices[1]] = [0, 1, 0]  # Green for matched points
    colors[max_indices[2]] = [0, 0, 1]  # Blue for matched points
    # colors[~np.isin(np.arange(2048), max_indices.numpy())] = [0, 1, 0]  # Green for others

    print(111)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_pcd.points)
    print(333)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(444)
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    print(555)
    vis.create_window()
    print(666)
    render_option = vis.get_render_option()
    print(777)
    render_option.point_size =20.0  # 设置点大小，默认值为1.0
    print(888)
    vis.add_geometry(pcd)
    print(999)
    vis.run()
    print(1000)
    vis.destroy_window()
    print("Visualization complete.")

