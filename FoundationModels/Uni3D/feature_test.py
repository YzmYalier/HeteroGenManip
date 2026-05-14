import torch
import numpy as np
import open3d as o3d
from models.uni3d import create_uni3d
from utils.params import parse_args
import models.uni3d as models

def load_pc(file_path, n_points=8192):
    # 读取点云并归一化
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 随机采样到指定点数
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    # 组合坐标和颜色 (N,6)
    return np.concatenate([points, colors], axis=1)

if __name__ == "__main__":
    # 1. 配置参数 (模拟scripts/inference.sh的large配置)
    args,ds_init = parse_args([])
    args.model = "create_uni3d"
    args.pc_model = "eva02_large_patch14_448"
    args.pc_feat_dim = 1024
    args.num_group = 512
    args.group_size = 64
    args.pc_encoder_dim = 512
    args.embed_dim = 1024
    
    # 2. 创建模型
    model = getattr(models, args.model)(args=args).cuda()
    # device = "cuda:0"
    # model.to(device)
    # 3. 加载预训练权重 (需要替换实际路径)
    ckpt = torch.load("logs/model.pt") 
    model.load_state_dict(ckpt['module'])
    model.eval()
    
    # 4. 处理点云
    pc_data = load_pc("/home/user/YzmCode/DexGarmentLab/data_0.ply")  # 替换实际点云路径
    pc_tensor = torch.from_numpy(pc_data).float().unsqueeze(0).cuda()  # 添加batch维度
    print(pc_tensor.shape)
    # 5. 特征提取
    with torch.no_grad():
        # 分割坐标和颜色
        xyz = pc_tensor[..., :3].contiguous()
        color = pc_tensor[..., 3:].contiguous()
        
        # 通过点云编码器
        feature = model.point_encoder(xyz, color)
        
    print(f"特征维度: {feature.shape}")
    print(f"示例特征值:\n{feature[0,:10]}") 
    torch.save(feature, "pc_feature.pt")  # 保存特征
