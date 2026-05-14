import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_foundation_model_diffusion_policy.model.vision.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2(nn.Module):
    
    def __init__(self, normal_channel=False, feature_dim=128):
        super(PointNet2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, feature_dim, 1)  # 输出 feature_dim 维特征向量

    def forward(self, xyz):
        # Set Abstraction layers
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        output = self.drop1(feat)
        output = self.conv2(output)
        output = output.permute(0, 2, 1)  # 输出形状 (B, N, feature_dim)
        return output


class PointNet2Global(nn.Module):
    def __init__(self, affordance_feature=False, feature_dim=32):
        super(PointNet2Global, self).__init__()
        if affordance_feature:
            additional_channel = 2
        else:
            additional_channel = 0
        self.affordance_feature = affordance_feature
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
    def forward(self, xyz):
        # Set Abstraction layers
        if self.affordance_feature:
            if xyz.shape[1] != 5:
                xyz = xyz.permute(0, 2, 1)
        else:
            if xyz.shape[1] != 3:
                xyz = xyz.permute(0, 2, 1)
                
        B,C,N = xyz.shape
        
        if self.affordance_feature:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
       
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # max_pooling
        global_feat = l3_points.squeeze(-1)  # (B, 512)
        # fully connected layers, downsample to feature_dim
        global_feat = self.fc(global_feat)
        return global_feat
    
class PointNet2GlobalFuser(nn.Module):
    def __init__(self, in_channel, feature_dim=32):
        super(PointNet2GlobalFuser, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + in_channel, mlp=[in_channel, in_channel, in_channel], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=in_channel + 3, mlp=[in_channel, in_channel, in_channel], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=in_channel + 3, mlp=[in_channel, in_channel, in_channel], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
                
        B,C,N = xyz.shape
        
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # max_pooling
        global_feat = l3_points.squeeze(-1)  # (B, 512)
        # fully connected layers, downsample to feature_dim
        global_feat = self.fc(global_feat)
        return global_feat
    
class PointNet2Global_light_v2(nn.Module):
    def __init__(self, in_channel, feature_dim=32):
        super(PointNet2Global_light_v2, self).__init__()
        self.down_dim = nn.Sequential(
            nn.Linear(in_channel, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        block_channel = [64, 128, 256]
        self.fc = nn.Sequential(
            nn.Linear(256+block_channel[2], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        self.point_xyz_encoder = nn.Sequential(
            nn.Linear(3, block_channel[0]),
            nn.LayerNorm(block_channel[0]),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]),
            nn.ReLU(),
        )
    def forward(self, xyz, point_feat):
        point_xyz_feat = self.point_xyz_encoder(xyz)
        point_feat = self.down_dim(point_feat)
        point_feat = torch.cat([point_xyz_feat,point_feat], dim=-1)
        max_feature = torch.max(point_feat, dim=1)[0]  # 输出(batch_size, feature_dim)

        # fully connected layers, downsample to feature_dim
        global_feat = self.fc(max_feature)
        return global_feat

class PointNet2Global_light(nn.Module):
    def __init__(self, in_channel, feature_dim=32):
        super(PointNet2Global_light, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    def forward(self, xyz):
        xyz = self.fc1(xyz)
        max_feature = torch.max(xyz, dim=1)[0]  # 输出(batch_size, feature_dim)

        # fully connected layers, downsample to feature_dim
        global_feat = self.fc2(max_feature)
        return global_feat
    
class CLSAggregator(nn.Module):
    def __init__(self, in_channel, feature_dim):

        super().__init__()
        num_heads = 2
        num_layers = 1
        dropout = 0.1
        self.in_channel = in_channel
        self.feature_dim = feature_dim
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channel))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channel,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channel, feature_dim-3),
            nn.ReLU(),
            nn.LayerNorm(feature_dim-3)
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, feat, pos, cls_token=None):
        feat = self.fc(feat)
        feat = torch.cat([feat, pos], dim=-1)
        B, P, C = feat.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, feat], dim=1)
        encoded = self.transformer(x_with_cls)
        global_feature = encoded[:, 0, :]
        global_feature = self.norm(global_feature)
        return global_feature

class Point_Downsampler(nn.Module):
    def __init__(self, in_channel, feature_dim):
        super().__init__()
        num_heads = 4
        num_layers = 1
        dropout = 0.1
        self.in_channel = in_channel
        self.feature_dim = feature_dim
        
        # self.cls_token = nn.Parameter(torch.randn(1, 1, in_channel))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channel,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channel, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
    def forward(self, pointList):
        point_feat = self.transformer(pointList)
        point_feat = torch.mean(point_feat, dim=1)
        point_feat = self.fc(point_feat)
        return point_feat
