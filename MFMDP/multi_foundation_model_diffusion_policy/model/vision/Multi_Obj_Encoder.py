import sys
import os
import copy
import pdb
from typing import Dict, List, Optional, Tuple, Type, Union
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from termcolor import cprint
from multi_foundation_model_diffusion_policy.model.vision.pointnet2 import PointNet2Global, PointNet2GlobalFuser, PointNet2Global_light,CLSAggregator
from multi_foundation_model_diffusion_policy.model.diffusion.conditional_unet1d import CrossAttention
sys.path.append("..")
import open3d as o3d

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        # assert in_channels == 3, cprint(
        #     f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red"
        # )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x
        
class DynamicFeatureFuser(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super().__init__()
        # 注意力融合层
        # self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 输出映射层
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 层归一化稳定训练
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, features):
        # 输入features为列表，长度1-3，每个元素形状为[batch_size, in_dim]
        if len(features) == 0:
            raise ValueError("至少输入1个特征")
        if len(features) == 1:
            return self.output_proj(features[0])
        feat = torch.stack(features, dim=1)
        # batch_size = feat.shape[0]
        
        # # 步骤1：添加分类令牌
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # feat = torch.cat([cls_tokens, feat], dim=1)
        
        # 步骤2：注意力融合
        encoded_feat = self.transformer(feat)
        
        global_feat = torch.max(encoded_feat, dim=1)[0]

        output = self.output_proj(global_feat)  # [batch_size, out_dim]

        return output
class FeatureFuser_v2(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super().__init__()
        self.cross_attention = CrossAttention(in_dim, in_dim, in_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 层归一化稳定训练
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, feat1, feat2 = None):
        if feat2 is None:
            return self.output_proj(feat1)

        feat1 = feat1.unsqueeze(1) 
        feat2 = feat2.unsqueeze(1)
        x = self.cross_attention(feat1,feat2)
        feat = x.squeeze(1)
        output = self.output_proj(feat)
        return output

def get_articulation_similarity(demo_feats:torch.tensor, input_feats:torch.tensor):
    # normalize feature
    demo_feats_normalized = torch.nn.functional.normalize(demo_feats, p=2, dim=1) # I * C
    input_feats_normalized = torch.nn.functional.normalize(input_feats, p=2, dim=2) # B * N * C

    # calculate cosine similarity
    similarity = torch.einsum("bnc, ic -> bni", input_feats_normalized, demo_feats_normalized) # B * N * I

    # normalize
    min_vals = similarity.min(dim=-2, keepdim=True)[0] # 
    max_vals = similarity.max(dim=-2, keepdim=True)[0]
    similarity = (similarity - min_vals) / (max_vals - min_vals + 1e-8)

    return similarity

class Limit_Cross_Attention(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super().__init__()
        self.cross_attention = CrossAttention(in_dim, in_dim, in_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, feat1, feat2):     
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(1)
        x = self.cross_attention(feat1, feat2)
        feat = x.squeeze(1)
        output = self.output_proj(feat)
        return output
class Ours_Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=128,
        state_mlp_size=(128, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
    ):
        super().__init__()

        self.state_key = "agent_pos"
        self.environment_point_cloud_key = "environment_point_cloud"
        self.active_object_point_cloud_key = "active_object_point_cloud"
        self.passive_object_point_cloud_key = "passive_object_point_cloud"
        self.active_object_point_feature_key = "active_object_point_feature"
        self.passive_object_point_feature_key = "passive_object_point_feature"

        self.n_output_channels = out_channel + out_channel//2
        # self.n_output_channels = out_channel
        self.use_pc_color = use_pc_color

        # get input {environment_point_cloud, garment_point_cloud, object_point_cloud, state}
        self.environment_point_cloud_shape = observation_space[self.environment_point_cloud_key]
        self.active_object_point_cloud_shape = observation_space[self.active_object_point_cloud_key]
        self.passive_object_point_cloud_shape = observation_space[self.passive_object_point_cloud_key]
        self.active_object_point_feature_shape = observation_space[self.active_object_point_feature_key]
        self.passive_object_point_feature_shape = observation_space[self.passive_object_point_feature_key]
        self.state_shape = observation_space[self.state_key]
        
        cprint(f"[MFMDP_Encoder] environment_point_cloud shape: {self.environment_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] active_object_point_cloud shape: {self.active_object_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] passive_object_point_cloud shape: {self.passive_object_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] active_object_point_feature shape: {self.active_object_point_feature_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] passive_object_point_feature shape: {self.passive_object_point_feature_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] state shape: {self.state_shape}", "yellow")

        self.active_object_feat_dim = self.active_object_point_feature_shape[-1]
        self.passive_object_feat_dim = self.passive_object_point_feature_shape[-1]

        self.extractor_active_object_feature_reducer = PointNetEncoderXYZ(
            in_channels=self.active_object_feat_dim+3,
            out_channels=out_channel//2,
            use_layernorm=True,
            final_norm="layernorm",
        )
        
        self.extractor_passive_object_feature_reducer = PointNetEncoderXYZ(
            in_channels=self.passive_object_feat_dim+3,
            out_channels=out_channel//2,
            use_layernorm=True,
            final_norm="layernorm",
        )
        
        self.extractor_passive_object_points_reducer = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=out_channel//2,
            use_layernorm=True,
            final_norm="layernorm",
        )
        
        self.passive_object_cross_attention = Limit_Cross_Attention(
            in_dim=out_channel//2,
            out_dim=out_channel//2,
        )
        
        self.extractor_env = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=out_channel,
            use_layernorm=True,
            final_norm="layernorm",
        )

        self.fuser = Limit_Cross_Attention(
            in_dim=out_channel//2,
            out_dim=out_channel//2,
        )

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        state_output_dim = state_mlp_size[-1]
        
        self.n_output_channels += state_output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(
                self.state_shape[0], state_output_dim, net_arch, state_mlp_activation_fn
            )
        )

        cprint(f"[MFMDP_Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:

        env_points = observations[self.environment_point_cloud_key]
 
        assert len(env_points.shape) == 3, cprint(
            f"env point cloud shape: {env_points.shape}, length should be 3", "red"
        )
        env_pn_feat = self.extractor_env(env_points)
        
        active_object_points = observations[self.active_object_point_cloud_key]

        assert len(active_object_points.shape) == 3, cprint(
        f"active_object point cloud shape: {active_object_points.shape}, length should be 3", "red"
        )
        active_object_feat = observations[self.active_object_point_feature_key]
        active_object_feat = torch.cat([active_object_feat, active_object_points], dim=-1)        
        active_object_pn_feat = self.extractor_active_object_feature_reducer(active_object_feat)
        # active_object_pn_points = self.extractor_active_object_points_reducer(active_object_points)
        # active_object_pn_all = self.active_object_cross_attention(active_object_pn_feat, active_object_pn_points)
        active_object_pn_all = active_object_pn_feat
        
        passive_object_points = observations[self.passive_object_point_cloud_key]

        assert len(passive_object_points.shape) == 3, cprint(
        f"passive_object point cloud shape: {passive_object_points.shape}, length should be 3", "red"
        )
        passive_object_feat = observations[self.passive_object_point_feature_key]
        passive_object_feat = torch.cat([passive_object_feat, passive_object_points], dim=-1)        
        passive_object_pn_feat = self.extractor_passive_object_feature_reducer(passive_object_feat)
        passive_object_pn_points = self.extractor_passive_object_points_reducer(passive_object_points)
        passive_object_pn_all = self.passive_object_cross_attention(passive_object_pn_feat, passive_object_pn_points)

        feature = self.fuser(active_object_pn_all, passive_object_pn_all)

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        
        final_feat = torch.cat([env_pn_feat, feature, state_feat], dim=-1)

        return final_feat

    def output_shape(self):
        return self.n_output_channels



if __name__ == '__main__':
    
    def test_encoder():
        observation_space = {
            "environment_point_cloud": (1024, 3),
            "garment_point_cloud": (1024, 3),
            "rigid_point_cloud": (1024, 3),
            "articulated_point_cloud": (1024, 3),
            "points_affordance_feature": (1024, 2),
            # "rigid_point_feature": (1024,),
            # "articulated_point_feature": (1024, 768),
            "agent_pos": (9,),
        }

        observations = {
            "environment_point_cloud": torch.rand(4, *observation_space["environment_point_cloud"]),
            "garment_point_cloud": torch.zeros(4, *observation_space["garment_point_cloud"]),
            "rigid_point_cloud": torch.rand(4, *observation_space["rigid_point_cloud"]),
            "articulated_point_cloud": torch.rand(4, *observation_space["articulated_point_cloud"]),
            "points_affordance_feature": torch.rand(4, *observation_space["points_affordance_feature"]),
            # "rigid_point_feature": torch.rand(4, *observation_space["rigid_point_feature"]),
            # "articulated_point_feature": torch.rand(4, *observation_space["articulated_point_feature"]),           
            "agent_pos": torch.rand(4, *observation_space["agent_pos"]),
        }

        encoder = MultiModal_Encoder(
            observation_space=observation_space,
            out_channel=128,
            state_mlp_size=(128, 64),
            state_mlp_activation_fn=nn.ReLU,
            use_pc_color=False,
        )

        output = encoder(observations)
        print("Output shape:", output.shape)
        assert output.shape[-1] == encoder.output_shape(), "Output shape mismatch!"

        print("所有测试通过！")

    test_encoder()
