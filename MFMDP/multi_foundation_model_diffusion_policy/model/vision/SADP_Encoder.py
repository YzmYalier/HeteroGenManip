import sys
import copy
import pdb
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from termcolor import cprint

from multi_foundation_model_diffusion_policy.model.vision.pointnet2 import PointNet2Global



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

        assert in_channels == 3, cprint(
            f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red"
        )

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

class MFMDP_Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        out_channel=128,
        state_mlp_size=(128, 64),
        state_mlp_activation_fn=nn.ReLU,
        use_pc_color=False,
    ):
        super().__init__()

        self.state_key = "agent_pos"
        self.environment_point_cloud_key = "environment_point_cloud"
        self.garment_point_cloud_key = "garment_point_cloud"
        self.object_point_cloud_key = "object_point_cloud"
        self.points_affordance_feature_key = "points_affordance_feature"
        self.rgb_image_key = "image"
        self.n_output_channels = out_channel + out_channel//2
        self.use_pc_color = use_pc_color

        # get input {environment_point_cloud, garment_point_cloud, object_point_cloud, state}
        self.environment_point_cloud_shape = observation_space[self.environment_point_cloud_key]
        self.garment_point_cloud_shape = observation_space[self.garment_point_cloud_key]
        self.object_point_cloud_shape = observation_space[self.object_point_cloud_key]
        self.points_affordance_feature_shape = observation_space[self.points_affordance_feature_key]
        self.state_shape = observation_space[self.state_key]
        
        cprint(f"[MFMDP_Encoder] environment_point_cloud shape: {self.environment_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] garment_point_cloud shape: {self.garment_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] object_point_cloud shape: {self.object_point_cloud_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] points_affordance_feature shape: {self.points_affordance_feature_shape}", "yellow")
        cprint(f"[MFMDP_Encoder] state shape: {self.state_shape}", "yellow")
        
        self.extractor_garment = PointNet2Global(
            affordance_feature=True,
            feature_dim=out_channel//4,
        )

        self.extractor_object = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=out_channel//4,
            use_layernorm=True,
            final_norm="layernorm",
        )
        
        self.extractor_env = PointNetEncoderXYZ(
            in_channels=3,
            out_channels=out_channel,
            use_layernorm=True,
            final_norm="layernorm",
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

        garment_points = observations[self.garment_point_cloud_key]
        affordance_feature = observations[self.points_affordance_feature_key]
        garment = torch.cat([garment_points, affordance_feature], dim=-1)
        assert len(garment.shape) == 3, cprint(
            f"garment point cloud shape: {garment.shape}, length should be 3", "red"
        )
        garment_pn_feat = self.extractor_garment(garment)  
        
        object_points = observations[self.object_point_cloud_key]
        assert len(object_points.shape) == 3, cprint(
            f"object point cloud shape: {object_points.shape}, length should be 3", "red"
        )
        object_pn_feat = self.extractor_object(object_points)  
        

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        final_feat = torch.cat([env_pn_feat, garment_pn_feat, object_pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels



if __name__ == '__main__':

    def test_encoder():
        observation_space = {
            "environment_point_cloud": (1024, 3),
            "garment_point_cloud": (1024, 3),
            "object_point_cloud": (1024, 3),
            "points_affordance_feature": (1024, 2),
            "agent_pos": (60,),
        }

        observations = {
            "environment_point_cloud": torch.rand(4, *observation_space["environment_point_cloud"]),
            "garment_point_cloud": torch.rand(4, *observation_space["garment_point_cloud"]),
            "object_point_cloud": torch.rand(4, *observation_space["object_point_cloud"]),
            "points_affordance_feature": torch.rand(4, *observation_space["points_affordance_feature"]),
            "agent_pos": torch.rand(4, *observation_space["agent_pos"]),
        }

        encoder = MFMDP_Encoder(
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
