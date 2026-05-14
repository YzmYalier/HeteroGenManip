import sys
import copy
from typing import Dict

import numpy as np
import torch
from multi_foundation_model_diffusion_policy.common.pytorch_util import dict_apply
from multi_foundation_model_diffusion_policy.common.replay_buffer import ReplayBuffer
from multi_foundation_model_diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from multi_foundation_model_diffusion_policy.dataset.base_dataset import BaseDataset
from multi_foundation_model_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class RobotDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        print(f"Loading dataset from {zarr_path}")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["state", "action", "environment_point_cloud",
                             "active_object_point_cloud", "passive_object_point_cloud",
                             "active_object_point_feature", "passive_object_point_feature"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "environment_point_cloud": self.replay_buffer["environment_point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        data = {
            "obs": {
                "environment_point_cloud": sample["environment_point_cloud"][:,],  # T, 2048, 6
                "active_object_point_cloud": sample["active_object_point_cloud"][:,],  # T, 2048, 3
                "passive_object_point_cloud": sample["passive_object_point_cloud"][:,],  # T, 2048, 3
                "active_object_point_feature": sample["active_object_point_feature"][:,],  # T, 2048, 2
                "passive_object_point_feature": sample["passive_object_point_feature"][:,],  # T, 1024, 2
                "agent_pos": sample["state"][:,],  # T, D_pos
            },
            "action": sample["action"],  # T, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
if __name__=="__main__":
    dataset = RobotDataset(
        zarr_path="/home/admin02/DexGarmentLab/MFMDP/data/Hang_Coat_stage_1_10.zarr",
        horizon=24,
        val_ratio=0.02,
    )
    print("ready")
