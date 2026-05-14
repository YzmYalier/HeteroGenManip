import os
import pathlib
import pdb

import dill
import hydra
import torch
from omegaconf import OmegaConf

from MFMDP.train import TrainMFMDPWorkspace


class MFMDP:
    def __init__(self, task_name, checkpoint_num, data_num, device="cuda:0") -> None:
        # load checkpoint
        checkpoint = f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt'
        payload = torch.load(open('./MFMDP/'+checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, checkpoint_num, task_name+"_"+str(data_num))
        self.policy.to(device)
        self.policy.eval()

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation):
        action = self.env_runner.get_action(self.policy, observation)
        return action

    def get_policy_and_runner(self, cfg, checkpoint_num, task_name):
        workspace = TrainMFMDPWorkspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, checkpoint_num, task_name)
        return policy, env_runner



