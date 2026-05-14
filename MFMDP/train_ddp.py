import copy
import os
import pathlib
import pdb
import random
import shutil
import sys
import threading
import time

import dill
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import tqdm
import wandb
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
import pdb

sys.path.append('MFMDP')

from multi_foundation_model_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from multi_foundation_model_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from multi_foundation_model_diffusion_policy.dataset.base_dataset import BaseDataset
from multi_foundation_model_diffusion_policy.env_runner.base_runner import BaseRunner
from multi_foundation_model_diffusion_policy.model.common.lr_scheduler import get_scheduler
from multi_foundation_model_diffusion_policy.model.diffusion.ema_model import EMAModel
from multi_foundation_model_diffusion_policy.policy.mfmdp import MFMDP

OmegaConf.register_new_resolver("eval", eval, replace=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class TrainMFMDPWorkspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, rank, world_size, output_dir=None):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if rank == 0:
            print("MFMDP:", os.getcwd())
        # configure model
        self.model: MFMDP = hydra.utils.instantiate(cfg.policy)
        self.model = self.model.to(rank)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])

        self.ema_model: MFMDP = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model.module)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)
            self.ema_model = self.ema_model.to(rank)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.logging.mode == "online" and self.rank == 0:
            WANDB = True
        else:
            WANDB = False

        if cfg.training.debug:
            cfg.training.num_epochs = 5
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 1
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 5
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = False
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_ROLLOUT = False
        RUN_VALIDATION = True  # reduce time cost

        # resume training
        if cfg.training.resume and self.rank == 0:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(
            f"dataset must be BaseDataset, got {type(dataset)}"
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)

        self.model.module.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None

        # cfg.logging.name = str(cfg.task.name)
        if self.rank == 0:
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        if self.rank == 0:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
            )

        # device transfer
        device = torch.device(self.rank)

        # save batch for sampling
        train_sampling_batch = None
        checkpoint_num = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        for local_epoch_idx in range(cfg.training.num_epochs):
            train_sampler.set_epoch(local_epoch_idx)
            val_sampler.set_epoch(local_epoch_idx)
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
                disable=self.rank != 0
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model.module)
                        # 同步 EMA 模型参数
                        self._sync_ema_model()
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose and self.rank == 0:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model.module
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=self.rank != 0
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True)
                            )
                            loss, loss_dict = self.model.module.compute_loss(batch)
                            val_losses.append(loss)
                            if self.rank == 0:
                                print(f"epoch {self.epoch}, eval loss: ", float(loss.cpu()))
                            if (
                                cfg.training.max_val_steps is not None
                            ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log["val_loss"] = val_loss

            # checkpoint
            if (
                (self.epoch + 1) % cfg.training.checkpoint_every
            ) == 0 and cfg.checkpoint.save_ckpt and self.rank == 0:

                if not cfg.policy.use_pc_color:
                    if not os.path.exists(f"./checkpoints/{self.cfg.task.name}"):
                        os.makedirs(f"./checkpoints/{self.cfg.task.name}")
                    save_path = (
                        f"./checkpoints/{self.cfg.task.name}/{self.epoch + 1}.ckpt"
                    )
                else:
                    if not os.path.exists(f"./checkpoints/{self.cfg.task.name}_w_rgb"):
                        os.makedirs(f"./checkpoints/{self.cfg.task.name}_w_rgb")
                    save_path = (
                        f"./checkpoints/{self.cfg.task.name}_w_rgb/{self.epoch + 1}.ckpt"
                    )

                self.save_checkpoint(save_path)

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
            
    def _sync_ema_model(self):
        """
        同步所有进程的 EMA 模型参数
        """
        for param in self.ema_model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

        for buffer in self.ema_model.buffers():
            dist.all_reduce(buffer.data, op=dist.ReduceOp.SUM)
            buffer.data /= dist.get_world_size()
            
    def get_policy_and_runner(self, cfg, checkpoint_num=3000, task_name="Hang_Tops_stage_1"):
        # load the latest checkpoint
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=None
        )
        assert isinstance(env_runner, BaseRunner)

        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(
                f"MFMDP/checkpoints/{task_name}/{checkpoint_num}.ckpt"
            )
        else:
            ckpt_file = pathlib.Path(
                f"MFMDP/checkpoints/{task_name}_w_rgb/{checkpoint_num}.ckpt"
            )

        print("ckpt file exist:", ckpt_file.is_file())

        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", "magenta")
            self.load_checkpoint(path=ckpt_file)

        policy = self.model.module
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda(self.rank)
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        print("saved in ", path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        if key == 'model':
                            payload["state_dicts"][key] = value.module.state_dict()
                        else:
                            payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                score = float(ckpt.split("test_mean_score=")[1].split(".ckpt")[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                if key == 'model':
                    self.__dict__[key].module.load_state_dict(value, **kwargs)
                else:
                    self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

def run_train(rank, world_size, cfg):
    setup(rank, world_size)
    workspace = TrainMFMDPWorkspace(cfg, rank, world_size)
    workspace.run()
    cleanup()

@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("multi_foundation_model_diffusion_policy", "config")
    ),
)
def main(cfg):
    world_size = torch.cuda.device_count()
    mp.spawn(run_train, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()