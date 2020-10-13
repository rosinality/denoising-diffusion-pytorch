from typing import Optional, List
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool
from tensorfn.config import Config, Optimizer, Scheduler, DataLoader, get_model

import diffusion
import model


class Dataset(Config):
    name: StrictStr
    path: StrictStr
    resolution: StrictInt


"""class Model(Config):
    in_channel: StrictInt
    channel: StrictInt
    channel_multiplier: List[StrictInt]
    n_res_blocks: StrictInt
    attn_strides: List[StrictInt]
    attn_heads: StrictInt
    use_affine_time: StrictBool
    dropout: StrictFloat
    fold: StrictInt"""

Model = get_model("UNet")


class Diffusion(Config):
    beta_schedule: get_model("make_beta_schedule")


class Training(Config):
    n_iter: StrictInt
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader


class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt


class DiffusionConfig(Config):
    n_gpu: Optional[StrictInt]
    n_machine: Optional[StrictInt]
    machine_rank: Optional[StrictInt]
    dist_url: Optional[StrictStr]
    distributed: Optional[StrictBool]
    ckpt: Optional[StrictStr]

    dataset: Dataset
    model: Model
    diffusion: Diffusion
    training: Training
    evaluate: Eval
