import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tensorfn import load_arg_config
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from dataset import MultiResolutionDataset
from config import DiffusionConfig


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device):
    loader = sample_data(loader)

    pbar = range(conf.training.n_iter + 1)

    for i in pbar:
        epoch, img = next(loader)
        img = img.to(device)
        time = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(device)
        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            if i % conf.evaluate.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}")

            if i % conf.evaluate.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "conf": conf,
                    },
                    f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
                )


def main(conf):
    device = "cuda"
    beta_schedule = "linear"
    beta_start = 1e-4
    beta_end = 2e-2
    n_timestep = 1000

    conf.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    train_set = MultiResolutionDataset(
        conf.dataset.path, transform, conf.dataset.resolution
    )
    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    model = UNet(
        conf.model.in_channel,
        conf.model.channel,
        channel_multiplier=conf.model.channel_multiplier,
        n_res_blocks=conf.model.n_res_blocks,
        attn_strides=conf.model.attn_strides,
        dropout=conf.model.dropout,
        fold=conf.model.fold,
    )
    model = model.to(device)
    ema = UNet(
        conf.model.in_channel,
        conf.model.channel,
        channel_multiplier=conf.model.channel_multiplier,
        n_res_blocks=conf.model.n_res_blocks,
        attn_strides=conf.model.attn_strides,
        dropout=conf.model.dropout,
        fold=conf.model.fold,
    )
    ema = ema.to(device)

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    betas = make_beta_schedule(beta_schedule, beta_start, beta_end, n_timestep)
    diffusion = GaussianDiffusion(betas).to(device)

    train(conf, train_loader, model, ema, diffusion, optimizer, scheduler, device)


if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
