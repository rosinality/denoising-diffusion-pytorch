import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config

from model import UNet
from config import DiffusionConfig
from diffusion import GaussianDiffusion, make_beta_schedule


@torch.no_grad()
def p_sample_loop(self, model, noise, device, noise_fn=torch.randn, capture_every=1000):
    img = noise
    imgs = []

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
        )

        if i % capture_every == 0:
            imgs.append(img)

    imgs.append(img)

    return imgs


if __name__ == "__main__":
    conf = load_config(DiffusionConfig, "config/diffusion.conf", show=False)
    ckpt = torch.load("ckpt-2400k.pt")
    model = conf.model.make()
    model.load_state_dict(ckpt["ema"])
    model = model.to("cuda")
    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to("cuda")
    noise = torch.randn([16, 3, 256, 256], device="cuda")
    imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=10)
    imgs = imgs[1:]

    save_image(imgs[-1], "sample.png", normalize=True, range=(-1, 1), nrow=4)
