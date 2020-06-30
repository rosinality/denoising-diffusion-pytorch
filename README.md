# denoising-diffusion-pytorch
Implementation of Denoising Diffusion Probabilistic Models in PyTorch

## Training

First prepare lmdb dataset:

```bash
python prepare_data.py --size [SIZES, e.g. 128,256] --out [LMDB NAME] [DATASET PATH]
```

Then run training looop!


```bash
python train.py --n_gpu [NUMBER OF GPUS FOR TRAINING] --conf diffusion.conf 
```