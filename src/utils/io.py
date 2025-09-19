import torch
import torchvision.utils as vutils
from pathlib import Path

def save_sample_grid(blur, sharp, pred, epoch, step, save_dir):
    B = min(4, blur.shape[0])
    tiles = []
    for i in range(B):
        tiles += [blur[i:i+1].clamp(0,1), pred[i:i+1].clamp(0,1), sharp[i:i+1].clamp(0,1)]
    grid = torch.cat(tiles, dim=0)
    grid = vutils.make_grid(grid, nrow=3, padding=2)
    out = Path(save_dir) / "samples" / f"epoch{epoch:03d}_step{step:06d}.png"
    vutils.save_image(grid, str(out))
