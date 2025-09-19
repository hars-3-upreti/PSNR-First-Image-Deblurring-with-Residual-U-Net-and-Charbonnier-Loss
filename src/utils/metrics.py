import torch
import torch.nn.functional as F
try:
    from pytorch_msssim import ssim as ssim_gpu
except Exception as e:
    ssim_gpu = None
    print("[WARN] install pytorch-msssim for SSIM metrics")

@torch.no_grad()
def _to01_safe(x: torch.Tensor) -> torch.Tensor:
    mn, mx = float(x.min()), float(x.max())
    frac = float(((x < 0) | (x > 1)).float().mean().item())
    if frac < 0.05: return x.clamp(0,1)
    if (mn < -0.8) and (mx > 0.8): return (x + 1) * 0.5
    return x.clamp(0,1)

@torch.no_grad()
def psnr_torch_batch(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr = sr.clamp(0,1); hr = hr.clamp(0,1)
    mse = F.mse_loss(sr, hr, reduction="none").mean(dim=(1,2,3)).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).mean().item())

@torch.no_grad()
def ssim_torch_batch(sr: torch.Tensor, hr: torch.Tensor) -> float:
    if ssim_gpu is None:
        return float('nan')
    sr = sr.clamp(0,1); hr = hr.clamp(0,1)
    return float(ssim_gpu(sr, hr, data_range=1.0, size_average=True).item())
