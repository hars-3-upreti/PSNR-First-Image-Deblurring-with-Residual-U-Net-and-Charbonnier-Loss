import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """sqrt((x - y)^2 + eps^2)"""
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps2 = eps * eps
        self.reduction = reduction
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

def build_charb_msssim(w_charb=1.0, w_msssim=0.05, eps=1e-3):
    if w_msssim <= 0:
        return CharbonnierLoss(eps=eps)
    try:
        from pytorch_msssim import MS_SSIM
        class CharbPlusMSSSIM(nn.Module):
            def __init__(self):
                super().__init__()
                self.charb = CharbonnierLoss(eps=eps)
                self.ms = MS_SSIM(data_range=1.0, size_average=True, channel=3)
            def forward(self, pred, target):
                pred = pred.clamp(0,1); target = target.clamp(0,1)
                l1 = self.charb(pred, target)
                l2 = 1.0 - self.ms(pred, target)
                return w_charb * l1 + w_msssim * l2
        return CharbPlusMSSSIM()
    except Exception:
        print("[WARN] pytorch-msssim not found; using pure Charbonnier")
        return CharbonnierLoss(eps=eps)
