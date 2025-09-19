import torch
import torch.nn as nn

class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9999, device=None):
        self.ema = type(model)().to(device or next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.ema.eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)
