import torch
from torch.cuda.amp import autocast

@torch.no_grad()
def infer_fp32_tta(model, x, mode='x8'):
    model.eval()
    outs = []
    rots  = (0,1,2,3) if mode == 'x8' else (0,2)
    flips = (False, True)
    for r in rots:
        xr = torch.rot90(x, k=r, dims=(-2,-1))
        for f in flips:
            xrf = torch.flip(xr, dims=(-1,)) if f else xr
            with autocast(enabled=False):
                y = model(xrf).clamp_(0,1)
            if f: y = torch.flip(y, dims=(-1,))
            outs.append(torch.rot90(y, k=(4-r)%4, dims=(-2,-1)))
    return torch.stack(outs, 0).mean(0)
