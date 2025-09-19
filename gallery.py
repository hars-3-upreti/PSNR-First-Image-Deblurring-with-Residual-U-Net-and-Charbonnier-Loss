import os, argparse
from pathlib import Path
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.gopro import GoProDataset
from src.models.psnr_unet import PSNRResUNet

# ===== helpers (inlined) =====
@torch.no_grad()
def _to01(x):
    mn, mx = float(x.min()), float(x.max())
    frac = float(((x < 0) | (x > 1)).float().mean().item())
    if frac < 0.05: return x.clamp(0,1)
    if (mn < -0.8) and (mx > 0.8): return (x + 1)*0.5
    return x.clamp(0,1)

def _ensure_bchw3(x):
    if x.dim()==3: x = x.unsqueeze(0)
    B,C,H,W = x.shape
    if C==3: return x
    if C==1: return x.repeat(1,3,1,1)
    if C%3==0: return x.view(B,3,C//3,H,W).mean(2)
    return x[:,:3]

@torch.no_grad()
def _psnr(a, b):
    a=_to01(a); b=_to01(b)
    mse = torch.mean((a-b)**2).clamp_min(1e-12)
    return float(10.0*torch.log10(1.0/mse).item())

@torch.no_grad()
def _sobel(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    y = x.mean(1, keepdim=True)
    gx = torch.nn.functional.conv2d(y, kx, padding=1)
    gy = torch.nn.functional.conv2d(y, ky, padding=1)
    return torch.sqrt(gx*gx+gy*gy)

@torch.no_grad()
def _crop_around_max_residual(sr, hr, size=256):
    r = (sr-hr).abs().mean(1, keepdim=True)
    H,W = r.shape[-2:]
    yx = torch.nonzero(r[0,0]==r[0,0].max(), as_tuple=False)[0]
    cy,cx = int(yx[0]), int(yx[1])
    h=w=min(size,H,W)
    y0=max(0,min(H-h,cy-h//2)); x0=max(0,min(W-w,cx-w//2))
    return sr[...,y0:y0+h,x0:x0+w], hr[...,y0:y0+h,x0:x0+w], (y0,x0,h,w)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--save', type=str, default='runs_deblur/gallery')
    ap.add_argument('--top_k', type=int, default=12)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_full  = Path(args.save)/"full";  out_full.mkdir(parents=True, exist_ok=True)
    out_crops = Path(args.save)/"crops"; out_crops.mkdir(parents=True, exist_ok=True)

    # model
    ck = torch.load(args.ckpt, map_location=device)
    state = ck["ema"] if isinstance(ck, dict) and "ema" in ck else ck
    net = PSNRResUNet(in_ch=3, out_ch=3, base=48, n_resblocks=5).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()

    # data
    test_set = GoProDataset(args.data, split='test', crop_size=None, use_flips=False, use_rot90=False, jitter_strength=0.0)
    loader   = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # pass 1: rank by ΔPSNR
    deltas = []
    names  = getattr(loader.dataset, "blur_paths", None)
    for i, (blur, sharp) in enumerate(tqdm(loader, desc="Ranking")):
        blur, sharp = blur.to(device), sharp.to(device)
        pred = net(blur).clamp(0,1)
        p0 = _psnr(blur, sharp); p1 = _psnr(pred, sharp)
        deltas.append((p1-p0, p1, p0, i))
    deltas.sort(key=lambda t:t[0], reverse=True)
    chosen = [idx for _,_,_,idx in deltas[:args.top_k]]

    # pass 2: save panels
    for i, (blur, sharp) in enumerate(tqdm(loader, desc="Saving")):
        if i not in chosen: continue
        blur, sharp = blur.to(device), sharp.to(device)
        pred = net(blur).clamp(0,1)
        name = os.path.splitext(os.path.basename(names[i]))[0] if names else f"idx{i:05d}"

        p_blur=_psnr(blur, sharp); p_pred=_psnr(pred, sharp); delta=p_pred-p_blur

        b3=_ensure_bchw3(_to01(blur)); p3=_ensure_bchw3(_to01(pred)); s3=_ensure_bchw3(_to01(sharp))
        ds=_ensure_bchw3((p3-s3).abs().mul(4).clamp(0,1))
        db=_ensure_bchw3((p3-b3).abs().mul(4).clamp(0,1))
        eb=_ensure_bchw3(_sobel(b3)); ep=_ensure_bchw3(_sobel(p3))

        panel_full=torch.cat([b3,p3,s3,ds,db,eb,ep], dim=0)
        grid_full=vutils.make_grid(panel_full, nrow=7, padding=2)
        vutils.save_image(grid_full, str(out_full/f"{name}_PSNR{p_pred:.2f}_d{delta:+.2f}.png"))

        pc,sc,(y0,x0,h,w)=_crop_around_max_residual(p3,s3,size=256)
        bc=b3[...,y0:y0+h,x0:x0+w]
        dsc=_ensure_bchw3((pc-sc).abs().mul(4).clamp(0,1))
        dbc=_ensure_bchw3((pc-bc).abs().mul(4).clamp(0,1))
        ebc=_ensure_bchw3(_sobel(bc)); epc=_ensure_bchw3(_sobel(pc))
        panel_crop=torch.cat([bc,pc,sc,dsc,dbc,ebc,epc], dim=0)
        grid_crop=vutils.make_grid(panel_crop, nrow=7, padding=2)
        vutils.save_image(grid_crop, str(out_crops/f"{name}_CROP_P{p_pred:.2f}_d{delta:+.2f}.png"))

    print(f"Saved -> {out_full.resolve()} and {out_crops.resolve()}")

if __name__ == "__main__":
    main()
