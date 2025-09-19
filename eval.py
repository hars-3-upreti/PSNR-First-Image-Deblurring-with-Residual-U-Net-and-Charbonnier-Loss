import os, csv, argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F

from src.data.gopro import GoProDataset
from src.models.psnr_unet import PSNRResUNet
from src.utils.metrics import _to01_safe, psnr_torch_batch, ssim_torch_batch
from src.utils.tta import infer_fp32_tta

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--save', type=str, default='runs_deblur/final_eval_rgb')
    ap.add_argument('--tta', action='store_true', help='use x8 TTA')
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.save); (out_dir/'images').mkdir(parents=True, exist_ok=True)

    # model
    ck = torch.load(args.ckpt, map_location=device)
    state = ck["ema"] if isinstance(ck, dict) and "ema" in ck else ck
    gen = PSNRResUNet(in_ch=3, out_ch=3, base=48, n_resblocks=5).to(device)
    gen.load_state_dict(state, strict=True)
    gen.eval()

    # data (FULL images)
    test_set = GoProDataset(args.data, split='test', crop_size=None, use_flips=False, use_rot90=False, jitter_strength=0.0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    ps_list, ss_list, rows = [], [], []

    print(f"[Eval] use_tta={args.tta}")
    for i, (blur, sharp) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Final eval"):
        blur, sharp = blur.to(device), sharp.to(device)
        if args.tta:
            pred = infer_fp32_tta(gen, blur, mode='x8')
        else:
            with torch.no_grad():
                pred = gen(blur)
        sr = _to01_safe(pred); hr = _to01_safe(sharp)
        p = psnr_torch_batch(sr, hr); s = ssim_torch_batch(sr, hr)
        ps_list.append(p); ss_list.append(s)

        # filename
        try:
            name = os.path.splitext(os.path.basename(test_loader.dataset.blur_paths[i]))[0]
        except Exception:
            name = f"idx{i:05d}"

        vutils.save_image(sr.clamp(0,1), str(out_dir/'images'/f"{name}.png"))
        rows.append([name, p, s])

    mean_psnr = float(sum(ps_list)/len(ps_list))
    mean_ssim = float(sum(ss_list)/len(ss_list))
    with open(out_dir/'metrics.csv', "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","psnr_rgb","ssim_rgb"])
        for r in rows: w.writerow(r)
        w.writerow(["MEAN", mean_psnr, mean_ssim])

    print(f"[Eval DONE] Mean PSNR={mean_psnr:.3f} dB, SSIM={mean_ssim:.4f}")
    print(f"Predictions -> {out_dir/'images'}")
    print(f"Metrics CSV -> {out_dir/'metrics.csv'}")

if __name__ == "__main__":
    main()
