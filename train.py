import os, math, time, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.gopro import GoProDataset
from src.models.psnr_unet import PSNRResUNet, init_weights
from src.losses.charbonnier import build_charb_msssim
from src.utils.ema import ModelEMA
from src.utils.metrics import psnr_torch_batch, ssim_torch_batch
from src.utils.sched import build_warmup_cosine_scheduler
from src.utils.io import save_sample_grid
from src.utils.tta import infer_fp32_tta

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to GOPRO_Large')
    ap.add_argument('--save', type=str, default='runs_deblur', help='Run dir')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--ms_ssim_w', type=float, default=0.05, help='0.0 to disable')
    ap.add_argument('--lambda_id', type=float, default=0.02, help='identity regularizer weight')
    ap.add_argument('--preview_freq', type=int, default=200)
    ap.add_argument('--validate_every', type=int, default=1)
    ap.add_argument('--tta_every', type=int, default=5)
    ap.add_argument('--num_workers', type=int, default=0)  # Windows/Notebook safe
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(args.save)
    (save_dir/'samples').mkdir(parents=True, exist_ok=True)
    (save_dir/'checkpoints').mkdir(parents=True, exist_ok=True)

    # Data
    train_set = GoProDataset(args.data, split='train', crop_size=args.crop,
                             use_flips=True, use_rot90=True, jitter_strength=0.05)
    val_set   = GoProDataset(args.data, split='test', crop_size=None,
                             use_flips=False, use_rot90=False, jitter_strength=0.0)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Model
    gen = PSNRResUNet(in_ch=3, out_ch=3, base=48, n_resblocks=5).to(device)
    init_weights(gen, 'kaiming')

    # Losses
    criterion = build_charb_msssim(w_charb=1.0, w_msssim=args.ms_ssim_w)

    # Optimizer/Sched/AMP/EMA
    g_optimizer = AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    steps_per_epoch   = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / args.accum)
    total_updates     = args.epochs * updates_per_epoch
    warmup_updates    = args.warmup_epochs * updates_per_epoch
    scheduler = build_warmup_cosine_scheduler(g_optimizer, total_updates, warmup_updates,
                                              base_lr=args.lr, min_lr=args.min_lr)
    scaler = GradScaler(enabled=(device=='cuda'))
    ema = ModelEMA(gen, decay=0.9999)

    best_psnr, no_improve, early_stop_pat = -1e9, 0, 40

    for epoch in range(1, args.epochs + 1):
        gen.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, (blur, sharp) in pbar:
            blur  = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)

            with autocast(enabled=(device=='cuda')):
                pred = gen(blur)
                pred_c  = pred.clamp(0,1)
                sharp_c = sharp.clamp(0,1)
                blur_c  = blur.clamp(0,1)
                # main + tiny identity
                loss_main = criterion(pred_c, sharp_c)
                loss_id   = F.smooth_l1_loss(pred_c, blur_c)  # simple L1-like identity
                loss = (loss_main + args.lambda_id * loss_id) / args.accum

            scaler.scale(loss).backward()

            if (step + 1) % args.accum == 0:
                scaler.unscale_(g_optimizer)
                if args.grad_clip is not None:
                    clip_grad_norm_(gen.parameters(), max_norm=args.grad_clip)
                scaler.step(g_optimizer)
                scaler.update()
                g_optimizer.zero_grad(set_to_none=True)

                ema.update(gen)
                scheduler.step()

            epoch_loss += loss.item() * args.accum

            if step % args.preview_freq == 0:
                with torch.no_grad():
                    pred_ema = ema.ema(blur).clamp(0,1)
                    p_blur = psnr_torch_batch(blur.clamp(0,1), sharp.clamp(0,1))
                    p_pred = psnr_torch_batch(pred_ema, sharp.clamp(0,1))
                    pbar.set_postfix(loss=f"{(epoch_loss/(step+1)):.4f}",
                                     lr=f"{g_optimizer.param_groups[0]['lr']:.2e}",
                                     psnr=f"{p_pred:.2f}")
                    save_sample_grid(blur, sharp, pred_ema, epoch, step, save_dir)

        # ---- Validation (fast most epochs; TTA every N) ----
        if (epoch % args.validate_every) == 0:
            gen.eval()
            m = ema.ema
            ps_list, ss_list = [], []
            for i, (blur, sharp) in enumerate(val_loader):
                blur, sharp = blur.to(device), sharp.to(device)
                if (epoch % args.tta_every) == 0:
                    pred = infer_fp32_tta(m, blur, mode='x8')
                else:
                    with torch.no_grad():
                        pred = m(blur)
                sr = pred.clamp(0,1); hr = sharp.clamp(0,1)
                ps_list.append(psnr_torch_batch(sr, hr))
                ss_list.append(ssim_torch_batch(sr, hr))
            val_psnr = float(sum(ps_list)/len(ps_list))
            val_ssim = float(sum(ss_list)/len(ss_list))
        else:
            val_psnr, val_ssim = float('nan'), float('nan')

        print(f"[Epoch {epoch:03d}] loss={epoch_loss/len(train_loader):.4f} | val PSNR={val_psnr:.3f} SSIM={val_ssim:.4f}")

        if not math.isnan(val_psnr) and (val_psnr > best_psnr + 1e-4):
            best_psnr, no_improve = val_psnr, 0
            torch.save({"ema": ema.ema.state_dict()}, str(save_dir/"checkpoints"/"best_ema.pth"))
            print(f"✅ Saved best EMA @ epoch {epoch} (PSNR={val_psnr:.3f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_pat:
                print(f"Early stopping (best PSNR {best_psnr:.3f} dB).")
                break

    print(f"Training complete. Best EMA PSNR: {best_psnr:.3f} dB")

if __name__ == "__main__":
    main()
