import os, glob, random
from typing import Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class GoProDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",
                 crop_size: Optional[int] = None,
                 use_flips: bool = True,
                 use_rot90: bool = True,
                 jitter_strength: float = 0.05):
        """
        Expects: {root_dir}/{split}/*/(blur|sharp)/*.png
        Returns tensors in [0,1].
        """
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.crop_size = crop_size
        self.use_flips = use_flips
        self.use_rot90 = use_rot90
        self.jitter_strength = jitter_strength if split == "train" else 0.0

        self.blur_paths = sorted(glob.glob(os.path.join(self.root_dir, "*", "blur", "*.png")))
        self.sharp_paths = [p.replace(os.sep+"blur"+os.sep, os.sep+"sharp"+os.sep) for p in self.blur_paths]
        print(f"[INFO] {split}: {len(self.blur_paths)} pairs | crop={crop_size}")

        # transforms
        self.to_tensor = T.ToTensor()
        if self.jitter_strength > 0:
            s = self.jitter_strength
            self.jitter = T.ColorJitter(brightness=s, contrast=s)
        else:
            self.jitter = None

    def __len__(self): return len(self.blur_paths)

    def _random_crop(self, b_img, s_img, size):
        W, H = b_img.size
        if (H < size) or (W < size):
            # center-crop to min size
            th = min(H, size); tw = min(W, size)
            i = max(0, (H - th)//2); j = max(0, (W - tw)//2)
        else:
            i = random.randint(0, H - size)
            j = random.randint(0, W - size)
            th = tw = size
        b = b_img.crop((j, i, j + tw, i + th))
        s = s_img.crop((j, i, j + tw, i + th))
        return b, s

    def __getitem__(self, idx):
        b_path = self.blur_paths[idx]; s_path = self.sharp_paths[idx]
        b_img = Image.open(b_path).convert("RGB")
        s_img = Image.open(s_path).convert("RGB")

        if self.crop_size is not None:
            b_img, s_img = self._random_crop(b_img, s_img, self.crop_size)

        # flips/rot90 (train-time or if explicitly enabled)
        if self.use_flips and random.random() < 0.5:
            b_img = b_img.transpose(Image.FLIP_LEFT_RIGHT)
            s_img = s_img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.use_rot90:
            k = random.randint(0, 3)
            if k:
                b_img = b_img.rotate(90*k, expand=True)
                s_img = s_img.rotate(90*k, expand=True)

        if self.jitter is not None:
            b_img = self.jitter(b_img)

        b, s = self.to_tensor(b_img), self.to_tensor(s_img)  # [0,1]
        return b, s
