import math
from torch.optim.lr_scheduler import LambdaLR

def build_warmup_cosine_scheduler(optimizer, total_updates, warmup_updates, base_lr=1e-4, min_lr=1e-5):
    ratio = min_lr / base_lr
    def lr_lambda(step):
        step = min(step, total_updates)
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        t = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda)
