import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=True),
              nn.InstanceNorm2d(out_ch, affine=True)]
    if act: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = conv_block(ch, ch, act=True)
        self.c2 = conv_block(ch, ch, act=False)
    def forward(self, x):
        return F.relu(x + self.c2(self.c1(x)), inplace=True)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class PSNRResUNet(nn.Module):
    """Linear output in [0,1] (clamp in loss/eval)."""
    def __init__(self, in_ch=3, out_ch=3, base=48, n_resblocks=5):
        super().__init__()
        self.e1 = conv_block(in_ch, base, act=True)         # H
        self.d1 = Down(base, base*2)                        # H/2
        self.d2 = Down(base*2, base*4)                      # H/4
        self.d3 = Down(base*4, base*8)                      # H/8

        self.b = nn.Sequential(*[ResidualBlock(base*8) for _ in range(n_resblocks)])

        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4*2, base*2)
        self.u1 = Up(base*2*2, base)

        self.f = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 3, 1, 1)
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.d1(e1)
        e3 = self.d2(e2)
        e4 = self.d3(e3)
        b  = self.b(e4)
        u3 = self.u3(b); u3 = torch.cat([u3, e3], dim=1)
        u2 = self.u2(u3); u2 = torch.cat([u2, e2], dim=1)
        u1 = self.u1(u2); u1 = torch.cat([u1, e1], dim=1)
        out = self.f(u1)
        return out

def init_weights(net, init_type='kaiming'):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0.2)
            else:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias is not None:   nn.init.zeros_(m.bias)
