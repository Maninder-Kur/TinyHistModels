import torch.nn as nn
import math

B0_CFG = [
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, kernel_size):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = in_ch == out_ch and stride == 1
        self.expand = (
            nn.Identity()
            if expand_ratio == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
            )
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                mid,
                mid,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid,
                bias=False,
            ),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        se_mid = max(1, mid // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, se_mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_mid, mid, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.dwconv(out)
        out = out * self.se(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out


class EfficientNetTinyStudent(nn.Module):
    def __init__(self, num_genes=845, phi=-5.0):
        super().__init__()
        (ALPHA, BETA) = (1.2, 1.1)
        depth_mult = ALPHA**phi
        width_mult = BETA**phi

        def scale_channels(ch):
            return max(4, int(ch * width_mult))

        def scale_repeats(r):
            return max(1, int(math.ceil(r * depth_mult)))

        self.stem = nn.Sequential(
            nn.Conv2d(3, scale_channels(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scale_channels(32)),
            nn.SiLU(inplace=True),
        )
        blocks = []
        in_ch = scale_channels(32)
        for (expand, ch, repeats, stride, k) in B0_CFG:
            out_ch = scale_channels(ch)
            r = scale_repeats(repeats)
            for i in range(r):
                blocks.append(MBConv(in_ch, out_ch, expand, stride if i == 0 else 1, k))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        hidden_dim = max(32, min(256, num_genes // 4))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
