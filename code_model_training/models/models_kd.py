import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from transformer import ViT
from torchvision.models.densenet import DenseNet


class TinyHisToGene(nn.Module):
    def __init__(self, patch_size=16, n_layers=2, n_genes=1000, dim=480, n_pos=64):
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim

        patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        self.vit = ViT(
            dim=dim,
            depth=n_layers,      # ↓ reduced depth
            heads=4,             # ↓ reduced heads
            mlp_dim=2 * dim,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, images):
        B, C, H, W = images.shape
        patch_size = self.patch_size

        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)

        grid_size = int((patches.shape[1]) ** 0.5)

        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=images.device),
            torch.arange(grid_size, device=images.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)

        coords = torch.clamp(coords, 0, 63)

        patches = self.patch_embedding(patches)

        x = patches + self.x_embed(coords[:, :, 0]) + self.y_embed(coords[:, :, 1])

        x = self.vit(x)

        x = self.head(x)
        x = x.mean(dim=1)

        return x


# class TinySTNet(nn.Module):
#     def __init__(self, num_genes=460, reduction=0.5, pretrained=False):
#         super().__init__()

#         base = models.densenet121(pretrained=pretrained)

#         self.features = base.features

#         reduced_channels = int(1024 * reduction)
#         print(reduced_channels)

#         self.channel_reduction = nn.Conv2d(1024, reduced_channels, kernel_size=1)

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.fc = nn.Linear(reduced_channels, num_genes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.channel_reduction(x)
#         print(x.shape)
#         x = self.pool(x).view(x.size(0), -1)
#         return self.fc(x)
    


class TinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=4, num_init_features=16):
        super().__init__()

        # smaller DenseNet (same pattern, reduced width)
        self.backbone = DenseNet(
            growth_rate=growth_rate,          # ↓ from 32
            block_config=(6, 12, 24, 16),     # SAME structure
            num_init_features=num_init_features,  # ↓ from 64
            bn_size=4,
            drop_rate=0.1
        )

        self.features = self.backbone.features

        # final channels depend on growth_rate
        # For growth_rate=16 → final ≈ 512
        # final_channels = int(num_init_features + growth_rate * (6 + 12 + 24 + 16))
        final_channels = self.backbone.classifier.in_features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(final_channels, num_genes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)