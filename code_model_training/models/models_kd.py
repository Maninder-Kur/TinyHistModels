import torch
import torch.nn as nn
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
            depth=n_layers,
            heads=4,
            mlp_dim=2 * dim,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, images):
        (B, C, H, W) = images.shape
        patch_size = self.patch_size
        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)
        grid_size = int(patches.shape[1] ** 0.5)
        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(grid_size, device=images.device),
                    torch.arange(grid_size, device=images.device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        coords = torch.clamp(coords, 0, 63)
        patches = self.patch_embedding(patches)
        x = patches + self.x_embed(coords[:, :, 0]) + self.y_embed(coords[:, :, 1])
        x = self.vit(x)
        x = self.head(x)
        x = x.mean(dim=1)
        return x


class TinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=4, num_init_features=16):
        super().__init__()
        self.backbone = DenseNet(
            growth_rate=growth_rate,
            block_config=(6, 12, 24, 16),
            num_init_features=num_init_features,
            bn_size=4,
            drop_rate=0.1,
        )
        self.features = self.backbone.features
        final_channels = self.backbone.classifier.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_genes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
