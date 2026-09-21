import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
for source_dir in (MODEL_DIR, MODEL_DIR / 'TCGN', MODEL_DIR / 'STPath'):
    sys.path.insert(0, str(source_dir))

from CMT_block import *
from gnn_block import Graph_Encoding_Block_big, Graph_Encoding_Block
from transformer_block import Channel_Attention
from functools import partial
from collections import OrderedDict

import timm
import torch
import os
import h5py
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0
from torchvision.models import densenet121
from torchvision.transforms.functional import crop

from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0


from transformer import ViT




class THItoGeneModel(nn.Module):
    def __init__(self, num_genes, learning_rate=1e-4, route_dim=64, caps=20, heads=[16, 8], n_layers=4):
        super().__init__()
        from THItoGene.vis_model import THItoGene

        self.model = THItoGene(
            n_genes=num_genes,
            learning_rate=learning_rate,
            route_dim=route_dim,
            caps=caps,
            heads=heads,
            n_layers=n_layers,
        )

    def forward(self, patches, centers, adj):
        """
        patches : (B, N, 3, 112, 112)
        centers : (B, N, 2)    # pixel_x, pixel_y
        adj     : (B, N, N)
        """
        return self.model(patches, centers, adj)

########################### GhostNetModel ################################################

class GhostNetModel(nn.Module):
    def __init__(self, num_genes=845, pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            "ghostnet_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        # self.regressor = nn.Linear(1280, num_genes)
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_genes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x
    

########################### UNI ################################################


class UNI(nn.Module):
    def __init__(
        self,
        num_genes=460,
        assets_dir="./assets",
        checkpoint="pytorch_model.bin",
        freeze_backbone=False
    ):
        super(UNI, self).__init__()

        # -------------------------------------------------
        # Checkpoint path
        # -------------------------------------------------
        assets_dir = os.path.expandvars(os.path.expanduser(assets_dir))
        ckpt_path = os.path.join(
            assets_dir,
            "uni",
            checkpoint
        )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"UNI checkpoint not found at:\n{ckpt_path}"
            )

        # -------------------------------------------------
        # Create UNI backbone
        # -------------------------------------------------
        uni_kwargs = {
            'model_name': 'vit_large_patch16_224',
            'img_size': 224,
            'patch_size': 16,
            'init_values': 1e-5,
            'num_classes': 0,
            'dynamic_img_size': True
        }

        self.backbone = timm.create_model(**uni_kwargs)

        # -------------------------------------------------
        # Load pretrained weights
        # -------------------------------------------------
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu"
        )

        self.backbone.load_state_dict(
            state_dict,
            strict=True
        )

        print("UNI pretrained weights loaded.")

        # -------------------------------------------------
        # Freeze backbone
        # -------------------------------------------------
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------
        # Regression head
        # -------------------------------------------------
        self.fc = nn.Linear(1024, num_genes)

    def forward(self, x):

        # (B, 1024)
        features = self.backbone(x)

        # (B, num_genes)
        out = self.fc(features)

        return out



########################### STNet ################################################

class STNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(STNet, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)

        self.features = densenet.features  # Output: (batch, 1024, 7, 7)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1024, num_genes)
        
    def forward(self, x):
        features = self.features(x)  # (batch, 1024, 7, 7)
        pooled = self.global_pool(features)  # (batch, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 1024)
        out = self.fc(pooled)  # (batch, num_genes)
        return out

########################### EfftNet ################################################

class EfficientNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(EfficientNet, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Remove classifier and replace with custom regression head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Regression head for gene expression
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output
    

class EfficientNetB4GeneRegressor(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(EfficientNetB4GeneRegressor, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=False)
        
        # Get number of input features from the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the classification head with a regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_genes)
        )

    def forward(self, x):
        return self.backbone(x)


###################################### DeepSpaCe #####################################################


class Custom_VGG16(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(Custom_VGG16, self).__init__()
        
        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=pretrained)
        
        # Remove original classifier (FC layers)
        self.features = vgg.features  # convolutional layers
        
        # Define a custom regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),  # VGG16 default flatten size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, num_genes)  # Output: 460 genes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
###################################### SqueezeNetResSE #####################################################   


#  Fire Module + Residual SE Attention
class FireSE(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # SE Attention
        out_channels = 2 * expand_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 16)
        self.fc2 = nn.Linear(out_channels // 16, out_channels)

    def forward(self, x):
        # Fire
        x = self.relu(self.squeeze(x))

        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)

        x = torch.cat([out1, out3], dim=1)
        x = self.relu(x)

        # SE Attention
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)

        # Residual attention (safe version)
        x = x + x * y

        return x


#  Full Modified SqueezeNet
class SqueezeNetResSE(nn.Module):
    def __init__(self, num_genes=460):
        super().__init__()

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        # FireSE blocks (following SqueezeNet 1.1 pattern)
        self.features = nn.Sequential(
            FireSE(64, 16, 64),    # → 128
            FireSE(128, 16, 64),   # → 128
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireSE(128, 32, 128),  # → 256
            FireSE(256, 32, 128),  # → 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireSE(256, 48, 192),  # → 384
            FireSE(384, 48, 192),  # → 384
            FireSE(384, 64, 256),  # → 512
            FireSE(512, 64, 256),  # → 512
        )

        # Final head (minimal, flexible)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_genes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
###################################### HisToGene #####################################################


class HisToGene(nn.Module):
    def __init__(self, patch_size=16, n_layers=4, n_genes=1000, dim=1024, dropout=0.1, n_pos=64):
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
            heads=16,
            mlp_dim=2 * dim,
            dropout=dropout,
            emb_dropout=dropout
        )

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        Automatically splits into patches and generates centers
        """
        B, C, H, W = images.shape
        patch_size = self.patch_size

        # Split image into non-overlapping patches
        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)  # [B, N_patches, 3*patch_size*patch_size]

        # Compute coordinates for each patch
        grid_size = int((patches.shape[1]) ** 0.5)
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=images.device),
            torch.arange(grid_size, device=images.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)  # [B, N_patches, 2]

        # Positional embeddings
        coords = torch.clamp(coords, 0, self.x_embed.num_embeddings - 1)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(coords[:, :, 0])
        centers_y = self.y_embed(coords[:, :, 1])

        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        x = x.mean(dim=1)  # mean pooling across patches
        return x
    
    
###################################### #####################################################

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))



class TCGN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=785, embed_dims=[46, 92, 184, 368], stem_channel=16,
                 fc_dim=1280,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None,
                 representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None,
                 depths=[2, 2, 10, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.patch_embed_a.num_patches,
            self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches,
            self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(
            num_heads[2], self.patch_embed_c.num_patches,
            self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(
            num_heads[3], self.patch_embed_d.num_patches,
            self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())#('act', nn.GELU())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Graph learning
        self.gnn_block0 = Graph_Encoding_Block(img_size=56, patch_size=4, num_feature_in=embed_dims[0],
                                               embed_dim=embed_dims[0]*2, num_feature_graph_hidden=embed_dims[0]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention0 = Channel_Attention(num_nodes=196)

        self.gnn_block1 = Graph_Encoding_Block(img_size=28, patch_size=2, num_feature_in=embed_dims[1],
                                               embed_dim=embed_dims[1]*2, num_feature_graph_hidden=embed_dims[1]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention1 = Channel_Attention(num_nodes=196)

        # Classifier head
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.GELU(),nn.Linear(fc_dim*4,num_classes, bias=True))
        # nn.Linear(fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        # nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.Linear(fc_dim*4,num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature0 = self.gnn_block0(x,self.relative_pos_a)

        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature1 = self.gnn_block1(x, self.relative_pos_c)

        x, (H, W) = self.patch_embed_c(x)

        x = self.channel_attention0(x, graph_feature0)


        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x=self.channel_attention1(x,graph_feature1)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)

        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        #print("+", end="")

        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

   
    
######################  EfficientNet_GeneCaptionContrastive  ######################### 

######################  EfficientNet_GeneCaptionContrastive  #########################

class EfficientNet_GeneCaptionContrastive(nn.Module):
    """
    EfficientNet-B0 + Biological Bottleneck + Contrastive Regularization
    Works for any gene count (171 / 250 / 560 / 845)
    """

    def __init__(
        self,
        num_genes,
        pretrained=True,
        contrastive_weight=0.1,
        temperature=0.07
    ):
        super().__init__()

        self.num_genes = num_genes
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        # =========================
        # Auto dimension scaling
        # =========================
        self.latent_dim = min(512, max(128, num_genes // 2))  # scales with gene count
        self.proj_dim   = min(256, max(64,  num_genes // 4))

        print(f"[INFO] Using latent_dim={self.latent_dim}, proj_dim={self.proj_dim} for {num_genes} genes")

        # =========================
        # EfficientNet-B0 Backbone
        # =========================
        backbone = models.efficientnet_b0(pretrained=pretrained)
        self.backbone = backbone.features            # (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.enc_dim = 1280

        # =========================
        # Biological Bottleneck
        # =========================
        self.bottleneck = nn.Sequential(
            nn.Linear(self.enc_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # =========================
        # Gene Decoder — direct, no redundant middle layer
        # =========================
        self.gene_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, num_genes)
        )

        # =========================
        # Projection Heads (Contrastive — training only)
        # =========================
        self.img_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.proj_dim),   # takes z, not raw feat
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

        self.gene_proj = nn.Sequential(
            nn.Linear(num_genes, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, images, gene_values=None):
        """
        images:      (B, 3, 224, 224)
        gene_values: (B, G) or None — pass during training, None at inference

        returns:
            gene_pred:        (B, G)
            contrastive_loss: scalar or None
        """

        B = images.size(0)

        # -------- Image Encoding --------
        feat = self.backbone(images)                 # (B, 1280, H', W')
        feat = self.pool(feat).view(B, -1)           # (B, 1280)

        # -------- Bottleneck --------
        z = self.bottleneck(feat)                    # (B, latent_dim)

        # -------- Gene Prediction --------
        gene_pred = self.gene_decoder(z)             # (B, num_genes)

        # -------- Contrastive (training only) --------
        contrastive_loss = None
        if gene_values is not None:
            img_emb  = self.img_proj(z)              # (B, proj_dim)
            gene_emb = self.gene_proj(gene_values)   # (B, proj_dim)
            contrastive_loss = self.info_nce(img_emb, gene_emb)

        return gene_pred, contrastive_loss

    # ============================================================
    # InfoNCE Loss
    # ============================================================
    def info_nce(self, img_emb, gene_emb):
        img_emb  = F.normalize(img_emb,  dim=1)
        gene_emb = F.normalize(gene_emb, dim=1)

        logits = torch.matmul(img_emb, gene_emb.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2g = F.cross_entropy(logits,   labels)
        loss_g2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2g + loss_g2i)
    
    
#######################################               #########################################################    
    

def compute_sobel_channels(image_tensor):
    # image_tensor: (B, 3, H, W) normalized
    sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]], dtype=torch.float32)
    sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]], dtype=torch.float32)

    sobel_kernel_x = sobel_kernel_x.expand(3, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.expand(3, 1, 3, 3)

    if image_tensor.is_cuda:
        sobel_kernel_x = sobel_kernel_x.cuda()
        sobel_kernel_y = sobel_kernel_y.cuda()

    grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1, groups=3)
    grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1, groups=3)

    grad_x = grad_x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    grad_y = grad_y.mean(dim=1, keepdim=True)

    return grad_x, grad_y


class GEMResNet18(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)

        # Modify first conv layer
        old_conv = base.conv1
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
            new_conv.weight[:, 3:] = old_conv.weight[:, :2]  # Init gradients

        base.conv1 = new_conv
        self.backbone = base

        # Replace final FC with gene regressor
        self.backbone.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x_rgb):
        grad_x, grad_y = compute_sobel_channels(x_rgb)
        x = torch.cat([x_rgb, grad_x, grad_y], dim=1)  # (B, 5, H, W)
        return self.backbone(x)


class GEMEfficientNetB0(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(GEMEfficientNetB0, self).__init__()
        base = efficientnet_b0(pretrained=pretrained)

        # Modify first conv layer for 5-channel input
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
            new_conv.weight[:, 3:] = old_conv.weight[:, :2]  # Init Sobel channels

        base.features[0][0] = new_conv
        self.encoder = base

        # Custom regression head
        in_features = base.classifier[1].in_features
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x_rgb):
        grad_x, grad_y = compute_sobel_channels(x_rgb)
        x = torch.cat([x_rgb, grad_x, grad_y], dim=1)  # (B, 5, H, W)
        features = self.encoder.features(x)
        features = self.encoder.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.regressor(features)
        return output

##########################################################################################    
    
from torchvision.models import mobilenet_v2

class MobileNetV2Regressor(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super().__init__()
        base = mobilenet_v2(pretrained=pretrained)
        
        self.backbone = base.features  # feature extractor
        in_features = base.classifier[1].in_features

        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, x):
        features = self.backbone(x)               # (B, C, H, W)
        pooled = features.mean([2, 3])            # Global average pooling
        out = self.regressor(pooled)              # (B, num_genes)
        return out


####################################################################################


def extract_center_crops(x, sizes=[64, 224]):
    """
    Extract center crops of sizes from a batch of 224x224 patches.

    Args:
        x: Tensor of shape (B, C, 224, 224)
        sizes: List of crop sizes

    Returns:
        Dict of {size: tensor of shape (B, C, size, size)}
    """
    B, C, H, W = x.shape
    assert H == 224 and W == 224, "Input must be 224x224"

    crops = {}
    for size in sizes:
        top = (H - size) // 2
        left = (W - size) // 2
        crops[size] = crop(x, top, left, size, size)
    return crops


class ScaleAttentionFusion(nn.Module):
    def __init__(self, in_dim=1024, num_scales=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        # features: list of [B, C]
        stacked = torch.stack(features, dim=1)  # [B, S, C]
        attn_weights = self.attn(stacked)       # [B, S, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)  # [B, C]
        return fused


class HierarchicalDenseNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True, shared_backbone=True):
        super(HierarchicalDenseNet, self).__init__()
        self.shared_backbone = shared_backbone

        def get_encoder():
            model = densenet121(pretrained=pretrained)
            features = model.features  # Exclude classifier
            return features

        if shared_backbone:
            self.encoder = get_encoder()
        else:
            # self.encoder_small = get_encoder()
            self.encoder_mid = get_encoder()
            self.encoder_large = get_encoder()

        # DenseNet121 output: (B, 1024, 7, 7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fused_dim = 1024  # concatenate 3 scale features
        
        # Attention-based fusion
        self.scale_fusion = ScaleAttentionFusion(in_dim=1024, num_scales=2)
        
        self.fusion_bn_dropout = nn.Sequential(
            nn.BatchNorm1d(self.fused_dim),
            nn.Dropout(p=0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        crops = extract_center_crops(x, sizes=[64, 224])
        
        # Resize to 224x224
        crops = {size: F.interpolate(c, size=(224, 224), mode='bilinear') for size, c in crops.items()}
        x_mid, x_large = crops[64], crops[224]

        if self.shared_backbone:
            # f_small = self.global_pool(self.encoder(x_small)).squeeze(-1).squeeze(-1)
            f_mid   = self.global_pool(self.encoder(x_mid)).squeeze(-1).squeeze(-1)
            f_large = self.global_pool(self.encoder(x_large)).squeeze(-1).squeeze(-1)
        else:
            # f_small = self.global_pool(self.encoder_small(x_small)).squeeze(-1).squeeze(-1)
            f_mid   = self.global_pool(self.encoder_mid(x_mid)).squeeze(-1).squeeze(-1)
            f_large = self.global_pool(self.encoder_large(x_large)).squeeze(-1).squeeze(-1)

        fused = self.scale_fusion([f_mid, f_large])  # [B, 1024]
        fused = self.fusion_bn_dropout(fused)
        out = self.regressor(fused)  # [B, num_genes]
        
        # fused = torch.cat([f_small, f_mid, f_large], dim=1)
        
        return out


#################################       TinySTNet        #########################################


class TinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=16, num_init_features=32, pretrained=False):
        super(TinySTNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense Blocks (fewer layers per block)
        num_features = num_init_features
        block_config = [3, 4, 6]  # Reduced layers from [6, 12, 24] in DenseNet121

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=4, growth_rate=growth_rate, drop_rate=0)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_genes)
        )

    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(F.relu(features)).view(x.size(0), -1)
        out = self.regressor(out)
        return out

#################################       TinySTNet        #########################################

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale

class RefinedTinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=24, num_init_features=48, pretrained=False):
        super(RefinedTinySTNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense Blocks with modified configuration
        num_features = num_init_features
        block_config = [4, 6, 8]

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=4, growth_rate=growth_rate, drop_rate=0)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final layers
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.se_block = SEBlock(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genes)
        )

    def forward(self, x):
        features = self.features(x)
        features = self.se_block(features)
        out = self.avg_pool(F.relu(features)).view(x.size(0), -1)
        out = self.regressor(out)
        return out

##########################################  TinyEfficientNet    ###########################################

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale

class TinyEfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_se=True):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = (in_ch == out_ch and stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, mid_ch, kernel_size=1, padding=0))

        layers.extend([
            ConvBNReLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
            SqueezeExcite(mid_ch) if use_se else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out += x
        return F.relu(out)

class TinyEfficientNet(nn.Module):
    def __init__(self, num_genes=460):
        super().__init__()

        base_channels = 16
        self.stem = ConvBNReLU(3, base_channels, stride=2)

        self.blocks = nn.Sequential(
            TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1),
            TinyEfficientBlock(24, 32, expand_ratio=2, stride=2),
            TinyEfficientBlock(32, 48, expand_ratio=2, stride=2),
            TinyEfficientBlock(48, 64, expand_ratio=2, stride=1),
            TinyEfficientBlock(64, 96, expand_ratio=2, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(96, 256),  #96 to 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_genes)  #256
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.head(x)
        return x

###################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

# class ECABlock(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # (B, C, 1, 1)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# class TinyEfficientBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_eca=True):
#         super().__init__()
#         # mid_ch = in_ch * expand_ratio
#         mid_ch = int(in_ch * expand_ratio)
#         self.use_res = (in_ch == out_ch and stride == 1)

#         layers = []
#         if expand_ratio != 1:
#             layers.append(ConvBNReLU(in_ch, mid_ch, kernel_size=1, padding=0))

#         layers.extend([
#             ConvBNReLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
#             ECABlock(mid_ch) if use_eca else nn.Identity(),
#             nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_ch)
#         ])

#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.block(x)
#         if self.use_res:
#             out += x
#         return F.relu(out)

# class DropBlock2D(nn.Module):
#     def __init__(self, block_size=3, drop_prob=0.1):
#         super().__init__()
#         self.block_size = block_size
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if not self.training or self.drop_prob == 0.:
#             return x
#         gamma = self.drop_prob * x.numel() / (x.shape[0] * x.shape[1] * (x.shape[2] - self.block_size + 1) * (x.shape[3] - self.block_size + 1))
#         mask = (torch.rand_like(x[:, :, self.block_size//2::, self.block_size//2::]) < gamma).float()
#         mask = F.pad(mask, [0, self.block_size//2, 0, self.block_size//2, 0, 0, 0, 0])
#         mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
#         return x * (1 - mask)

# # class RefinedTinyEfficientNet(nn.Module):
# #     def __init__(self, num_genes=460):
# #         super().__init__()

# #         base_channels = 12
# #         self.stem = ConvBNReLU(3, base_channels, stride=2)

# #         self.blocks = nn.Sequential(
# #             TinyEfficientBlock(base_channels, 16, expand_ratio=1, stride=1),
# #             TinyEfficientBlock(16, 24, expand_ratio=1.25, stride=2),
# #             TinyEfficientBlock(24, 32, expand_ratio=1.25, stride=2),
# #             TinyEfficientBlock(32, 48, expand_ratio=1.5, stride=1),
# #             TinyEfficientBlock(48, 64, expand_ratio=1.5, stride=1),
# #         )

# #         self.dropblock = DropBlock2D(block_size=3, drop_prob=0.1)
# #         self.pool = nn.AdaptiveAvgPool2d(1)
# #         self.head = nn.Sequential(
# #             nn.Linear(64, 256),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(256, num_genes)
# #         )

# #     def forward(self, x):
# #         x = self.stem(x)
# #         x = self.blocks(x)
# #         x = self.dropblock(x)
# #         x = self.pool(x).view(x.size(0), -1)
# #         x = self.head(x)
# #         return x

# class RefinedTinyEfficientNet(nn.Module):
#     def __init__(self, num_genes=460):
#         super().__init__()

#         base_channels = 16
#         self.stem = ConvBNReLU(3, base_channels, stride=2)

#         self.blocks = nn.Sequential(
#             TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1, use_eca=False),
#             TinyEfficientBlock(24, 32, expand_ratio=2, stride=2, use_eca=False),
#             TinyEfficientBlock(32, 48, expand_ratio=2, stride=2, use_eca=True),
#             TinyEfficientBlock(48, 64, expand_ratio=2, stride=1, use_eca=True),
#             TinyEfficientBlock(64, 80, expand_ratio=2, stride=1, use_eca=True),
#         )

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.head = nn.Sequential(
#             nn.Linear(80, 384),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(384, num_genes)
#         )

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.head(x)
#         return x

class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TinyEfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_eca=False):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_res = (in_ch == out_ch and stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNSiLU(in_ch, mid_ch, kernel_size=1, padding=0))

        layers.extend([
            ConvBNSiLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
            ECABlock(mid_ch) if use_eca else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out += x
        return F.silu(out)

class RefinedTinyEfficientNet(nn.Module):
    def __init__(self, num_genes=460):
        super().__init__()
        base_channels = 16

        self.stem = ConvBNSiLU(3, base_channels, stride=2)

        self.blocks = nn.Sequential(
            TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1, use_eca=False),
            TinyEfficientBlock(24, 32, expand_ratio=2, stride=2, use_eca=False),
            TinyEfficientBlock(32, 48, expand_ratio=2, stride=2, use_eca=True),
            TinyEfficientBlock(48, 64, expand_ratio=2, stride=1, use_eca=True),
            TinyEfficientBlock(64, 80, expand_ratio=2, stride=1, use_eca=True)
        )

        self.conv_head = ConvBNSiLU(80, 96, kernel_size=1, stride=1, padding=0)
        self.dropout2d = nn.Dropout2d(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(96, 192),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(192, num_genes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.dropout2d(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.head(x)
        return x


################################################  GeneCLIPRegressor  ##########################################

class GeneCLIPRegressor(nn.Module):

    def __init__(self, num_genes, pretrained=True, temperature=0.07):
        super().__init__()

        self.num_genes = num_genes
        self.temperature = temperature

        # ---------------------------------
        # EfficientNet Backbone
        # ---------------------------------
        backbone = models.efficientnet_b0(pretrained=pretrained)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.img_dim = 1280

        # ---------------------------------
        # Gene Encoder
        # ---------------------------------
        self.gene_encoder = nn.Sequential(
            nn.Linear(num_genes, 1280)
        )

        # ---------------------------------
        # Regression Head
        # ---------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(1280, num_genes)
        )

    def encode_image(self, images):
        feat = self.backbone(images)
        feat = self.pool(feat).flatten(1)   # (B,1280)
        return feat

    def encode_gene(self, genes):
        return self.gene_encoder(genes)     # (B,1280)

    def clip_loss(self, img_emb, gene_emb):

        img_emb = F.normalize(img_emb, dim=1)
        gene_emb = F.normalize(gene_emb, dim=1)

        logits = img_emb @ gene_emb.T
        logits = logits / self.temperature

        labels = torch.arange(logits.size(0)).to(logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_g = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_g) / 2

    def forward(self, images, genes=None):

        img_emb = self.encode_image(images)

        pred = self.regressor(img_emb)

        contrastive_loss = None

        if genes is not None:
            gene_emb = self.encode_gene(genes)
            contrastive_loss = self.clip_loss(img_emb, gene_emb)

        return pred, contrastive_loss


######################  GeneGuidedImageTransformer_with_sigmoid  #########################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1. Sigmoid Cross-Attention
# ============================================================

class SigmoidCrossAttention(nn.Module):
    """
    Cross-attention with sigmoid instead of softmax.
    Each image token attends to ALL gene tokens independently.
    Better for gene expression — multiple genes/pathways relevant simultaneously.
    """

    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()

        assert d_model % nhead == 0, \
            f"d_model {d_model} must be divisible by nhead {nhead}"

        self.d_model = d_model
        self.nhead   = nhead
        self.d_head  = d_model // nhead
        self.scale   = self.d_head ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.o_proj  = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        query : (B, Tq, d_model)  image tokens  Tq=49
        key   : (B, Tk, d_model)  gene tokens   Tk=num_tokens
        value : (B, Tk, d_model)  gene tokens

        returns
            out  : (B, Tq, d_model)
            attn : (B, nhead, Tq, Tk)
        """
        B, Tq, _ = query.shape
        _, Tk, _ = key.shape

        Q = self.q_proj(query)                                              # (B,Tq,d)
        K = self.k_proj(key)                                                # (B,Tk,d)
        V = self.v_proj(value)                                              # (B,Tk,d)

        Q = Q.view(B, Tq, self.nhead, self.d_head).transpose(1, 2)         # (B,h,Tq,dh)
        K = K.view(B, Tk, self.nhead, self.d_head).transpose(1, 2)         # (B,h,Tk,dh)
        V = V.view(B, Tk, self.nhead, self.d_head).transpose(1, 2)         # (B,h,Tk,dh)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale           # (B,h,Tq,Tk)
        attn = torch.sigmoid(attn)                                          # independent
        attn = self.dropout(attn)

        out  = torch.matmul(attn, V)                                        # (B,h,Tq,dh)
        out  = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # (B,Tq,d)
        out  = self.o_proj(out)

        return out, attn


# ============================================================
# 2. Multi-Scale Gene Encoder
# ============================================================

class MultiScaleGeneEncoder(nn.Module):
    """
    (B, num_genes) → (B, num_tokens, d_model)

    Three parallel depthwise-separable Conv1D branches:
      k=3  → local gene-gene correlations
      k=7  → pathway-level patterns
      k=15 → global expression signatures

    Training only — dropped at inference.
    """

    def __init__(self, num_genes, d_model=512, num_tokens=32):
        super().__init__()

        self.num_genes  = num_genes
        self.d_model    = d_model
        self.num_tokens = num_tokens

        # gene-wise mixing + norm before conv
        self.input_proj = nn.Sequential(
            nn.Linear(num_genes, num_genes),
            nn.LayerNorm(num_genes)
        )

        branch_dim       = d_model // 3
        branch_remainder = d_model - 2 * branch_dim

        self.branch_small  = self._make_branch(branch_dim,       kernel=3)
        self.branch_medium = self._make_branch(branch_dim,       kernel=7)
        self.branch_large  = self._make_branch(branch_remainder, kernel=15)

        self.channel_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.token_pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.token_norm = nn.LayerNorm(d_model)

    def _make_branch(self, out_ch, kernel):
        pad = kernel // 2
        return nn.Sequential(
            nn.Conv1d(1, 1,      kernel_size=kernel, padding=pad, groups=1),
            nn.Conv1d(1, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, genes):
        """
        genes   : (B, num_genes)
        returns : (B, num_tokens, d_model)
        """
        x = self.input_proj(genes)               # (B, num_genes)
        x = x.unsqueeze(1)                       # (B, 1, num_genes)

        s = self.branch_small(x)                 # (B, branch_dim, num_genes)
        m = self.branch_medium(x)                # (B, branch_dim, num_genes)
        l = self.branch_large(x)                 # (B, branch_remainder, num_genes)

        x = torch.cat([s, m, l], dim=1)          # (B, d_model, num_genes)
        x = self.channel_proj(x)                 # (B, d_model, num_genes)
        x = self.token_pool(x)                   # (B, d_model, num_tokens)
        x = x.permute(0, 2, 1)                   # (B, num_tokens, d_model)
        x = self.token_norm(x)

        return x


# ============================================================
# 3. Image Token Encoder
# ============================================================

class ImageTokenEncoder(nn.Module):
    """
    EfficientNet-B0 → (B, 49, d_model) spatial token sequence.

    Preserves 7x7 spatial structure instead of global average pooling.
    Projects 1280 → d_model for attention compatibility.
    Adds learnable positional encoding.
    """

    def __init__(self, pretrained=False, d_model=512):
        super().__init__()

        backbone      = models.efficientnet_b0(pretrained=pretrained)
        self.backbone = backbone.features              # (B,1280,7,7) for 224x224
        self.feat_dim = 1280

        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # learnable positional encoding for 7x7 grid
        self.pos_embed = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)

    def forward(self, images):
        """
        images  : (B, 3, 224, 224)
        returns : (B, 49, d_model)
        """
        feat = self.backbone(images)                        # (B,1280,7,7)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)     # (B,49,1280)
        feat = self.proj(feat)                              # (B,49,d_model)
        feat = feat + self.pos_embed
        return feat


# ============================================================
# 4. Cross-Attention Block  (training path)
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Training path only.
    Step 1 — Sigmoid cross-attention : image tokens attend to gene tokens
    Step 2 — Softmax self-attention  : image tokens refine spatial context
    Step 3 — FFN
    """

    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()

        self.cross_attn = SigmoidCrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, img_tokens, gene_tokens):
        """
        img_tokens  : (B, 49, d_model)
        gene_tokens : (B, num_tokens, d_model)
        returns     : (B, 49, d_model)
        """
        x, _ = self.cross_attn(
            query=img_tokens,
            key=gene_tokens,
            value=gene_tokens
        )
        img_tokens = self.norm1(img_tokens + x)

        x, _ = self.self_attn(img_tokens, img_tokens, img_tokens)
        img_tokens = self.norm2(img_tokens + x)

        img_tokens = self.norm3(img_tokens + self.ffn(img_tokens))

        return img_tokens


# ============================================================
# 5. Self-Attention Block  (inference path)
# ============================================================

class SelfAttentionBlock(nn.Module):
    """
    Inference path only.
    No gene tokens — image tokens attend to each other.
    Softmax self-attention + FFN.
    """

    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, img_tokens):
        """
        img_tokens : (B, 49, d_model)
        returns    : (B, 49, d_model)
        """
        x, _ = self.self_attn(img_tokens, img_tokens, img_tokens)
        img_tokens = self.norm1(img_tokens + x)
        img_tokens = self.norm2(img_tokens + self.ffn(img_tokens))
        return img_tokens


# ============================================================
# 6. Full Model
# ============================================================

class GeneGuidedImageTransformer(nn.Module):
    """
    Gene-Guided Image Transformer for spatial gene expression prediction.
    Inspired by BLEEP (NeurIPS 2023) with three key improvements:
      1. Multi-scale Conv1D gene encoder  vs  simple MLP
      2. Sigmoid cross-attention          vs  global contrastive only
      3. Direct regression at inference   vs  kNN retrieval

    TRAINING path:
        image  → EfficientNet-B0 → (B, 49, d_model)
        genes  → MultiScale Conv1D → (B, num_tokens, d_model)
                         ↓
        BLEEP-style InfoNCE on global embeddings  (alignment)
                         ↓
        Sigmoid Cross-Attention  (fine-grained spatial guidance)
                         ↓
        Softmax Self-Attention   (spatial refinement)
                         ↓
        Pool → Regressor → (B, num_genes)

    INFERENCE path:
        image  → EfficientNet-B0 → (B, 49, d_model)
                         ↓
        Softmax Self-Attention only
                         ↓
        Pool → Regressor → (B, num_genes)

    Args:
        num_genes         : output gene count  (171 / 541 / 845)
        pretrained        : EfficientNet pretrained weights  (default False)
        d_model           : attention dimension              (default 512)
        nhead             : attention heads                  (default 8)
        num_layers        : number of attention blocks       (default 2)
        num_tokens        : gene sequence length             (default 32)
        dropout           : dropout rate                     (default 0.1)
        temperature       : InfoNCE temperature              (default 0.07)
        contrastive_weight: InfoNCE loss weight              (default 0.1)
    """

    def __init__(
        self,
        num_genes,
        pretrained=False,
        d_model=512,
        nhead=8,
        num_layers=2,
        num_tokens=32,
        dropout=0.1,
        temperature=0.07,
        contrastive_weight=0.1
    ):
        super().__init__()

        self.num_genes          = num_genes
        self.d_model            = d_model
        self.num_layers         = num_layers
        self.temperature        = temperature
        self.contrastive_weight = contrastive_weight

        # --- image branch ---
        self.image_encoder = ImageTokenEncoder(
            pretrained=pretrained,
            d_model=d_model
        )

        # --- gene branch (training only) ---
        self.gene_encoder = MultiScaleGeneEncoder(
            num_genes=num_genes,
            d_model=d_model,
            num_tokens=num_tokens
        )

        # --- cross-attention blocks (training path) ---
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- self-attention blocks (inference path) ---
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- regression head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_genes)
        )

        self._print_info()

    # ----------------------------------------------------------
    def _print_info(self):
        total   = sum(p.numel() for p in self.parameters())
        img_p   = sum(p.numel() for p in self.image_encoder.parameters())
        gene_p  = sum(p.numel() for p in self.gene_encoder.parameters())
        cross_p = sum(p.numel() for p in self.cross_blocks.parameters())
        self_p  = sum(p.numel() for p in self.self_blocks.parameters())
        head_p  = sum(p.numel() for p in self.head.parameters())

        print(f"\n[GeneGuidedImageTransformer]")
        print(f"  num_genes          = {self.num_genes}")
        print(f"  d_model            = {self.d_model}")
        print(f"  num_layers         = {self.num_layers}")
        print(f"  temperature        = {self.temperature}")
        print(f"  contrastive_weight = {self.contrastive_weight}")
        print(f"  image encoder      = {img_p:,}  params")
        print(f"  gene encoder       = {gene_p:,}  params  (training only)")
        print(f"  cross blocks       = {cross_p:,}  params  (training path)")
        print(f"  self  blocks       = {self_p:,}  params  (inference path)")
        print(f"  head               = {head_p:,}  params")
        print(f"  total              = {total:,}  params\n")

    # ----------------------------------------------------------
    def info_nce(self, img_emb, gene_emb):
        """
        BLEEP-style InfoNCE on global embeddings.
        Aligns image and gene latent spaces at the batch level.
        """
        img_emb  = F.normalize(img_emb,  dim=1)
        gene_emb = F.normalize(gene_emb, dim=1)

        logits = torch.matmul(img_emb, gene_emb.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2g = F.cross_entropy(logits,   labels)
        loss_g2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2g + loss_g2i)

    # ----------------------------------------------------------
    def forward(self, images, gene_values=None):
        """
        images      : (B, 3, 224, 224)
        gene_values : (B, num_genes)  — training only, None at inference

        returns
            gene_pred        : (B, num_genes)
            contrastive_loss : scalar or None
        """

        # --- image tokens ---
        img_tokens = self.image_encoder(images)              # (B, 49, d_model)

        contrastive_loss = None

        if gene_values is not None:
            # ============ TRAINING PATH ============

            # gene tokens from multi-scale conv1d
            gene_tokens = self.gene_encoder(gene_values)     # (B, num_tokens, d_model)

            # BLEEP-style global contrastive alignment
            img_global  = img_tokens.mean(dim=1)             # (B, d_model)
            gene_global = gene_tokens.mean(dim=1)            # (B, d_model)
            contrastive_loss = self.info_nce(img_global, gene_global)

            # fine-grained sigmoid cross-attention + self-attention
            for block in self.cross_blocks:
                img_tokens = block(img_tokens, gene_tokens)

        else:
            # ============ INFERENCE PATH ============
            for block in self.self_blocks:
                img_tokens = block(img_tokens)

        # --- pool spatial tokens → single vector ---
        x = img_tokens.mean(dim=1)                           # (B, d_model)

        # --- predict gene expression ---
        gene_pred = self.head(x)                             # (B, num_genes)

        return gene_pred, contrastive_loss


    
########################### STPath (Spatial Transcriptomics Foundation Model) ################################
#
# Paper : "STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics
#           and Whole Slide Images" (Huang et al., 2025)
#           https://www.biorxiv.org/content/10.1101/2025.04.19.649665v2
#
# Architecture:
#   • Stage 1  — GigaPath ViT-Giant backbone (frozen by default) → 1536-dim CLS features
#   • Stage 2  — STPath STFM spatial transformer (fine-tuned from pretrained stfm.pth)
#                Uses STFM.inference() which calls:
#                   input_encoder (image_embed + gene_embed + tech_embed + organ_embed)
#                   → spatial SpatialTransformer (4 layers, d_model=512)
#                   → (N_spots, d_model=512) features
#   • Stage 3  — Task-specific MLP head (d_model=512 → hidden_dim → num_genes)
#                This is trained for your specific gene count (e.g., 785 for HER2).
#
# Key design decisions:
#   • ge_tokens  : zeroed (pure image → gene prediction, no ground truth gene input)
#   • coords     : zeroed (single-spot mode; each batch image = one spot at origin)
#   • batch_idx  : [0, 1, 2, ..., B-1] (each spot is its own sample)
#   • STFM weights are LOADED (pretrained) but NOT frozen → full fine-tuning
#   • GigaPath is FROZEN by default (serves purely as feature extractor)
#
# Checkpoint paths (relative to assets_dir):
#   gigapath/pytorch_model.bin   ← GigaPath ViT-Giant tile encoder (4.5 GB)
#   stpath/stfm.pth              ← STPath STFM pretrained model (187 MB)
#
# Gene vocabulary:
#   models/STPath/utils_data/symbol2ensembl.json  (auto-detected if gene_voc_path=None)
#
####################################################################################
    
    
    
class STPathModel(nn.Module):
    """
    STPath foundation model for spatial gene expression prediction from image patches.

    Pipeline:
      image (B,3,H,W)
        → GigaPath ViT-Giant [frozen]   → (B, 1536)  CLS features
        → STPath STFM spatial transformer [fine-tuned from stfm.pth]
             uses STFM.inference(img_tokens, coords, ge_tokens, batch_idx, ...)
             → (B, 512)  spatial-aware features
        → Task MLP head [trained from scratch]
             → (B, num_genes)  gene expression predictions

    Parameters
    ----------
    num_genes         : number of output gene targets (e.g. 785 for HER2)
    assets_dir        : root directory containing 'gigapath/' and 'stpath/' sub-folders
    stpath_ckpt       : STFM weight filename under assets_dir/stpath/ (default: stfm.pth)
    gigapath_ckpt     : GigaPath weight filename under assets_dir/gigapath/ (default: pytorch_model.bin)
    gene_voc_path     : path to symbol2ensembl.json (auto-found if None)
    freeze_gigapath   : freeze GigaPath backbone (default: True — use as fixed feature extractor)
    hidden_dim        : hidden units in the task-specific MLP head
    dropout           : dropout rate in the task head
    organ_type        : organ conditioning string, e.g. 'Breast', 'Skin', None
    tech_type         : sequencing technology string, e.g. 'Visium', 'ST', 'Xenium'
    """

    def __init__(
        self,
        num_genes: int = 785,
        assets_dir: str = "./assets",
        stpath_ckpt: str = "stfm.pth",
        gigapath_ckpt: str = "pytorch_model.bin",
        gene_voc_path: str = None,
        freeze_gigapath: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        organ_type: str = None,
        tech_type: str = "Visium",
    ):
        super(STPathModel, self).__init__()

        self.num_genes  = num_genes
        self.organ_type = organ_type
        self.tech_type  = tech_type

        # -----------------------------------------------------------------------
        # Resolve checkpoint paths
        # -----------------------------------------------------------------------
        assets_dir = os.path.expandvars(os.path.expanduser(assets_dir))
        gigapath_ckpt_path = os.path.join(assets_dir, "gigapath", gigapath_ckpt)
        stpath_ckpt_path   = os.path.join(assets_dir, "stpath",   stpath_ckpt)

        if not os.path.isfile(gigapath_ckpt_path):
            raise FileNotFoundError(
                f"[STPathModel] GigaPath checkpoint not found at:\n{gigapath_ckpt_path}\n"
                "Please download: huggingface-cli download prov-gigapath/prov-gigapath pytorch_model.bin"
            )
        if not os.path.isfile(stpath_ckpt_path):
            raise FileNotFoundError(
                f"[STPathModel] STPath checkpoint not found at:\n{stpath_ckpt_path}\n"
                "Please ensure assets/stpath/stfm.pth exists."
            )

        # Auto-detect gene vocabulary
        if gene_voc_path is None:
            _default_voc = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "STPath", "utils_data", "symbol2ensembl.json"
            )
            if os.path.isfile(_default_voc):
                gene_voc_path = _default_voc
            else:
                raise FileNotFoundError(
                    "[STPathModel] gene_voc_path not supplied and default path not found.\n"
                    "Expected: models/STPath/utils_data/symbol2ensembl.json"
                )
        print(f"[STPathModel] Using gene vocabulary: {gene_voc_path}")

        # -----------------------------------------------------------------------
        # 1. STPath imports (lazy to avoid import errors if STPath not installed)
        # -----------------------------------------------------------------------
        try:
            from stpath.model.model import STFM
            from stpath.model.nn_utils.config import ModelConfig
            from stpath.tokenization import (
                GeneExpTokenizer, ImageTokenizer,
                IDTokenizer, TokenizerTools, AnnotationTokenizer
            )
        except ImportError as exc:
            raise ImportError(
                "[STPathModel] STPath package not found.\n"
                "Install it: pip install -e models/STPath\n"
                f"Original: {exc}"
            )

        # -----------------------------------------------------------------------
        # 2. GigaPath ViT-Giant backbone  (1536-dim CLS token, frozen by default)
        # -----------------------------------------------------------------------
        print("[STPathModel] Building GigaPath ViT-Giant backbone …")
        self.gigapath_backbone = timm.create_model(
            "vit_giant_patch14_dinov2",
            patch_size=16,
            img_size=224,
            in_chans=3,
            num_classes=0,           # output raw CLS token (1536-dim)
            dynamic_img_size=False,
        )

        print(f"[STPathModel] Loading GigaPath from: {gigapath_ckpt_path}")
        gp_sd = torch.load(gigapath_ckpt_path, map_location="cpu")
        if "model" in gp_sd and isinstance(gp_sd.get("model"), dict):
            gp_sd = gp_sd["model"]
        missing, unexpected = self.gigapath_backbone.load_state_dict(gp_sd, strict=False)
        if missing:
            print(f"[STPathModel/GigaPath] Missing ({len(missing)}): {missing[:3]} …")
        if unexpected:
            print(f"[STPathModel/GigaPath] Unexpected ({len(unexpected)}): {unexpected[:3]} …")
        print("[STPathModel] GigaPath weights loaded.")

        if freeze_gigapath:
            for p in self.gigapath_backbone.parameters():
                p.requires_grad = False
            print("[STPathModel] GigaPath backbone FROZEN (feature extractor only).")

        # -----------------------------------------------------------------------
        # 3. STPath tokeniser — for organ/tech token id look-up
        # -----------------------------------------------------------------------
        print("[STPathModel] Building STPath tokeniser …")
        tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )
        self.tokenizer = tokenizer

        # Resolve organ / tech token ids
        self._organ_id = self._resolve_token_id(tokenizer.organ_tokenizer, organ_type)
        self._tech_id  = self._resolve_token_id(tokenizer.tech_tokenizer,  tech_type)
        print(f"[STPathModel] Conditioning — organ: '{organ_type}' (id={self._organ_id}), "
              f"tech: '{tech_type}' (id={self._tech_id})")

        # -----------------------------------------------------------------------
        # 4. STPath STFM spatial transformer  (fine-tuned from pretrained weights)
        # -----------------------------------------------------------------------
        stfm_config = ModelConfig.get_default_config()
        # Override to match tokeniser sizes
        stfm_config.feature_dim = 1536
        stfm_config.activation  = "gelu"
        stfm_config.n_genes     = tokenizer.ge_tokenizer.n_tokens
        stfm_config.n_tech      = tokenizer.tech_tokenizer.n_tokens
        stfm_config.n_species   = tokenizer.specie_tokenizer.n_tokens
        stfm_config.n_organs    = tokenizer.organ_tokenizer.n_tokens
        stfm_config.backbone    = "spatial_transformer"

        self._d_model = stfm_config.d_model   # 512

        print(f"[STPathModel] Building STFM (d_model={self._d_model}, "
              f"n_layers={stfm_config.n_layers}, n_heads={stfm_config.n_heads}) …")
        self.stfm = STFM(stfm_config)

        print(f"[STPathModel] Loading STFM pretrained weights from: {stpath_ckpt_path}")
        stfm_sd = torch.load(stpath_ckpt_path, map_location="cpu")
        self.stfm.load_state_dict(stfm_sd)
        print("[STPathModel] STFM weights loaded — model will be FINE-TUNED (not frozen).")

        # -----------------------------------------------------------------------
        # 5. Task-specific regression head  (d_model → hidden_dim → num_genes)
        #    This is always trained from scratch for the downstream gene set.
        # -----------------------------------------------------------------------
        self.task_head = nn.Sequential(
            nn.Linear(self._d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )

        # Print param counts
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[STPathModel] ✓ Initialized.")
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print(f"  (GigaPath frozen={freeze_gigapath}, STFM fine-tuned, task_head trained from scratch)\n")

    # -------------------------------------------------------------------
    @staticmethod
    def _resolve_token_id(tokenizer_obj, label: str) -> int:
        """Map a string label to its integer token id (0 = pad/unknown)."""
        if label is None:
            return 0
        # The vendored IDTokenizer exposes encode/token2id, not ``voc``.
        # Align aliases (e.g. ST) before retrieving the conditioning token.
        return int(tokenizer_obj.encode(label, align_first=True))

    # -------------------------------------------------------------------
    def _make_stfm_inputs(self, img_features: torch.Tensor):
        """
        Prepare the flat tensors required by STFM.inference().

        The STPath SpatialTransformer processes a *flat* sequence of spots
        across the whole batch:

          img_tokens  : (N, 1536)     N = B (one spot per image in batch)
          coords      : (N, 2)        zero coords — single-spot context
          ge_tokens   : (N, n_genes)  zeroed — image-only prediction
          batch_idx   : (N,)          [0,1,…,B-1]
          tech_tokens : (N,)
          organ_tokens: (N,)

        Returns all six tensors.
        """
        B = img_features.size(0)
        device = img_features.device

        img_tokens  = img_features                                    # (B, 1536)
        coords      = torch.zeros(B, 2,  device=device)              # (B, 2)
        n_gene_voc  = self.stfm.input_encoder.gene_embed.weight.shape[1]  # in_features = 38984
        ge_tokens   = torch.zeros(B, n_gene_voc, device=device)      # (B, n_gene_voc)
        batch_idx   = torch.arange(B, device=device, dtype=torch.long)
        tech_tok    = torch.full((B,), self._tech_id,  device=device, dtype=torch.long)
        organ_tok   = torch.full((B,), self._organ_id, device=device, dtype=torch.long)

        return img_tokens, coords, ge_tokens, batch_idx, tech_tok, organ_tok

    # -------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images  : (B, 3, H, W)   — typically H=W=224
        returns : (B, num_genes)
        """
        # ── Step 1: GigaPath feature extraction  (B, 1536)
        img_features = self.gigapath_backbone(images)

        # ── Step 2: STPath STFM spatial transformer  (B, d_model=512)
        img_tokens, coords, ge_tokens, batch_idx, tech_tok, organ_tok = \
            self._make_stfm_inputs(img_features)

        # STFM.inference() calls input_encoder then the spatial transformer backbone
        spatial_feats = self.stfm.inference(
            img_tokens=img_tokens,
            coords=coords,
            ge_tokens=ge_tokens,
            batch_idx=batch_idx,
            tech_tokens=tech_tok,
            organ_tokens=organ_tok,
        )                                                             # (B, d_model)

        # ── Step 3: Task-specific gene expression head  (B, num_genes)
        gene_pred = self.task_head(spatial_feats)
        return gene_pred
