import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
for source_dir in (MODEL_DIR, MODEL_DIR / "TCGN", MODEL_DIR / "STPath"):
    sys.path.insert(0, str(source_dir))
from CMT_block import *
from gnn_block import Graph_Encoding_Block
from transformer_block import Channel_Attention
from functools import partial
from collections import OrderedDict
import timm
import torch
import os
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformer import ViT


class THItoGeneModel(nn.Module):
    def __init__(
        self,
        num_genes,
        learning_rate=0.0001,
        route_dim=64,
        caps=20,
        heads=[16, 8],
        n_layers=4,
    ):
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


class UNI(nn.Module):
    def __init__(
        self,
        num_genes=460,
        assets_dir="./assets",
        checkpoint="pytorch_model.bin",
        freeze_backbone=False,
    ):
        super(UNI, self).__init__()
        assets_dir = os.path.expandvars(os.path.expanduser(assets_dir))
        ckpt_path = os.path.join(assets_dir, "uni", checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"UNI checkpoint not found at:\n{ckpt_path}")
        uni_kwargs = {
            "model_name": "vit_large_patch16_224",
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-05,
            "num_classes": 0,
            "dynamic_img_size": True,
        }
        self.backbone = timm.create_model(**uni_kwargs)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        print("UNI pretrained weights loaded.")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(1024, num_genes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out


class STNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(STNet, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_genes)

    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc(pooled)
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(EfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Linear(1024, num_genes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output


class Custom_VGG16(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(Custom_VGG16, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, num_genes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class HisToGene(nn.Module):
    def __init__(
        self, patch_size=16, n_layers=4, n_genes=1000, dim=1024, dropout=0.1, n_pos=64
    ):
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
            emb_dropout=dropout,
        )
        self.gene_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        Automatically splits into patches and generates centers
        """
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
        coords = torch.clamp(coords, 0, self.x_embed.num_embeddings - 1)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(coords[:, :, 0])
        centers_y = self.y_embed(coords[:, :, 1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        x = x.mean(dim=1)
        return x


class TCGN(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=785,
        embed_dims=[46, 92, 184, 368],
        stem_channel=16,
        fc_dim=1280,
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[3.6, 3.6, 3.6, 3.6],
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        depths=[2, 2, 10, 2],
        qk_ratio=1,
        sr_ratios=[8, 4, 2, 1],
        dp=0.1,
    ):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.stem_conv1 = nn.Conv2d(
            3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.stem_conv2 = nn.Conv2d(
            stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.stem_conv3 = nn.Conv2d(
            stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2,
            patch_size=2,
            in_chans=stem_channel,
            embed_dim=embed_dims[0],
        )
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )
        self.relative_pos_a = nn.Parameter(
            torch.randn(
                num_heads[0],
                self.patch_embed_a.num_patches,
                self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0],
            )
        )
        self.relative_pos_b = nn.Parameter(
            torch.randn(
                num_heads[1],
                self.patch_embed_b.num_patches,
                self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1],
            )
        )
        self.relative_pos_c = nn.Parameter(
            torch.randn(
                num_heads[2],
                self.patch_embed_c.num_patches,
                self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2],
            )
        )
        self.relative_pos_d = nn.Parameter(
            torch.randn(
                num_heads[3],
                self.patch_embed_d.num_patches,
                self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3],
            )
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_a = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    qk_ratio=qk_ratio,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        cur += depths[0]
        self.blocks_b = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    qk_ratio=qk_ratio,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        cur += depths[1]
        self.blocks_c = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    qk_ratio=qk_ratio,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        cur += depths[2]
        self.blocks_d = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    qk_ratio=qk_ratio,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(self.embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()
        self.gnn_block0 = Graph_Encoding_Block(
            img_size=56,
            patch_size=4,
            num_feature_in=embed_dims[0],
            embed_dim=embed_dims[0] * 2,
            num_feature_graph_hidden=embed_dims[0] * 2,
            num_feature_out=48,
            flatten=False,
            num_heads=2,
        )
        self.channel_attention0 = Channel_Attention(num_nodes=196)
        self.gnn_block1 = Graph_Encoding_Block(
            img_size=28,
            patch_size=2,
            num_feature_in=embed_dims[1],
            embed_dim=embed_dims[1] * 2,
            num_feature_graph_hidden=embed_dims[1] * 2,
            num_feature_out=48,
            flatten=False,
            num_heads=2,
        )
        self.channel_attention1 = Channel_Attention(num_nodes=196)
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-05)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Sequential(
            nn.Linear(fc_dim, fc_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(fc_dim * 4, num_classes, bias=True),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
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
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.fc_dim, num_classes, bias=True)
            if num_classes > 0
            else nn.Identity()
        )

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
        (x, (H, W)) = self.patch_embed_a(x)
        for (i, blk) in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        graph_feature0 = self.gnn_block0(x, self.relative_pos_a)
        (x, (H, W)) = self.patch_embed_b(x)
        for (i, blk) in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        graph_feature1 = self.gnn_block1(x, self.relative_pos_c)
        (x, (H, W)) = self.patch_embed_c(x)
        x = self.channel_attention0(x, graph_feature0)
        for (i, blk) in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)
        x = self.channel_attention1(x, graph_feature1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        (x, (H, W)) = self.patch_embed_d(x)
        for (i, blk) in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)
        (B, N, C) = x.shape
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


class GigaPathFlash(nn.Module):
    """Pretrained GigaPath-Flash tile encoder with a gene-regression head.

    Builds its tile encoder from the local ``models/prov-gigapath`` checkout and
    loads only the local tile checkpoint; no slide encoder or Hub download.
    """

    def __init__(
        self,
        num_genes=460,
        assets_dir="./assets",
        checkpoint="pytorch_model.bin",
        freeze_backbone=False,
    ):
        super().__init__()
        gigapath_repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prov-gigapath"
        )
        tile_encoder_path = os.path.join(gigapath_repo, "gigapath", "tile_encoder.py")
        if not os.path.isfile(tile_encoder_path):
            raise FileNotFoundError(
                f"Local GigaPath tile encoder not found at:\n{tile_encoder_path}"
            )
        if gigapath_repo not in sys.path:
            sys.path.insert(0, gigapath_repo)
        try:
            from gigapath import tile_encoder as gigapath_tile_encoder
        except ImportError as exc:
            raise ImportError(
                f"Could not import the local GigaPath tile encoder from {tile_encoder_path}: {exc}"
            ) from exc
        ckpt_path = os.path.join(assets_dir, "gigapath_flash", checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"GigaPath Flash checkpoint not found at:\n{ckpt_path}"
            )
        self.backbone = gigapath_tile_encoder.gigapath_tile_enc_dinov2s()
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
            state_dict = state_dict["model"]
        self.backbone.load_state_dict(state_dict, strict=True)
        print(f"GigaPath Flash pretrained weights loaded from: {ckpt_path}")
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.fc = nn.Linear(self.backbone.num_features, num_genes)

    @staticmethod
    def create_transform():
        """Official tile preprocessing for the datasets' uint8 RGB arrays."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, x):
        return self.fc(self.backbone(x))


class STPathZeroModel(nn.Module):
    """Legacy STPath with zero spatial/gene inputs and padding metadata IDs."""

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
        super(STPathZeroModel, self).__init__()
        self.num_genes = num_genes
        self.organ_type = organ_type
        self.tech_type = tech_type
        gigapath_ckpt_path = os.path.join(assets_dir, "gigapath", gigapath_ckpt)
        stpath_ckpt_path = os.path.join(assets_dir, "stpath", stpath_ckpt)
        if not os.path.isfile(gigapath_ckpt_path):
            raise FileNotFoundError(
                f"[STPathZeroModel] GigaPath checkpoint not found at:\n{gigapath_ckpt_path}\nPlease download: huggingface-cli download prov-gigapath/prov-gigapath pytorch_model.bin"
            )
        if not os.path.isfile(stpath_ckpt_path):
            raise FileNotFoundError(
                f"[STPathZeroModel] STPath checkpoint not found at:\n{stpath_ckpt_path}\nPlease ensure assets/stpath/stfm.pth exists."
            )
        if gene_voc_path is None:
            _default_voc = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "STPath",
                "utils_data",
                "symbol2ensembl.json",
            )
            if os.path.isfile(_default_voc):
                gene_voc_path = _default_voc
            else:
                raise FileNotFoundError(
                    "[STPathZeroModel] gene_voc_path not supplied and default path not found.\nExpected: models/STPath/utils_data/symbol2ensembl.json"
                )
        print(f"[STPathZeroModel] Using gene vocabulary: {gene_voc_path}")
        try:
            from stpath.model.model import STFM
            from stpath.model.nn_utils.config import ModelConfig
            from stpath.tokenization import (
                GeneExpTokenizer,
                ImageTokenizer,
                IDTokenizer,
                TokenizerTools,
                AnnotationTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                f"[STPathZeroModel] STPath package not found.\nInstall it: pip install -e models/STPath\nOriginal: {exc}"
            )
        print("[STPathZeroModel] Building GigaPath ViT-Giant backbone …")
        self.gigapath_backbone = timm.create_model(
            "vit_giant_patch14_dinov2",
            patch_size=16,
            img_size=224,
            in_chans=3,
            num_classes=0,
            dynamic_img_size=False,
        )
        print(f"[STPathZeroModel] Loading GigaPath from: {gigapath_ckpt_path}")
        gp_sd = torch.load(gigapath_ckpt_path, map_location="cpu")
        if "model" in gp_sd and isinstance(gp_sd.get("model"), dict):
            gp_sd = gp_sd["model"]
        (missing, unexpected) = self.gigapath_backbone.load_state_dict(
            gp_sd, strict=False
        )
        if missing:
            print(
                f"[STPathZeroModel/GigaPath] Missing ({len(missing)}): {missing[:3]} …"
            )
        if unexpected:
            print(
                f"[STPathZeroModel/GigaPath] Unexpected ({len(unexpected)}): {unexpected[:3]} …"
            )
        print("[STPathZeroModel] GigaPath weights loaded.")
        if freeze_gigapath:
            for p in self.gigapath_backbone.parameters():
                p.requires_grad = False
            print(
                "[STPathZeroModel] GigaPath backbone FROZEN (feature extractor only)."
            )
        print("[STPathZeroModel] Building STPath tokeniser …")
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
        self._organ_id = self._resolve_token_id(tokenizer.organ_tokenizer, organ_type)
        self._tech_id = self._resolve_token_id(tokenizer.tech_tokenizer, tech_type)
        print(
            f"[STPathZeroModel] Conditioning — organ: '{organ_type}' (id={self._organ_id}), tech: '{tech_type}' (id={self._tech_id})"
        )
        stfm_config = ModelConfig.get_default_config()
        stfm_config.feature_dim = 1536
        stfm_config.activation = "gelu"
        stfm_config.n_genes = tokenizer.ge_tokenizer.n_tokens
        stfm_config.n_tech = tokenizer.tech_tokenizer.n_tokens
        stfm_config.n_species = tokenizer.specie_tokenizer.n_tokens
        stfm_config.n_organs = tokenizer.organ_tokenizer.n_tokens
        stfm_config.backbone = "spatial_transformer"
        self._d_model = stfm_config.d_model
        print(
            f"[STPathZeroModel] Building STFM (d_model={self._d_model}, n_layers={stfm_config.n_layers}, n_heads={stfm_config.n_heads}) …"
        )
        self.stfm = STFM(stfm_config)
        print(
            f"[STPathZeroModel] Loading STFM pretrained weights from: {stpath_ckpt_path}"
        )
        stfm_sd = torch.load(stpath_ckpt_path, map_location="cpu")
        self.stfm.load_state_dict(stfm_sd)
        print(
            "[STPathZeroModel] STFM weights loaded — model will be FINE-TUNED (not frozen)."
        )
        self.task_head = nn.Sequential(
            nn.Linear(self._d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )
        total = sum((p.numel() for p in self.parameters()))
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        print(f"\n[STPathZeroModel] ✓ Initialized.")
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print(
            f"  (GigaPath frozen={freeze_gigapath}, STFM fine-tuned, task_head trained from scratch)\n"
        )

    @staticmethod
    def _resolve_token_id(tokenizer_obj, label: str) -> int:
        """Map a string label to its integer token id (0 = pad/unknown)."""
        if label is None:
            return 0
        try:
            voc = tokenizer_obj.voc
            return int(voc.index(label)) if label in voc else 0
        except Exception:
            return 0

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
        img_tokens = img_features
        coords = torch.zeros(B, 2, device=device)
        n_gene_voc = self.stfm.input_encoder.gene_embed.weight.shape[1]
        ge_tokens = torch.zeros(B, n_gene_voc, device=device)
        batch_idx = torch.arange(B, device=device, dtype=torch.long)
        tech_tok = torch.full((B,), self._tech_id, device=device, dtype=torch.long)
        organ_tok = torch.full((B,), self._organ_id, device=device, dtype=torch.long)
        return (img_tokens, coords, ge_tokens, batch_idx, tech_tok, organ_tok)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images  : (B, 3, H, W)   — typically H=W=224
        returns : (B, num_genes)
        """
        img_features = self.gigapath_backbone(images)
        (
            img_tokens,
            coords,
            ge_tokens,
            batch_idx,
            tech_tok,
            organ_tok,
        ) = self._make_stfm_inputs(img_features)
        spatial_feats = self.stfm.inference(
            img_tokens=img_tokens,
            coords=coords,
            ge_tokens=ge_tokens,
            batch_idx=batch_idx,
            tech_tokens=tech_tok,
            organ_tokens=organ_tok,
        )
        gene_pred = self.task_head(spatial_feats)
        return gene_pred


class STPathOriginalModel(nn.Module):
    """STPath with spatial windows and a pretrained gene-prediction head."""

    def __init__(
        self,
        panel_gene_names=None,
        assets_dir="./assets",
        stpath_ckpt="stfm.pth",
        gigapath_ckpt="pytorch_model.bin",
        gene_voc_path=None,
        organ_type=None,
        tech_type="Visium",
        freeze_gigapath=True,
        allow_missing_panel_genes=True,
    ):
        super().__init__()
        if not freeze_gigapath:
            raise ValueError(
                "STPathOriginalModel intentionally freezes GigaPath. Use the existing STPathModel if you need a trainable image encoder."
            )
        gigapath_ckpt_path = os.path.join(assets_dir, "gigapath", gigapath_ckpt)
        stpath_ckpt_path = os.path.join(assets_dir, "stpath", stpath_ckpt)
        if not os.path.isfile(gigapath_ckpt_path):
            raise FileNotFoundError(
                f"GigaPath checkpoint not found: {gigapath_ckpt_path}"
            )
        if not os.path.isfile(stpath_ckpt_path):
            raise FileNotFoundError(f"STPath checkpoint not found: {stpath_ckpt_path}")
        if gene_voc_path is None:
            gene_voc_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "STPath",
                "utils_data",
                "symbol2ensembl.json",
            )
        if not os.path.isfile(gene_voc_path):
            raise FileNotFoundError(
                f"STPath gene vocabulary not found: {gene_voc_path}"
            )
        from stpath.model.model import STFM
        from stpath.model.nn_utils.config import ModelConfig
        from stpath.tokenization import (
            AnnotationTokenizer,
            GeneExpTokenizer,
            IDTokenizer,
            ImageTokenizer,
            TokenizerTools,
        )

        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )
        self.organ_type = organ_type
        self.tech_type = tech_type
        self._organ_id = self._encode_id(
            self.tokenizer.organ_tokenizer, organ_type, default="Others"
        )
        self._tech_id = self._encode_id(
            self.tokenizer.tech_tokenizer, tech_type, default=None
        )
        print(
            f"[STPathOriginalModel] Conditioning: organ={organ_type!r} (id={self._organ_id}), tech={tech_type!r} (id={self._tech_id})"
        )
        self.gigapath_backbone = timm.create_model(
            "vit_giant_patch14_dinov2",
            patch_size=16,
            img_size=224,
            in_chans=3,
            num_classes=0,
            dynamic_img_size=False,
        )
        gigapath_state = torch.load(gigapath_ckpt_path, map_location="cpu")
        if isinstance(gigapath_state, dict) and isinstance(
            gigapath_state.get("model"), dict
        ):
            gigapath_state = gigapath_state["model"]
        self.gigapath_backbone.load_state_dict(gigapath_state, strict=False)
        for parameter in self.gigapath_backbone.parameters():
            parameter.requires_grad = False
        self.gigapath_backbone.eval()
        stfm_config = ModelConfig.get_default_config()
        stfm_config.feature_dim = 1536
        stfm_config.activation = "gelu"
        stfm_config.n_genes = self.tokenizer.ge_tokenizer.n_tokens
        stfm_config.n_tech = self.tokenizer.tech_tokenizer.n_tokens
        stfm_config.n_species = self.tokenizer.specie_tokenizer.n_tokens
        stfm_config.n_organs = self.tokenizer.organ_tokenizer.n_tokens
        stfm_config.backbone = "spatial_transformer"
        self.stfm = STFM(stfm_config)
        self.stfm.load_state_dict(torch.load(stpath_ckpt_path, map_location="cpu"))
        for parameter in self.stfm.parameters():
            parameter.requires_grad = True
        self.panel_gene_names = None
        self.panel_gene_ids = None
        self.panel_missing_positions = []
        self.panel_missing_names = []
        self.panel_missing_head = None
        if panel_gene_names is not None:
            self.panel_gene_names = [str(gene) for gene in panel_gene_names]
            (
                self.panel_gene_ids,
                self.panel_missing_positions,
                self.panel_missing_names,
            ) = self._resolve_panel_gene_ids(self.panel_gene_names)
            if self.panel_missing_names and (not allow_missing_panel_genes):
                raise ValueError(
                    f"The STPath vocabulary does not contain requested genes: {self.panel_missing_names[:10]}"
                )
            if self.panel_missing_names:
                self.panel_missing_head = nn.Linear(
                    stfm_config.d_model, len(self.panel_missing_names)
                )
        self.num_genes = (
            len(self.panel_gene_names)
            if self.panel_gene_names is not None
            else self.tokenizer.ge_tokenizer.n_tokens
        )
        self._d_model = stfm_config.d_model
        trainable = sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
        total = sum((parameter.numel() for parameter in self.parameters()))
        print(
            f"[STPathOriginalModel] Initialized with frozen GigaPath and trainable STFM: total={total:,}, trainable={trainable:,}, panel_genes={self.num_genes}, missing_panel_genes={len(self.panel_missing_names)}"
        )

    @staticmethod
    def _encode_id(tokenizer_obj, value, default=None):
        if value is None:
            if default is None:
                return tokenizer_obj.pad_token_id
            value = default
        try:
            if isinstance(value, str):
                candidates = list(getattr(tokenizer_obj, "token_aligner", {}).keys())
                match = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.lower() == value.lower()
                    ),
                    None,
                )
                if match is not None:
                    value = match
            aligned = tokenizer_obj.align(value)
            return int(tokenizer_obj.encode(aligned))
        except Exception:
            return int(tokenizer_obj.unk_token_id)

    def _resolve_panel_gene_ids(self, panel_gene_names):
        gene_tokenizer = self.tokenizer.ge_tokenizer
        gene_ids = []
        missing_positions = []
        missing_names = []
        for (position, name) in enumerate(panel_gene_names):
            if name in gene_tokenizer.gene2id:
                gene_ids.append(int(gene_tokenizer.gene2id[name]))
            elif name in gene_tokenizer.symbol2gene:
                ensembl_id = gene_tokenizer.symbol2gene[name]
                if ensembl_id in gene_tokenizer.gene2id:
                    gene_ids.append(int(gene_tokenizer.gene2id[ensembl_id]))
                else:
                    gene_ids.append(-1)
                    missing_positions.append(position)
                    missing_names.append(name)
            else:
                gene_ids.append(-1)
                missing_positions.append(position)
                missing_names.append(name)
        return (gene_ids, missing_positions, missing_names)

    def train(self, mode=True):
        """Keep the frozen GigaPath encoder deterministic during fine-tuning."""
        super().train(mode)
        self.gigapath_backbone.eval()
        return self

    def _encode_images(self, images):
        with torch.no_grad():
            return self.gigapath_backbone(images)

    def forward(
        self,
        images=None,
        coords=None,
        batch_idx=None,
        tech_tokens=None,
        organ_tokens=None,
        ge_tokens=None,
        img_features=None,
        return_full=False,
    ):
        if img_features is None:
            if images is None:
                raise ValueError("Provide either images or precomputed img_features")
            img_features = self._encode_images(images)
        if img_features.ndim != 2 or img_features.shape[1] != 1536:
            raise ValueError(
                f"Expected image features [N, 1536], got {tuple(img_features.shape)}"
            )
        n_spots = img_features.shape[0]
        device = img_features.device
        if coords is None:
            raise ValueError("STPathOriginalModel requires real spatial coordinates")
        if coords.shape != (n_spots, 2):
            raise ValueError(
                f"Expected coords [{n_spots}, 2], got {tuple(coords.shape)}"
            )
        if batch_idx is None:
            batch_idx = torch.zeros(n_spots, dtype=torch.long, device=device)
        if batch_idx.shape != (n_spots,):
            raise ValueError(
                f"Expected batch_idx [{n_spots}], got {tuple(batch_idx.shape)}"
            )
        if ge_tokens is None:
            mask = self.tokenizer.ge_tokenizer.mask_token.to(device=device)
            ge_tokens = mask.unsqueeze(0).repeat(n_spots, 1)
        else:
            ge_tokens = ge_tokens.to(device=device, dtype=torch.float32)
        if tech_tokens is None:
            tech_tokens = torch.full(
                (n_spots,), self._tech_id, dtype=torch.long, device=device
            )
        else:
            tech_tokens = tech_tokens.to(device=device, dtype=torch.long)
        if organ_tokens is None:
            organ_tokens = torch.full(
                (n_spots,), self._organ_id, dtype=torch.long, device=device
            )
        else:
            organ_tokens = organ_tokens.to(device=device, dtype=torch.long)
        (full_prediction, features) = self.stfm.prediction_head(
            img_tokens=img_features,
            coords=coords.to(device=device, dtype=torch.float32),
            ge_tokens=ge_tokens,
            batch_idx=batch_idx.to(device=device, dtype=torch.long),
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
            return_all=True,
        )
        if self.panel_gene_ids is None:
            return (full_prediction, features) if return_full else full_prediction
        panel_prediction = full_prediction.new_zeros(
            (n_spots, len(self.panel_gene_names))
        )
        valid_positions = [
            position
            for (position, gene_id) in enumerate(self.panel_gene_ids)
            if gene_id >= 0
        ]
        valid_ids = [self.panel_gene_ids[position] for position in valid_positions]
        if valid_positions:
            panel_prediction[:, valid_positions] = full_prediction[:, valid_ids]
        if self.panel_missing_head is not None:
            panel_prediction[:, self.panel_missing_positions] = self.panel_missing_head(
                features
            )
        return (panel_prediction, features) if return_full else panel_prediction


class STPathSmallModel(nn.Module):
    """Compact spatial STPath with transferred pretrained STFM layers."""

    def __init__(
        self,
        panel_gene_names,
        assets_dir="./assets",
        stpath_ckpt="stfm.pth",
        gigapath_ckpt="pytorch_model.bin",
        gene_voc_path=None,
        organ_type=None,
        tech_type="Visium",
        freeze_gigapath=True,
        head_hidden_dim=1536,
        dropout=0.1,
        stpath_small_d_model=512,
        stpath_small_layers=1,
        stpath_small_heads=4,
        stpath_small_mlp_ratio=2.0,
    ):
        super().__init__()
        if not freeze_gigapath:
            raise ValueError("STPathSmallModel requires freeze_gigapath=True")
        if panel_gene_names is None or len(panel_gene_names) == 0:
            raise ValueError(
                "STPathSmallModel requires panel_gene_names so its output head can match the dataset .npy gene order."
            )
        if int(stpath_small_d_model) != 512:
            raise ValueError(
                "stpath_small_d_model must be 512 because the compact model transfers the 512-dimensional STPath checkpoint weights."
            )
        if int(stpath_small_layers) != 1:
            raise ValueError(
                "stpath_small_layers must be 1 for the current checkpoint-compatible compact design."
            )
        if int(stpath_small_heads) != 4:
            raise ValueError(
                "stpath_small_heads must be 4 for the current checkpoint-compatible compact design."
            )
        if float(stpath_small_mlp_ratio) != 2.0:
            raise ValueError(
                "stpath_small_mlp_ratio must be 2.0 for the current checkpoint-compatible compact design."
            )
        gigapath_ckpt_path = os.path.join(assets_dir, "gigapath", gigapath_ckpt)
        stpath_ckpt_path = os.path.join(assets_dir, "stpath", stpath_ckpt)
        if not os.path.isfile(gigapath_ckpt_path):
            raise FileNotFoundError(
                f"GigaPath checkpoint not found: {gigapath_ckpt_path}"
            )
        if not os.path.isfile(stpath_ckpt_path):
            raise FileNotFoundError(f"STPath checkpoint not found: {stpath_ckpt_path}")
        if gene_voc_path is None:
            gene_voc_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "STPath",
                "utils_data",
                "symbol2ensembl.json",
            )
        if not os.path.isfile(gene_voc_path):
            raise FileNotFoundError(
                f"STPath gene vocabulary not found: {gene_voc_path}"
            )
        from stpath.model.nn_utils.config import ModelConfig
        from stpath.model.encoder.spatial_transformer import SpatialTransformer
        from stpath.tokenization import (
            AnnotationTokenizer,
            GeneExpTokenizer,
            IDTokenizer,
            ImageTokenizer,
            TokenizerTools,
        )

        self.panel_gene_names = [str(gene) for gene in panel_gene_names]
        self.num_genes = len(self.panel_gene_names)
        self.organ_type = organ_type
        self.tech_type = tech_type
        self._d_model = 512
        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )
        self._organ_id = self._encode_id(
            self.tokenizer.organ_tokenizer, organ_type, default="Others"
        )
        self._tech_id = self._encode_id(
            self.tokenizer.tech_tokenizer, tech_type, default=None
        )
        print(
            f"[STPathSmallModel] Conditioning: organ={organ_type!r} (id={self._organ_id}), tech={tech_type!r} (id={self._tech_id})"
        )
        self.gigapath_backbone = timm.create_model(
            "vit_giant_patch14_dinov2",
            patch_size=16,
            img_size=224,
            in_chans=3,
            num_classes=0,
            dynamic_img_size=False,
        )
        gigapath_state = torch.load(gigapath_ckpt_path, map_location="cpu")
        if isinstance(gigapath_state, dict) and isinstance(
            gigapath_state.get("model"), dict
        ):
            gigapath_state = gigapath_state["model"]
        self.gigapath_backbone.load_state_dict(gigapath_state, strict=False)
        for parameter in self.gigapath_backbone.parameters():
            parameter.requires_grad = False
        self.gigapath_backbone.eval()
        spatial_config = ModelConfig(
            d_input=self._d_model,
            d_model=self._d_model,
            n_layers=1,
            n_heads=4,
            dropout=dropout,
            attn_dropout=dropout,
            act="gelu",
            mlp_ratio=2.0,
        )
        self.spatial_transformer = SpatialTransformer(spatial_config)
        self.image_embed = nn.Linear(1536, self._d_model)
        self.tech_embed = nn.Embedding(
            self.tokenizer.tech_tokenizer.n_tokens, self._d_model
        )
        self.organ_embed = nn.Embedding(
            self.tokenizer.organ_tokenizer.n_tokens, self._d_model
        )
        self.mask_gene_embed = nn.Parameter(torch.empty(self._d_model))
        self.task_head = nn.Sequential(
            nn.Linear(self._d_model, int(head_hidden_dim)),
            nn.LayerNorm(int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(head_hidden_dim), self.num_genes),
        )
        stfm_state = torch.load(stpath_ckpt_path, map_location="cpu")
        if isinstance(stfm_state, dict) and isinstance(
            stfm_state.get("state_dict"), dict
        ):
            stfm_state = stfm_state["state_dict"]
        required_keys = [
            "input_encoder.image_embed.weight",
            "input_encoder.image_embed.bias",
            "input_encoder.gene_embed.weight",
            "input_encoder.tech_embed.weight",
            "input_encoder.organ_embed.weight",
        ]
        missing_keys = [key for key in required_keys if key not in stfm_state]
        if missing_keys:
            raise KeyError(
                f"The STPath checkpoint is missing compact-transfer keys: {missing_keys}"
            )
        with torch.no_grad():
            self.image_embed.weight.copy_(
                stfm_state["input_encoder.image_embed.weight"]
            )
            self.image_embed.bias.copy_(stfm_state["input_encoder.image_embed.bias"])
            self.tech_embed.weight.copy_(stfm_state["input_encoder.tech_embed.weight"])
            self.organ_embed.weight.copy_(
                stfm_state["input_encoder.organ_embed.weight"]
            )
            mask_id = self.tokenizer.ge_tokenizer.mask_token_id
            self.mask_gene_embed.copy_(
                stfm_state["input_encoder.gene_embed.weight"][:, mask_id]
            )
        first_block_state = {
            key[len("model.") :]: value
            for (key, value) in stfm_state.items()
            if key.startswith("model.blks.0.")
        }
        if not first_block_state:
            raise KeyError(
                "The STPath checkpoint does not contain model.blks.0 weights"
            )
        (missing, unexpected) = self.spatial_transformer.load_state_dict(
            first_block_state, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Compact STPath spatial-block transfer was incomplete: missing={missing}, unexpected={unexpected}"
            )
        del stfm_state
        total = sum((parameter.numel() for parameter in self.parameters()))
        trainable = sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
        print(
            f"[STPathSmallModel] Initialized with frozen GigaPath, one pretrained spatial block, and panel head: total={total:,}, trainable={trainable:,}, panel_genes={self.num_genes}"
        )

    @staticmethod
    def _encode_id(tokenizer_obj, value, default=None):
        if value is None:
            if default is None:
                return tokenizer_obj.pad_token_id
            value = default
        try:
            if isinstance(value, str):
                candidates = list(getattr(tokenizer_obj, "token_aligner", {}).keys())
                match = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.lower() == value.lower()
                    ),
                    None,
                )
                if match is not None:
                    value = match
            aligned = tokenizer_obj.align(value)
            return int(tokenizer_obj.encode(aligned))
        except Exception:
            return int(tokenizer_obj.unk_token_id)

    def train(self, mode=True):
        super().train(mode)
        self.gigapath_backbone.eval()
        return self

    def _encode_images(self, images):
        with torch.no_grad():
            return self.gigapath_backbone(images)

    def forward(
        self,
        images=None,
        coords=None,
        batch_idx=None,
        tech_tokens=None,
        organ_tokens=None,
        ge_tokens=None,
        img_features=None,
        return_full=False,
    ):
        if img_features is None:
            if images is None:
                raise ValueError("Provide either images or precomputed img_features")
            img_features = self._encode_images(images)
        if img_features.ndim != 2 or img_features.shape[1] != 1536:
            raise ValueError(
                f"Expected image features [N, 1536], got {tuple(img_features.shape)}"
            )
        n_spots = img_features.shape[0]
        device = img_features.device
        if coords is None:
            raise ValueError("STPathSmallModel requires real spatial coordinates")
        if coords.shape != (n_spots, 2):
            raise ValueError(
                f"Expected coords [{n_spots}, 2], got {tuple(coords.shape)}"
            )
        if batch_idx is None:
            batch_idx = torch.zeros(n_spots, dtype=torch.long, device=device)
        if batch_idx.shape != (n_spots,):
            raise ValueError(
                f"Expected batch_idx [{n_spots}], got {tuple(batch_idx.shape)}"
            )
        if ge_tokens is not None:
            raise ValueError(
                "STPathSmallModel is an image-only compact model and uses its learned mask token; do not pass measured gene-expression tokens."
            )
        if tech_tokens is None:
            tech_tokens = torch.full(
                (n_spots,), self._tech_id, dtype=torch.long, device=device
            )
        else:
            tech_tokens = tech_tokens.to(device=device, dtype=torch.long)
        if organ_tokens is None:
            organ_tokens = torch.full(
                (n_spots,), self._organ_id, dtype=torch.long, device=device
            )
        else:
            organ_tokens = organ_tokens.to(device=device, dtype=torch.long)
        x = self.image_embed(img_features)
        x = x + self.mask_gene_embed.unsqueeze(0)
        x = x + self.tech_embed(tech_tokens)
        x = x + self.organ_embed(organ_tokens)
        features = self.spatial_transformer(
            x,
            coords.to(device=device, dtype=torch.float32),
            batch_idx.to(device=device, dtype=torch.long),
        )
        panel_prediction = self.task_head(features)
        return (panel_prediction, features) if return_full else panel_prediction


class STPathSmallScratchModel(STPathSmallModel):
    """Random compact spatial model on frozen pretrained GigaPath features."""

    def __init__(
        self,
        panel_gene_names,
        assets_dir="./assets",
        gigapath_ckpt="pytorch_model.bin",
        organ_type=None,
        tech_type="Visium",
        freeze_gigapath=True,
        head_hidden_dim=1536,
        dropout=0.1,
        stpath_small_d_model=512,
        stpath_small_layers=1,
        stpath_small_heads=4,
        stpath_small_mlp_ratio=2.0,
    ):
        nn.Module.__init__(self)
        if not freeze_gigapath:
            raise ValueError("STPathSmallScratchModel requires freeze_gigapath=True")
        if panel_gene_names is None or len(panel_gene_names) == 0:
            raise ValueError(
                "STPathSmallScratchModel requires panel_gene_names so its output head matches the dataset .npy gene order."
            )
        if int(stpath_small_d_model) != 512:
            raise ValueError("stpath_small_d_model must be 512")
        if int(stpath_small_layers) != 1:
            raise ValueError("stpath_small_layers must be 1")
        if int(stpath_small_heads) != 4:
            raise ValueError("stpath_small_heads must be 4")
        if float(stpath_small_mlp_ratio) != 2.0:
            raise ValueError("stpath_small_mlp_ratio must be 2.0")
        gigapath_ckpt_path = os.path.join(assets_dir, "gigapath", gigapath_ckpt)
        if not os.path.isfile(gigapath_ckpt_path):
            raise FileNotFoundError(
                f"GigaPath checkpoint not found: {gigapath_ckpt_path}"
            )
        from stpath.model.nn_utils.config import ModelConfig
        from stpath.model.encoder.spatial_transformer import SpatialTransformer
        from stpath.tokenization import IDTokenizer, TokenizerTools

        self.panel_gene_names = [str(gene) for gene in panel_gene_names]
        self.num_genes = len(self.panel_gene_names)
        self.organ_type = organ_type
        self.tech_type = tech_type
        self._d_model = 512
        self.tokenizer = TokenizerTools(
            ge_tokenizer=None,
            image_tokenizer=None,
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=None,
            domain_anno_tokenizer=None,
        )
        self._organ_id = STPathSmallModel._encode_id(
            self.tokenizer.organ_tokenizer, organ_type, default="Others"
        )
        self._tech_id = STPathSmallModel._encode_id(
            self.tokenizer.tech_tokenizer, tech_type, default=None
        )
        print(
            f"[STPathSmallScratchModel] Conditioning: organ={organ_type!r} (id={self._organ_id}), tech={tech_type!r} (id={self._tech_id})"
        )
        self.gigapath_backbone = timm.create_model(
            "vit_giant_patch14_dinov2",
            patch_size=16,
            img_size=224,
            in_chans=3,
            num_classes=0,
            dynamic_img_size=False,
        )
        gigapath_state = torch.load(gigapath_ckpt_path, map_location="cpu")
        if isinstance(gigapath_state, dict) and isinstance(
            gigapath_state.get("model"), dict
        ):
            gigapath_state = gigapath_state["model"]
        self.gigapath_backbone.load_state_dict(gigapath_state, strict=False)
        for parameter in self.gigapath_backbone.parameters():
            parameter.requires_grad = False
        self.gigapath_backbone.eval()
        spatial_config = ModelConfig(
            d_input=self._d_model,
            d_model=self._d_model,
            n_layers=1,
            n_heads=4,
            dropout=dropout,
            attn_dropout=dropout,
            act="gelu",
            mlp_ratio=2.0,
        )
        self.spatial_transformer = SpatialTransformer(spatial_config)
        self.image_embed = nn.Linear(1536, self._d_model)
        self.tech_embed = nn.Embedding(
            self.tokenizer.tech_tokenizer.n_tokens, self._d_model
        )
        self.organ_embed = nn.Embedding(
            self.tokenizer.organ_tokenizer.n_tokens, self._d_model
        )
        self.mask_gene_embed = nn.Parameter(torch.empty(self._d_model))
        nn.init.normal_(self.mask_gene_embed, mean=0.0, std=0.02)
        self.task_head = nn.Sequential(
            nn.Linear(self._d_model, int(head_hidden_dim)),
            nn.LayerNorm(int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(head_hidden_dim), self.num_genes),
        )
        total = sum((parameter.numel() for parameter in self.parameters()))
        trainable = sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
        print(
            f"[STPathSmallScratchModel] Initialized with random compact weights, frozen GigaPath, and no STFM checkpoint: total={total:,}, trainable={trainable:,}, panel_genes={self.num_genes}"
        )
