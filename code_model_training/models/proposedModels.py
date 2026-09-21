'''
cscc dataset: "data/cscc_dataset/224"
her2 dataset: "data/her2st_dataset/224"



'''

import torch
import torch.nn as nn
import torchvision.models as models


#Best Model
# ImageGeneCrossTransformer

# --- Cross-Attention Block --------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries from one modality, keys/values from another."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context):
        # query: (B, G, d) | context: (B, N, d)
        attn_out, _ = self.cross_attn(query, context, context)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# --- Image Projector (EfficientNet-B4) -------------------------------------
class EfficientNetImageProjector(nn.Module):
    """Extracts visual tokens from EfficientNet and projects to d_model."""
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        effnet = models.efficientnet_b4(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(effnet.features.children()))
        self.proj = nn.Linear(1792, d_model)  # EfficientNet-B4 output channels
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images):
        feats = self.backbone(images)                 # (B, 1792, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)     # (B, N=H'*W', C)
        tokens = self.proj(tokens)                    # (B, N, d_model)
        tokens = self.norm(tokens)
        return tokens


# class EfficientNetImageProjector(nn.Module):
#     """
#     Extracts visual tokens from EfficientNet-B0 and projects to d_model,
#     with learnable positional embeddings.
#     """
#     def __init__(self, d_model=512, pretrained=True):
#         super().__init__()
        
#         # Load EfficientNet-B0 backbone
#         effnet = models.efficientnet_b0(pretrained=pretrained)
#         print(f"Loaded EfficientNet-B0 backbone (pretrained={pretrained})")

#         # Use only convolutional feature extractor (exclude classifier)
#         self.backbone = nn.Sequential(*list(effnet.features.children()))

#         # EfficientNet-B0 last stage output channels = 1280
#         self.proj = nn.Linear(1280, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.d_model = d_model

#         # Placeholder for positional embedding (created dynamically)
#         self.register_parameter("pos_embed", None)

#     def forward(self, images):
#         """
#         Args:
#             images: (B, 3, H, W)
#         Returns:
#             tokens: (B, N=H'*W', d_model)
#         """
#         feats = self.backbone(images)  # (B, 1280, H', W')
#         B, C, H, W = feats.shape

#         # Flatten spatial dimensions → sequence of tokens
#         tokens = feats.flatten(2).transpose(1, 2)  # (B, N=H'*W', C)
#         tokens = self.proj(tokens)                 # (B, N, d_model)

#         N = H * W

#         # Create or update positional embeddings dynamically
#         if self.pos_embed is None or self.pos_embed.shape[1] != N:
#             pos_embed = torch.zeros(1, N, self.d_model, device=tokens.device)
#             nn.init.trunc_normal_(pos_embed, std=0.02)
#             self.pos_embed = nn.Parameter(pos_embed)

#         # Add positional encoding
#         tokens = tokens + self.pos_embed

#         # Normalize tokens for stable transformer input
#         tokens = self.norm(tokens)

#         return tokens  # (B, N, d_model)


class DenseNetImageProjector(nn.Module):
    """
    Extracts visual tokens from DenseNet121 and projects to d_model,
    with learnable positional embeddings.
    """
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()

        # ---- Load DenseNet-121 backbone ----
        densenet = models.densenet121(pretrained=pretrained)
        print(f"Loaded DenseNet-121 backbone (pretrained={pretrained})")

        # Use only convolutional feature extractor (exclude classifier)
        self.backbone = densenet.features  # Output: (B, 1024, H', W')

        # ---- Projection and normalization ----
        self.proj = nn.Linear(1024, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # Placeholder for positional embeddings (created dynamically)
        self.register_parameter("pos_embed", None)

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (B, 3, H, W)

        Returns:
            tokens: Tensor of shape (B, N=H'*W', d_model)
        """
        feats = self.backbone(images)  # (B, 1024, H', W')
        B, C, H, W = feats.shape

        # ---- Flatten spatial dimensions to sequence ----
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N=H'*W', C)
        tokens = self.proj(tokens)                 # (B, N, d_model)

        N = H * W

        # ---- Create or update positional embeddings dynamically ----
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            pos_embed = torch.zeros(1, N, self.d_model, device=tokens.device)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = nn.Parameter(pos_embed)

        # ---- Add positional embeddings ----
        tokens = tokens + self.pos_embed

        # ---- Normalize for stable downstream models ----
        tokens = self.norm(tokens)

        return tokens  # (B, N, d_model)


# class DenseNetImageProjector(nn.Module):
#     """
#     Extracts visual tokens from DenseNet121 and projects to d_model,
#     with learnable positional embeddings.
#     """
#     def __init__(self, d_model=512, pretrained=True):
#         super().__init__()

#         # ---- Load DenseNet-121 backbone ----
#         densenet = models.densenet121(pretrained=pretrained)
#         print(f"Loaded DenseNet-121 backbone (pretrained={pretrained})")

#         # 2. Extract feature extractor
#         self.backbone = densenet.features

#         # 3. NOW freeze layers (you changed to last 5)
#         modules = list(self.backbone.children())
#         total_modules = len(modules)
#         train_from = total_modules - 5

#         for idx, module in enumerate(modules):
#             for param in module.parameters():
#                 param.requires_grad = (idx >= train_from)

#         print(f"Trainable DenseNet modules: {list(range(train_from, total_modules))}")



#         # ---- Projection + normalization ----
#         self.proj = nn.Linear(1024, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.d_model = d_model
#         self.register_parameter("pos_embed", None)

#     def forward(self, images):
#         feats = self.backbone(images)  
#         B, C, H, W = feats.shape

#         tokens = feats.flatten(2).transpose(1, 2)  
#         tokens = self.proj(tokens)

#         N = H * W

#         if self.pos_embed is None or self.pos_embed.shape[1] != N:
#             pos_embed = torch.zeros(1, N, self.d_model, device=tokens.device)
#             nn.init.trunc_normal_(pos_embed, std=0.02)
#             self.pos_embed = nn.Parameter(pos_embed)

#         tokens = tokens + self.pos_embed
#         tokens = self.norm(tokens)

#         return tokens




class VGG16ImageProjector(nn.Module):
    """
    Extracts visual tokens from VGG16 and projects to d_model,
    with learnable positional embeddings.
    """
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()

        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=pretrained)
        print(f"Loaded VGG16 backbone (pretrained={pretrained})")

        # Use only convolutional feature extractor (exclude classifier)
        self.backbone = vgg.features  # Output: (B, 512, H/32, W/32)

        # Since VGG16 already outputs 512 channels, projection not needed
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # Placeholder for positional embedding (created dynamically)
        self.register_parameter("pos_embed", None)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            tokens: (B, N=H'*W', d_model)
        """
        feats = self.backbone(images)  # (B, 512, H', W')
        B, C, H, W = feats.shape

        # Flatten spatial dimensions → sequence of tokens
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N=H'*W', C)

        N = H * W

        # Create or update positional embeddings dynamically
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            pos_embed = torch.zeros(1, N, self.d_model, device=tokens.device)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = nn.Parameter(pos_embed)
            print(f"Initialized pos_embed with shape: {self.pos_embed.shape}")

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Normalize tokens for stable transformer input
        tokens = self.norm(tokens)

        return tokens  # (B, N, d_model)



# --- Gene Projector ---------------------------------------------------------
class GeneProjector(nn.Module):
    """Project scalar gene values to d_model embeddings."""
    def __init__(self, num_genes, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_genes, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_values):
        if gene_values.dim() == 2:
            gene_values = gene_values.unsqueeze(-1)
        g = self.linear(gene_values) + self.pos_embed
        return self.norm(g)  # (B, G, d_model)


# --- Simplified Transformer -------------------------------------------------
class ImageGeneCrossTransformer(nn.Module):
    """
    Simplified version (no self-attention):
      1. Extract image features with EfficientNet
      2. Project genes to embeddings
      3. Perform cross-attention (genes query image tokens)
      4. Predict per-gene expression
    """
    def __init__(
        self,
        num_genes,
        d_model=512,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        pretrained=True,
        img_proj ="effNet"
    ):
        super().__init__()

        if img_proj == "effNet":
            self.img_proj = EfficientNetImageProjector(d_model=d_model, pretrained=pretrained)
            print(f"Model EfficientNet is used as image encoder!!!")
        elif img_proj == "vgg16":
            self.img_proj = VGG16ImageProjector(d_model=d_model, pretrained=pretrained)
            print(f"Model VGG16 is used as image encoder!!!")
        elif img_proj == "denseNet":
            self.img_proj = DenseNetImageProjector(d_model=d_model, pretrained=pretrained)
            print(f"Model stNet is used as image encoder!!!")
        else:
            print("No Modal found!!!")


        self.gene_proj = GeneProjector(num_genes, d_model)

        # Learned query tokens for inference
        self.gene_query_tokens = nn.Parameter(torch.randn(1, num_genes, d_model))
        nn.init.trunc_normal_(self.gene_query_tokens, std=0.02)

        # Single cross-attention layer
        self.cross_block = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)

        # Output head
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, images, gene_values=None):
        B = images.shape[0]


        img_tokens = self.img_proj(images)  # (B, N, d)


        if gene_values is not None:
            gene_tokens = self.gene_proj(gene_values)  # (B, G, d)
        else:
            gene_tokens = self.gene_query_tokens.expand(B, -1, -1)

        # 3️⃣ Cross-attention (genes attend to image)
        H_cross = self.cross_block(gene_tokens, img_tokens)  # (B, G, d)

        # 4️⃣ Predict per-gene expression
        y_hat = self.output_head(H_cross).squeeze(-1)  # (B, G)
        return y_hat, H_cross
