import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePatchProjector(nn.Module):
    """
    Projects image patches into d-dim embeddings:
      z_i = W_e x_i + b_e
    Adds learnable positional embeddings p_i.
    Accepts:
      - Raw images shaped (B, C, H, W)
      - patches shaped (B, N, H, W, C)
      - flattened patches shaped (B, N, Din)
    Returns: Z_tilde shaped (B, N, d)
    """
    def __init__(self, H, W, C, d, patch_size=16):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.patch_size = patch_size
        self.Din = patch_size * patch_size * C
        self.d = d
        self.proj = nn.Linear(self.Din, d)   # W_e and b_e
        self.pos_embed = None
        
        # Calculate number of patches
        self.n_patches_h = H // patch_size
        self.n_patches_w = W // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

    def _create_positional_embeddings(self, N, device):
        self.pos_embed = nn.Parameter(torch.randn(1, N, self.d, device=device))
        
    def _image_to_patches(self, images):
        # images: (B, C, H, W)
        B, C, H, W = images.shape
        
        # Ensure image dimensions are compatible with patch size
        if H != self.H or W != self.W:
            images = F.interpolate(images, size=(self.H, self.W), mode='bilinear', align_corners=False)
            
        # Unfold into patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, n_patches_h, n_patches_w, C, patch_size, patch_size]
        patches = patches.reshape(B, -1, C, self.patch_size, self.patch_size)  # [B, N, C, patch_size, patch_size]
        return patches.reshape(B, -1, self.Din)  # [B, N, Din]

    def forward(self, x):
        # Handle different input formats
        if x.dim() == 4:  # Raw image input (B, C, H, W)
            x = self._image_to_patches(x)  # Convert to (B, N, Din)
        elif x.dim() == 5:  # Patch input (B, N, H, W, patchesC)
            B, N, H, W, C = x.shape
            x = x.reshape(B, N, -1)  # Flatten to (B, N, Din)
        
        # Verify dimensions
        if x.shape[-1] != self.Din:
            raise ValueError(f"Expected patches with dimension {self.Din}, got {x.shape[-1]}")
            
        B, N = x.shape[:2]
        
        # Create or verify positional embeddings
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            self._create_positional_embeddings(N, x.device)

        # Project and add positional embeddings
        z = self.proj(x)  # (B, N, d)
        z_tilde = z + self.pos_embed  # (B, N, d)
        
        return z_tilde

        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            # Create new positional embedding for this N (re-initialize if N changes)
            # (Alternatively you can demand fixed N known at init.)
            self._create_positional_embeddings(N, device=x.device)

        z = self.proj(x)                 # (B, N, d)
        z_tilde = z + self.pos_embed     # broadcast pos embeddings
        return z_tilde                   # (B, N, d)


class GeneQueryProjector(nn.Module):
    """
    Projects scalar gene value y_j to d-dim embedding:
      g_j = W_g * y_j + b_g
    y_j is scalar per gene. Input: (B, G) -> output (B, G, d)
    """
    def __init__(self, d):
        super().__init__()
        # W_g is shape (d, 1) implemented as Linear(1, d)
        self.scalar_proj = nn.Linear(1, d)
    
    def forward(self, gene_values):
        # gene_values: (B, G) or (B, G, 1)
        if gene_values.dim() == 2:
            gene_values = gene_values.unsqueeze(-1)  # (B, G, 1)
        B, G, _ = gene_values.shape
        # Flatten to (B*G, 1), project, then reshape
        g = self.scalar_proj(gene_values)  # (B, G, d)
        return g


class ImageToGeneTransformer(nn.Module):
    """
    Implements the full model:
      - encoder: TransformerEncoder over spot embeddings Z_tilde (B, N, d)
      - decoder: TransformerDecoder where tgt is gene token embeddings (B, G, d)
                 and memory is encoder output (B, N, d)
      - final projection: h_j -> y_hat_j = W_o h_j + b_o  (per-gene scalar)
    Behavior:
      - training: pass gene_values (B, G) -> project to queries with GeneQueryProjector
                  (this matches your equations using ground-truth y)
      - inference: pass gene_values=None -> uses learned gene-query tokens (learnable)
    """
    def __init__(
        self, H, W, C,
        num_genes,
        d_model=768,
        encoder_layers=6,
        decoder_layers=4,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        layer_norm_eps=1e-5,
        batch_first=True,
        freeze_encoder=False,
        patch_size=16
    ):
        super().__init__()
        self.num_genes = num_genes
        self.d_model = d_model

        # Image patch projector + positional embeddings
        self.patch_proj = ImagePatchProjector(H, W, C, d_model, patch_size=patch_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Decoder: transformer decoder layers (self-attn among gene tokens, cross-attn to encoder outputs)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=batch_first
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Gene projection (for training; projects scalar y_j -> d-dim g_j)
        self.g_proj = GeneQueryProjector(d_model)

        # Learned gene query tokens for inference (if gene_values not provided)
        # shape (1, G, d)
        self.gene_query_tokens = nn.Parameter(torch.randn(1, num_genes, d_model))

        # Output projection per gene: h_j (d) -> scalar
        self.output_head = nn.Linear(d_model, 1)

        # Optionally freeze encoder weights (useful if using pretrained patch projector/encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.patch_proj.parameters():
                p.requires_grad = True

    def forward(self, patches, gene_values=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        patches: (B, N, H, W, C) OR (B, N, Din)
        gene_values: (B, G) used during training for teacher forcing. If None -> use learned tokens.
        src_key_padding_mask: (B, N) bool mask where True indicates padding (optional)
        tgt_key_padding_mask: (B, G) bool mask for gene tokens
        Returns:
          y_hat: (B, G) predicted gene scalar values
          decoder_states: (B, G, d) optional latent vectors (before final projection)
        """
        B = patches.shape[0]
        # 1) Encoder input Z_tilde: (B, N, d)
        Z_tilde = self.patch_proj(patches)  # includes positional embeddings

        # 2) Encoder output He: (B, N, d)
        He = self.encoder(Z_tilde, src_key_padding_mask=src_key_padding_mask)

        # 3) Decoder input G:
        if gene_values is not None:
            # Use ground-truth gene scalars (training mode) to create g_j as in your paper
            G_t = self.g_proj(gene_values)   # (B, G, d)
        else:
            # Inference: use learned gene query tokens (same for all batches)
            G_t = self.gene_query_tokens.expand(B, -1, -1)  # (B, G, d)

        # 4) Transformer decoder: tgt=G_t, memory=He. Output Hd: (B, G, d)
        Hd = self.decoder(tgt=G_t, memory=He,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=src_key_padding_mask)

        # 5) Project each hd_j to scalar y_hat_j
        y_hat = self.output_head(Hd).squeeze(-1)   # (B, G)

        return y_hat, Hd