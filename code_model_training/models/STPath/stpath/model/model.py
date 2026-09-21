import torch
import torch.nn as nn
import torch.nn.functional as F

from stpath.model.nn_utils.config import ModelConfig
from stpath.model.nn_utils import RegressionHead


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.head = RegressionHead(config.d_model, config.n_genes)
    
    def forward(self, x, *args, **kwargs):
        return self.head(x)


def get_backbone(config):
    if config.backbone == "spatial_transformer":
        from stpath.model.encoder.spatial_transformer import SpatialTransformer

        return SpatialTransformer(
                ModelConfig(
                    n_genes=config.n_genes,
                    d_input=config.d_model,
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    act=config.activation,
                    mlp_ratio=config.mlp_ratio,
                )
            )
    else:
        raise ValueError(f"Backbone {config.backbone} not supported")


class EncodeInputs(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_embed = nn.Linear(config.feature_dim, config.d_model)
        self.gene_embed = nn.Linear(config.n_genes, config.d_model, bias=False)
        self.tech_embed = nn.Embedding(config.n_tech, config.d_model)
        self.organ_embed = nn.Embedding(config.n_organs, config.d_model)

    def forward(
        self,
        img_tokens, 
        ge_tokens,
        tech_tokens,
        organ_tokens,
    ):
        img_embed = self.image_embed(img_tokens)
        ge_embed = self.gene_embed(ge_tokens)
        tech_embed = self.tech_embed(tech_tokens)
        organ_embed = self.organ_embed(organ_tokens)

        return (
            img_embed
            + ge_embed
            + tech_embed
            + organ_embed
        )


class STFM(nn.Module):
    def __init__(self, config) -> None:
        super(STFM, self).__init__()

        self.backbone = config.backbone
        self.input_encoder = EncodeInputs(config)
        self.model = get_backbone(config)
        
        self.gene_exp_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.n_genes),
        ) if config.backbone != "MLP" else nn.Identity()

        self.regression_loss_func = nn.MSELoss()

    def inference(
        self,         
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: torch.Tensor | None = None,
        organ_tokens: torch.Tensor | None = None,
    ):
        x = self.input_encoder(
            img_tokens=img_tokens,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )
        return self.model(x, coords, batch_idx)

    def prediction_head(
        self,         
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: torch.Tensor | None = None,
        organ_tokens: torch.Tensor | None = None,
        return_all=False,
    ):

        x = self.inference(
            img_tokens=img_tokens,
            coords=coords,
            batch_idx=batch_idx,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )

        if return_all:
            return self.gene_exp_head(x), x
        return self.gene_exp_head(x)

    def forward(        
        self,         
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        obs_gene_ids: torch.Tensor,
        ge_masked_tokens=None,
        tech_tokens: torch.Tensor | None = None,
        organ_tokens: torch.Tensor | None = None,
    ):

        x = self.inference(
            img_tokens=img_tokens,
            coords=coords,
            ge_tokens=ge_tokens,
            batch_idx=batch_idx,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )

        if ge_masked_tokens is not None:
            ge_masked_ids, ge_labels = ge_masked_tokens.position, ge_masked_tokens.groundtruth
            prediction = self.gene_exp_head(x)
            prediction = prediction[ge_masked_ids]
            masked_obs_gene_ids = obs_gene_ids[ge_masked_ids]
            assert prediction.shape[0] == ge_labels.shape[0]

            # calculate the gene expression on observed genes
            prediction = prediction[torch.arange(prediction.shape[0]).unsqueeze(-1), masked_obs_gene_ids]
            ge_labels = ge_labels[torch.arange(ge_labels.shape[0]).unsqueeze(-1), masked_obs_gene_ids]
            padding_mask = masked_obs_gene_ids == 0

            loss = self.regression_loss_func(prediction[~padding_mask], ge_labels[~padding_mask])
        else:
            raise NotImplementedError("No ge_masked_tokens provided, please provide it for training")

        return loss


