"""Trainable model definition for the published UNI-only Stem ablation.

Stem is a conditional diffusion model.  Its normal forward inputs are a noisy
gene-expression vector, a diffusion timestep, and a frozen UNI embedding.  The
convenience methods below expose the official diffusion training loss and the
iterative sample-mean prediction used at evaluation time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).resolve().parent
STEM_SOURCE_ROOT = MODEL_DIR / "Stem"
if not STEM_SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"Stem source not found at {STEM_SOURCE_ROOT}. "
        "Copy the official Stem package into models/Stem first."
    )
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from Stem.diffusion import create_diffusion  # noqa: E402
from Stem.models import Stem_models  # noqa: E402
from models import UNI  # noqa: E402


class StemUNI(nn.Module):
    """Official Stem DiT conditioned on frozen 1024-dimensional UNI features."""

    def __init__(
        self,
        num_genes: int,
        depth: int = 12,
        hidden_size: int = 384,
        num_heads: int = 6,
        condition_size: int = 1024,
    ) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        self.condition_size = int(condition_size)
        self.dit = Stem_models["Stem"](
            input_size=self.num_genes,
            depth=int(depth),
            hidden_size=int(hidden_size),
            num_heads=int(num_heads),
            label_size=self.condition_size,
        )
        self.train_diffusion = create_diffusion(timestep_respacing="")

    def forward(
        self,
        noisy_expression: torch.Tensor,
        timestep: torch.Tensor,
        uni_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict diffusion noise and variance using the official Stem DiT."""
        if uni_embedding.ndim != 2 or uni_embedding.shape[1] != self.condition_size:
            raise ValueError(
                f"Expected UNI conditions shaped [batch,{self.condition_size}], "
                f"got {tuple(uni_embedding.shape)}"
            )
        return self.dit(noisy_expression, timestep, uni_embedding)

    def diffusion_loss(
        self,
        expression: torch.Tensor,
        uni_embedding: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the official DDPM training objective for one mini-batch."""
        if expression.ndim == 2:
            expression = expression.unsqueeze(1)
        if expression.ndim != 3 or expression.shape[1:] != (1, self.num_genes):
            raise ValueError(
                f"Expected expression shaped [batch,1,{self.num_genes}], "
                f"got {tuple(expression.shape)}"
            )
        if timestep is None:
            timestep = torch.randint(
                0,
                self.train_diffusion.num_timesteps,
                (expression.shape[0],),
                device=expression.device,
            )
        return self.train_diffusion.training_losses(
            self,
            expression,
            timestep,
            {"uni_embedding": uni_embedding},
        )["loss"].mean()

    @torch.inference_mode()
    def sample(
        self,
        uni_embedding: torch.Tensor,
        num_samples_per_condition: int = 20,
        num_sampling_steps: int = 1000,
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate expression samples shaped [batch, samples, genes]."""
        samples_per_condition = int(num_samples_per_condition)
        if samples_per_condition < 1:
            raise ValueError("num_samples_per_condition must be positive")
        diffusion = create_diffusion(str(int(num_sampling_steps)))
        condition = uni_embedding.repeat_interleave(samples_per_condition, dim=0)
        noise = torch.randn(
            condition.shape[0],
            1,
            self.num_genes,
            device=uni_embedding.device,
            dtype=torch.float32,
        )
        generated = diffusion.p_sample_loop(
            self.forward,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs={"uni_embedding": condition},
            progress=progress,
            device=uni_embedding.device,
        )
        return generated.squeeze(1).reshape(
            uni_embedding.shape[0], samples_per_condition, self.num_genes
        )

    @torch.inference_mode()
    def predict(
        self,
        uni_embedding: torch.Tensor,
        num_samples_per_condition: int = 20,
        num_sampling_steps: int = 1000,
        clamp_nonnegative: bool = True,
    ) -> torch.Tensor:
        """Return Stem's sample-mean gene-expression prediction."""
        prediction = self.sample(
            uni_embedding,
            num_samples_per_condition=num_samples_per_condition,
            num_sampling_steps=num_sampling_steps,
        ).mean(dim=1)
        return prediction.clamp_min(0) if clamp_nonnegative else prediction


class StemUNIEvaluationModel(nn.Module):
    """End-to-end H&E evaluator: frozen UNI followed by trained Stem sampling."""

    def __init__(
        self,
        num_genes: int,
        uni_checkpoint: str,
        stem_depth: int = 12,
        stem_hidden_size: int = 384,
        stem_num_heads: int = 6,
        num_sampling_steps: int = 1000,
        num_samples_per_condition: int = 20,
        clamp_predictions_nonnegative: bool = True,
    ) -> None:
        super().__init__()
        checkpoint = Path(uni_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint}")

        uni_container = UNI(
            num_genes=1,
            assets_dir=str(checkpoint.parent.parent),
            checkpoint=checkpoint.name,
            freeze_backbone=True,
        )
        self.uni_backbone = uni_container.backbone
        self.stem = StemUNI(
            num_genes=num_genes,
            depth=stem_depth,
            hidden_size=stem_hidden_size,
            num_heads=stem_num_heads,
            condition_size=1024,
        )
        self.num_sampling_steps = int(num_sampling_steps)
        self.num_samples_per_condition = int(num_samples_per_condition)
        self.clamp_predictions_nonnegative = bool(clamp_predictions_nonnegative)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        uni_embeddings = self.uni_backbone(images)
        return self.stem.predict(
            uni_embeddings,
            num_samples_per_condition=self.num_samples_per_condition,
            num_sampling_steps=self.num_sampling_steps,
            clamp_nonnegative=self.clamp_predictions_nonnegative,
        )
