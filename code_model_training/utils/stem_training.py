"""Dataset-independent training utilities for the Stem-UNI runners.

This module contains only shared model training, sampling, metrics, logging,
gene-list, and frozen-UNI helpers. Dataset discovery and preprocessing remain
inside their respective Xenium and Visium entry points.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
STEM_SOURCE_ROOT = MODEL_DIR / "Stem"
if not STEM_SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"Stem source not found at {STEM_SOURCE_ROOT}. "
        "Copy the official Stem package into models/Stem first."
    )
for source_path in (PROJECT_ROOT, MODEL_DIR, PROJECT_ROOT / "utils"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from Stem.train_helper import requires_grad, update_ema  # noqa: E402
from models import UNI  # noqa: E402
from stem_uni import StemUNI  # noqa: E402
from compute_metrics import spearmanrr  # noqa: E402


UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)
METRICS_CSV_FIELDS = [
    "fold",
    "epoch",
    "phase",
    "custom_loss",
    "pearson_mean_cell_wise",
    "spearman_mean_cell_wise",
    "pearson_mean_genewise",
    "spearman_mean_genewise",
    "l1_error_mean",
    "l2_errors_mean",
    "r2_scores_mean",
    "pearson_std",
    "l2_error_q1",
    "l2_error_q2",
    "l2_error_q3",
    "r2_score_q1",
    "r2_score_q2",
    "r2_score_q3",
    "mape_mean",
    "mape_std",
    "rmse_mean",
    "rmse_std",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def batchwise_validation_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_size: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Reproduce the validation loss/PCC/SRCC aggregation used by main2."""
    mse_values: list[float] = []
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    rank_device = device or torch.device("cpu")
    for start in range(0, len(y_true), batch_size):
        true_batch = y_true[start : start + batch_size]
        pred_batch = y_pred[start : start + batch_size]
        mse_values.append(float(np.mean((pred_batch - true_batch) ** 2)))
        try:
            value = float(pearsonr(pred_batch.reshape(-1), true_batch.reshape(-1))[0])
            pearson_values.append(value if np.isfinite(value) else 0.0)
        except Exception:
            pearson_values.append(0.0)
        try:
            with torch.no_grad():
                value = float(
                    spearmanrr(
                        torch.from_numpy(pred_batch).to(rank_device, dtype=torch.float32),
                        torch.from_numpy(true_batch).to(rank_device, dtype=torch.float32),
                    ).item()
                )
            spearman_values.append(value if np.isfinite(value) else 0.0)
        except Exception:
            spearman_values.append(0.0)
    return {
        "custom_loss": float(np.mean(mse_values)) if mse_values else 0.0,
        "pearson_mean_cell_wise": float(np.mean(pearson_values)) if pearson_values else 0.0,
        "spearman_mean_cell_wise": float(np.mean(spearman_values)) if spearman_values else 0.0,
    }


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"stem_{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_gene_names(params: dict[str, Any]) -> list[str]:
    gene_path = Path(params["genes"])
    if gene_path.is_dir():
        gene_path = gene_path / f"{params['dataset_name']}.npy"
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene list not found: {gene_path}")
    genes = [str(gene) for gene in np.load(gene_path, allow_pickle=True).tolist()]
    if not genes:
        raise ValueError(f"Gene list is empty: {gene_path}")
    return genes


def uni_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=UNI_MEAN, std=UNI_STD),
        ]
    )

def load_uni_backbone(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint}")
    model = UNI(num_genes=1, assets_dir=str(checkpoint.parent.parent), freeze_backbone=True)
    backbone = model.backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


class StemCacheDataset(Dataset):
    def __init__(self, cache_path: Path):
        try:
            payload = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        self.embeddings = payload["embeddings"]
        self.targets = payload["targets"]
        self.indices = payload["indices"]
        self.barcodes = payload.get("barcodes", [])
        self.patient = payload["patient"]

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[index].float(), self.targets[index]


def _new_stem(params: dict[str, Any], n_genes: int) -> torch.nn.Module:
    return StemUNI(
        num_genes=n_genes,
        depth=int(params.get("stem_depth", 12)),
        hidden_size=int(params.get("stem_hidden_size", 384)),
        num_heads=int(params.get("stem_num_heads", 6)),
        condition_size=1024,
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict()},
        temporary_path,
    )
    os.replace(temporary_path, path)


def _save_model_weights(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_path)
    os.replace(temporary_path, path)


def train_fold(
    train_dataset: Dataset,
    params: dict[str, Any],
    n_genes: int,
    device: torch.device,
    fold_dir: Path,
    fold: int,
    train_patients: list[str],
    val_patients: list[str],
    logger: logging.Logger,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    del train_patients, val_patients
    model = _new_stem(params, n_genes).to(device)
    ema = deepcopy(model).to(device).eval()
    requires_grad(ema, False)
    update_ema(ema, model, decay=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get("learning_rate", 1e-4)), weight_decay=0.0
    )
    batch_size = int(params.get("batch_size", 32))
    if "epochs" not in params:
        raise ValueError("epochs must be set explicitly")
    epochs = int(params["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    log_every = int(params.get("log_every", 100))
    checkpoint_every = int(params.get("checkpoint_every", 0))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected_optimizer_steps = epochs * len(loader)
    logger.info(
        "Fold %d: %d samples, %d trainable Stem parameters, batch-size=%d, "
        "epochs=%d, expected-optimizer-steps=%d, diffusion-timesteps=%d",
        fold,
        len(train_dataset),
        trainable_params,
        batch_size,
        epochs,
        expected_optimizer_steps,
        model.train_diffusion.num_timesteps,
    )
    loss_csv = fold_dir / "training_loss.csv"
    with loss_csv.open("w", newline="") as handle:
        csv.writer(handle).writerow(["step", "epoch", "diffusion_loss"])
    model.train()
    global_step = 0
    running_loss = 0.0
    running_batches = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        for embedding, expression in loader:
            expression = expression.to(device, non_blocking=True).unsqueeze(1)
            embedding = embedding.to(device, non_blocking=True)
            loss = model.diffusion_loss(expression, embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model, decay=float(params.get("ema_decay", 0.9999)))
            running_loss += float(loss.detach().item())
            running_batches += 1
            global_step += 1
            if global_step % log_every == 0:
                average_loss = running_loss / max(1, running_batches)
                logger.info(
                    "Fold %d step %d epoch %d/%d diffusion loss %.6f",
                    fold,
                    global_step,
                    epoch + 1,
                    epochs,
                    average_loss,
                )
                with loss_csv.open("a", newline="") as handle:
                    csv.writer(handle).writerow([global_step, epoch + 1, average_loss])
                running_loss = 0.0
                running_batches = 0
            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                checkpoint_path = fold_dir / f"{global_step:07d}.pt"
                _save_checkpoint(checkpoint_path, model, ema, optimizer)
                logger.info("Saved checkpoint to %s", checkpoint_path)
    _save_model_weights(fold_dir / "best_model.pth", ema)
    _save_model_weights(fold_dir / "epoch_model.pth", model)
    elapsed = time.perf_counter() - started
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ema, {
        "train_steps": global_step,
        "train_epochs": epochs,
        "training_seconds": elapsed,
        "trainable_parameters": trainable_params,
    }


@torch.inference_mode()
def sample_fold(
    model: torch.nn.Module,
    dataset: Dataset,
    params: dict[str, Any],
    n_genes: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del n_genes
    sampling_steps = int(params.get("num_sampling_steps", 1_000))
    samples_per_condition = int(params.get("num_samples_per_condition", 20))
    batch_size = int(params.get("sampling_batch_size", 8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_index, (embedding, target) in enumerate(loader):
        embedding = embedding.to(device, non_blocking=True)
        prediction = model.predict(
            embedding,
            num_samples_per_condition=samples_per_condition,
            num_sampling_steps=sampling_steps,
            clamp_nonnegative=bool(params.get("clamp_predictions_nonnegative", True)),
        )
        predictions.append(prediction.cpu().numpy())
        targets.append(target.numpy())
        if batch_index % int(params.get("sampling_log_every", 25)) == 0:
            logger.info("Sampled %d/%d validation batches", batch_index + 1, len(loader))
    elapsed = time.perf_counter() - started
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    return np.vstack(targets), np.vstack(predictions), {
        "sampling_seconds": elapsed,
        "sampling_seconds_per_cell": elapsed / max(1, len(dataset)),
        "sampling_steps": sampling_steps,
        "samples_per_condition": samples_per_condition,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }


def relative_variation_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_variance = np.var(y_true, axis=0)
    predicted_variance = np.var(y_pred, axis=0)
    valid = true_variance > 1e-12
    if not np.any(valid):
        return 0.0
    return float(
        np.mean(((predicted_variance[valid] - true_variance[valid]) ** 2) / true_variance[valid] ** 2)
    )
