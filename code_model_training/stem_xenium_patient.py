#!/usr/bin/env python3
"""Patient-wise Xenium training for the published UNI-only Stem ablation.

The pathology foundation model is frozen and used once to cache a 1024-D UNI
embedding per cell patch.  The trainable portion is the official Stem
conditional diffusion transformer stored under models/Stem.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
STEM_SOURCE_ROOT = SCRIPT_DIR / "models" / "Stem"
if not STEM_SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"Stem source not found at {STEM_SOURCE_ROOT}. "
        "Copy the official Stem package into models/Stem first."
    )
if str(STEM_SOURCE_ROOT.parent) not in sys.path:
    sys.path.append(str(STEM_SOURCE_ROOT.parent))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "utils"))

from Stem.train_helper import requires_grad, update_ema  # noqa: E402
from models import UNI  # noqa: E402
from stem_uni import StemUNI  # noqa: E402
from compute_metrics import compute_metrics, spearmanrr  # noqa: E402
from dataPrep_xenium import XeniumPatientDataset  # noqa: E402


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _batchwise_validation_statistics(
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


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"stem_xenium_{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _load_gene_names(params: dict[str, Any]) -> list[str]:
    gene_path = Path(params["genes"])
    if gene_path.is_dir():
        gene_path = gene_path / f"{params['dataset_name']}.npy"
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene list not found: {gene_path}")
    genes = [str(g) for g in np.load(gene_path, allow_pickle=True).tolist()]
    if not genes:
        raise ValueError(f"Gene list is empty: {gene_path}")
    return genes


def _discover_patient_ids(dataset_path: Path) -> list[str]:
    h5_ids = {p.stem for p in dataset_path.glob("*.h5")}
    h5ad_ids = {p.stem for p in dataset_path.glob("*.h5ad")}
    patients = sorted(h5_ids & h5ad_ids)
    if not patients:
        raise FileNotFoundError(f"No paired .h5/.h5ad patients found in {dataset_path}")
    return patients


def _image_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        key = "patches" if "patches" in handle else "img"
        return int(handle[key].shape[0])


def _fixed_patient_subsets(
    dataset_path: Path,
    patients: list[str],
    spots_per_patient: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, Any]] = {}
    for patient in patients:
        h5_path = dataset_path / f"{patient}.h5"
        available = _image_count(h5_path)
        n_take = min(spots_per_patient, available)
        indices = np.sort(rng.choice(available, size=n_take, replace=False)).astype(np.int64)
        selected[patient] = indices
        summary[patient] = {
            "available_spots": available,
            "selected_spots": n_take,
            "patch_file": str(h5_path),
            "adata_file": str(dataset_path / f"{patient}.h5ad"),
        }
    return selected, summary


def _uni_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=UNI_MEAN, std=UNI_STD),
        ]
    )


def _cache_key(
    patient: str,
    indices: np.ndarray,
    genes: list[str],
    params: dict[str, Any],
) -> str:
    checkpoint = Path(params["uni_checkpoint"])
    payload = {
        "patient": patient,
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "genes": genes,
        "log_norm": bool(params.get("log_norm", True)),
        "scale_factor": float(params.get("scale_factor", 1_000_000)),
        "uni_checkpoint": str(checkpoint.resolve()),
        "uni_checkpoint_size": checkpoint.stat().st_size,
        "preprocessing": {"size": 224, "mean": UNI_MEAN, "std": UNI_STD},
        "embedding_dtype": "float32",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_uni_backbone(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint}")
    # The local UNI class is already validated by the existing UNI baseline.
    model = UNI(num_genes=1, assets_dir=str(checkpoint.parent.parent), freeze_backbone=True)
    backbone = model.backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


@torch.inference_mode()
def _build_patient_cache(
    patient: str,
    indices: np.ndarray,
    genes: list[str],
    params: dict[str, Any],
    uni_backbone: torch.nn.Module,
    device: torch.device,
    cache_path: Path,
    logger: logging.Logger,
) -> None:
    dataset_path = Path(params["dataset_path"])
    dataset = XeniumPatientDataset(
        gene_path=str(dataset_path / f"{patient}.h5ad"),
        img_path=str(dataset_path / f"{patient}.h5"),
        indices=indices,
        gene_names=genes,
        transform=_uni_transform(),
        log_norm=bool(params.get("log_norm", True)),
        scale_factor=float(params.get("scale_factor", 1_000_000)),
    )
    if dataset.num_genes != len(genes):
        raise ValueError(
            f"{patient}: expected {len(genes)} genes but dataset produced {dataset.num_genes}. "
            "Every configured gene must exist in every patient."
        )

    loader = DataLoader(
        dataset,
        batch_size=int(params.get("uni_embedding_batch_size", 64)),
        shuffle=False,
        num_workers=int(params.get("embedding_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    logger.info("Extracting frozen UNI embeddings for %s (%d cells)", patient, len(dataset))
    for images, expression in loader:
        images = images.to(device, non_blocking=True)
        feature = uni_backbone(images)
        embeddings.append(feature.float().cpu())
        targets.append(expression.float().cpu())

    embedding_tensor = torch.cat(embeddings, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    if embedding_tensor.shape != (len(dataset), 1024):
        raise ValueError(f"Unexpected UNI embedding shape for {patient}: {embedding_tensor.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "patient": patient,
            "embeddings": embedding_tensor,
            "targets": target_tensor,
            "indices": torch.from_numpy(dataset.indices.copy()),
            "barcodes": list(dataset.selected_barcodes or []),
            "genes": genes,
            "uni_frozen": True,
            "uni_embedding_dim": 1024,
        },
        temporary_path,
    )
    os.replace(temporary_path, cache_path)
    logger.info("Saved %s", cache_path)


def _prepare_embedding_caches(
    patients: list[str],
    selected: dict[str, np.ndarray],
    genes: list[str],
    params: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Path]:
    cache_root = Path(params.get("embedding_cache_dir", SCRIPT_DIR / "embedding_cache"))
    paths = {
        patient: cache_root
        / params["dataset_name"]
        / f"{patient}_stem_uni_{_cache_key(patient, selected[patient], genes, params)}.pt"
        for patient in patients
    }
    missing = [patient for patient in patients if not paths[patient].is_file()]
    if not missing:
        logger.info("All frozen UNI embedding caches already exist.")
        return paths

    uni = _load_uni_backbone(Path(params["uni_checkpoint"]), device)
    for patient in missing:
        _build_patient_cache(
            patient, selected[patient], genes, params, uni, device, paths[patient], logger
        )
    del uni
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return paths


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
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def _save_model_weights(path: Path, model: torch.nn.Module) -> None:
    """Save a plain state dict using the same convention as main2_xenium_patient.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_path)
    os.replace(temporary_path, path)


def _train_fold(
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
    model = _new_stem(params, n_genes).to(device)
    ema = deepcopy(model).to(device).eval()
    requires_grad(ema, False)
    update_ema(ema, model, decay=0.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get("learning_rate", 1e-4)), weight_decay=0.0
    )
    batch_size = int(params.get("batch_size", 32))
    if "epochs" not in params:
        raise ValueError(
            "epochs must be set explicitly; it controls training duration and is independent "
            "of Stem's 1000 diffusion timesteps."
        )
    epochs = int(params["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    log_every = int(params.get("log_every", 100))
    checkpoint_every = int(params.get("checkpoint_every", 25_000))

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_optimizer_steps = epochs * len(loader)
    logger.info(
        "Fold %d: %d cells, %d trainable Stem parameters, batch-size=%d, "
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
                checkpoint_name = f"{global_step:07d}.pt"
                _save_checkpoint(
                    fold_dir / checkpoint_name,
                    model,
                    ema,
                    optimizer,
                )
                logger.info("Saved checkpoint to %s", fold_dir / checkpoint_name)
    # Match the established patient-wise output contract. Stem does not use
    # epoch-wise validation/early stopping, so its EMA model is the preferred
    # inference checkpoint and is saved as best_model.pth. The final raw model
    # weights are retained as epoch_model.pth.
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
def _sample_fold(
    model: torch.nn.Module,
    dataset: Dataset,
    params: dict[str, Any],
    n_genes: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sampling_steps = int(params.get("num_sampling_steps", 1_000))
    samples_per_cell = int(params.get("num_samples_per_condition", 20))
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
            num_samples_per_condition=samples_per_cell,
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
        "samples_per_condition": samples_per_cell,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }


def _relative_variation_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_variance = np.var(y_true, axis=0)
    predicted_variance = np.var(y_pred, axis=0)
    valid = true_variance > 1e-12
    if not np.any(valid):
        return 0.0
    return float(
        np.mean(((predicted_variance[valid] - true_variance[valid]) ** 2) / true_variance[valid] ** 2)
    )


def run(params: dict[str, Any], config_path: str | None = None) -> Path:
    if params.get("model") not in {"StemUNI", "Stem-UNI"}:
        raise ValueError("Stem runner requires model=StemUNI or model=Stem-UNI")
    if int(params.get("augmentation_ratio", 0)) != 0:
        raise ValueError(
            "This runner implements the published UNI-only/no-augmentation Stem ablation; "
            "augmentation_ratio must be 0."
        )
    seed = int(params.get("seed", 123))
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(params.get("require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("Stem training requires a CUDA GPU, but CUDA is unavailable.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Keep the same directory naming convention as main2_xenium_patient.py so
    # the later evaluation workflow can discover the run in the usual way.
    result_dir = Path(params["result_path"]) / f"result_vit_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = _setup_logger(result_dir / "results_log.txt")
    logger.info("Starting Stem-UNI on %s", device)
    logger.info("Official Stem source: %s", STEM_SOURCE_ROOT)
    logger.info("UNI checkpoint: %s (frozen)", params["uni_checkpoint"])
    with (result_dir / "experiment_info.json").open("w") as handle:
        json.dump(params, handle, indent=2)
    metrics_csv_path = result_dir / "metrics_result.csv"
    with metrics_csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS).writeheader()

    dataset_path = Path(params["dataset_path"])
    patients = list(params.get("patient_ids") or _discover_patient_ids(dataset_path))
    genes = _load_gene_names(params)
    selected, subset_summary = _fixed_patient_subsets(
        dataset_path,
        patients,
        int(params.get("spots_per_patient", 30_000)),
        int(params.get("subset_seed", seed)),
    )
    np.savez_compressed(result_dir / "patient_fixed_spot_indices.npz", **selected)
    with (result_dir / "patient_fixed_spot_summary.json").open("w") as handle:
        json.dump(subset_summary, handle, indent=2)

    cache_paths = _prepare_embedding_caches(
        patients, selected, genes, params, device, logger
    )
    n_splits = int(params.get("k_folds", len(patients)))
    splitter = KFold(
        n_splits=n_splits,
        shuffle=bool(params.get("patient_shuffle", False)),
        random_state=int(params.get("fold_seed", 42))
        if bool(params.get("patient_shuffle", False))
        else None,
    )
    folds_to_run = {int(fold) for fold in params.get("folds_to_run", range(1, n_splits + 1))}
    metric_rows: list[dict[str, Any]] = []
    stem_summary_rows: list[dict[str, Any]] = []

    for fold, (train_ids, val_ids) in enumerate(splitter.split(patients), start=1):
        if fold not in folds_to_run:
            continue
        fold_seed = seed + fold
        _set_seed(fold_seed)
        train_patients = [patients[i] for i in train_ids]
        val_patients = [patients[i] for i in val_ids]
        fold_dir = result_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Fold %d/%d train=%s validation=%s", fold, n_splits, train_patients, val_patients)
        train_dataset = ConcatDataset([StemCacheDataset(cache_paths[p]) for p in train_patients])
        val_dataset = ConcatDataset([StemCacheDataset(cache_paths[p]) for p in val_patients])

        ema, train_summary = _train_fold(
            train_dataset,
            params,
            len(genes),
            device,
            fold_dir,
            fold,
            train_patients,
            val_patients,
            logger,
        )
        y_true, y_pred, sampling_summary = _sample_fold(
            ema, val_dataset, params, len(genes), device, logger
        )
        metrics = compute_metrics(y_true, y_pred, genes=genes)
        validation_statistics = _batchwise_validation_statistics(
            y_true,
            y_pred,
            batch_size=int(params.get("batch_size", 32)),
            device=device,
        )
        metric_row = {
            "fold": fold,
            "epoch": train_summary["train_epochs"],
            "phase": "val",
            **validation_statistics,
            **metrics,
        }
        metric_rows.append(_jsonable(metric_row))
        csv_metric_row = {
            **metric_row,
            "custom_loss": f"{metric_row['custom_loss']:.4f}",
            "pearson_mean_cell_wise": f"{metric_row['pearson_mean_cell_wise']:.4f}",
            "spearman_mean_cell_wise": f"{metric_row['spearman_mean_cell_wise']:.4f}",
        }
        with metrics_csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS).writerow(csv_metric_row)

        stem_summary = {
            "fold": fold,
            "train_patients": ",".join(train_patients),
            "val_patients": ",".join(val_patients),
            "n_train_cells": len(train_dataset),
            "n_val_cells": len(val_dataset),
            **train_summary,
            **sampling_summary,
            "relative_variation_distance": _relative_variation_distance(y_true, y_pred),
        }
        stem_summary_rows.append(_jsonable(stem_summary))
        pd.DataFrame(stem_summary_rows).to_csv(result_dir / "stem_fold_summary.csv", index=False)
        with (fold_dir / "metrics.json").open("w") as handle:
            json.dump(
                _jsonable({"metrics": metric_row, "stem_summary": stem_summary}),
                handle,
                indent=2,
            )
        if bool(params.get("save_predictions", True)):
            np.savez_compressed(fold_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)
        logger.info("Fold %d validation metrics: %s", fold, metric_row)

        del ema, train_dataset, val_dataset, y_true, y_pred
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if not metric_rows:
        raise ValueError("No folds were run; check folds_to_run")
    numeric = pd.DataFrame(metric_rows).select_dtypes(include=[np.number])
    aggregate = {
        "mean": _jsonable(numeric.mean().to_dict()),
        "std": _jsonable(numeric.std(ddof=0).to_dict()),
        "completed_folds": sorted(folds_to_run),
    }
    with (result_dir / "aggregate_metrics.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)
    with (result_dir / "folds_summary.txt").open("w") as handle:
        handle.write(
            f"Completed {len(metric_rows)} patient-wise folds. Models saved in {result_dir}\n"
        )
        handle.write(f"Patients: {patients}\n")
        handle.write(
            f"Requested spots_per_patient: {int(params.get('spots_per_patient', 30_000))}\n"
        )
    logger.info("Stem-UNI completed. Results: %s", result_dir)
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-wise Xenium Stem-UNI training")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    args = parser.parse_args()
    with open(args.config) as handle:
        params = json.load(handle)
    run(params, config_path=args.config)


if __name__ == "__main__":
    main()
