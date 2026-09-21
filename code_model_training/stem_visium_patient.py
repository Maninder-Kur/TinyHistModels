#!/usr/bin/env python3
"""Patient-wise Stem-UNI training for spot-level Visium datasets.

This entry point preserves the patient folds and preprocessing used by main2.py,
while using Stem's official diffusion objective, AdamW optimizer, and EMA update.
Frozen UNI embeddings are extracted once per slide and cached before training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "utils"))

from dataPrep import PatchDataset  # noqa: E402
from stem_training import (  # noqa: E402
    METRICS_CSV_FIELDS,
    StemCacheDataset,
    batchwise_validation_statistics,
    jsonable,
    load_gene_names,
    load_uni_backbone,
    relative_variation_distance,
    sample_fold,
    set_seed,
    setup_logger,
    train_fold,
    uni_transform,
)
from compute_metrics import compute_metrics  # noqa: E402


DATASET_GENE_COUNTS = {
    "her2": 785,
    "hist2st_845": 845,
}


def _num_genes_for_dataset(dataset_name: str) -> int:
    try:
        return DATASET_GENE_COUNTS[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset_name: {dataset_name}") from error


def _patient_id(sample: str, dataset_name: str) -> str:
    if dataset_name == "her2":
        return sample[0]
    if dataset_name == "hist2st_845":
        return sample.split("_", maxsplit=1)[0]
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def _discover_sample_ids(dataset_path: Path) -> list[str]:
    h5_ids = {path.stem for path in dataset_path.glob("*.h5")}
    h5ad_ids = {path.stem for path in dataset_path.glob("*.h5ad")}
    samples = sorted(h5_ids & h5ad_ids)
    if not samples:
        raise FileNotFoundError(f"No paired .h5/.h5ad slides found in {dataset_path}")
    return samples


def _sample_cache_key(sample: str, genes: list[str], params: dict[str, Any]) -> str:
    dataset_path = Path(params["dataset_path"])
    checkpoint = Path(params["uni_checkpoint"])
    patch_path = dataset_path / f"{sample}.h5"
    adata_path = dataset_path / f"{sample}.h5ad"
    payload = {
        "sample": sample,
        "genes": genes,
        "log_norm": bool(params.get("log_norm", True)),
        "scale_factor": float(params.get("scale_factor", 1_000_000)),
        "spots_per_slide": int(params.get("spots_per_slide", 30_000)),
        "patch_path": str(patch_path.resolve()),
        "patch_size": patch_path.stat().st_size,
        "adata_path": str(adata_path.resolve()),
        "adata_size": adata_path.stat().st_size,
        "uni_checkpoint": str(checkpoint.resolve()),
        "uni_checkpoint_size": checkpoint.stat().st_size,
        "preprocessing": {
            "size": 224,
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        "embedding_dtype": "float32",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@torch.inference_mode()
def _build_slide_cache(
    sample: str,
    genes: list[str],
    params: dict[str, Any],
    uni_backbone: torch.nn.Module,
    device: torch.device,
    cache_path: Path,
    logger,
) -> None:
    dataset_path = Path(params["dataset_path"])
    dataset = PatchDataset(
        gene_path=str(dataset_path / f"{sample}.h5ad"),
        img_path=str(dataset_path / f"{sample}.h5"),
        gene_names=genes,
        transform=uni_transform(),
        log_norm=bool(params.get("log_norm", True)),
        scale_factor=float(params.get("scale_factor", 1_000_000)),
    )
    if dataset.gene_data.shape[1] != len(genes):
        raise ValueError(
            f"{sample}: expected {len(genes)} genes but PatchDataset produced "
            f"{dataset.gene_data.shape[1]}. Every selected gene must exist in every slide."
        )

    max_spots = int(params.get("spots_per_slide", len(dataset)))
    n_take = min(max_spots, len(dataset))
    indices = np.arange(n_take, dtype=np.int64)
    selected_dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        selected_dataset,
        batch_size=int(params.get("uni_embedding_batch_size", 64)),
        shuffle=False,
        num_workers=int(params.get("embedding_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    logger.info("Extracting frozen UNI embeddings for %s (%d spots)", sample, n_take)
    for images, expression in loader:
        feature = uni_backbone(images.to(device, non_blocking=True))
        embeddings.append(feature.float().cpu())
        targets.append(expression.float().cpu())

    embedding_tensor = torch.cat(embeddings, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    if embedding_tensor.shape != (n_take, 1024):
        raise ValueError(f"Unexpected UNI embedding shape for {sample}: {embedding_tensor.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "patient": sample,
            "embeddings": embedding_tensor,
            "targets": target_tensor,
            "indices": torch.from_numpy(indices),
            "barcodes": [],
            "genes": genes,
            "uni_frozen": True,
            "uni_embedding_dim": 1024,
        },
        temporary_path,
    )
    os.replace(temporary_path, cache_path)
    logger.info("Saved %s", cache_path)
    del dataset, selected_dataset, loader, embeddings, targets
    gc.collect()


def _prepare_slide_caches(
    samples: list[str],
    genes: list[str],
    params: dict[str, Any],
    device: torch.device,
    logger,
) -> dict[str, Path]:
    cache_root = Path(params.get("embedding_cache_dir", SCRIPT_DIR / "embedding_cache"))
    paths = {
        sample: cache_root
        / params["dataset_name"]
        / f"{sample}_stem_uni_{_sample_cache_key(sample, genes, params)}.pt"
        for sample in samples
    }
    missing = [sample for sample in samples if not paths[sample].is_file()]
    if not missing:
        logger.info("All frozen UNI slide embedding caches already exist.")
        return paths

    uni_backbone = load_uni_backbone(Path(params["uni_checkpoint"]), device)
    for sample in missing:
        _build_slide_cache(
            sample, genes, params, uni_backbone, device, paths[sample], logger
        )
    del uni_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return paths


def _resolve_patient_folds(
    samples: list[str], params: dict[str, Any]
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    dataset_name = params["dataset_name"]
    patient_for_sample = {
        sample: _patient_id(sample, dataset_name) for sample in samples
    }
    all_patients = sorted(set(patient_for_sample.values()))
    k_folds = int(params.get("k_folds", 5))
    if not 2 <= k_folds <= len(all_patients):
        raise ValueError(
            f"k_folds must be between 2 and {len(all_patients)}, got {k_folds}"
        )

    shuffled_patients = np.asarray(all_patients, dtype=object)
    if bool(params.get("patient_shuffle", True)):
        fold_rng = np.random.RandomState(int(params.get("fold_seed", 42)))
        fold_rng.shuffle(shuffled_patients)
    validation_folds = np.array_split(shuffled_patients, k_folds)

    folds = []
    for validation_fold in validation_folds:
        validation_patients = [str(patient) for patient in validation_fold.tolist()]
        train_patients = [p for p in all_patients if p not in validation_patients]
        train_samples = [s for s in samples if patient_for_sample[s] in train_patients]
        validation_samples = [s for s in samples if patient_for_sample[s] in validation_patients]
        folds.append((train_samples, validation_samples, train_patients, validation_patients))
    return folds


def run(params: dict[str, Any]) -> Path:
    if params.get("model") not in {"StemUNI", "Stem-UNI"}:
        raise ValueError("Stem Visium runner requires model=StemUNI or model=Stem-UNI")
    if params.get("dataset_name") not in DATASET_GENE_COUNTS:
        raise ValueError(
            "This runner supports dataset_name=her2 or dataset_name=hist2st_845"
        )
    if int(params.get("augmentation_ratio", 0)) != 0:
        raise ValueError("The UNI-only Stem run requires augmentation_ratio=0")

    seed = int(params.get("seed", 123))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(params.get("require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("Stem training requires a CUDA GPU, but CUDA is unavailable")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = Path(params["result_path"]) / f"result_vit_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(result_dir / "results_log.txt")
    logger.info(
        "Starting Stem-UNI %s training on %s", params["dataset_name"], device
    )
    logger.info("UNI checkpoint: %s (frozen)", params["uni_checkpoint"])
    with (result_dir / "experiment_info.json").open("w") as handle:
        json.dump(params, handle, indent=2)
    metrics_csv_path = result_dir / "metrics_result.csv"
    with metrics_csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS).writeheader()

    dataset_path = Path(params["dataset_path"])
    samples = list(params.get("sample_ids") or _discover_sample_ids(dataset_path))
    if len(samples) != len(set(samples)):
        raise ValueError("sample_ids contains duplicates; each Visium slide must appear once")
    missing_pairs = [
        sample
        for sample in samples
        if not (dataset_path / f"{sample}.h5").is_file()
        or not (dataset_path / f"{sample}.h5ad").is_file()
    ]
    if missing_pairs:
        raise FileNotFoundError(f"Missing H5/H5AD pairs for: {missing_pairs}")

    num_genes = _num_genes_for_dataset(params["dataset_name"])
    genes = load_gene_names(params)
    if len(genes) != num_genes:
        raise ValueError(
            f"Dataset {params['dataset_name']} requires {num_genes} genes, "
            f"but the loaded gene list contains {len(genes)}"
        )
    folds = _resolve_patient_folds(samples, params)
    caches = _prepare_slide_caches(samples, genes, params, device, logger)
    cache_summary = {
        sample: {"cache": str(caches[sample]), "spots": len(StemCacheDataset(caches[sample]))}
        for sample in samples
    }
    with (result_dir / "slide_cache_summary.json").open("w") as handle:
        json.dump(cache_summary, handle, indent=2)

    metric_rows: list[dict[str, Any]] = []
    stem_summary_rows: list[dict[str, Any]] = []
    for fold, (train_samples, val_samples, train_patients, val_patients) in enumerate(
        folds, start=1
    ):
        set_seed(seed + fold)
        fold_dir = result_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info("===== Starting Fold %d/%d =====", fold, len(folds))
        logger.info("Train patients: %s; slides: %s", train_patients, train_samples)
        logger.info("Val patients: %s; slides: %s", val_patients, val_samples)

        train_dataset = ConcatDataset([StemCacheDataset(caches[s]) for s in train_samples])
        val_dataset = ConcatDataset([StemCacheDataset(caches[s]) for s in val_samples])
        logger.info("Fold %d: Train=%d, Val=%d", fold, len(train_dataset), len(val_dataset))

        ema, train_summary = train_fold(
            train_dataset,
            params,
            num_genes,
            device,
            fold_dir,
            fold,
            train_patients,
            val_patients,
            logger,
        )
        y_true, y_pred, sampling_summary = sample_fold(
            ema, val_dataset, params, num_genes, device, logger
        )
        metrics = compute_metrics(y_true, y_pred, genes=genes)
        validation_statistics = batchwise_validation_statistics(
            y_true, y_pred, batch_size=int(params.get("batch_size", 128)), device=device
        )
        metric_row = {
            "fold": fold,
            "epoch": train_summary["train_epochs"],
            "phase": "val",
            **validation_statistics,
            **metrics,
        }
        metric_rows.append(jsonable(metric_row))
        csv_row = {
            **metric_row,
            "custom_loss": f"{metric_row['custom_loss']:.4f}",
            "pearson_mean_cell_wise": f"{metric_row['pearson_mean_cell_wise']:.4f}",
            "spearman_mean_cell_wise": f"{metric_row['spearman_mean_cell_wise']:.4f}",
        }
        with metrics_csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS).writerow(csv_row)

        stem_summary = {
            "fold": fold,
            "train_patients": ",".join(train_patients),
            "val_patients": ",".join(val_patients),
            "train_slides": ",".join(train_samples),
            "val_slides": ",".join(val_samples),
            "n_train_spots": len(train_dataset),
            "n_val_spots": len(val_dataset),
            **train_summary,
            **sampling_summary,
            "relative_variation_distance": relative_variation_distance(y_true, y_pred),
        }
        stem_summary_rows.append(jsonable(stem_summary))
        pd.DataFrame(stem_summary_rows).to_csv(
            result_dir / "stem_fold_summary.csv", index=False
        )
        with (fold_dir / "metrics.json").open("w") as handle:
            json.dump(
                jsonable({"metrics": metric_row, "stem_summary": stem_summary}),
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

    numeric = pd.DataFrame(metric_rows).select_dtypes(include=[np.number])
    aggregate = {
        "mean": jsonable(numeric.mean().to_dict()),
        "std": jsonable(numeric.std(ddof=0).to_dict()),
        "completed_folds": list(range(1, len(folds) + 1)),
    }
    with (result_dir / "aggregate_metrics.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)
    with (result_dir / "folds_summary.txt").open("w") as handle:
        handle.write(
            f"Completed {len(folds)} {params['dataset_name']} patient-wise folds.\n"
        )
        handle.write(f"Samples: {samples}\n")
        handle.write(f"Validation patient folds: {[fold[3] for fold in folds]}\n")
        handle.write(
            f"Patient shuffle: {bool(params.get('patient_shuffle', True))}; "
            f"fold seed: {int(params.get('fold_seed', 42))}\n"
        )
    logger.info("Stem-UNI %s completed. Results: %s", params["dataset_name"], result_dir)
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-wise Visium Stem-UNI training")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    args = parser.parse_args()
    with open(args.config) as handle:
        params = json.load(handle)
    run(params)


if __name__ == "__main__":
    main()
