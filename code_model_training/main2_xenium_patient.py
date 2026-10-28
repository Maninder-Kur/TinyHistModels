import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (SCRIPT_DIR, SCRIPT_DIR / "models", SCRIPT_DIR / "utils"):
    sys.path.insert(0, str(source_dir))
import warnings

warnings.filterwarnings("ignore")
import gc
import csv
import h5py
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from datetime import datetime
import argparse
import json
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from utils.dataPrep_xenium import (
    PatchDataset,
    FoldDataset,
    log1p_normalization,
    XeniumPatientDataset,
    SpatialXeniumPatientDataset,
    spatial_collate_fn,
)
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging
from set_deterministic_seed import set_deterministic_seed
from models import STPathZeroModel
from models import (
    STNet,
    EfficientNet,
    Custom_VGG16,
    HisToGene,
    TCGN,
    UNI,
    GigaPathFlash,
    STPathOriginalModel,
    STPathSmallModel,
    STPathSmallScratchModel,
)
from model_eff_net_versions import EfficientNetTinyStudent


def main():
    (g, seed_worker) = set_deterministic_seed(123)
    parser = argparse.ArgumentParser(
        description="Script that uses a JSON configuration file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    try:
        with open(args.config, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file at {args.config}")
        return
    model_name = str(params["model"]).strip().lower()
    is_stpath = model_name in {
        "stpath",
        "stpathorg",
        "stpathsmall",
        "stpathtiny",
        "stpathsmallscratch",
        "stpathscratch",
    }
    gene_names = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join(params["result_path"], f"result_vit_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    json_file = os.path.join(model_save_dir, f"experiment_info.json")
    with open(json_file, "w") as exp:
        json.dump(params, exp, indent=4)
    log_file = os.path.join(model_save_dir, "results_log.txt")
    logger = setup_logging(log_file)
    logger.info("Logging setup complete.")
    logger.info(f"Experiment information saved to the path: {json_file}")
    metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")
    with open(metrics_csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
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
            ],
        )
        writer.writeheader()

    def train_regression(
        train_datasets,
        val_datasets,
        fold_idx,
        lambda_reg=params.get("lambda_reg", 0.0),
        learning_rate=params["learning_rate"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
    ):
        models = {}
        logger.info(f"train_datasets length:  {len(train_datasets)}")
        loader_batch_size = (
            int(params.get("stpath_window_batch_size", 1)) if is_stpath else batch_size
        )
        collate_fn = spatial_collate_fn if is_stpath else None
        train_dataloader = DataLoader(
            train_datasets,
            batch_size=loader_batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_datasets,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn,
        )
        logger.info(f"Number of train batches: {len(train_dataloader)}")
        logger.info("Initializing model...")
        if params.get("num_genes") is not None:
            num_genes = int(params["num_genes"])
        elif params["dataset_name"] == "cscc":
            num_genes = 171
        elif params["dataset_name"] == "her2":
            num_genes = 785
        elif params["dataset_name"] == "hist2st_845":
            num_genes = 845
        elif params["dataset_name"] == "10X_HumandBreastCancer":
            num_genes = 541
        elif params["dataset_name"] == "xenium_breast_patient":
            num_genes = 500
        elif params["dataset_name"] == "xenium_lung_patient":
            num_genes = 541
        elif params["dataset_name"] == "10X_HumanLungCancer":
            num_genes = 480
        elif params["dataset_name"] == "10X_HumanPancreasCancer":
            num_genes = 377
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")
        if params["model"] == "STNet":
            pretrained = params.get("pretrained", False)
            model = STNet(num_genes=num_genes, pretrained=pretrained)
            logger.info(
                f"STNet initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "VGG":
            pretrained = params.get("pretrained", True)
            model = Custom_VGG16(num_genes=num_genes, pretrained=pretrained)
            logger.info(
                f"VGG initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "Effnet":
            pretrained = params.get("pretrained", False)
            model = EfficientNet(num_genes=num_genes, pretrained=pretrained)
            logger.info(
                f"EFFnet initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "HisToGene":
            model = HisToGene(patch_size=16, n_layers=8, n_genes=num_genes)
        elif params["model"] == "EfficientNetTinyStudent":
            logger.info(
                "EfficientNetTinyStudent (tiny) for gene expression prediction."
            )
            phi = params.get("phi", -7.0)
            model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)
            logger.info(
                f"EfficientNetTinyStudent initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "TCGN":
            logger.info("Initializing TCGN model...")
            model = TCGN(
                num_classes=num_genes,
                embed_dims=params.get("embed_dims", [46, 92, 184, 368]),
            )
            logger.info(
                f"TCGN initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif model_name in {"gigapathflash", "gigapath_flash"}:
            model = GigaPathFlash(
                num_genes=num_genes,
                assets_dir=params["assets_dir"],
                checkpoint=params.get("gigapath_flash_ckpt", "pytorch_model.bin"),
                freeze_backbone=params.get("freeze_backbone", False),
            )
            logger.info(
                f"GigaPath Flash initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "UNI":
            model = UNI(
                num_genes=num_genes,
                assets_dir=params["assets_dir"],
                freeze_backbone=params.get("freeze_backbone", False),
            )
            logger.info(
                f"UNI initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif model_name in {"stpathsmallscratch", "stpathscratch"}:
            model = STPathSmallScratchModel(
                panel_gene_names=gene_names,
                assets_dir=params["assets_dir"],
                gigapath_ckpt=params["gigapath_ckpt"],
                freeze_gigapath=params["freeze_gigapath"],
                organ_type=params["organ_type"],
                tech_type=params["tech_type"],
                head_hidden_dim=params.get("stpath_small_head_dim", 1536),
                dropout=params.get("stpath_small_dropout", 0.1),
                stpath_small_d_model=params.get("stpath_small_d_model", 512),
                stpath_small_layers=params.get("stpath_small_layers", 1),
                stpath_small_heads=params.get("stpath_small_heads", 4),
                stpath_small_mlp_ratio=params.get("stpath_small_mlp_ratio", 2.0),
            )
            logger.info(
                f"STPathSmallScratch initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif model_name in {"stpathsmall", "stpathtiny"}:
            model = STPathSmallModel(
                panel_gene_names=gene_names,
                assets_dir=params["assets_dir"],
                stpath_ckpt=params["stpath_ckpt"],
                gigapath_ckpt=params["gigapath_ckpt"],
                gene_voc_path=params["gene_voc_path"],
                freeze_gigapath=params["freeze_gigapath"],
                organ_type=params["organ_type"],
                tech_type=params["tech_type"],
                head_hidden_dim=params.get("stpath_small_head_dim", 1536),
                dropout=params.get("stpath_small_dropout", 0.1),
                stpath_small_d_model=params.get("stpath_small_d_model", 512),
                stpath_small_layers=params.get("stpath_small_layers", 1),
                stpath_small_heads=params.get("stpath_small_heads", 4),
                stpath_small_mlp_ratio=params.get("stpath_small_mlp_ratio", 2.0),
            )
            logger.info(
                f"STPathSmall initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        elif params["model"] == "STPathZero":
            model = STPathZeroModel(
                num_genes=num_genes,
                assets_dir=params["assets_dir"],
                stpath_ckpt=params.get("stpath_ckpt", "stfm.pth"),
                gigapath_ckpt=params.get("gigapath_ckpt", "pytorch_model.bin"),
                gene_voc_path=params.get("gene_voc_path"),
                freeze_gigapath=params.get("freeze_gigapath", True),
                hidden_dim=params.get("hidden_dim", 256),
                dropout=params.get("dropout", 0.1),
                organ_type=params.get("organ_type"),
                tech_type=params.get("tech_type", "Visium"),
            )
        elif model_name in {"stpath", "stpathorg"}:
            model = STPathOriginalModel(
                panel_gene_names=gene_names,
                assets_dir=params["assets_dir"],
                stpath_ckpt=params["stpath_ckpt"],
                gigapath_ckpt=params["gigapath_ckpt"],
                gene_voc_path=params["gene_voc_path"],
                freeze_gigapath=params["freeze_gigapath"],
                organ_type=params["organ_type"],
                tech_type=params["tech_type"],
            )
            logger.info(
                f"STPath initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
            )
        else:
            logger.error("No model found with name: %s", params["model"])
            raise ValueError("No model found")
        logger.info("Model\n%s", model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)
        if params.get("optimizer", "Adam").lower() == "adam":
            optimizer = Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif params.get("optimizer", "sgd").lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            logger.error("Optimizer not defined!!! Defaulting to Adam")
            optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params.get("lr_patience", 5),
            factor=params.get("lr_factor", 0.5),
            min_lr=params.get("min_lr", 1e-07),
        )
        train_losses = []
        val_losses = []
        best_spearman = -np.inf
        patience_counter = 0
        patience = params.get("early_stopping_patience", 15)
        mse_criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            all_y_true = []
            all_y_pred = []
            pearson_train_corr = []
            spearman_train_corr = []
            for (batch_idx, batch) in enumerate(train_dataloader):
                if is_stpath:
                    (images, coords, targets, spatial_batch_idx) = batch
                    coords = coords.to(device)
                    spatial_batch_idx = spatial_batch_idx.to(device)
                else:
                    (images, targets) = batch
                images = images.to(device)
                y_true = targets.to(device)
                if not torch.isfinite(images).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in input images. Skipping batch."
                    )
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch."
                    )
                    continue
                if is_stpath:
                    y_pred = model(
                        images=images, coords=coords, batch_idx=spatial_batch_idx
                    )
                else:
                    y_pred = model(images)
                y_pred = torch.where(
                    y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred
                )
                if not torch.isfinite(y_pred).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in model output. Zeroing out predictions."
                    )
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"image : {images.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}"
                )
                if batch_idx == 0:
                    print(
                        "\n===================== DEBUG: Epoch {} =====================".format(
                            epoch + 1
                        )
                    )
                    print("Sample y_true values (first sample, first 10 genes):")
                    print(y_true[0, :10].detach().cpu().numpy())
                    print("Sample y_pred values (first sample, first 10 genes):")
                    print(y_pred[0, :10].detach().cpu().numpy())
                    y_true_np = y_true.detach().cpu().numpy()
                    y_pred_np = y_pred.detach().cpu().numpy()
                    print(
                        f"y_true  -> mean={np.nanmean(y_true_np):.4f}, std={np.nanstd(y_true_np):.4f}, min={np.nanmin(y_true_np):.4f}, max={np.nanmax(y_true_np):.4f}"
                    )
                    print(
                        f"y_pred  -> mean={np.nanmean(y_pred_np):.4f}, std={np.nanstd(y_pred_np):.4f}, min={np.nanmin(y_pred_np):.4f}, max={np.nanmax(y_pred_np):.4f}"
                    )
                loss_fn_name = params.get("loss_fn", "")
                if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                    mse_criterion = torch.nn.MSELoss()
                    mse_loss = mse_criterion(y_pred, y_true)
                    loss = mse_loss
                else:
                    logger.error("Loss Function not defined!!!")
                    raise ValueError("Loss Function not defined")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += float(loss.item())
                all_y_true.append(y_true.detach().cpu().numpy())
                all_y_pred.append(y_pred.detach().cpu().numpy())
                try:
                    batch_pearson = pearsonr(
                        y_pred.detach().cpu().numpy().flatten(),
                        y_true.detach().cpu().numpy().flatten(),
                    )[0]
                    print(f"Batch {batch_idx} Pearson correlation: {batch_pearson:.4f}")
                except Exception as e:
                    print(
                        "Exception+++++++++++++++++++++++++++++++++++++++++++++ ", {e}
                    )
                    batch_pearson = 0.0
                pearson_train_corr.append(float(batch_pearson.mean().item()))
                try:
                    batch_spearman = spearmanrr(y_pred, y_true)
                    spearman_train_corr.append(batch_spearman.mean().item())
                except Exception as e:
                    print(f"code is in Exception++++++++++++++++++++++++++++++++ {e}")
                    spearman_train_corr.append(0.0)
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Fold {fold_idx + 1} Train Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_train_corr[-1]:.4f}, Spearman: {spearman_train_corr[-1]:.4f}"
                    )
            all_y_true = np.vstack(all_y_true) if len(all_y_true) > 0 else np.array([])
            all_y_pred = np.vstack(all_y_pred) if len(all_y_pred) > 0 else np.array([])
            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")
            print(
                f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}"
            )
            results_train = (
                compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}
            )
            del loss, y_pred, y_true, images, all_y_true, all_y_pred
            torch.cuda.empty_cache()
            gc.collect()
            avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
            logger.info(
                f"Fold {fold_idx + 1} Train Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Pearson Mean: {np.mean(pearson_train_corr):.4f}, Spearman Mean: {np.mean(spearman_train_corr):.4f}"
            )
            if results_train:
                logger.info(f"Training Metrics: {results_train}")
            with open(metrics_csv_path, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
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
                    ],
                )
                row = {
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "phase": "train",
                    "custom_loss": f"{avg_train_loss:.4f}",
                    "pearson_mean_cell_wise": f"{np.mean(pearson_train_corr):.4f}",
                    "spearman_mean_cell_wise": f"{np.mean(spearman_train_corr):.4f}",
                }
                if results_train:
                    for k in results_train:
                        if k in writer.fieldnames:
                            row[k] = results_train[k]
                writer.writerow(row)
            train_losses.append(avg_train_loss)
            model.eval()
            epoch_val_loss = 0.0
            val_y_true = []
            val_y_pred = []
            spearman_val_corr = []
            pearson_val_corr = []
            with torch.no_grad():
                for (batch_idx, batch) in enumerate(val_dataloader):
                    if is_stpath:
                        (images, coords, targets, spatial_batch_idx) = batch
                        coords = coords.to(device)
                        spatial_batch_idx = spatial_batch_idx.to(device)
                    else:
                        (images, targets) = batch
                    images = images.to(device)
                    y_true = targets.to(device)
                    if is_stpath:
                        y_pred = model(
                            images=images, coords=coords, batch_idx=spatial_batch_idx
                        )
                    else:
                        y_pred = model(images)
                    assert torch.is_tensor(
                        y_pred
                    ), f"y_pred is not tensor, got {type(y_pred)}"
                    y_pred = torch.where(
                        y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred
                    )
                    loss_fn_name = params.get("loss_fn", "")
                    if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                        mse_criterion = torch.nn.MSELoss()
                        loss = mse_criterion(y_pred, y_true)
                    else:
                        logger.error("Loss Function not defined!!!")
                        raise ValueError("Loss Function not defined")
                    epoch_val_loss += float(loss.item())
                    val_y_true.append(y_true.detach().cpu().numpy())
                    val_y_pred.append(y_pred.detach().cpu().numpy())
                    try:
                        batch_pearson = pearsonr(
                            y_pred.detach().cpu().numpy().flatten(),
                            y_true.detach().cpu().numpy().flatten(),
                        )[0]
                    except Exception:
                        batch_pearson = 0.0
                    pearson_val_corr.append(float(batch_pearson.mean().item()))
                    try:
                        batch_spearman = spearmanrr(y_pred, y_true)
                        spearman_val_corr.append(float(batch_spearman.mean().item()))
                    except Exception:
                        spearman_val_corr.append(0.0)
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Fold {fold_idx + 1} Val Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_val_corr[-1]:.4f}, Spearman: {spearman_val_corr[-1]:.4f}"
                        )
                val_y_true = (
                    np.vstack(val_y_true) if len(val_y_true) > 0 else np.array([])
                )
                val_y_pred = (
                    np.vstack(val_y_pred) if len(val_y_pred) > 0 else np.array([])
                )
                if val_y_true.size:
                    n_display = min(25, val_y_true.shape[0])
                    val_y_true_25 = val_y_true[:n_display, :]
                    val_y_pred_25 = val_y_pred[:n_display, :]
                    grid_sz = int(np.ceil(np.sqrt(n_display)))
                    (fig, axes) = plt.subplots(
                        nrows=grid_sz, ncols=grid_sz, figsize=(4 * grid_sz, 4 * grid_sz)
                    )
                    axes = np.atleast_2d(axes)
                    for idx in range(n_display):
                        r = idx // grid_sz
                        c = idx % grid_sz
                        axes[r, c].scatter(
                            val_y_true_25[idx], val_y_pred_25[idx], alpha=0.6
                        )
                        min_val = min(
                            val_y_true_25[idx].min(), val_y_pred_25[idx].min()
                        )
                        max_val = max(
                            val_y_true_25[idx].max(), val_y_pred_25[idx].max()
                        )
                        axes[r, c].plot(
                            [min_val, max_val],
                            [min_val, max_val],
                            linestyle="--",
                            color="red",
                        )
                        axes[r, c].set_title(f"Sample {idx + 1}")
                    plt.tight_layout()
                    scatter_save_path = os.path.join(
                        model_save_dir,
                        f"fold_{fold_idx + 1}",
                        f"scatter_plot_25_samples_epoch_{epoch + 1}.png",
                    )
                    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
                    plt.savefig(scatter_save_path)
                    plt.close(fig)
                results_val = (
                    compute_metrics(val_y_true, val_y_pred) if val_y_true.size else {}
                )
                del loss, y_pred, y_true, images
                torch.cuda.empty_cache()
                gc.collect()
                avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))
                logger.info(
                    f"Fold {fold_idx + 1} Val Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Pearson Mean: {np.mean(pearson_val_corr):.4f}, Spearman Mean: {np.mean(spearman_val_corr):.4f}"
                )
                if results_val:
                    logger.info(f"Validation Metrics: {results_val}")
                with open(metrics_csv_path, mode="a", newline="") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=[
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
                        ],
                    )
                    row = {
                        "fold": fold_idx + 1,
                        "epoch": epoch + 1,
                        "phase": "val",
                        "custom_loss": f"{avg_val_loss:.4f}",
                        "pearson_mean_cell_wise": f"{np.mean(pearson_val_corr):.4f}",
                        "spearman_mean_cell_wise": f"{np.mean(spearman_val_corr):.4f}",
                    }
                    if results_val:
                        for k in results_val:
                            if k in writer.fieldnames:
                                row[k] = results_val[k]
                    writer.writerow(row)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Learning rate for epoch {epoch + 1}: {current_lr}")
                current_spearman = float(
                    results_val.get(
                        "spearman_mean_genewise",
                        np.mean(spearman_val_corr) if spearman_val_corr else -np.inf,
                    )
                )
                if current_spearman > best_spearman:
                    best_spearman = current_spearman
                    patience_counter = 0
                    best_model_path = os.path.join(
                        model_save_dir, f"fold_{fold_idx + 1}", "best_model.pth"
                    )
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1
                    logger.info(
                        f"No improvement in spearman genewise. Patience: {patience_counter}/{patience}"
                    )
                if patience_counter >= patience:
                    logger.info("Early stopping triggered. Breaking training loop.")
                    break
                try:
                    plt.figure(figsize=(8, 5))
                    plt.plot(
                        range(1, len(train_losses) + 1),
                        train_losses,
                        label="Training Loss",
                        marker="o",
                    )
                    plt.plot(
                        range(1, len(val_losses) + 1),
                        val_losses,
                        label="Validation Loss",
                        marker="o",
                    )
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Fold {fold_idx + 1} Loss")
                    plt.legend()
                    plt.grid(True)
                    graph_save_path = os.path.join(
                        model_save_dir, f"fold_{fold_idx + 1}", f"loss_plot_epoch.png"
                    )
                    plt.savefig(graph_save_path)
                    plt.close("all")
                except Exception as e:
                    logger.warning(f"Could not plot losses: {e}")
            snapshot_path = os.path.join(
                model_save_dir, f"fold_{fold_idx + 1}", f"epoch_model.pth"
            )
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            torch.save(model.state_dict(), snapshot_path)
        del model, optimizer, scheduler, train_dataloader, val_dataloader
        torch.cuda.empty_cache()
        gc.collect()
        return None

    def load_gene_names_from_params():
        genes_param = params.get("genes")
        if not genes_param:
            return None
        genes_file = genes_param
        if os.path.isdir(genes_file):
            genes_file = os.path.join(genes_file, f"{params['dataset_name']}.npy")
        logger.info(f"Using gene list from: {genes_file}")
        if not os.path.exists(genes_file):
            logger.warning(f"Gene list not found: {genes_file}")
            return None
        gene_names = np.load(genes_file, allow_pickle=True).tolist()
        logger.info(f"Loaded {len(gene_names)} genes from {genes_file}")
        return gene_names

    def get_image_count(img_path):
        with h5py.File(img_path, "r") as f:
            if "patches" in f:
                return f["patches"].shape[0]
            if "img" in f:
                return f["img"].shape[0]
            raise KeyError(f"No valid image dataset found in {img_path}")

    def discover_patient_ids(dataset_path):
        h5_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(dataset_path)
            if f.endswith(".h5")
        }
        h5ad_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(dataset_path)
            if f.endswith(".h5ad")
        }
        patient_ids = sorted(h5_ids & h5ad_ids)
        if not patient_ids:
            raise FileNotFoundError(
                f"No matching .h5/.h5ad patient pairs found in {dataset_path}"
            )
        return patient_ids

    from torchvision import transforms

    transform_steps = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if is_stpath:
        transform_steps.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    transform = transforms.Compose(transform_steps)
    if model_name in {"gigapathflash", "gigapath_flash"}:
        transform = GigaPathFlash.create_transform()
    if params.get("patient_wise_cv", False):
        dataset_path = params["dataset_path"]
        patient_ids = params.get("patient_ids") or discover_patient_ids(dataset_path)
        k = params.get("k_folds", len(patient_ids))
        if k > len(patient_ids):
            logger.warning(
                "k_folds > number of patients. Setting k = number of patients."
            )
            k = len(patient_ids)
        spots_per_patient = int(params.get("spots_per_patient", 30000))
        subset_seed = int(params.get("subset_seed", params.get("seed", 123)))
        rng = np.random.default_rng(subset_seed)
        gene_names = load_gene_names_from_params()
        if is_stpath and gene_names is None:
            raise FileNotFoundError(
                "STPath Xenium training requires a gene panel .npy file in the 'genes' setting."
            )
        if gene_names is not None:
            logger.info(f"Using provided gene list with {len(gene_names)} genes")
        else:
            logger.info("No gene list provided; using all genes from the dataset")
        patch_paths = {
            patient: os.path.join(dataset_path, f"{patient}.h5")
            for patient in patient_ids
        }
        adata_paths = {
            patient: os.path.join(dataset_path, f"{patient}.h5ad")
            for patient in patient_ids
        }
        selected_indices = {}
        subset_summary = {}
        for patient in patient_ids:
            n_spots = get_image_count(patch_paths[patient])
            n_take = min(spots_per_patient, n_spots)
            indices = np.sort(rng.choice(n_spots, size=n_take, replace=False)).astype(
                np.int64
            )
            selected_indices[patient] = indices
            subset_summary[patient] = {
                "available_spots": int(n_spots),
                "selected_spots": int(n_take),
                "patch_file": patch_paths[patient],
                "adata_file": adata_paths[patient],
            }
            if n_take < spots_per_patient:
                logger.warning(
                    f"{patient}: requested {spots_per_patient} spots, but only {n_spots} are available. Using {n_take}."
                )
            else:
                logger.info(
                    f"{patient}: selected fixed subset of {n_take}/{n_spots} spots."
                )
        np.savez_compressed(
            os.path.join(model_save_dir, "patient_fixed_spot_indices.npz"),
            **selected_indices,
        )
        with open(
            os.path.join(model_save_dir, "patient_fixed_spot_summary.json"), "w"
        ) as sf:
            json.dump(subset_summary, sf, indent=4)
        logger.info(f"Total patients: {len(patient_ids)}")
        logger.info(f"Patient order: {patient_ids}")
        patient_shuffle = bool(params.get("patient_shuffle", False))
        if patient_shuffle:
            kf = KFold(
                n_splits=k, shuffle=True, random_state=int(params.get("fold_seed", 42))
            )
        else:
            kf = KFold(n_splits=k, shuffle=False)
        for (fold_idx, (train_patient_idx, val_patient_idx)) in enumerate(
            kf.split(patient_ids)
        ):
            train_patients = [patient_ids[i] for i in train_patient_idx]
            val_patients = [patient_ids[i] for i in val_patient_idx]
            logger.info(f"\n===== Starting Fold {fold_idx + 1}/{k} =====")
            logger.info(f"Train patients: {train_patients}")
            logger.info(f"Val patients: {val_patients}")
            if is_stpath:
                spatial_kwargs = dict(
                    gene_names=gene_names,
                    transform=transform,
                    log_norm=params.get("log_norm", True),
                    scale_factor=params.get("scale_factor", 1000000),
                    window_size=params.get("stpath_window_size", 256),
                    window_stride=params.get("stpath_window_stride", 256),
                )
                train_datasets_list = [
                    SpatialXeniumPatientDataset(
                        gene_path=adata_paths[patient],
                        img_path=patch_paths[patient],
                        spot_indices=selected_indices[patient],
                        **spatial_kwargs,
                    )
                    for patient in train_patients
                ]
                val_datasets_list = [
                    SpatialXeniumPatientDataset(
                        gene_path=adata_paths[patient],
                        img_path=patch_paths[patient],
                        spot_indices=selected_indices[patient],
                        **spatial_kwargs,
                    )
                    for patient in val_patients
                ]
            else:
                train_datasets_list = [
                    XeniumPatientDataset(
                        gene_path=adata_paths[patient],
                        img_path=patch_paths[patient],
                        indices=selected_indices[patient],
                        gene_names=gene_names,
                        transform=transform,
                        log_norm=params.get("log_norm", True),
                        scale_factor=params.get("scale_factor", 1000000),
                    )
                    for patient in train_patients
                ]
                val_datasets_list = [
                    XeniumPatientDataset(
                        gene_path=adata_paths[patient],
                        img_path=patch_paths[patient],
                        indices=selected_indices[patient],
                        gene_names=gene_names,
                        transform=transform,
                        log_norm=params.get("log_norm", True),
                        scale_factor=params.get("scale_factor", 1000000),
                    )
                    for patient in val_patients
                ]
            if params.get("num_genes") is None:
                params["num_genes"] = int(train_datasets_list[0].num_genes)
                logger.info(
                    f"Inferred num_genes={params['num_genes']} from {train_patients[0]}"
                )
            fold_num_genes = {
                ds.num_genes for ds in train_datasets_list + val_datasets_list
            }
            if len(fold_num_genes) != 1:
                raise ValueError(
                    f"Fold {fold_idx + 1} patients have different gene counts after filtering: {fold_num_genes}"
                )
            train_dataset = ConcatDataset(train_datasets_list)
            val_dataset = ConcatDataset(val_datasets_list)
            logger.info(
                f"Fold {fold_idx + 1}: Train spots={len(train_dataset)}, Val spots={len(val_dataset)}"
            )
            train_regression(
                train_dataset,
                val_dataset,
                fold_idx,
                learning_rate=params["learning_rate"],
                epochs=params["epochs"],
                batch_size=params["batch_size"],
            )
            del train_dataset, val_dataset, train_datasets_list, val_datasets_list
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"===== Completed Fold {fold_idx + 1}/{k} =====")
        summary_file = os.path.join(model_save_dir, "folds_summary.txt")
        with open(summary_file, "w") as sf:
            sf.write(
                f"Completed {k} patient-wise folds. Models saved in {model_save_dir}\n"
            )
            sf.write(f"Patients: {patient_ids}\n")
            sf.write(f"Requested spots_per_patient: {spots_per_patient}\n")
        logger.info("All patient-wise folds completed successfully.")
        return
    patch_file = f"{params['dataset_path']}/cells_patches.h5"
    adata_file = f"{params['dataset_path']}/cells_expression.h5ad"
    single_dataset = PatchDataset(
        gene_path=adata_file, img_path=patch_file, gene_names=None, transform=transform
    )
    k = params.get("k_folds", 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_models = []
    raw_images = single_dataset.image
    raw_genes = single_dataset.gene_data
    for (fold_idx, (train_idx, val_idx)) in enumerate(
        kf.split(range(len(single_dataset)))
    ):
        logger.info(f"\n===== Starting Fold {fold_idx + 1} =====")
        train_norm_full = np.zeros_like(raw_genes, dtype=np.float32)
        val_norm_full = np.zeros_like(raw_genes, dtype=np.float32)
        train_norm = log1p_normalization(
            raw_genes[train_idx], scale_factor=params.get("scale_factor", 10000)
        )
        val_norm = log1p_normalization(
            raw_genes[val_idx], scale_factor=params.get("scale_factor", 10000)
        )
        train_norm_full[train_idx] = train_norm
        val_norm_full[val_idx] = val_norm
        train_dataset = FoldDataset(
            image_data=raw_images,
            gene_data=train_norm_full,
            indices=train_idx,
            transform=transform,
        )
        val_dataset = FoldDataset(
            image_data=raw_images,
            gene_data=val_norm_full,
            indices=val_idx,
            transform=transform,
        )
        logger.info(
            f"Fold {fold_idx + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}"
        )
        fold_model = train_regression(
            train_dataset,
            val_dataset,
            fold_idx,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"===== Completed Fold {fold_idx + 1} =====")
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")
    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
