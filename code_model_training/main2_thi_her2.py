import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (SCRIPT_DIR, SCRIPT_DIR / "models", SCRIPT_DIR / "utils"):
    sys.path.insert(0, str(source_dir))
import warnings

warnings.filterwarnings("ignore")
import gc
import csv
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torch
import os
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from datetime import datetime
import argparse
import json
from scipy.stats import pearsonr
from utils.thiToGene import THItoGeneHER2Dataset
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging
from set_deterministic_seed import set_deterministic_seed
from models import THItoGeneModel
from sklearn.model_selection import GroupKFold


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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join(params["result_path"], f"result_{timestamp}")
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
        train_dataloader = DataLoader(
            train_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_dataloader = DataLoader(
            val_datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        logger.info(f"Number of train batches: {len(train_dataloader)}")
        logger.info("Initializing model...")
        if params["dataset_name"] == "cscc":
            num_genes = 171
        elif params["dataset_name"] == "her2":
            num_genes = 785
        elif params["dataset_name"] == "hist2st_845":
            num_genes = 845
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")
        if params["model"] == "THItoGene":
            logger.info("Initializing THItoGene model for gene expression prediction.")
            model = THItoGeneModel(
                num_genes=num_genes, learning_rate=params.get("learning_rate", 1e-05)
            )
            print("learning_rate", learning_rate)
            logger.info(
                f"THItoGene initialized with {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} trainable parameters."
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
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            all_y_true = []
            all_y_pred = []
            pearson_train_corr = []
            spearman_train_corr = []
            for (batch_idx, batch) in enumerate(train_dataloader):
                (patches, centers, y_true, adj) = batch
                patches = patches.to(device)
                centers = centers.to(device)
                adj = adj.to(device)
                y_true = y_true.to(device)
                y_pred = model(patches, centers, adj)
                if y_pred.dim() == 2:
                    y_pred = y_pred.unsqueeze(0)
                if not torch.isfinite(patches).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in input patches. Skipping batch."
                    )
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch."
                    )
                    continue
                y_pred = torch.where(
                    y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred
                )
                if not torch.isfinite(y_pred).all():
                    logger.warning(
                        f"[Epoch {epoch + 1} | Batch {batch_idx}] NaN/Inf detected in model output. Zeroing out predictions."
                    )
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"image : {patches.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}"
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
                torch.cuda.empty_cache()
                gc.collect()
                epoch_train_loss += float(loss.item())
                y_true_2d = y_true.squeeze(0)
                y_pred_2d = y_pred.squeeze(0)
                all_y_true.append(y_true_2d.detach().cpu().numpy())
                all_y_pred.append(y_pred_2d.detach().cpu().numpy())
                try:
                    batch_pearson = pearsonr(
                        y_pred_2d.detach().cpu().numpy().flatten(),
                        y_true_2d.detach().cpu().numpy().flatten(),
                    )[0]
                    print(f"Batch {batch_idx} Pearson correlation: {batch_pearson:.4f}")
                except Exception as e:
                    print("Pearson Exception:", e)
                    batch_pearson = 0.0
                pearson_train_corr.append(float(batch_pearson))
                try:
                    batch_spearman = spearmanrr(y_pred_2d, y_true_2d)
                    spearman_train_corr.append(float(batch_spearman.mean().item()))
                except Exception as e:
                    print("Spearman Exception:", e)
                    spearman_train_corr.append(0.0)
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Fold {fold_idx + 1} Train Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_train_corr[-1]:.4f}, Spearman: {spearman_train_corr[-1]:.4f}"
                    )
            all_y_true = (
                np.concatenate(all_y_true, axis=0)
                if len(all_y_true) > 0
                else np.array([])
            )
            all_y_pred = (
                np.concatenate(all_y_pred, axis=0)
                if len(all_y_pred) > 0
                else np.array([])
            )
            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")
            print(
                f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}"
            )
            results_train = (
                compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}
            )
            del loss, y_pred, y_true, all_y_true, all_y_pred, patches, centers, adj
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
                    (patches, centers, y_true, adj) = batch
                    patches = patches.to(device)
                    centers = centers.to(device)
                    adj = adj.to(device)
                    y_true = y_true.to(device)
                    y_pred = model(patches, centers, adj)
                    if y_pred.dim() == 2:
                        y_pred = y_pred.unsqueeze(0)
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
                    y_true_2d = y_true.squeeze(0)
                    y_pred_2d = y_pred.squeeze(0)
                    val_y_true.append(y_true_2d.detach().cpu().numpy())
                    val_y_pred.append(y_pred_2d.detach().cpu().numpy())
                    try:
                        batch_pearson = pearsonr(
                            y_pred_2d.detach().cpu().numpy().flatten(),
                            y_true_2d.detach().cpu().numpy().flatten(),
                        )[0]
                    except Exception as e:
                        print("Pearson Exception (val):", e)
                        batch_pearson = 0.0
                    pearson_val_corr.append(float(batch_pearson))
                    try:
                        batch_spearman = spearmanrr(y_pred_2d, y_true_2d)
                        spearman_val_corr.append(float(batch_spearman.mean().item()))
                    except Exception as e:
                        print("Spearman Exception (val):", e)
                        spearman_val_corr.append(0.0)
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Fold {fold_idx + 1} Val Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_val_corr[-1]:.4f}, Spearman: {spearman_val_corr[-1]:.4f}"
                        )
                val_y_true = (
                    np.concatenate(val_y_true, axis=0)
                    if len(val_y_true) > 0
                    else np.array([])
                )
                val_y_pred = (
                    np.concatenate(val_y_pred, axis=0)
                    if len(val_y_pred) > 0
                    else np.array([])
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
                del loss, y_pred, y_true, patches, centers, adj
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

    if params["dataset_name"] == "cscc":
        gsm_samples = [
            "GSM4284316_P2",
            "GSM4284317_P2",
            "GSM4284318_P2",
            "GSM4284319_P5",
            "GSM4284320_P5",
            "GSM4284321_P5",
            "GSM4284322_P9",
            "GSM4284323_P9",
            "GSM4284324_P9",
            "GSM4284325_P10",
            "GSM4284326_P10",
            "GSM4284327_P10",
        ]
    if params["dataset_name"] == "hist2st_845":
        gsm_samples = [
            "23209_C1",
            "23209_C2",
            "23209_D1",
            "23268_C1",
            "23268_C2",
            "23268_D1",
            "23269_C1",
            "23269_C2",
            "23269_D1",
            "23270_D2",
            "23270_E1",
            "23270_E2",
            "23272_D2",
            "23272_E1",
            "23272_E2",
            "23277_D2",
            "23277_E1",
            "23277_E2",
            "23287_C1",
            "23287_C2",
            "23287_D1",
            "23288_D2",
            "23288_E1",
            "23288_E2",
            "23377_C1",
            "23377_C2",
            "23377_D1",
            "23450_D2",
            "23450_E1",
            "23450_E2",
            "23506_C1",
            "23506_C2",
            "23506_D1",
            "23508_D2",
            "23508_E1",
            "23508_E2",
            "23567_D2",
            "23567_E1",
            "23567_E2",
            "23803_D2",
            "23803_E1",
            "23803_E2",
            "23810_D2",
            "23810_E1",
            "23810_E2",
            "23895_C1",
            "23895_C2",
            "23895_D1",
            "23901_C2",
            "23901_D1",
            "23903_C1",
            "23903_C2",
            "23903_D1",
            "23944_D2",
            "23944_E1",
            "23944_E2",
            "24044_D2",
            "24044_E1",
            "24044_E2",
            "24105_C1",
            "24105_C2",
            "24105_D1",
            "24220_D2",
            "24220_E1",
            "24220_E2",
            "24223_D2",
            "24223_E1",
            "24223_E2",
        ]
    if params["dataset_name"] == "her2":
        gsm_samples = [
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "E1",
            "E2",
            "E3",
            "F1",
            "F2",
            "F3",
            "G1",
            "G2",
            "G3",
            "H1",
            "H2",
            "H3",
        ]

    def get_patient_id(sample, dataset_name):
        if dataset_name == "cscc":
            return sample.split("_")[1]
        elif dataset_name == "hist2st_845":
            return sample.split("_")[0]
        elif dataset_name == "her2":
            return sample[0]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_path = params.get("dataset_path", "")
    if params["dataset_name"] == "cscc":
        patch_paths = [
            os.path.join(dataset_path, f"{sample}_patches.h5") for sample in gsm_samples
        ]
        adata_paths = [
            os.path.join(dataset_path, f"{sample}_spots.h5ad") for sample in gsm_samples
        ]
    if params["dataset_name"] == "her2":
        patch_paths = [
            os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples
        ]
        adata_paths = [
            os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples
        ]
    if params["dataset_name"] == "hist2st_845":
        patch_paths = [
            os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples
        ]
        adata_paths = [
            os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples
        ]
    logger.info(f"Total samples: {len(gsm_samples)}")
    for (p, a) in zip(patch_paths, adata_paths):
        logger.info(f"  - {os.path.basename(p)} ↔ {os.path.basename(a)}")
    for (p, a) in zip(patch_paths, adata_paths):
        if not os.path.exists(p):
            logger.warning(f"Missing patch file: {p}")
        if not os.path.exists(a):
            logger.warning(f"Missing spot file: {a}")
    patient_ids = [get_patient_id(s, params["dataset_name"]) for s in gsm_samples]
    unique_patients = sorted(set(patient_ids))
    logger.info(f"Total patients: {len(unique_patients)} → {unique_patients}")
    genes_file = os.path.join(params["genes"], f"{params['dataset_name']}.npy")
    print("+++++++++++++++++++++++++++++++", genes_file)
    gene_names = np.load(genes_file, allow_pickle=True).tolist()
    logger.info(f" Using gene list from: {gene_names}")
    if not os.path.exists(genes_file):
        print(genes_file)
        print(gene_names)
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((112, 112)), transforms.ToTensor()]
    )
    k = params.get("k_folds", 5)
    n_patients = len(unique_patients)
    if k > n_patients:
        logger.warning(
            f"k_folds ({k}) > number of patients ({n_patients}). Setting k = {n_patients}"
        )
        k = n_patients
    kf = GroupKFold(n_splits=k, shuffle=True, random_state=42)
    fold_models = []
    for (fold_idx, (train_idx, val_idx)) in enumerate(
        kf.split(gsm_samples, groups=patient_ids)
    ):
        logger.info(f"\n===== Starting Fold {fold_idx + 1}/{k} =====")
        train_patients = sorted(set((patient_ids[i] for i in train_idx)))
        val_patients = sorted(set((patient_ids[i] for i in val_idx)))
        logger.info(f"Train patients: {len(train_patients)}, {train_patients}")
        logger.info(f"Val patients: {len(val_patients)},  {val_patients}")
        train_patch_files = [patch_paths[i] for i in train_idx]
        train_adata_files = [adata_paths[i] for i in train_idx]
        val_patch_files = [patch_paths[i] for i in val_idx]
        val_adata_files = [adata_paths[i] for i in val_idx]
        logger.info(
            f"Fold {fold_idx + 1}: Train={len(train_patch_files)}, Val={len(val_patch_files)}"
        )
        train_datasets_list = [
            THItoGeneHER2Dataset(
                h5_img_path=img_p,
                h5ad_path=adata_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000),
            )
            for (adata_p, img_p) in zip(train_adata_files, train_patch_files)
        ]
        val_datasets_list = [
            THItoGeneHER2Dataset(
                h5_img_path=img_p,
                h5ad_path=adata_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000),
            )
            for (adata_p, img_p) in zip(val_adata_files, val_patch_files)
        ]
        train_dataset_concat = ConcatDataset(train_datasets_list)
        val_dataset_concat = ConcatDataset(val_datasets_list)
        logger.info(
            f"Fold {fold_idx + 1}: Train={len(train_dataset_concat)}, Val={len(val_dataset_concat)}"
        )
        fold_dir = os.path.join(model_save_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        models = train_regression(
            train_dataset_concat,
            val_dataset_concat,
            fold_idx,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )
        del train_dataset_concat, train_datasets_list
        del val_dataset_concat, val_datasets_list
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"===== Completed Fold {fold_idx + 1}/{k} =====")
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")
    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
