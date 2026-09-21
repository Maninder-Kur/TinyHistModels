import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (SCRIPT_DIR, SCRIPT_DIR / 'models', SCRIPT_DIR / 'utils'):
    sys.path.insert(0, str(source_dir))

import warnings
warnings.filterwarnings("ignore")

import gc
import csv
import anndata as ad
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from datetime import datetime
import argparse
import json
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from torch.nn import DataParallel
from sklearn.model_selection import KFold
from utils.dataPrep_xenium import PatchDataset, FoldDataset, log1p_normalization, XeniumPatientDataset
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging 
from set_deterministic_seed import set_deterministic_seed
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive, GeneGuidedImageTransformer, UNI
from torch.utils.data import ConcatDataset
from proposedModels import ImageGeneCrossTransformer
from sklearn.model_selection import KFold
from proposedModels2 import ImageToGeneTransformer
from models import STPathModel
from model_eff_net_versions import EfficientNetTinyStudent


def main():
     
    g, seed_worker = set_deterministic_seed(123)   
     
    parser = argparse.ArgumentParser(description="Script that uses a JSON configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file at {args.config}")
        return

    # g, seed_worker = set_deterministic_seed(params.get("seed", 123))
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join(params['result_path'], f"result_vit_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)

    json_file = os.path.join(model_save_dir, f"experiment_info.json")
    with open(json_file, "w") as exp:
        json.dump(params, exp, indent=4)

    log_file = os.path.join(model_save_dir, "results_log.txt")
    logger = setup_logging(log_file)
    logger.info("Logging setup complete.")
    logger.info(f"Experiment information saved to the path: {json_file}")

    metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")

    # Write CSV headers (only once before the loop starts)
    # We'll append per-fold & per-epoch rows later
    with open(metrics_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 
            'spearman_mean_genewise','l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
            'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
            'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
        ])
        writer.writeheader()

    # --- Training routine (mostly unchanged) ---
    def train_regression(train_datasets, val_datasets, fold_idx,
                                          lambda_reg=params.get("lambda_reg", 0.0),
                                          learning_rate=params['learning_rate'],
                                          epochs=params['epochs'],
                                          batch_size=params['batch_size'],
                                          weight_decay=params['weight_decay']):

        models = {}

        logger.info(f"train_datasets length:  {len(train_datasets)}")
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)

        logger.info(f"Number of train batches: {len(train_dataloader)}")
        logger.info("Initializing model...")

        if params.get("num_genes") is not None:
            num_genes = int(params["num_genes"])
        elif params['dataset_name'] == "cscc":
            num_genes = 171
        elif params['dataset_name'] == "her2":
            num_genes = 785
        elif params['dataset_name'] == "hist2st_845":
            num_genes = 845
        elif params['dataset_name'] == "10X_HumandBreastCancer":
            num_genes = 541
        elif params['dataset_name'] == "xenium_breast_patient":
            num_genes = 500
        elif params['dataset_name'] == "xenium_lung_patient":
            num_genes = 541
        elif params['dataset_name'] == "10X_HumanLungCancer":    
            num_genes = 480
        elif params['dataset_name'] == "10X_HumanPancreasCancer":    
            num_genes = 377
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")

        if params["model"] == "STNet":
            pretrained = params.get("pretrained", False)
            model = STNet(num_genes=num_genes, pretrained=pretrained)
            logger.info(f"STNet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        elif params["model"] == "VGG":
            pretrained = params.get("pretrained", True)
            model = Custom_VGG16(num_genes=num_genes, pretrained=pretrained)
            
            logger.info(f"VGG initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
            
        elif params["model"] == "Effnet":
            pretrained = params.get("pretrained", False)
            model = EfficientNet(num_genes=num_genes, pretrained=pretrained) 
            logger.info(f"EFFnet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
            
        elif params["model"] == "HisToGene":
            model = HisToGene(patch_size=16, n_layers= 8, n_genes=num_genes)
            
        elif params["model"] == "EfficientNetTinyStudent":
            logger.info("EfficientNetTinyStudent (tiny) for gene expression prediction.")
            phi = params.get("phi", -7.0)
            model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)
            
            logger.info(f"EfficientNetTinyStudent initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        
        elif params["model"] == "TCGN":            
            logger.info("Initializing TCGN model...")

            # # Set up model parameters
            # tcgn_kwargs = dict(
            #     img_size=params.get("image_height", 224),
            #     in_chans=params.get("image_channels", 3),
            #     num_classes=num_genes,
            #     embed_dims=params.get("embed_dims", [52, 104, 208, 416]),
            #     stem_channel=params.get("stem_channel", 16),
            #     fc_dim=params.get("fc_dim", 1280),
            #     num_heads=params.get("num_heads", [1, 2, 4, 8]),
            #     mlp_ratios=params.get("mlp_ratios", [3.6, 3.6, 3.6, 3.6]),
            #     qkv_bias=True,
            #     qk_scale=None,
            #     representation_size=None,
            #     drop_rate=params.get("drop_rate", 0.0),
            #     attn_drop_rate=params.get("attn_drop_rate", 0.0),
            #     drop_path_rate=params.get("drop_path_rate", 0.0),
            #     hybrid_backbone=None,
            #     norm_layer=None,
            #     depths=params.get("depths", [2, 2, 10, 2]),
            #     qk_ratio=params.get("qk_ratio", 1),
            #     sr_ratios=params.get("sr_ratios", [8, 4, 2, 1]),
            #     dp=params.get("dropout", 0.1)
            # )

            # Instantiate the model
            model = TCGN(
                num_classes=num_genes,
                embed_dims=params.get("embed_dims", [46, 92, 184, 368]),
            )
            logger.info(f"TCGN initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        elif params["model"] == "EfficientNet_GeneCaptionContrastive":
            logger.info("Initializing EfficientNet Gene Caption + Contrastive model...")

            model = EfficientNet_GeneCaptionContrastive(
                num_genes=num_genes,
                pretrained=params.get("pretrained", True),
                contrastive_weight=params.get("contrastive_weight", 0.1),
                temperature=params.get("temperature", 0.07)
            )
            
            logger.info(f"EfficientNet_GeneCaptionContrastive initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        elif params["model"] == "UNI":

            model = UNI(
                num_genes=num_genes,
                assets_dir=params["assets_dir"],
                freeze_backbone=params.get("freeze_backbone", False)
            )

            logger.info(
                f"UNI initialized with "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
                f"trainable parameters."
            )

        elif params["model"] == "STPath":
            # Release adapter: the archived model and run configuration survive,
            # but the historical STPath entry point was not in the source tree.
            model = STPathModel(
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

        elif params["model"] == "GeneGuidedImageTransformer":
            logger.info("Initializing GeneGuidedImageTransformer...")
            model = GeneGuidedImageTransformer(num_genes = num_genes, pretrained = params.get("pretrained", False))
            logger.info(
                f"GeneGuidedImageTransformer initialized with "
                f"{sum(p.numel() for p in model.parameters()):,} trainable parameters.")
        
        
        elif params["model"] == "ImageGeneCrossTransformer":
            # --- Get model hyperparameters ---
            embed_dim = params.get("embed_dim", 512)
            nhead = params.get("nhead", 4)
            dim_feedforward = params.get("dim_feedforward", 1024)
            dropout = params.get("dropout", 0.1)
            pretrained = params.get("pretrained", False)
            H = params.get("image_height", 224)
            W = params.get("image_width", 224)
            image_channels = params.get("image_channels", 3)

            # --- Logging configuration ---
            logger.info("Initializing ImageGeneCrossTransformer with:")
            logger.info(f"- Image size: {H}x{W}")
            logger.info(f"- Embedding dimension (d_model): {embed_dim}")
            logger.info(f"- Attention heads: {nhead}")
            logger.info(f"- Feedforward dim: {dim_feedforward}")
            logger.info(f"- Dropout: {dropout}")
            logger.info(f"- Pretrained EfficientNet: {pretrained}")
            logger.info(f"- Number of genes: {num_genes}")
            logger.info(f"- Image channels: {image_channels}")

            # --- Model initialization ---
            model = ImageGeneCrossTransformer(
                num_genes=num_genes,
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pretrained=pretrained,
            )

        elif params["model"] == "ImageToGeneTransformer":
            # --- Get model hyperparameters ---
            embed_dim = params.get("embed_dim", 768)
            nhead = params.get("nhead", 8)
            encoder_layers = params.get("encoder_layers", 6)
            decoder_layers = params.get("decoder_layers", 4)
            dim_feedforward = params.get("dim_feedforward", 2048)
            dropout = params.get("dropout", 0.1)
            patch_size = params.get("patch_size", 16)
            activation = params.get("activation", "relu")
            layer_norm_eps = params.get("layer_norm_eps", 1e-5)
            batch_first = params.get("batch_first", True)
            freeze_encoder = params.get("freeze_encoder", False)
            H = params.get("image_height", 128)
            W = params.get("image_width", 128)
            image_channels = params.get("image_channels", 3)

            # --- Logging configuration ---
            logger.info("Initializing ImageToGeneTransformer with:")
            logger.info(f"- Image size: {H}x{W}")
            logger.info(f"- Embedding dimension (d_model): {embed_dim}")
            logger.info(f"- Encoder layers: {encoder_layers}")
            logger.info(f"- Decoder layers: {decoder_layers}")
            logger.info(f"- Attention heads: {nhead}")
            logger.info(f"- Feedforward dim: {dim_feedforward}")
            logger.info(f"- Dropout: {dropout}")
            logger.info(f"- Patch size: {patch_size}")
            logger.info(f"- Activation: {activation}")
            logger.info(f"- Layer norm eps: {layer_norm_eps}")
            logger.info(f"- Batch first: {batch_first}")
            logger.info(f"- Freeze encoder: {freeze_encoder}")
            logger.info(f"- Number of genes: {num_genes}")
            logger.info(f"- Image channels: {image_channels}")

            # --- Model initialization ---
            model = ImageToGeneTransformer(
                H=H,
                W=W,
                C=image_channels,
                num_genes=num_genes,
                d_model=embed_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                freeze_encoder=freeze_encoder,
                patch_size=patch_size,
            )
        else:
            logger.error('No model found with name: %s', params["model"])
            raise ValueError("No model found")

        logger.info("Model\n%s", model)


        # Device / GPU logic (robust). Always set a device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)

        # Optimizer
        if params.get("optimizer", "Adam").lower() == "adam":
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
        elif params.get("optimizer", "sgd").lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            logger.error("Optimizer not defined!!! Defaulting to Adam")
            optimizer = Adam(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params.get('lr_patience', 5),
            factor=params.get('lr_factor', 0.5),
            min_lr=params.get('min_lr', 1e-7),
        )

        train_losses = []
        val_losses = []

        best_spearman = -np.inf
        patience_counter = 0
        patience = params.get('early_stopping_patience', 15)
        
        mse_criterion = torch.nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            all_y_true = []
            all_y_pred = []
            pearson_train_corr = []
            spearman_train_corr = []

            for batch_idx, (images, targets) in enumerate(train_dataloader):
                # Move data to device
                images = images.to(device)             # shape: (B, 3, 224, 224)
                y_true = targets.to(device)            # shape: (B, num_genes)
                
                
                # === Sanity check: Detect NaNs or Infs in input data ===
                if not torch.isfinite(images).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in input images. Skipping batch.")
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch.")
                    continue


                if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer", "EfficientNet_GeneCaptionContrastive","GeneGuidedImageTransformer"]:
                    y_pred, contrastive_loss = model(images, gene_values=y_true)  # Pass gene values during training
                else:
                    y_pred = model(images)                 # shape: (B, num_genes)


                y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)
                
                # === Sanity check: Detect NaNs/Infs in predictions ===
                if not torch.isfinite(y_pred).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in model output. Zeroing out predictions.")
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)


                print(f"image : {images.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}")
                
                if batch_idx == 0:
                    print("\n===================== DEBUG: Epoch {} =====================".format(epoch + 1))
                    print("Sample y_true values (first sample, first 10 genes):")
                    print(y_true[0, :10].detach().cpu().numpy())
                    print("Sample y_pred values (first sample, first 10 genes):")
                    print(y_pred[0, :10].detach().cpu().numpy())

                    # Compute basic stats
                    y_true_np = y_true.detach().cpu().numpy()
                    y_pred_np = y_pred.detach().cpu().numpy()

                    print(f"y_true  -> mean={np.nanmean(y_true_np):.4f}, std={np.nanstd(y_true_np):.4f}, "
                        f"min={np.nanmin(y_true_np):.4f}, max={np.nanmax(y_true_np):.4f}")

                    print(f"y_pred  -> mean={np.nanmean(y_pred_np):.4f}, std={np.nanstd(y_pred_np):.4f}, "
                        f"min={np.nanmin(y_pred_np):.4f}, max={np.nanmax(y_pred_np):.4f}")

                # choose loss
                # loss = None
                loss_fn_name = params.get("loss_fn", "")
                if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                    mse_criterion = torch.nn.MSELoss()
                    mse_loss = mse_criterion(y_pred, y_true)
                    # if contrastive_loss is not None:
                    #     loss = mse_loss + model.contrastive_weight * contrastive_loss
                    # else:
                    loss = mse_loss
                    
                elif loss_fn_name == "clip" or loss_fn_name == "CLIP":
                    mse_loss = mse_criterion(y_pred, y_true)
    
                    if contrastive_loss is not None:
                        loss = mse_loss + params.get("contrastive_weight", 0.1) * contrastive_loss
                    else:
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
                    batch_pearson = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())[0]
                    print(f"Batch {batch_idx} Pearson correlation: {batch_pearson:.4f}")
                except Exception as e:
                    print("Exception+++++++++++++++++++++++++++++++++++++++++++++ ",{e})
                    batch_pearson = 0.0
                pearson_train_corr.append(float(batch_pearson.mean().item()))

                # Spearman (use your function that returns vector)
                try:
                    batch_spearman = spearmanrr(y_pred, y_true)
                    spearman_train_corr.append(batch_spearman.mean().item())

                except Exception as e:
                    print(f"code is in Exception++++++++++++++++++++++++++++++++ {e}")
                    spearman_train_corr.append(0.0)

                if batch_idx % 10 == 0:
                    logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_train_corr[-1]:.4f}, Spearman: {spearman_train_corr[-1]:.4f}")

            # epoch-level aggregation
            all_y_true = np.vstack(all_y_true) if len(all_y_true) > 0 else np.array([])
            all_y_pred = np.vstack(all_y_pred) if len(all_y_pred) > 0 else np.array([])

            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")

            print(f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}")
            results_train = compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}

            del loss, y_pred, y_true, images, all_y_true, all_y_pred
            torch.cuda.empty_cache()
            gc.collect()
            
            avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
            logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Pearson Mean: {np.mean(pearson_train_corr):.4f}, Spearman Mean: {np.mean(spearman_train_corr):.4f}")
            if results_train:
                logger.info(f"Training Metrics: {results_train}")

            # Write train row to CSV
            with open(metrics_csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 'spearman_mean_genewise',
                    'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
                    'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                    'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                ])
                row = {'fold': fold_idx + 1, 'epoch': epoch + 1, 'phase': 'train', 'custom_loss': f"{avg_train_loss:.4f}", 'pearson_mean_cell_wise': f"{np.mean(pearson_train_corr):.4f}", 'spearman_mean_cell_wise': f"{np.mean(spearman_train_corr):.4f}"}
                # add other metrics if present
                if results_train:
                    for k in results_train:
                        if k in writer.fieldnames:
                            row[k] = results_train[k]
                writer.writerow(row)

            train_losses.append(avg_train_loss)
            
            

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            val_y_true = []
            val_y_pred = []
            spearman_val_corr = []
            pearson_val_corr = []

            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_dataloader):
                    # Move images and targets to GPU
                    images = images.to(device)                          # shape: (B, 3, 224, 224)
                    y_true = targets.to(device)                         # shape: (B, num_genes)

                    # --- Forward pass through model ---
                    if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer", "EfficientNet_GeneCaptionContrastive","GeneGuidedImageTransformer"]:
                        y_pred,_ = model(images)
                    else:
                        y_pred = model(images) 
                        
                    assert torch.is_tensor(y_pred), f"y_pred is not tensor, got {type(y_pred)}"
   
                    y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)

                    loss_fn_name = params.get("loss_fn", "")
                    if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                        mse_criterion = torch.nn.MSELoss()
                        loss = mse_criterion(y_pred, y_true)
                        
                    elif loss_fn_name == "clip" or loss_fn_name == "CLIP":
                        mse_loss = mse_criterion(y_pred, y_true)
        
                        if contrastive_loss is not None:
                            loss = mse_loss + params.get("contrastive_weight", 0.1) * contrastive_loss
                        else:
                            loss = mse_loss
                    
                    else:
                        logger.error("Loss Function not defined!!!")
                        raise ValueError("Loss Function not defined")

                    epoch_val_loss += float(loss.item())
                    val_y_true.append(y_true.detach().cpu().numpy())
                    val_y_pred.append(y_pred.detach().cpu().numpy())

                    try:
                        batch_pearson = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())[0]
                    except Exception:
                        batch_pearson = 0.0
                    pearson_val_corr.append(float(batch_pearson.mean().item()))

                    try:
                        batch_spearman = spearmanrr(y_pred, y_true)
                        spearman_val_corr.append(float(batch_spearman.mean().item()))
                    except Exception:
                        spearman_val_corr.append(0.0)

                    if batch_idx % 10 == 0:
                        logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_val_corr[-1]:.4f}, Spearman: {spearman_val_corr[-1]:.4f}")

                val_y_true = np.vstack(val_y_true) if len(val_y_true) > 0 else np.array([])
                val_y_pred = np.vstack(val_y_pred) if len(val_y_pred) > 0 else np.array([])

                # Save scatter of first up-to-25 samples (if present)
                if val_y_true.size:
                    n_display = min(25, val_y_true.shape[0])
                    val_y_true_25 = val_y_true[:n_display, :]
                    val_y_pred_25 = val_y_pred[:n_display, :]

                    # Create grid (square)
                    grid_sz = int(np.ceil(np.sqrt(n_display)))
                    fig, axes = plt.subplots(nrows=grid_sz, ncols=grid_sz, figsize=(4 * grid_sz, 4 * grid_sz))
                    axes = np.atleast_2d(axes)
                    for idx in range(n_display):
                        r = idx // grid_sz
                        c = idx % grid_sz
                        axes[r, c].scatter(val_y_true_25[idx], val_y_pred_25[idx], alpha=0.6)
                        min_val = min(val_y_true_25[idx].min(), val_y_pred_25[idx].min())
                        max_val = max(val_y_true_25[idx].max(), val_y_pred_25[idx].max())
                        axes[r, c].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
                        axes[r, c].set_title(f"Sample {idx+1}")
                    plt.tight_layout()
                    scatter_save_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"scatter_plot_25_samples_epoch_{epoch+1}.png")
                    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
                    plt.savefig(scatter_save_path)
                    plt.close(fig)

                results_val = compute_metrics(val_y_true, val_y_pred) if val_y_true.size else {}


                # Memory cleanup per validation batch
                del loss, y_pred, y_true, images
                torch.cuda.empty_cache()
                gc.collect()


                avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))
                logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Pearson Mean: {np.mean(pearson_val_corr):.4f}, Spearman Mean: {np.mean(spearman_val_corr):.4f}")
                if results_val:
                    logger.info(f"Validation Metrics: {results_val}")

                # Write val row to CSV
                with open(metrics_csv_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=[
                        'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 'spearman_mean_genewise',
                        'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
                        'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                        'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                    ])
                    row = {'fold': fold_idx + 1, 'epoch': epoch + 1, 'phase': 'val', 'custom_loss': f"{avg_val_loss:.4f}", 'pearson_mean_cell_wise': f"{np.mean(pearson_val_corr):.4f}", 'spearman_mean_cell_wise':  f"{np.mean(spearman_val_corr):.4f}"}
                    if results_val:
                        for k in results_val:
                            if k in writer.fieldnames:
                                row[k] = results_val[k]
                    writer.writerow(row)

                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate for epoch {epoch + 1}: {current_lr}")

                # Early stopping based on spearman_mean_cell_wise if present
                current_spearman = float(results_val.get('spearman_mean_genewise', np.mean(spearman_val_corr) if spearman_val_corr else -np.inf))
                if current_spearman > best_spearman:
                    best_spearman = current_spearman
                    patience_counter = 0
                    # Save best
                    best_model_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", "best_model.pth")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in spearman genewise. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered. Breaking training loop.")
                    break

                # plot losses
                try:
                    plt.figure(figsize=(8, 5))
                    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
                    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Fold {fold_idx+1} Loss")
                    plt.legend()
                    plt.grid(True)
                    graph_save_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"loss_plot_epoch.png")
                    plt.savefig(graph_save_path)
                    plt.close('all')  # Close all figures to prevent memory leaks
                except Exception as e:
                    logger.warning(f"Could not plot losses: {e}")

            # Save snapshot per epoch
            snapshot_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"epoch_model.pth")
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
        h5_ids = {os.path.splitext(f)[0] for f in os.listdir(dataset_path) if f.endswith(".h5")}
        h5ad_ids = {os.path.splitext(f)[0] for f in os.listdir(dataset_path) if f.endswith(".h5ad")}
        patient_ids = sorted(h5_ids & h5ad_ids)
        if not patient_ids:
            raise FileNotFoundError(
                f"No matching .h5/.h5ad patient pairs found in {dataset_path}"
            )
        return patient_ids

    # --- Optional image transforms ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if params.get("patient_wise_cv", False):
        dataset_path = params["dataset_path"]
        patient_ids = params.get("patient_ids") or discover_patient_ids(dataset_path)
        k = params.get("k_folds", len(patient_ids))
        if k > len(patient_ids):
            logger.warning("k_folds > number of patients. Setting k = number of patients.")
            k = len(patient_ids)

        spots_per_patient = int(params.get("spots_per_patient", 30000))
        subset_seed = int(params.get("subset_seed", params.get("seed", 123)))
        rng = np.random.default_rng(subset_seed)
        gene_names = load_gene_names_from_params()
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
            indices = np.sort(rng.choice(n_spots, size=n_take, replace=False)).astype(np.int64)
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
                logger.info(f"{patient}: selected fixed subset of {n_take}/{n_spots} spots.")

        np.savez_compressed(
            os.path.join(model_save_dir, "patient_fixed_spot_indices.npz"),
            **selected_indices
        )
        with open(os.path.join(model_save_dir, "patient_fixed_spot_summary.json"), "w") as sf:
            json.dump(subset_summary, sf, indent=4)

        logger.info(f"Total patients: {len(patient_ids)}")
        logger.info(f"Patient order: {patient_ids}")

        patient_shuffle = bool(params.get("patient_shuffle", False))
        if patient_shuffle:
            kf = KFold(n_splits=k, shuffle=True, random_state=int(params.get("fold_seed", 42)))
        else:
            kf = KFold(n_splits=k, shuffle=False)

        for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(kf.split(patient_ids)):
            train_patients = [patient_ids[i] for i in train_patient_idx]
            val_patients = [patient_ids[i] for i in val_patient_idx]

            logger.info(f"\n===== Starting Fold {fold_idx+1}/{k} =====")
            logger.info(f"Train patients: {train_patients}")
            logger.info(f"Val patients: {val_patients}")

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
                logger.info(f"Inferred num_genes={params['num_genes']} from {train_patients[0]}")

            fold_num_genes = {ds.num_genes for ds in train_datasets_list + val_datasets_list}
            if len(fold_num_genes) != 1:
                raise ValueError(
                    f"Fold {fold_idx+1} patients have different gene counts after filtering: {fold_num_genes}"
                )

            train_dataset = ConcatDataset(train_datasets_list)
            val_dataset = ConcatDataset(val_datasets_list)

            logger.info(f"Fold {fold_idx+1}: Train spots={len(train_dataset)}, Val spots={len(val_dataset)}")

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
            logger.info(f"===== Completed Fold {fold_idx+1}/{k} =====")

        summary_file = os.path.join(model_save_dir, "folds_summary.txt")
        with open(summary_file, "w") as sf:
            sf.write(f"Completed {k} patient-wise folds. Models saved in {model_save_dir}\n")
            sf.write(f"Patients: {patient_ids}\n")
            sf.write(f"Requested spots_per_patient: {spots_per_patient}\n")

        logger.info("All patient-wise folds completed successfully.")
        return

    # --- Define paths --- 
    patch_file = f"{params['dataset_path']}/cells_patches.h5" 
    adata_file = f"{params['dataset_path']}/cells_expression.h5ad"
    
    single_dataset = PatchDataset(
        gene_path=adata_file,
        img_path=patch_file,
        gene_names=None,
        transform=transform
    )

    # --- K-Fold setup ---
    k = params.get("k_folds", 5)

    kf = KFold(n_splits= k, shuffle=True, random_state=42)
    fold_models = []
    
    raw_images = single_dataset.image
    raw_genes = single_dataset.gene_data
    
    # results_path_fold = os.path.join(params["result_path"], "result_vit_2025-11-22_23-27-42")  
    # # find all existing fold_* folders
    # completed_folds = {
    #     int(name.split("_")[1]) - 1
    #     for name in os.listdir(results_path_fold)
    #     if name.startswith("fold_")
    # }
    
    # start_fold = 1
    # --- K-Fold Loop ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(single_dataset)))):
        # if fold_idx in completed_folds:
        #     logger.info(f"Skipping Fold {fold_idx+1} (already completed)")
        #     continue
        # if fold_idx < start_fold:
        #     logger.info(f"Skipping Fold {fold_idx+1} (already completed)")
        #     continue
        
    
        logger.info(f"\n===== Starting Fold {fold_idx+1} =====")

        # ---------- Create FULL gene matrix for this fold ----------
        train_norm_full = np.zeros_like(raw_genes, dtype=np.float32)
        val_norm_full   = np.zeros_like(raw_genes, dtype=np.float32)

        # Normalize only train/val rows
        train_norm = log1p_normalization(raw_genes[train_idx], scale_factor=params.get("scale_factor", 10000))
        val_norm   = log1p_normalization(raw_genes[val_idx],   scale_factor=params.get("scale_factor", 10000))

        # Insert back into full matrices
        train_norm_full[train_idx] = train_norm
        val_norm_full[val_idx] = val_norm

        # ---------- Build fold datasets ----------
        train_dataset = FoldDataset(
            image_data=raw_images,
            gene_data=train_norm_full,     # ✔ FULL matrix
            indices=train_idx,
            transform=transform
        )

        val_dataset = FoldDataset(
            image_data=raw_images,
            gene_data=val_norm_full,       # ✔ FULL matrix
            indices=val_idx,
            transform=transform
        )

        logger.info(f"Fold {fold_idx+1}: Train={len(train_dataset)}, Val={len(val_dataset)}")

        # ---------- Train ----------
        fold_model = train_regression(
            train_dataset, val_dataset, fold_idx,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"]
        )

        del train_dataset
        del val_dataset

        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"===== Completed Fold {fold_idx+1} =====")

    # --- Save summary ---
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")

    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
