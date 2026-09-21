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
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import os
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
from utils.dataPrep import PatchDataset
from utils.thiToGene import THItoGeneH5Dataset, THItoGeneHER2Dataset
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging
from set_deterministic_seed import set_deterministic_seed
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive, GEMResNet18, GEMEfficientNetB0, MobileNetV2Regressor, HierarchicalDenseNet
from models import TinySTNet, RefinedTinySTNet, TinyEfficientNet, RefinedTinyEfficientNet, THItoGeneModel
from model_eff_net_versions import EfficientNetTinyStudent
from models_kd import TinyHisToGene, TinySTNet
from torch.utils.data import ConcatDataset
from proposedModels import ImageGeneCrossTransformer
from sklearn.model_selection import KFold, GroupKFold
from proposedModels2 import ImageToGeneTransformer
from losses_kd import KDLoss

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
    model_save_dir = os.path.join(params['result_path'], f"result_{timestamp}")
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
        
        teacher = None

        logger.info(f"train_datasets length:  {len(train_datasets)}")
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)

        logger.info(f"Number of train batches: {len(train_dataloader)}")
        logger.info("Initializing model...")

        if params['dataset_name'] == "cscc":
            num_genes = 171
        elif params['dataset_name'] == "her2":
            num_genes = 785
        elif params['dataset_name'] == "hist2st_845":
            num_genes = 845
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")

        if params["model"] == "STNet":
            pretrained = params.get("pretrained", False)
            model = STNet(num_genes=num_genes, pretrained=pretrained)
        
        elif params["model"] == "VGG":
            pretrained = params.get("pretrained", True)
            model = Custom_VGG16(num_genes=num_genes, pretrained=pretrained)
            
        elif params["model"] == "Effnet":
            pretrained = params.get("pretrained", False)
            model = EfficientNet(num_genes=num_genes, pretrained=pretrained) 
            
        elif params["model"] == "Resnet_GEM": 
            pretrained = params.get("pretrained", True)
            model = GEMResNet18(num_genes=num_genes, pretrained=pretrained)
        
        elif params["model"] == "GEMEfficientNetB0": 
            pretrained = params.get("pretrained", True)
            model = GEMEfficientNetB0(num_genes=num_genes, pretrained=pretrained)
            
        elif params["model"] == "MobileNet":
            pretrained = params.get("pretrained", True)
            model = MobileNetV2Regressor(num_genes=num_genes, pretrained=pretrained)
            
        elif params["model"] == "TinyHisToGene":
            model = TinyHisToGene(n_genes=num_genes)

            if params.get("use_kd", False):
                teacher = HisToGene(patch_size=16, n_layers=8, n_genes=num_genes)
                
            logger.info(f"TinyHisToGene initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
            
        elif params["model"] == "TinySTNet":
            pretrained = params.get("pretrained", False)
            model = TinySTNet(num_genes=num_genes)  # no pretrained weights

            if params.get("use_kd", False):  # if distillation is enabled
                # Load the teacher model
                teacher = STNet(num_genes=num_genes, pretrained=pretrained)
                # teacher.eval()
                # for p in teacher.parameters():
                #     p.requires_grad = False  # freeze teacher
                
            logger.info(f"TinySTNet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        elif params["model"] == "ReTinySTNet":
            pretrained = params.get("pretrained", True)
            model = RefinedTinySTNet(num_genes=num_genes)  # no pretrained weights

            if params.get("use_kd", False):  # if distillation is enabled
                # Load the teacher model
                teacher = STNet(num_genes=num_genes, pretrained=True)
        
        elif params["model"] == "TinyEffNet_kd":
            pretrained = params.get("pretrained", True)
            
            phi = params.get("phi", -7.0)   # default phi if not provided
            print("phi:", phi)

            model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)

            if params.get("use_kd", False):
                teacher = EfficientNet(num_genes=num_genes, pretrained=pretrained)
                
            logger.info(f"TinyEffNet_kd initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        elif params["model"] == "RefinedTinyEfficientNet":
            logger.info(f"RefinedTinyEfficientNet model for gene expression prediction.")
            pretrained = params.get("pretrained", True)
            model = RefinedTinyEfficientNet(num_genes=num_genes)

            if params.get("use_kd", False):
                teacher = EfficientNet(num_genes=num_genes, pretrained=True)
        
        # elif params["model"] == "EfficientNetTinyStudent":
        #     logger.info("EfficientNetTinyStudent (compound-scaled) for gene expression prediction.")

        #     pretrained = params.get("pretrained", True)
        #     phi = params.get("phi", -7.0)   # default phi if not provided
        #     print("phi:", phi)
        #     use_kd = params.get("use_kd", False)

        #     model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)

        #     teacher = None
        #     if use_kd:
        #         logger.info("Using EfficientNet-B0 teacher for knowledge distillation.")
        #         teacher = EfficientNet(num_genes=num_genes, pretrained=pretrained)
            
        #     logger.info(f"EfficientNetTinyStudent initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        elif params["model"] == "EfficientNetTinyStudent":
            logger.info("EfficientNetTinyStudent (tiny) for gene expression prediction.")
            phi = params.get("phi", -7.0)
            model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)
            
            logger.info(f"EfficientNetTinyStudent initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        elif params["model"] == "HierarchicalDenseNet": 
            pretrained = params.get("pretrained", True)
            model = HierarchicalDenseNet(num_genes=num_genes, pretrained=pretrained, shared_backbone=False)
            
        elif params["model"] == "HisToGene":
            model = HisToGene(patch_size=16, n_layers= 8, n_genes=num_genes)
            
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

            # # Instantiate the model
            # model = TCGN(**tcgn_kwargs)
            model = TCGN(num_classes=num_genes)

            logger.info(f"TCGN initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        elif params["model"] == "EfficientNet_GeneCaptionContrastive":
            logger.info("Initializing EfficientNet Gene Caption + Contrastive model...")

            model = EfficientNet_GeneCaptionContrastive(
                num_genes=num_genes,
                pretrained=params.get("pretrained", True),
                contrastive_weight=params.get("contrastive_weight", 0.02),
                temperature=params.get("temperature", 0.07)
            )
            
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


        elif params["model"] == "THItoGene":
            logger.info("Initializing THItoGene model for gene expression prediction.")
            model = THItoGeneModel(
                num_genes=num_genes,
                learning_rate=params.get("learning_rate", 1e-5))
            
            print("learning_rate", learning_rate)
            
            logger.info(f"THItoGene initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
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
        
        print("use_kd:", params.get("use_kd"))
        print("model:", params.get("model"))
        print("teacher:", teacher)

        if params.get("use_kd", False):
            if teacher is None:
                raise ValueError("KD enabled but teacher model not defined!")

            fold_name = f"fold_{fold_idx+1}"
    
            teacher_path = os.path.join(
                params["teacher_base_path"],
                fold_name,
                "best_model.pth"
            )
            
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

            logger.info(f"[Fold {fold_idx+1}] Loading teacher: {teacher_path}")

            checkpoint = torch.load(teacher_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                teacher.load_state_dict(checkpoint["model_state_dict"])
            else:
                teacher.load_state_dict(checkpoint)

            teacher = teacher.to(device)
            teacher.eval()

            for p in teacher.parameters():
                p.requires_grad = False
            
        mse_criterion = torch.nn.MSELoss()
        kd_criterion = KDLoss()

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
                # ---- Unpack THItoGene batch ----
                
                images = images.to(device)             # shape: (B, 3, 224, 224)
                y_true = targets.to(device)

                # Forward
                y_pred = model(images)         

                
                # ---- Sanity checks ----
                if not torch.isfinite(images).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in input patches. Skipping batch.")
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch.")
                    continue
                

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
                
                teacher_pred = None

                if params.get("use_kd", False):
                    with torch.no_grad():
                        teacher_pred = teacher(images)

                    if not torch.isfinite(teacher_pred).all():
                        logger.warning(f"[Batch {batch_idx}] Teacher produced NaNs. Skipping KD.")
                        teacher_pred = None
            
                loss_fn_name = params.get("loss_fn", "mse")
                if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                    loss = mse_criterion(y_pred, y_true)
                    
                elif loss_fn_name == "kd_loss":
                    loss = kd_criterion(y_pred, teacher_pred, y_true)
                    
                # elif loss_fn_name == "dist_loss":
                #     loss = distil_loss(y_pred, teacher_pred, y_true)
                elif loss_fn_name =="pearson_loss":
                    
                    def pearson_loss(pred, target):
                        pred = pred - pred.mean(dim=1, keepdim=True)
                        target = target - target.mean(dim=1, keepdim=True)
                        numerator = (pred * target).sum(dim=1)
                        denominator = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
                        loss = 1 - numerator / denominator
                        return loss.mean()
                    
                    mse = F.mse_loss(y_pred, y_true)
                    loss = 0.5 * pearson_loss(y_pred, y_true) + 0.5 * mse

                # elif loss_fn_name == "gene_weighted_loss":
                #     loss = gene_weighted_mse_loss(y_pred, y_true)
                                  
                else:
                    logger.error("Loss Function not defined!!!")
                    raise ValueError("Loss Function not defined")

                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()

                epoch_train_loss += float(loss.item())

                # ---- Remove batch dimension (THItoGene gives (1, N, G)) ----
                y_true_2d = y_true.squeeze(0)   # (N_spots, G)
                y_pred_2d = y_pred.squeeze(0)   # (N_spots, G)

                # ---- Collect for epoch metrics (flatten over spots) ----
                all_y_true.append(y_true_2d.detach().cpu().numpy())
                all_y_pred.append(y_pred_2d.detach().cpu().numpy())

                # ---- Pearson (cell-wise flattened) ----
                try:
                    batch_pearson = pearsonr(
                        y_pred_2d.detach().cpu().numpy().flatten(),
                        y_true_2d.detach().cpu().numpy().flatten()
                    )[0]
                    print(f"Batch {batch_idx} Pearson correlation: {batch_pearson:.4f}")
                except Exception as e:
                    print("Pearson Exception:", e)
                    batch_pearson = 0.0

                pearson_train_corr.append(float(batch_pearson))

                # ---- Spearman (your custom function expects 2D) ----
                try:
                    batch_spearman = spearmanrr(y_pred_2d, y_true_2d)
                    spearman_train_corr.append(float(batch_spearman.mean().item()))
                except Exception as e:
                    print("Spearman Exception:", e)
                    spearman_train_corr.append(0.0)

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, "
                        f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                        f"Pearson: {pearson_train_corr[-1]:.4f}, "
                        f"Spearman: {spearman_train_corr[-1]:.4f}"
                    )
            # epoch-level aggregation
            all_y_true = np.concatenate(all_y_true, axis=0) if len(all_y_true) > 0 else np.array([])
            all_y_pred = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) > 0 else np.array([])
            
            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")

            print(f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}")
            results_train = compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}

            
            del loss, y_pred, y_true,images
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
                    images = images.to(device)                          # shape: (B, 3, 224, 224)
                    y_true = targets.to(device)                         # shape: (B, num_genes)

                    y_pred = model(images)

                    y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)


                    loss_fn_name = params.get("loss_fn", "")
                    if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                        loss = mse_criterion(y_pred, y_true)
                        
                    elif loss_fn_name == "kd_loss":
                        loss = mse_criterion(y_pred, y_true)
                    
                    # elif loss_fn_name == "dist_loss":
                    #     if "teacher" not in globals() and teacher is None:
                    #         raise RuntimeError("Teacher model is not loaded for KD evaluation.")
                        
                    #     teacher_pred = teacher(images)
                        
                    #     loss = distil_loss(y_pred, teacher_pred, y_true)    
                    
                    elif loss_fn_name =="pearson_loss":
                    
                        def pearson_loss(pred, target):
                            pred = pred - pred.mean(dim=1, keepdim=True)
                            target = target - target.mean(dim=1, keepdim=True)
                            numerator = (pred * target).sum(dim=1)
                            denominator = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
                            loss = 1 - numerator / denominator
                            return loss.mean()
                        
                        mse = F.mse_loss(y_pred, y_true)
                        loss = 0.5 * pearson_loss(y_pred, y_true) + 0.5 * mse
                        
                    # elif loss_fn_name == "gene_weighted_loss":
                    #     loss = gene_weighted_mse_loss(y_pred, y_true)
          
                    else:
                        logger.error("Loss Function not defined!!!")
                        raise ValueError("Loss Function not defined")

                    epoch_val_loss += float(loss.item())
                    # ---- Remove batch dim for metrics ----
                    y_true_2d = y_true.squeeze(0)   # (N_spots, G)
                    y_pred_2d = y_pred.squeeze(0)   # (N_spots, G)

                    val_y_true.append(y_true_2d.detach().cpu().numpy())
                    val_y_pred.append(y_pred_2d.detach().cpu().numpy())

                    # ---- Pearson (flattened) ----
                    try:
                        batch_pearson = pearsonr(
                            y_pred_2d.detach().cpu().numpy().flatten(),
                            y_true_2d.detach().cpu().numpy().flatten()
                        )[0]
                    except Exception as e:
                        print("Pearson Exception (val):", e)
                        batch_pearson = 0.0

                    pearson_val_corr.append(float(batch_pearson))

                    # ---- Spearman (2D) ----
                    try:
                        batch_spearman = spearmanrr(y_pred_2d, y_true_2d)
                        spearman_val_corr.append(float(batch_spearman.mean().item()))
                    except Exception as e:
                        print("Spearman Exception (val):", e)
                        spearman_val_corr.append(0.0)

                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, "
                            f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                            f"Pearson: {pearson_val_corr[-1]:.4f}, "
                            f"Spearman: {spearman_val_corr[-1]:.4f}"
                        )

                # ---- Epoch-level aggregation (concat, not vstack!) ----
                val_y_true = np.concatenate(val_y_true, axis=0) if len(val_y_true) > 0 else np.array([])
                val_y_pred = np.concatenate(val_y_pred, axis=0) if len(val_y_pred) > 0 else np.array([])


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

        del model, teacher, optimizer, scheduler, train_dataloader, val_dataloader
        torch.cuda.empty_cache()
        gc.collect()
        return None

    if params["dataset_name"] == "cscc":
        gsm_samples = [ 'GSM4284316_P2','GSM4284317_P2','GSM4284318_P2',
                        'GSM4284319_P5','GSM4284320_P5','GSM4284321_P5',
                        'GSM4284322_P9','GSM4284323_P9','GSM4284324_P9',
                        'GSM4284325_P10','GSM4284326_P10','GSM4284327_P10']
        
    if params["dataset_name"] == "hist2st_845":
        gsm_samples = [
            '23209_C1', '23209_C2', '23209_D1', '23268_C1', '23268_C2', '23268_D1', '23269_C1', '23269_C2',
            '23269_D1', '23270_D2', '23270_E1', '23270_E2', '23272_D2', '23272_E1', '23272_E2', '23277_D2',
            '23277_E1', '23277_E2', '23287_C1', '23287_C2', '23287_D1', '23288_D2', '23288_E1', '23288_E2',
            '23377_C1', '23377_C2', '23377_D1', '23450_D2', '23450_E1', '23450_E2', '23506_C1', '23506_C2',
            '23506_D1', '23508_D2', '23508_E1', '23508_E2', '23567_D2', '23567_E1', '23567_E2', '23803_D2',
            '23803_E1', '23803_E2', '23810_D2', '23810_E1', '23810_E2', '23895_C1', '23895_C2', '23895_D1',
            '23901_C2', '23901_D1', '23903_C1', '23903_C2', '23903_D1', '23944_D2', '23944_E1', '23944_E2',
            '24044_D2', '24044_E1', '24044_E2', '24105_C1', '24105_C2', '24105_D1', '24220_D2', '24220_E1',
            '24220_E2', '24223_D2', '24223_E1', '24223_E2'
        ]

    if params["dataset_name"] == "her2":
        gsm_samples =  [
            "A1", "A2", "A3", "A4", "A5", "A6",
            "B1", "B2", "B3", "B4", "B5", "B6",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "D1", "D2", "D3", "D4", "D5", "D6",
            "E1", "E2", "E3",
            "F1", "F2", "F3",
            "G1", "G2", "G3",
            "H1", "H2", "H3"
        ]
        # gsm_samples =  [
        #     "A1", "A2", "A3", "A4", "A5", "A6"
        # ]      

    
    def get_patient_id(sample, dataset_name):
        if dataset_name == "cscc":
            # GSM4284316_P2 → P2
            return sample.split("_")[1]

        elif dataset_name == "hist2st_845":
            # 23209_C1 → 23209
            return sample.split("_")[0]

        elif dataset_name == "her2":
            # A1 → A
            return sample[0]

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        
    # --- Define paths ---
    dataset_path = params.get("dataset_path", "")

    if params["dataset_name"] == "cscc":
        patch_paths = [os.path.join(dataset_path, f"{sample}_patches.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}_spots.h5ad") for sample in gsm_samples]
    if params["dataset_name"] == "her2":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]
    if params["dataset_name"] == "hist2st_845":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]
    logger.info(f"Total samples: {len(gsm_samples)}")
    for p, a in zip(patch_paths, adata_paths):
        logger.info(f"  - {os.path.basename(p)} ↔ {os.path.basename(a)}")

    # --- Check file existence ---
    for p, a in zip(patch_paths, adata_paths):
        if not os.path.exists(p):
            logger.warning(f"Missing patch file: {p}")
        if not os.path.exists(a):
            logger.warning(f"Missing spot file: {a}")
            
    
    # --- Patient grouping (for leakage-free CV) ---
    patient_ids = [get_patient_id(s, params["dataset_name"]) for s in gsm_samples]
    unique_patients = sorted(set(patient_ids))

    logger.info(f"Total patients: {len(unique_patients)} → {unique_patients}")


    # --- Gene CSV file ---
    genes_file = os.path.join(params['genes'], f"{params['dataset_name']}.npy")
    print("+++++++++++++++++++++++++++++++",genes_file)
    gene_names = np.load(genes_file, allow_pickle=True).tolist()
    logger.info(f" Using gene list from: {gene_names}")
    if not os.path.exists(genes_file):
        # raise FileNotFoundError(f"Gene list not found: {genes_file}")
        print(genes_file)
        print(gene_names)

    # --- Optional image transforms ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
 
    # --- K-Fold setup ---
    k = params.get("k_folds", 5)
    n_patients = len(unique_patients)

    if k > n_patients:
        logger.warning(
            f"k_folds ({k}) > number of patients ({n_patients}). "
            f"Setting k = {n_patients}"
        )
        k = n_patients

    kf = GroupKFold(n_splits=k, shuffle=True, random_state=42)

    fold_models = []

    # --- K-Fold Loop ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(gsm_samples, groups=patient_ids)):
        logger.info(f"\n===== Starting Fold {fold_idx+1}/{k} =====")
        
        train_patients = sorted(set(patient_ids[i] for i in train_idx))
        val_patients   = sorted(set(patient_ids[i] for i in val_idx))

        logger.info(f"Train patients: {len(train_patients)}, {train_patients}")
        logger.info(f"Val patients: {len(val_patients)},  {val_patients}")


        train_patch_files = [patch_paths[i] for i in train_idx]
        train_adata_files = [adata_paths[i] for i in train_idx]
        val_patch_files   = [patch_paths[i] for i in val_idx]
        val_adata_files   = [adata_paths[i] for i in val_idx]

        logger.info(f"Fold {fold_idx+1}: Train={len(train_patch_files)}, Val={len(val_patch_files)}")  
        
        train_datasets_list = [
            PatchDataset(
                gene_path=adata_p,
                img_path=img_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000)
            )
            for adata_p, img_p in zip(train_adata_files, train_patch_files)
        ]

        val_datasets_list = [
            PatchDataset(
                gene_path=adata_p,
                img_path=img_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000)
            )
            for adata_p, img_p in zip(val_adata_files, val_patch_files)
        ]

        # --- Combine datasets ---

        train_dataset_concat = ConcatDataset(train_datasets_list)
        val_dataset_concat   = ConcatDataset(val_datasets_list)

        logger.info(f"Fold {fold_idx+1}: Train={len(train_dataset_concat)}, Val={len(val_dataset_concat)}")

        # --- Create fold directory ---
        fold_dir = os.path.join(model_save_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # --- Train model for this fold ---
        models = train_regression(
            train_dataset_concat,
            val_dataset_concat,
            fold_idx,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"]
        )

        del train_dataset_concat, train_datasets_list
        del val_dataset_concat, val_datasets_list

        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"===== Completed Fold {fold_idx+1}/{k} =====")

    # --- Save summary ---
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")

    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
