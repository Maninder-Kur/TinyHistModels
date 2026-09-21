import h5py
import anndata as ad
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import ConcatDataset


import torch
from torch.utils.data import Dataset
import h5py
import scanpy as sc
import numpy as np
import pandas as pd

def log1p_normalization(arr, scale_factor=1000000):
    """
    Apply log1p normalization to rows of arr:
      normalized_row = log1p( (row / row.sum()) * scale_factor )

    Handles rows with sum 0 by leaving them zero (avoids NaN).
    """
    arr = np.asarray(arr, dtype=np.float64)
    lib = arr.sum(axis=1, keepdims=True)
    # avoid divide by zero
    lib[lib == 0] = 1.0
    scaled = (arr / lib) * float(scale_factor)
    return np.log1p(scaled)

class PatchDataset(Dataset):
    def __init__(self, gene_path, img_path, gene_names=None, transform=None):
        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform

        print("Initializing PatchDataset...")
        print(f"Image file: {self.img_path}")
        print(f"Gene file:  {self.gene_path}")

        # === Load images ===
        with h5py.File(self.img_path, "r") as f:
            if "patches" in f:
                self.image = f["patches"][:]
            elif "img" in f:
                self.image = f["img"][:]
            else:
                raise KeyError(f"No valid image dataset found in {self.img_path}")
        print(f" Loaded images: {self.image.shape}")

        # ============================================================
        # LOAD GENE EXPRESSION
        # ============================================================

        adata = sc.read_h5ad(self.gene_path)

        print(f"\nLoaded AnnData: {adata.shape}")

        # ============================================================
        # KEEP ONLY BIOLOGICAL GENES
        # ============================================================

        if "feature_types" in adata.var.columns:

            print("\nFeature type counts:")
            print(
                adata.var["feature_types"]
                .value_counts()
            )

            adata = adata[
                :,
                adata.var["feature_types"] == "Gene Expression"
            ].copy()

            print(
                f"\nFiltered to biological genes only: "
                f"{adata.shape}"
            )

        # ============================================================
        # OPTIONAL GENE SUBSETTING
        # ============================================================
        if gene_names:

            valid_genes = [
                g for g in gene_names
                if g in adata.var_names
            ]

            adata = adata[:, valid_genes].copy()

            print(
                f"\nAfter gene subsetting: "
                f"{adata.shape}"
            )

        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        # Store raw counts only
        self.gene_data = gene_exp.astype(np.float32)
        print(f" Raw gene matrix: {self.gene_data.shape}")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx].astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        gene_raw = torch.tensor(self.gene_data[idx], dtype=torch.float32)

        return img, gene_raw


class FoldDataset(Dataset):
    def __init__(self, image_data, gene_data, indices, transform=None):
        """
        image_data: numpy array of shape (N, H, W, 3)
        gene_data: numpy array of shape (N, G)
        indices: list or array of sample indices for this fold
        transform: torchvision transforms
        """
        self.image_data = image_data
        self.gene_data = gene_data
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = self.image_data[real_idx].astype(np.uint8)
        genes = torch.tensor(self.gene_data[real_idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, genes


class XeniumPatientDataset(Dataset):
    def __init__(
        self,
        gene_path,
        img_path,
        indices,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
    ):
        self.img_path = img_path
        self.gene_path = gene_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self._h5_file = None
        self._image_key = None
        self.selected_barcodes = None

        print("Initializing XeniumPatientDataset...")
        print(f"Image file: {self.img_path}")
        print(f"Gene file:  {self.gene_path}")
        print(f"Selected spots: {len(self.indices)}")

        with h5py.File(self.img_path, "r") as f:
            if "patches" in f:
                self._image_key = "patches"
            elif "img" in f:
                self._image_key = "img"
            else:
                raise KeyError(f"No valid image dataset found in {self.img_path}")

            n_images = f[self._image_key].shape[0]
            if len(self.indices) and self.indices.max() >= n_images:
                raise IndexError(
                    f"Selected index {self.indices.max()} is outside image dataset with {n_images} rows"
                )

            if "barcode" in f:
                raw_barcodes = f["barcode"][self.indices]
                self.selected_barcodes = [
                    x[0].decode() if isinstance(x[0], bytes) else str(x[0])
                    for x in raw_barcodes
                ]

        adata = sc.read_h5ad(self.gene_path)
        print(f"\nLoaded AnnData: {adata.shape}")

        if self.selected_barcodes is not None:
            available_barcodes = set(adata.obs_names)

            valid_mask = np.array(
                [barcode in available_barcodes for barcode in self.selected_barcodes],
                dtype=bool,
            )

            n_missing = int((~valid_mask).sum())
            if n_missing > 0:
                missing_barcodes = np.asarray(self.selected_barcodes)[~valid_mask]

                print(
                    f"WARNING: Skipping {n_missing} image spot(s) without gene expression "
                    f"from {self.gene_path}. First missing barcode: {missing_barcodes[0]}"
                )

                # Crucial: filter image indices and barcodes together, preserving alignment.
                self.indices = self.indices[valid_mask]
                self.selected_barcodes = list(
                    np.asarray(self.selected_barcodes)[valid_mask]
                )

            if len(self.indices) == 0:
                raise ValueError(
                    f"No selected image spots have matching gene expression in {self.gene_path}"
                )

            adata = adata[self.selected_barcodes, :].copy()
            print(f"Aligned AnnData to selected H5 barcodes: {adata.shape}")

        if "feature_types" in adata.var.columns:
            print("\nFeature type counts:")
            print(adata.var["feature_types"].value_counts())
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            print(f"\nFiltered to biological genes only: {adata.shape}")

        if gene_names:
            valid_genes = [g for g in gene_names if g in adata.var_names]
            adata = adata[:, valid_genes].copy()
            print(f"\nAfter gene subsetting: {adata.shape}")

        gene_exp = adata.X if self.selected_barcodes is not None else adata.X[self.indices]
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        gene_exp = gene_exp.astype(np.float32)
        if log_norm:
            gene_exp = log1p_normalization(gene_exp, scale_factor=scale_factor).astype(np.float32)

        self.gene_data = gene_exp
        self.num_genes = self.gene_data.shape[1]
        print(f"Gene matrix for selected spots: {self.gene_data.shape}")

    def __len__(self):
        return len(self.indices)

    def _images(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.img_path, "r")
        return self._h5_file[self._image_key]

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        img = self._images()[real_idx].astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        genes = torch.tensor(self.gene_data[idx], dtype=torch.float32)
        return img, genes

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
