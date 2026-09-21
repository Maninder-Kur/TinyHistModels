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
    """Apply log1p normalization to the given array."""
    eps = 1e-8  # small constant to avoid divide-by-zero
    return np.log1p((arr / (np.sum(arr, axis=1, keepdims=True) + eps)) * scale_factor)


# class PatchDataset(Dataset):
#     def __init__(self, gene_path, img_path, gene_names=None, transform=None, log_norm=True, scale_factor=1000000):
#         self.img_path = img_path
#         self.gene_path = gene_path
#         self.gene_names = gene_names
#         self.transform = transform
#         self.log_norm = log_norm
#         self.scale_factor = scale_factor

#         print("Initializing PatchDataset...")
#         print(f"Image file: {self.img_path}")
#         print(f"Gene file:  {self.gene_path}")


#         # === Load image data from HDF5 ===
#         with h5py.File(self.img_path, "r") as f:
#             # check for possible keys like "patches" or "img"
#             if "patches" in f:
#                 self.image = f["patches"][:]
#             elif "img" in f:
#                 self.image = f["img"][:]
#             else:
#                 raise KeyError(f"No valid image dataset found in {self.img_path}")
#         print(f" Loaded images: {self.image.shape}")

#         # === Load gene expression data ===
#         adata = sc.read_h5ad(self.gene_path)

#         # --- If a gene list is provided, subset to those genes ---
#         if self.gene_names:
#             # gene_list = pd.read_csv(self.gene_csv).iloc[:, 0].tolist()
#             gene_list = self.gene_names
#             valid_genes = [g for g in gene_list if g in adata.var_names]
#             adata = adata[:, valid_genes].copy()
#             print(f" Using {len(valid_genes)} valid genes from gene list ({len(gene_list)} total).")

#         # --- Convert expression matrix to dense ---
#         gene_exp = adata.X
#         if not isinstance(gene_exp, np.ndarray):
#             gene_exp = gene_exp.toarray()

#         # --- Apply log1p normalization ---
#         if self.log_norm:
#             print(" Applying custom log1p normalization...")
#             gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
#             print(f" Applied log1p normalization with scale_factor={self.scale_factor}.")

#         self.gene_data = gene_exp
#         print(f" Final gene matrix shape: {self.gene_data.shape}")

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, idx):
#         image = self.image[idx].astype(np.uint8)

#         # Apply transformation if provided (resize, normalization, etc.)
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

#         # Corresponding gene expression vector
#         gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)

#         return image, gene_exp


##############################################patch_new###############################################

class PatchDataset(Dataset):

    def __init__(
        self,
        gene_path,
        img_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        align_by_barcode=True
    ):

        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform
        self.log_norm = log_norm
        self.scale_factor = scale_factor

        print("=" * 60)
        print("Initializing PatchDataset")
        print("=" * 60)

        print(f"Image file : {self.img_path}")
        print(f"Gene file  : {self.gene_path}")

        ############################################################
        # LOAD IMAGE FILE
        ############################################################

        with h5py.File(self.img_path, "r") as f:

            print("\nAvailable HDF5 keys:")
            print(list(f.keys()))

            # ------------------------------------------------------
            # Flexible image key support
            # ------------------------------------------------------

            possible_img_keys = [
                "patches",
                "img",
                "images"
            ]

            img_key = None

            for k in possible_img_keys:
                if k in f:
                    img_key = k
                    break

            if img_key is None:
                raise KeyError(
                    f"No valid image dataset found.\n"
                    f"Available keys: {list(f.keys())}"
                )

            self.image = f[img_key][:]

            print(f"\nUsing image key : {img_key}")
            print(f"Image shape     : {self.image.shape}")
            print(f"Image dtype     : {self.image.dtype}")

            # ------------------------------------------------------
            # Optional barcode loading
            # ------------------------------------------------------

            self.img_barcodes = None

            if "barcodes" in f:

                self.img_barcodes = [
                    b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    for b in f["barcodes"][:]
                ]

                print(f"Loaded {len(self.img_barcodes)} image barcodes")

        ############################################################
        # LOAD GENE FILE
        ############################################################

        print("\nLoading AnnData...")

        adata = sc.read_h5ad(self.gene_path)

        print(f"AnnData shape : {adata.shape}")

        # ----------------------------------------------------------
        # IMPORTANT:
        # Make duplicate gene names unique
        # ----------------------------------------------------------

        adata.var_names_make_unique()

        ############################################################
        # OPTIONAL GENE SUBSETTING
        ############################################################

        if self.gene_names is not None:

            valid_genes = [
                g for g in self.gene_names
                if g in adata.var_names
            ]

            print(
                f"\nKeeping {len(valid_genes)} / "
                f"{len(self.gene_names)} genes"
            )

            adata = adata[:, valid_genes].copy()

        ############################################################
        # ALIGN IMAGE + GENE DATA USING BARCODES
        ############################################################

        if align_by_barcode and self.img_barcodes is not None:

            print("\nAligning image and gene data using barcodes...")

            gene_barcodes = adata.obs_names.tolist()

            barcode_to_idx = {
                bc: i for i, bc in enumerate(gene_barcodes)
            }

            matched_indices = []

            missing_barcodes = []

            for bc in self.img_barcodes:

                if bc in barcode_to_idx:
                    matched_indices.append(barcode_to_idx[bc])

                else:
                    missing_barcodes.append(bc)

            if len(missing_barcodes) > 0:

                raise ValueError(
                    f"{len(missing_barcodes)} barcodes "
                    f"missing in AnnData.\n"
                    f"Example: {missing_barcodes[:5]}"
                )

            adata = adata[matched_indices].copy()

            print("Barcode alignment complete")

        ############################################################
        # EXTRACT GENE MATRIX
        ############################################################

        print("\nExtracting gene matrix...")

        gene_exp = adata.X

        # Sparse -> Dense
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        gene_exp = gene_exp.astype(np.float32)

        print(f"Gene matrix shape : {gene_exp.shape}")

        ############################################################
        # NORMALIZATION
        ############################################################

        if self.log_norm:

            print("\nApplying log1p normalization...")

            gene_exp = log1p_normalization(
                gene_exp,
                scale_factor=self.scale_factor
            )

            print(
                f"Normalization complete "
                f"(scale_factor={self.scale_factor})"
            )

        self.gene_data = gene_exp

        ############################################################
        # FINAL SAFETY CHECK
        ############################################################

        if len(self.image) != len(self.gene_data):

            raise ValueError(
                f"\nImage count ({len(self.image)}) != "
                f"Gene count ({len(self.gene_data)})"
            )

        print("\nFinal dataset summary")
        print("-" * 40)
        print(f"Images      : {self.image.shape}")
        print(f"Gene matrix : {self.gene_data.shape}")
        print(f"Dataset size: {len(self.image)}")

        print("=" * 60)

    ############################################################
    # DATASET LENGTH
    ############################################################

    def __len__(self):
        return len(self.image)

    ############################################################
    # GET ITEM
    ############################################################

    def __getitem__(self, idx):

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        image = self.image[idx].astype(np.uint8)

        if self.transform:

            image = self.transform(image)

        else:

            image = (
                torch.tensor(
                    image,
                    dtype=torch.float32
                )
                .permute(2, 0, 1) / 255.0
            )

        # --------------------------------------------------------
        # GENE EXPRESSION
        # --------------------------------------------------------

        gene_exp = torch.tensor(
            self.gene_data[idx],
            dtype=torch.float32
        )

        return image, gene_exp

# class PatchDataset_2(Dataset):
#     def __init__(self, gene_path, img_path, gene_names=None, transform=None, log_norm=True, scale_factor=1000000):
#         self.img_path = img_path
#         self.gene_path = gene_path
#         self.gene_names = gene_names
#         self.transform = transform
#         self.log_norm = log_norm
#         self.scale_factor = scale_factor

#         print("Initializing PatchDataset...")
#         print(f"Image file: {self.img_path}")
#         print(f"Gene file:  {self.gene_path}")


#         # === Load image data from HDF5 ===
#         with h5py.File(self.img_path, "r") as f:
#             # check for possible keys like "patches" or "img"
#             if "patches" in f:
#                 self.image = f["patches"][:]
#                 self.x = f["x"][:]
#                 self.y = f["y"][:]
#             elif "img" in f:
#                 self.image = f["img"][:]
#             else:
#                 raise KeyError(f"No valid image dataset found in {self.img_path}")
#         print(f" Loaded images: {self.image.shape}")

#         # === Load gene expression data ===
#         adata = sc.read_h5ad(self.gene_path)

#         # --- If a gene list is provided, subset to those genes ---
#         if self.gene_names:
#             # gene_list = pd.read_csv(self.gene_csv).iloc[:, 0].tolist()
#             gene_list = self.gene_names
#             valid_genes = [g for g in gene_list if g in adata.var_names]
#             adata = adata[:, valid_genes].copy()
#             print(f" Using {len(valid_genes)} valid genes from gene list ({len(gene_list)} total).")

#         # --- Convert expression matrix to dense ---
#         gene_exp = adata.X
#         if not isinstance(gene_exp, np.ndarray):
#             gene_exp = gene_exp.toarray()

#         # --- Apply log1p normalization ---
#         if self.log_norm:
#             print(" Applying custom log1p normalization...")
#             gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
#             print(f" Applied log1p normalization with scale_factor={self.scale_factor}.")

#         self.gene_data = gene_exp
#         print(f" Final gene matrix shape: {self.gene_data.shape}")

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, idx):
#         image = self.image[idx].astype(np.uint8)
#         x = self.x[idx].astype(np.uint8)
#         y = self.y[idx].astype(np.uint8)

#         # Apply transformation if provided (resize, normalization, etc.)
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

#         # Corresponding gene expression vector
#         gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)
#         x = torch.tensor(self.x[idx], dtype=torch.float32)
#         y = torch.tensor(self.y[idx], dtype=torch.float32)

#         return image, gene_exp, x, y