import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))

import h5py
import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset
from THItoGene.graph_construction import calcADJ

from sklearn.neighbors import NearestNeighbors
from torchvision import transforms


def log1p_normalization(arr, scale_factor=1000000):
    """Apply log1p normalization to the given array."""
    eps = 1e-8
    return np.log1p((arr / (np.sum(arr, axis=1, keepdims=True) + eps)) * scale_factor)


class THItoGeneH5Dataset(Dataset):
    """
    Final THItoGene-compatible dataset:
      - aligns patches ↔ coords ↔ gene expression by spot_id
      - uses normalized tissue coords for embeddings
      - builds adjacency on aligned spots
    Returns:
      patches, centers, exps, adj
    """

    def __init__(
        self,
        h5_img_path,
        h5ad_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        train=True,
        k_nn=4,
        n_pos=64,   # must match THItoGene n_pos
    ):
        self.h5_img_path = h5_img_path
        self.h5ad_path = h5ad_path
        self.gene_names = gene_names
        self.transform = transform
        self.log_norm = log_norm
        self.scale_factor = scale_factor
        self.train = train
        self.k_nn = k_nn
        self.n_pos = n_pos

        # ---- Load H5 patches + coords + spot_ids ----
       
        # ---- Load H5 patches + coords + spot_ids ----
        with h5py.File(self.h5_img_path, "r") as f:
            patches_all = f["patches"][:]  # (N_all, 224, 224, 3)
            spot_ids_h5 = [s.decode("utf-8") for s in f["spot_id"][:]]

            coords_grp = f["coords"]
            raw_xcoord = [x.decode("utf-8") for x in coords_grp["xcoord"][:]]
            raw_ycoord = [y.decode("utf-8") for y in coords_grp["ycoord"][:]]

        # ---- Parse numeric tissue coords ----
        x_vals_all, y_vals_all = [], []
        for s in raw_xcoord:
            parts = s.split("_")
            x_vals_all.append(float(parts[1]))
            y_vals_all.append(float(parts[2]))

        x_vals_all = np.array(x_vals_all, dtype=np.float32)
        y_vals_all = np.array(y_vals_all, dtype=np.float32)

        # ---- Load gene expression ----
        adata = sc.read_h5ad(self.h5ad_path)
        spot_ids_adata = adata.obs_names.astype(str).tolist()

        # ---- Build safe index maps ----
        valid_h5_indices = []
        valid_adata_indices = []

        h5_id_to_idx = {sid: i for i, sid in enumerate(spot_ids_h5)}
        adata_id_to_idx = {sid: i for i, sid in enumerate(spot_ids_adata)}

        for sid in spot_ids_h5:
            if sid in adata_id_to_idx:
                h5_i = h5_id_to_idx[sid]

                # 🔐 Critical safety check
                if h5_i < len(x_vals_all) and h5_i < len(y_vals_all):
                    valid_h5_indices.append(h5_i)
                    valid_adata_indices.append(adata_id_to_idx[sid])

        print(f"[THItoGeneH5Dataset] Aligned spots: {len(valid_h5_indices)}")

        # ---- Final aligned tensors ----
        self.patches_np = patches_all[valid_h5_indices]
        x_vals = x_vals_all[valid_h5_indices]
        y_vals = y_vals_all[valid_h5_indices]
        # ---- Subset genes to gene_names (IMPORTANT) ----
        if self.gene_names is not None:
            valid_genes = [g for g in self.gene_names if g in adata.var_names]
            print(f"[THItoGeneH5Dataset] Using {len(valid_genes)} genes from provided list.")
            adata = adata[:, valid_genes].copy()

        gene_exp = adata.X[valid_adata_indices]
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        print(f"[THItoGeneH5Dataset] Loaded patches after alignment: {self.patches_np.shape}")

        # ---- Log1p normalization ----
        if self.log_norm:
            print("[THItoGeneH5Dataset] Applying custom log1p normalization...")
            gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
            print(f"[THItoGeneH5Dataset] Applied log1p normalization with scale_factor={self.scale_factor}.")

        self.exps_np = gene_exp
        print(f"[THItoGeneH5Dataset] Final gene matrix shape: {self.exps_np.shape}")

        # ---- Normalize coords into embedding indices [0, n_pos-1] ----
        x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-8)
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)

        x_idx = (x_norm * (self.n_pos - 1)).astype(np.int64)
        y_idx = (y_norm * (self.n_pos - 1)).astype(np.int64)

        self.centers_np = np.stack([x_idx, y_idx], axis=1)
        self.positions_np = self.centers_np.copy()

        assert self.centers_np.min() >= 0 and self.centers_np.max() < self.n_pos, \
            f"Center indices out of range: min={self.centers_np.min()}, max={self.centers_np.max()}"

        # ---- Build adjacency ----
        self.adj = calcADJ(self.positions_np, k=self.k_nn, pruneTag='NA')

        # ---- Final sanity check ----
        N = self.patches_np.shape[0]
        assert self.exps_np.shape[0] == N and self.centers_np.shape[0] == N, \
            "Mismatch between patches, centers, and gene expression"

    def __len__(self):
        return 1  # one graph per slide

    def __getitem__(self, idx):
        patches_list = []
        for i in range(self.patches_np.shape[0]):
            img = self.patches_np[i]  # (224,224,3)

            if self.transform is not None:
                img = self.transform(img)  # (3,112,112)
            else:
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

            patches_list.append(img)

        patches = torch.stack(patches_list, dim=0)          # (N, 3, 112, 112)
        centers = torch.from_numpy(self.centers_np).long()  # (N, 2)
        exps = torch.from_numpy(self.exps_np).float()       # (N, G)
        adj = self.adj.float()                              # (N, N)

        return patches, centers, exps, adj
    

class THItoGeneHER2Dataset(Dataset):
    """
    THItoGene-compatible dataset for HER2 H5 format:
      - patches: (N, H, W, 3)
      - x, y    : normalized grid coords
      - genes   : from h5ad
    Returns:
      patches, centers, exps, adj
    """

    def __init__(
        self,
        h5_img_path,
        h5ad_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        k_nn=4,
        n_pos=64,   # must match THItoGene n_pos
    ):
        self.transform = transform
        self.log_norm = log_norm
        self.scale_factor = scale_factor
        self.k_nn = k_nn
        self.n_pos = n_pos

        # ---- Load H5 ----
        with h5py.File(h5_img_path, "r") as f:
            self.patches_np = f["patches"][:]   # (N, H, W, 3)
            x_vals = f["x"][:].astype(np.float32)
            y_vals = f["y"][:].astype(np.float32)

        print(f"[THItoGeneHER2Dataset] Loaded patches: {self.patches_np.shape}")

        # ---- Load gene expression ----
        adata = sc.read_h5ad(h5ad_path)

        if gene_names is not None:
            valid = [g for g in gene_names if g in adata.var_names]
            adata = adata[:, valid].copy()
            print(f"[THItoGeneHER2Dataset] Using {len(valid)} genes from provided list.")

        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        if self.log_norm:
            print("[THItoGeneHER2Dataset] Applying log1p normalization...")
            gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)

        self.exps_np = gene_exp
        print(f"[THItoGeneHER2Dataset] Final gene matrix shape: {self.exps_np.shape}")

        # ---- Normalize coords → embedding indices ----
        x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-8)
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)

        x_idx = (x_norm * (self.n_pos - 1)).astype(np.int64)
        y_idx = (y_norm * (self.n_pos - 1)).astype(np.int64)

        self.centers_np = np.stack([x_idx, y_idx], axis=1)

        # ---- Build adjacency on grid coords ----
        coords_for_adj = np.stack([x_vals, y_vals], axis=1)
        self.adj = calcADJ(coords_for_adj, k=self.k_nn, pruneTag='NA')

        # ---- Sanity check ----
        N = self.patches_np.shape[0]
        assert self.exps_np.shape[0] == N, "Mismatch between patches and gene rows"

    def __len__(self):
        return 1  # one slide = one graph

    def __getitem__(self, idx):
        patches_list = []
        for i in range(self.patches_np.shape[0]):
            img = self.patches_np[i]  # (H, W, 3)

            if self.transform is not None:
                img = self.transform(img)  # (3, 112, 112)
            else:
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

            patches_list.append(img)

        patches = torch.stack(patches_list, dim=0)            # (N, 3, 112, 112)
        centers = torch.from_numpy(self.centers_np).long()    # (N, 2)
        exps = torch.from_numpy(self.exps_np).float()         # (N, G)
        adj = self.adj.float()                                # (N, N)

        return patches, centers, exps, adj
    

class THItoGeneXeniumDataset(Dataset):
    """
    Xenium dataset adapter for THItoGene using LOCAL subgraphs.
    Each __getitem__ returns a local spatial neighborhood (subgraph) instead of full tissue.

    Returns:
        patches: (N_local, 3, H, W)
        centers: (N_local, 2)  [normalized spatial coords mapped to [0, n_pos-1]]
        exps:    (N_local, G)
        adj:     (N_local, N_local)
    """

    def __init__(
        self,
        h5_img_path,
        h5ad_path,
        gene_names=None,
        transform=None,
        n_neighbors=128,     # size of local subgraph
        k_nn=4,
        n_pos=64,            # must match THItoGene's positional embedding size
        anchor_indices=None,
        log_norm=True,
        scale_factor=1000000,
    ):
        self.h5_img_path = h5_img_path
        self.h5ad_path = h5ad_path
        self.gene_names = gene_names
        self.transform = transform
        self.n_neighbors = n_neighbors
        self.k_nn = k_nn
        self.n_pos = n_pos
        self._h5_file = None

        print("Loading Xenium H5...")
        with h5py.File(self.h5_img_path, "r") as f:
            image_key = "patches" if "patches" in f else "img"
            coord_key = "centroid_um" if "centroid_um" in f else "coords"
            if image_key not in f or coord_key not in f:
                raise KeyError(
                    "Xenium H5 must contain patches/img and centroid_um/coords datasets"
                )
            # Do not load all image patches: a 20k-cell Xenium slide can take
            # multiple GB.  Patches are read only for the current local graph.
            n_h5_cells = f[image_key].shape[0]
            self._image_key = image_key
            self.centroid_um = f[coord_key][:].astype(np.float32)
            h5_barcodes = None
            if "barcode" in f:
                h5_barcodes = [
                    (x[0] if isinstance(x, np.ndarray) else x).decode()
                    if isinstance((x[0] if isinstance(x, np.ndarray) else x), bytes)
                    else str(x[0] if isinstance(x, np.ndarray) else x)
                    for x in f["barcode"][:]
                ]

        print("Loading Xenium gene expression...")
        adata = sc.read_h5ad(self.h5ad_path)

        if h5_barcodes is not None:
            adata_index = {barcode: i for i, barcode in enumerate(adata.obs_names.astype(str))}
            keep_h5 = [i for i, barcode in enumerate(h5_barcodes) if barcode in adata_index]
            if not keep_h5:
                raise ValueError("No H5 barcodes matched AnnData obs_names")
            adata = adata[[adata_index[h5_barcodes[i]] for i in keep_h5]].copy()
            self.centroid_um = self.centroid_um[keep_h5]
            self.h5_indices = np.asarray(keep_h5, dtype=np.int64)
            print(f"Aligned {len(keep_h5)} Xenium cells by barcode.")
        else:
            self.h5_indices = np.arange(n_h5_cells, dtype=np.int64)

        if "feature_types" in adata.var.columns:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()

        if self.gene_names is not None:
            valid = [g for g in self.gene_names if g in adata.var_names]
            if not valid:
                raise ValueError("None of the requested genes are present in this patient")
            adata = adata[:, valid].copy()
            print(f"Using {len(valid)} genes.")

        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        self.gene_exp = gene_exp.astype(np.float32)
        if log_norm:
            self.gene_exp = log1p_normalization(
                self.gene_exp, scale_factor=scale_factor
            ).astype(np.float32)

        assert self.h5_indices.shape[0] == self.centroid_um.shape[0] == self.gene_exp.shape[0], \
            "Mismatch between patches, coords, and gene expression rows"

        self.N = self.h5_indices.shape[0]
        self.num_genes = self.gene_exp.shape[1]
        if self.N < 2:
            raise ValueError("At least two aligned cells are required for a local graph")
        self.n_neighbors = min(self.n_neighbors, self.N)
        if anchor_indices is None:
            self.anchor_indices = np.arange(self.N, dtype=np.int64)
        else:
            self.anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
            if self.anchor_indices.ndim != 1 or len(self.anchor_indices) == 0:
                raise ValueError("anchor_indices must be a non-empty one-dimensional array")
            if self.anchor_indices.min() < 0 or self.anchor_indices.max() >= self.N:
                raise IndexError("anchor_indices contain values outside the aligned Xenium cells")
        print(f"Loaded Xenium: {self.N} cells, {len(self.anchor_indices)} anchor graphs")

        # Pre-build KNN index for fast subgraph sampling
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.nn.fit(self.centroid_um)

        # Normalize coords globally (only for positional embedding indices)
        x = self.centroid_um[:, 0]
        y = self.centroid_um[:, 1]

        self.x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        self.y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    def __len__(self):
        return len(self.anchor_indices)  # each anchor is a local subgraph

    def _image_dataset(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_img_path, "r")
        return self._h5_file[self._image_key]

    def _read_patches(self, aligned_indices):
        """Read a local graph's images while preserving KNN neighbour order."""
        source_indices = self.h5_indices[aligned_indices]
        order = np.argsort(source_indices)
        inverse_order = np.argsort(order)
        sorted_patches = self._image_dataset()[source_indices[order]]
        return sorted_patches[inverse_order]

    def __getitem__(self, idx):
        # ---- Sample local neighborhood ----
        anchor_idx = int(self.anchor_indices[idx])
        center_coord = self.centroid_um[anchor_idx:anchor_idx + 1]
        _, nbr_idx = self.nn.kneighbors(center_coord, return_distance=True)
        nbr_idx = nbr_idx[0]

        patches_local = self._read_patches(nbr_idx)  # (N_local, H, W, 3)
        genes_local   = self.gene_exp[nbr_idx]       # (N_local, G)

        # ---- Build positional embeddings ----
        x_vals = self.x_norm[nbr_idx]
        y_vals = self.y_norm[nbr_idx]

        x_idx = (x_vals * (self.n_pos - 1)).astype(np.int64)
        y_idx = (y_vals * (self.n_pos - 1)).astype(np.int64)

        centers = np.stack([x_idx, y_idx], axis=1)

        # ---- Build adjacency (unchanged calcADJ) ----
        adj = calcADJ(centers, k=self.k_nn, pruneTag='NA')

        # ---- Tensor conversion ----
        patches_list = []
        for img in patches_local:
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            patches_list.append(img)

        patches = torch.stack(patches_list, dim=0)          # (N_local, 3, H, W)
        centers = torch.from_numpy(centers).long()          # (N_local, 2)
        genes   = torch.from_numpy(genes_local).float()    # (N_local, G)
        adj     = adj.float()                               # (N_local, N_local)

        return patches, centers, genes, adj

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
