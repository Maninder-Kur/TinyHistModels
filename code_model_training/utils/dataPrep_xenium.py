import h5py
from torch.utils.data import Dataset
import torch
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import numpy as np

try:
    from dataPrep import (
        load_spatial_patch_data,
        spatial_collate_fn as spatial_collate_fn,
        _read_adata_spatial_coordinates,
        _read_h5_ids,
        _read_h5_spatial_coordinates,
        normalize_spatial_coordinates,
    )
except ImportError:
    from .dataPrep import (
        spatial_collate_fn as spatial_collate_fn,
        load_spatial_patch_data,
        _read_adata_spatial_coordinates,
        _read_h5_ids,
        _read_h5_spatial_coordinates,
        normalize_spatial_coordinates,
    )


def log1p_normalization(arr, scale_factor=1000000):
    """
    Apply log1p normalization to rows of arr:
      normalized_row = log1p( (row / row.sum()) * scale_factor )

    Handles rows with sum 0 by leaving them zero (avoids NaN).
    """
    arr = np.asarray(arr, dtype=np.float64)
    lib = arr.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    scaled = arr / lib * float(scale_factor)
    return np.log1p(scaled)


def load_xenium_spatial_data(
    gene_path, img_path, gene_names=None, log_norm=True, scale_factor=1000000
):
    """Load Xenium images, barcodes, coordinates, and ordered gene targets.

    Xenium H5 files contain ``img``, ``coords`` and ``barcode`` while the
    corresponding H5AD stores expression in ``X`` and spatial metadata in
    ``obsm['spatial']``/``obs``.  The shared loader aligns H5 rows to H5AD
    rows by barcode before returning the modalities.
    """
    return load_spatial_patch_data(
        gene_path=gene_path,
        img_path=img_path,
        gene_names=gene_names,
        log_norm=log_norm,
        scale_factor=scale_factor,
    )


class SpatialXeniumDataset(Dataset):
    """STPath Xenium dataset returning nearest-neighbor spatial windows."""

    def __init__(
        self,
        gene_path,
        img_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        window_size=256,
        window_stride=None,
        spot_indices=None,
    ):
        from sklearn.neighbors import NearestNeighbors

        (
            self.image,
            self.coords,
            self.gene_data,
            self.barcodes,
            self.gene_names,
        ) = load_xenium_spatial_data(
            gene_path=gene_path,
            img_path=img_path,
            gene_names=gene_names,
            log_norm=log_norm,
            scale_factor=scale_factor,
        )
        if spot_indices is not None:
            spot_indices = np.asarray(spot_indices, dtype=np.int64)
            if spot_indices.ndim != 1 or len(spot_indices) == 0:
                raise ValueError(
                    "spot_indices must be a non-empty one-dimensional array"
                )
            if spot_indices.min() < 0 or spot_indices.max() >= len(self.image):
                raise IndexError("spot_indices contain an out-of-range spot")
            self.image = self.image[spot_indices]
            self.coords = self.coords[spot_indices]
            self.gene_data = self.gene_data[spot_indices]
            self.barcodes = [self.barcodes[index] for index in spot_indices]
        self.transform = transform
        self.window_size = min(int(window_size), len(self.image))
        if self.window_size < 1:
            raise ValueError("SpatialXeniumDataset contains no spots")
        self.window_stride = int(window_stride or self.window_size)
        n_centers = max(1, int(np.ceil(len(self.image) / self.window_stride)))
        centers = np.linspace(0, len(self.image) - 1, n_centers, dtype=np.int64)
        self._neighbor_model = NearestNeighbors(
            n_neighbors=self.window_size, metric="euclidean"
        ).fit(self.coords)
        (_, self.window_indices) = self._neighbor_model.kneighbors(self.coords[centers])
        self.window_indices = self.window_indices.astype(np.int64)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, index):
        spot_indices = self.window_indices[index]
        images = []
        for spot_index in spot_indices:
            image = self.image[int(spot_index)].astype(np.uint8)
            if self.transform:
                image = self.transform(image)
            else:
                image = (
                    torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
            images.append(image)
        return (
            torch.stack(images),
            torch.from_numpy(self.coords[spot_indices]).float(),
            torch.from_numpy(self.gene_data[spot_indices]).float(),
        )


class SpatialXeniumPatientDataset(Dataset):
    """Memory-efficient STPath dataset for one Xenium patient.

    Unlike ``SpatialXeniumDataset``, this class keeps the H5 image dataset
    lazy.  Xenium patient files can contain several GB of image patches, so
    loading every training patient fully into RAM would be unnecessary and
    can cause an out-of-memory failure before the model starts.
    """

    def __init__(
        self,
        gene_path,
        img_path,
        spot_indices=None,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        window_size=256,
        window_stride=None,
    ):
        self.img_path = img_path
        self.gene_path = gene_path
        self.transform = transform
        self._h5_file = None
        with h5py.File(img_path, "r") as handle:
            self._image_key = next(
                (key for key in ("patches", "img", "images") if key in handle), None
            )
            if self._image_key is None:
                raise KeyError(f"No image dataset found in {img_path}")
            n_images = int(handle[self._image_key].shape[0])
            h5_ids = _read_h5_ids(handle)
            h5_coords = _read_h5_spatial_coordinates(handle)
        if spot_indices is None:
            spot_indices = np.arange(n_images, dtype=np.int64)
        else:
            spot_indices = np.asarray(spot_indices, dtype=np.int64)
        if spot_indices.ndim != 1 or len(spot_indices) == 0:
            raise ValueError("spot_indices must be a non-empty one-dimensional array")
        if spot_indices.min() < 0 or spot_indices.max() >= n_images:
            raise IndexError("spot_indices contain an out-of-range image index")
        adata = sc.read_h5ad(gene_path)
        adata.var_names_make_unique()
        if h5_ids is not None:
            if len(h5_ids) != n_images:
                raise ValueError(
                    f"Barcode count ({len(h5_ids)}) and image count ({n_images}) differ: {img_path}"
                )
            obs_to_idx = {
                str(value): index for (index, value) in enumerate(adata.obs_names)
            }
            selected_ids = [str(h5_ids[index]) for index in spot_indices]
            valid_mask = np.asarray(
                [value in obs_to_idx for value in selected_ids], dtype=bool
            )
            if not valid_mask.all():
                missing_ids = np.asarray(selected_ids, dtype=object)[~valid_mask]
                print(
                    f"[WARN] Dropping {len(missing_ids)} image spot(s) without gene expression from {gene_path}; examples={missing_ids[:5].tolist()}"
                )
                spot_indices = spot_indices[valid_mask]
                selected_ids = list(np.asarray(selected_ids, dtype=object)[valid_mask])
                if len(spot_indices) == 0:
                    raise ValueError(
                        f"All selected Xenium barcodes are missing from {gene_path}"
                    )
            adata_rows = np.asarray(
                [obs_to_idx[value] for value in selected_ids], dtype=np.int64
            )
        else:
            if adata.n_obs != n_images:
                raise ValueError(
                    f"Image rows ({n_images}) and H5AD rows ({adata.n_obs}) differ without barcodes"
                )
            adata_rows = spot_indices
        if h5_coords is None:
            adata_coords = _read_adata_spatial_coordinates(adata)
            if adata_coords is None:
                raise KeyError(
                    f"No spatial coordinates found in {img_path} or {gene_path}"
                )
            if h5_ids is not None:
                full_coords = np.asarray(adata_coords[adata_rows], dtype=np.float32)
            else:
                full_coords = np.asarray(adata_coords, dtype=np.float32)
            selected_coords = full_coords
        else:
            if len(h5_coords) != n_images:
                raise ValueError(
                    f"Coordinate rows ({len(h5_coords)}) and image rows ({n_images}) differ"
                )
            normalized_all_coords = normalize_spatial_coordinates(h5_coords)
            selected_coords = normalized_all_coords[spot_indices]
        if gene_names is not None:
            requested = [str(value) for value in gene_names]
            available = set(map(str, adata.var_names))
            missing_genes = [gene for gene in requested if gene not in available]
            if missing_genes:
                raise ValueError(
                    f"{len(missing_genes)} requested genes are absent from {gene_path}; examples={missing_genes[:10]}"
                )
            adata = adata[adata_rows, requested].copy()
            self.gene_names = requested
        else:
            adata = adata[adata_rows].copy()
            self.gene_names = [str(value) for value in adata.var_names]
        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()
        gene_exp = np.asarray(gene_exp, dtype=np.float32)
        if log_norm:
            gene_exp = log1p_normalization(gene_exp, scale_factor=scale_factor).astype(
                np.float32
            )
        self.image_indices = spot_indices
        self.coords = np.asarray(selected_coords, dtype=np.float32)
        self.gene_data = gene_exp
        self.barcodes = (
            [str(h5_ids[index]) for index in spot_indices]
            if h5_ids is not None
            else [str(index) for index in spot_indices]
        )
        self.num_genes = int(self.gene_data.shape[1])
        self.window_size = min(int(window_size), len(self.image_indices))
        if self.window_size < 1:
            raise ValueError("SpatialXeniumPatientDataset contains no spots")
        self.window_stride = int(window_stride or self.window_size)
        n_centers = max(1, int(np.ceil(len(self.image_indices) / self.window_stride)))
        centers = np.linspace(0, len(self.image_indices) - 1, n_centers, dtype=np.int64)
        self._neighbor_model = NearestNeighbors(
            n_neighbors=self.window_size, metric="euclidean"
        ).fit(self.coords)
        (_, self.window_indices) = self._neighbor_model.kneighbors(self.coords[centers])
        self.window_indices = self.window_indices.astype(np.int64)

    def __len__(self):
        return len(self.window_indices)

    def _images(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.img_path, "r")
        return self._h5_file[self._image_key]

    def __getitem__(self, index):
        spot_indices = self.window_indices[index]
        images = []
        image_dataset = self._images()
        for local_index in spot_indices:
            image_index = int(self.image_indices[int(local_index)])
            image = np.asarray(image_dataset[image_index], dtype=np.uint8)
            if self.transform:
                image = self.transform(image)
            else:
                image = (
                    torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
            images.append(image)
        return (
            torch.stack(images),
            torch.from_numpy(self.coords[spot_indices]).float(),
            torch.from_numpy(self.gene_data[spot_indices]).float(),
        )

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()


class PatchDataset(Dataset):
    def __init__(self, gene_path, img_path, gene_names=None, transform=None):
        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform
        print("Initializing PatchDataset...")
        print(f"Image file: {self.img_path}")
        print(f"Gene file:  {self.gene_path}")
        with h5py.File(self.img_path, "r") as f:
            if "patches" in f:
                self.image = f["patches"][:]
            elif "img" in f:
                self.image = f["img"][:]
            else:
                raise KeyError(f"No valid image dataset found in {self.img_path}")
        print(f" Loaded images: {self.image.shape}")
        adata = sc.read_h5ad(self.gene_path)
        print(f"\nLoaded AnnData: {adata.shape}")
        if "feature_types" in adata.var.columns:
            print("\nFeature type counts:")
            print(adata.var["feature_types"].value_counts())
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            print(f"\nFiltered to biological genes only: {adata.shape}")
        if gene_names:
            valid_genes = [g for g in gene_names if g in adata.var_names]
            adata = adata[:, valid_genes].copy()
            print(f"\nAfter gene subsetting: {adata.shape}")
        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()
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
        return (img, gene_raw)


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
        return (img, genes)


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
                    f"WARNING: Skipping {n_missing} image spot(s) without gene expression from {self.gene_path}. First missing barcode: {missing_barcodes[0]}"
                )
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
        gene_exp = (
            adata.X if self.selected_barcodes is not None else adata.X[self.indices]
        )
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()
        gene_exp = gene_exp.astype(np.float32)
        if log_norm:
            gene_exp = log1p_normalization(gene_exp, scale_factor=scale_factor).astype(
                np.float32
            )
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
        return (img, genes)

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
