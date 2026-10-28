import h5py
from torch.utils.data import Dataset
import torch
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import numpy as np


def log1p_normalization(arr, scale_factor=1000000):
    """Apply log1p normalization to the given array."""
    eps = 1e-08
    return np.log1p(arr / (np.sum(arr, axis=1, keepdims=True) + eps) * scale_factor)


def _decode_h5_value(value):
    """Convert the scalar/string representations used by h5py to ``str``."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return str(value)
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_h5_ids(handle):
    """Read a spot identifier vector when one is present in an image H5."""
    for key in ("barcode", "barcodes", "spot_id"):
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return [_decode_h5_value(x) for x in handle[key][:]]
    return None


def _read_h5_spatial_coordinates(handle):
    """Read coordinates stored alongside Visium/Hist2ST image patches."""
    for (x_key, y_key) in (("pixel_x", "pixel_y"), ("x", "y")):
        if x_key in handle and y_key in handle:
            return np.column_stack(
                [np.asarray(handle[x_key][:]), np.asarray(handle[y_key][:])]
            ).astype(np.float32)
    if "coords" in handle and isinstance(handle["coords"], h5py.Dataset):
        coords = np.asarray(handle["coords"][:])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].astype(np.float32)
    if "coords" in handle and isinstance(handle["coords"], h5py.Group):
        group = handle["coords"]
        if "xcoord" in group and "ycoord" in group:
            x_values = []
            for value in group["xcoord"][:]:
                text = _decode_h5_value(value)
                parts = text.split("_")
                x_values.append(float(parts[-2] if len(parts) >= 2 else text))
            y_values = [float(_decode_h5_value(value)) for value in group["ycoord"][:]]
            return np.column_stack([x_values, y_values]).astype(np.float32)
    return None


def _read_adata_spatial_coordinates(adata):
    """Fallback coordinate reader for H5AD files."""
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"][:, :2], dtype=np.float32)
    for (x_key, y_key) in (
        ("pixel_x", "pixel_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("X", "Y"),
        ("x", "y"),
    ):
        if x_key in adata.obs.columns and y_key in adata.obs.columns:
            return np.column_stack(
                [adata.obs[x_key].to_numpy(), adata.obs[y_key].to_numpy()]
            ).astype(np.float32)
    return None


def normalize_spatial_coordinates(coords, new_max=100.0):
    """Match STPath's per-slide coordinate normalization without NaNs."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape [N, 2], got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain NaN or infinite values")
    coords = coords - coords.min(axis=0, keepdims=True)
    coord_range = coords.max(axis=0, keepdims=True)
    coord_range[coord_range < 1e-08] = 1.0
    return (coords / coord_range * float(new_max)).astype(np.float32)


def load_spatial_patch_data(
    gene_path, img_path, gene_names=None, log_norm=True, scale_factor=1000000
):
    """Load image patches, aligned expression, identifiers, and real coordinates.

    This is intentionally separate from the legacy ``PatchDataset``.  It is
    used only by the STPath branch, where spot identity and geometry are part
    of the model input.
    """
    with h5py.File(img_path, "r") as handle:
        image_key = next(
            (key for key in ("patches", "img", "images") if key in handle), None
        )
        if image_key is None:
            raise KeyError(
                f"No image dataset found in {img_path}; keys={list(handle.keys())}"
            )
        images = np.asarray(handle[image_key][:])
        h5_ids = _read_h5_ids(handle)
        coords = _read_h5_spatial_coordinates(handle)
    adata = sc.read_h5ad(gene_path)
    adata.var_names_make_unique()
    if h5_ids is not None:
        obs_ids = [str(value) for value in adata.obs_names]
        obs_to_idx = {value: index for (index, value) in enumerate(obs_ids)}
        missing = [value for value in h5_ids if value not in obs_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} H5 spot identifiers are missing from {gene_path}; examples={missing[:5]}"
            )
        row_indices = np.asarray(
            [obs_to_idx[value] for value in h5_ids], dtype=np.int64
        )
        adata = adata[row_indices].copy()
        spot_ids = h5_ids
    else:
        if len(images) != adata.n_obs:
            raise ValueError(
                f"Image rows ({len(images)}) and H5AD rows ({adata.n_obs}) differ without identifiers: {img_path} / {gene_path}"
            )
        spot_ids = [str(value) for value in adata.obs_names]
    if coords is None:
        coords = _read_adata_spatial_coordinates(adata)
    if coords is None:
        raise KeyError(f"No spatial coordinates found in {img_path} or {gene_path}")
    if len(coords) != len(images):
        raise ValueError(
            f"Coordinate rows ({len(coords)}) and image rows ({len(images)}) differ: {img_path}"
        )
    if gene_names is not None:
        requested = [str(value) for value in gene_names]
        available = set(map(str, adata.var_names))
        missing_genes = [gene for gene in requested if gene not in available]
        if missing_genes:
            raise ValueError(
                f"{len(missing_genes)} requested genes are absent from {gene_path}; examples={missing_genes[:10]}"
            )
        adata = adata[:, requested].copy()
        ordered_gene_names = requested
    else:
        ordered_gene_names = [str(value) for value in adata.var_names]
    gene_exp = adata.X
    if not isinstance(gene_exp, np.ndarray):
        gene_exp = gene_exp.toarray()
    gene_exp = np.asarray(gene_exp, dtype=np.float32)
    if log_norm:
        gene_exp = log1p_normalization(gene_exp, scale_factor=scale_factor).astype(
            np.float32
        )
    if len(images) != len(coords) or len(images) != len(gene_exp):
        raise ValueError(
            f"Unaligned spatial modalities: images={len(images)}, coords={len(coords)}, genes={len(gene_exp)}"
        )
    return (
        images,
        normalize_spatial_coordinates(coords),
        gene_exp,
        spot_ids,
        ordered_gene_names,
    )


class SpatialPatchDataset(Dataset):
    """STPath dataset returning contiguous spatial spot windows."""

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
    ):
        (
            self.image,
            self.coords,
            self.gene_data,
            self.spot_ids,
            self.gene_names,
        ) = load_spatial_patch_data(
            gene_path=gene_path,
            img_path=img_path,
            gene_names=gene_names,
            log_norm=log_norm,
            scale_factor=scale_factor,
        )
        self.transform = transform
        self.window_size = min(int(window_size), len(self.image))
        if self.window_size < 1:
            raise ValueError("SpatialPatchDataset contains no spots")
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


def spatial_collate_fn(batch):
    """Concatenate windows and preserve their separate attention groups."""
    (images, coords, targets) = zip(*batch)
    group_ids = [
        torch.full((item.shape[0],), index, dtype=torch.long)
        for (index, item) in enumerate(images)
    ]
    return (
        torch.cat(images, dim=0),
        torch.cat(coords, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(group_ids, dim=0),
    )


class PatchDataset(Dataset):
    def __init__(
        self,
        gene_path,
        img_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000,
        align_by_barcode=True,
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
        with h5py.File(self.img_path, "r") as f:
            print("\nAvailable HDF5 keys:")
            print(list(f.keys()))
            possible_img_keys = ["patches", "img", "images"]
            img_key = None
            for k in possible_img_keys:
                if k in f:
                    img_key = k
                    break
            if img_key is None:
                raise KeyError(
                    f"No valid image dataset found.\nAvailable keys: {list(f.keys())}"
                )
            self.image = f[img_key][:]
            print(f"\nUsing image key : {img_key}")
            print(f"Image shape     : {self.image.shape}")
            print(f"Image dtype     : {self.image.dtype}")
            self.img_barcodes = None
            if "barcodes" in f:
                self.img_barcodes = [
                    b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    for b in f["barcodes"][:]
                ]
                print(f"Loaded {len(self.img_barcodes)} image barcodes")
        print("\nLoading AnnData...")
        adata = sc.read_h5ad(self.gene_path)
        print(f"AnnData shape : {adata.shape}")
        adata.var_names_make_unique()
        if self.gene_names is not None:
            valid_genes = [g for g in self.gene_names if g in adata.var_names]
            print(f"\nKeeping {len(valid_genes)} / {len(self.gene_names)} genes")
            adata = adata[:, valid_genes].copy()
        if align_by_barcode and self.img_barcodes is not None:
            print("\nAligning image and gene data using barcodes...")
            gene_barcodes = adata.obs_names.tolist()
            barcode_to_idx = {bc: i for (i, bc) in enumerate(gene_barcodes)}
            matched_indices = []
            missing_barcodes = []
            for bc in self.img_barcodes:
                if bc in barcode_to_idx:
                    matched_indices.append(barcode_to_idx[bc])
                else:
                    missing_barcodes.append(bc)
            if len(missing_barcodes) > 0:
                raise ValueError(
                    f"{len(missing_barcodes)} barcodes missing in AnnData.\nExample: {missing_barcodes[:5]}"
                )
            adata = adata[matched_indices].copy()
            print("Barcode alignment complete")
        print("\nExtracting gene matrix...")
        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()
        gene_exp = gene_exp.astype(np.float32)
        print(f"Gene matrix shape : {gene_exp.shape}")
        if self.log_norm:
            print("\nApplying log1p normalization...")
            gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
            print(f"Normalization complete (scale_factor={self.scale_factor})")
        self.gene_data = gene_exp
        if len(self.image) != len(self.gene_data):
            raise ValueError(
                f"\nImage count ({len(self.image)}) != Gene count ({len(self.gene_data)})"
            )
        print("\nFinal dataset summary")
        print("-" * 40)
        print(f"Images      : {self.image.shape}")
        print(f"Gene matrix : {self.gene_data.shape}")
        print(f"Dataset size: {len(self.image)}")
        print("=" * 60)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx].astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)
        return (image, gene_exp)
