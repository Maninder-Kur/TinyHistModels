"""HER2-ST patch generator adapted from the original project converter.

Preserves RGB conversion, coordinate matching, and boundary exclusion.
"""
import os
from pathlib import Path
import argparse
import logging

# ======================================================
# ARGUMENTS
# ======================================================
parser = argparse.ArgumentParser(description="Generate image patches and H5/H5AD for HER2-ST samples.")
parser.add_argument("--base_dir", required=True,
                    help="Extracted HER2-ST data directory containing ST-cnts, ST-spotfiles, ST-imgs.")
parser.add_argument("--sample_ids", nargs="+", default=None,
                    help="List of sample IDs to process (e.g., --sample_ids A1 A2 B1)")
parser.add_argument("--patch_size", type=int, default=224,
                    help="Patch size in pixels (default: 224)")
parser.add_argument("--out_dir", type=str, required=True,
                    help="Output directory to store H5 and H5AD files.")
args = parser.parse_args()
if args.patch_size <= 0 or args.patch_size % 2:
    parser.error("--patch_size must be a positive even integer")
if args.sample_ids is None:
    import csv
    with (Path(__file__).parent / "manifests" / "her2.csv").open() as handle:
        args.sample_ids = [row["sample_id"] for row in csv.DictReader(handle)]

import h5py
import cv2
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm

# ======================================================
# PATHS
# ======================================================
BASE_PATH = args.base_dir
CNT_DIR = os.path.join(BASE_PATH, "ST-cnts")
SPOT_DIR = os.path.join(BASE_PATH, "ST-spotfiles")
IMG_DIR = os.path.join(BASE_PATH, "ST-imgs")
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# LOGGER SETUP
# ======================================================
logging.basicConfig(
    filename=os.path.join(OUT_DIR, "patch_generation.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ======================================================
# MAIN LOOP
# ======================================================
for SAMPLE_ID in args.sample_ids:
    logging.info("=" * 70)
    logging.info(f"Processing sample: {SAMPLE_ID}")
    logging.info("=" * 70)

    cnt_path = os.path.join(CNT_DIR, f"{SAMPLE_ID}.tsv.gz")
    spot_path = os.path.join(SPOT_DIR, f"{SAMPLE_ID}_selection.tsv")

    if not os.path.exists(cnt_path) or not os.path.exists(spot_path):
        raise FileNotFoundError(f"Missing counts or spot coordinates for sample {SAMPLE_ID}")

    try:
        # --------------------------
        # Load gene expression
        # --------------------------
        expr = pd.read_csv(cnt_path, sep="\t", index_col=0)
        expr.index = expr.index.astype(str)
        expr_spots = expr.index.str.extract(r'(?P<x>\d+)x(?P<y>\d+)').astype(int)
        expr_spots.index = expr.index
        expr_spots["spot_id"] = expr.index  # keep reference to original IDs

        # --------------------------
        # Load spot coordinates
        # --------------------------
        spots = pd.read_csv(spot_path, sep="\t")

        # Merge by x,y — keep spot_id for expression reference
        merged = pd.merge(expr_spots, spots, on=["x", "y"], how="inner")
        logging.info(f"Matched {len(merged)} spots (expression + coordinates)")

        # --------------------------
        # Locate image file (recursive search)
        # --------------------------
        img_root = os.path.join(IMG_DIR, SAMPLE_ID[0])  # e.g. A1 → ST-imgs/A
        image_file = None
        for root, _, files in os.walk(img_root):
            for f in files:
                if any(f.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]):
                    if SAMPLE_ID in Path(root).parts:  # e.g., .../A1/...
                        image_file = os.path.join(root, f)
                        break
            if image_file:
                break

        if image_file is None:
            raise FileNotFoundError(f"No image found for {SAMPLE_ID} in {img_root}")

        img = cv2.imread(image_file)
        if img is None:
            raise ValueError(f"Could not read image for {SAMPLE_ID}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        logging.info(f"Loaded image {image_file} with shape {img.shape}")

        # --------------------------
        # Extract patches
        # --------------------------
        PATCH_SIZE = args.patch_size
        half = PATCH_SIZE // 2
        patches, px, py, xs, ys, spot_ids = [], [], [], [], [], []

        for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"{SAMPLE_ID} patches"):
            cx, cy = int(row["pixel_x"]), int(row["pixel_y"])
            x1, x2 = cx - half, cx + half
            y1, y2 = cy - half, cy + half

            # Ensure within image bounds
            if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                continue

            patch = img[y1:y2, x1:x2, :]
            if patch.shape[0] == PATCH_SIZE and patch.shape[1] == PATCH_SIZE:
                patches.append(patch)
                px.append(cx)
                py.append(cy)
                xs.append(row["x"])
                ys.append(row["y"])
                spot_ids.append(row["spot_id"])

        logging.info(f"Extracted {len(patches)} patches for {SAMPLE_ID}")
        if not patches:
            raise ValueError(f"No valid patches extracted for {SAMPLE_ID}")

        # --------------------------
        # Save H5 file
        # --------------------------
        h5_path = os.path.join(OUT_DIR, f"{SAMPLE_ID}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("patches", data=np.array(patches), compression="gzip")
            f.create_dataset("pixel_x", data=np.array(px))
            f.create_dataset("pixel_y", data=np.array(py))
            f.create_dataset("x", data=np.array(xs))
            f.create_dataset("y", data=np.array(ys))
        logging.info(f" Saved H5 file: {h5_path}")

        # --------------------------
        # Save H5AD file (for Scanpy)
        # --------------------------
        expr_common = expr.loc[spot_ids]  # select using original spot IDs
        adata = sc.AnnData(X=expr_common.values)
        adata.var_names = expr_common.columns
        adata.obs_names = [str(i) for i in range(adata.n_obs)]
        adata.obs["spot_id"] = spot_ids 
        adata.obs["x"] = xs
        adata.obs["y"] = ys
        adata.obs["pixel_x"] = px
        adata.obs["pixel_y"] = py

        h5ad_path = os.path.join(OUT_DIR, f"{SAMPLE_ID}.h5ad")
        adata.write(h5ad_path)
        logging.info(f" Saved H5AD file: {h5ad_path}")

    except Exception as e:
        logging.exception(f"Error processing {SAMPLE_ID}: {e}")
        raise

logging.info(" Patch and H5/H5AD generation complete for all samples.")
print(f"All results saved to: {OUT_DIR}")
