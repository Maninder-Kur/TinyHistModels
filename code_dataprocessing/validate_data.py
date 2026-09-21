"""Validate the training file contract without loading image/expression matrices."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("hist2st_845", "her2", "xenium_breast_patient", "xenium_lung_patient")


def strings(node):
    """Read string arrays, including AnnData categorical encoding."""
    if isinstance(node, h5py.Group):
        categories = strings(node["categories"])
        return [categories[int(i)] if i >= 0 else "" for i in node["codes"][:]]
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in np.asarray(node[:]).reshape(-1)]


def index_names(group):
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return strings(group[key])


def validate_pair(data_dir, row, genes, xenium):
    patch_path = data_dir / row["patch_file"]
    expr_path = data_dir / row["expression_file"]
    errors, warnings = [], []
    result = {"sample_id": row["sample_id"], "errors": errors, "warnings": warnings}
    for path in (patch_path, expr_path):
        if not path.is_file():
            errors.append(f"Missing file: {path}")
    if errors:
        return result

    with h5py.File(patch_path, "r") as images, h5py.File(expr_path, "r") as expr:
        key = next((key for key in ("patches", "img", "images") if key in images), None)
        if key is None:
            errors.append("No image dataset: expected patches, img, or images")
            return result
        shape = images[key].shape
        result["image_shape"] = list(shape)
        if len(shape) != 4 or shape[-1] != 3 or shape[0] == 0:
            errors.append(f"Expected nonempty [N, H, W, 3] image array; got {shape}")
        if images[key].dtype != np.uint8:
            errors.append("Image array must contain uint8 RGB pixels")
        obs = index_names(expr["obs"])
        var = index_names(expr["var"])
        result["expression_shape"] = [len(obs), len(var)]
        if len(obs) != len(set(obs)):
            errors.append("Expression observation names are not unique")
        if len(var) != len(set(var)):
            errors.append("Expression gene names are not unique")
        if "feature_types" in expr["var"]:
            kinds = strings(expr["var/feature_types"])
            var = [gene for gene, kind in zip(var, kinds) if kind == "Gene Expression"]
        missing_genes = sorted(set(genes) - set(var))
        if missing_genes:
            errors.append(f"Missing {len(missing_genes)} target genes: {missing_genes[:10]}")

        if xenium:
            # XeniumPatientDataset expects the HEST singular key and Nx1 encoding.
            if "barcode" not in images or images["barcode"].shape != (shape[0], 1):
                errors.append("Xenium requires HEST barcode array with shape [N, 1]")
            else:
                barcodes = strings(images["barcode"])
                if len(barcodes) != len(set(barcodes)):
                    errors.append("Image barcodes are not unique")
                available = set(obs)
                absent = [barcode for barcode in barcodes if barcode not in available]
                result["unmatched_image_barcodes"] = len(absent)
                if len(absent) == len(barcodes):
                    errors.append("No image barcodes match expression observations")
                elif absent:
                    warnings.append(f"{len(absent)} unmatched image barcodes; the training loader skips them")
        else:
            if shape[0] != len(obs):
                errors.append("Spot image/expression row counts differ")
            if "spot_id" in images:
                expected = strings(expr["obs/spot_id"]) if "spot_id" in expr["obs"] else obs
                if strings(images["spot_id"]) != expected:
                    errors.append("Spot identifiers do not match expression row order")
            for axis in ("pixel_x", "pixel_y"):
                if axis in images and axis in expr["obs"]:
                    image_coords = images[axis][:]
                    expression_coords = expr["obs"][axis][:]
                    if image_coords.shape != expression_coords.shape or not np.allclose(image_coords, expression_coords):
                        errors.append(f"{axis} differs between image and expression rows")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    with (ROOT / "code_dataprocessing/manifests" / f"{args.cohort}.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    genes = (ROOT / "code_model_training/gene_outputs" / f"{args.cohort}.txt").read_text().splitlines()
    results = []
    for row in rows:
        try:
            result = validate_pair(args.data_dir, row, genes, args.cohort.startswith("xenium_"))
        except (KeyError, OSError, ValueError, TypeError) as exc:
            result = {"sample_id": row["sample_id"], "errors": [str(exc)], "warnings": []}
        results.append(result)
        status = "FAIL" if result["errors"] else "OK"
        print(f"{status} {row['sample_id']}: {result.get('image_shape', '?')} images; "
              f"{result.get('expression_shape', '?')} expression")
        for error in result["errors"]:
            print(f"  ERROR: {error}")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")
    errors = sum(bool(result["errors"]) for result in results)
    print(f"{len(results) - errors}/{len(results)} file pairs passed metadata checks; {len(genes)} target genes.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"cohort": args.cohort, "samples": results}, indent=2) + "\n")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
