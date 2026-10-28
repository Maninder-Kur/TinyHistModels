#!/usr/bin/env python3
"""Run the manuscript training scripts with repository-relative configuration paths."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "her2": "her2",
    "hist2st_845": "hist2st",
    "xenium_breast_patient": "xenium_breast",
    "xenium_lung_patient": "xenium_lung",
}
PATH_KEYS = {
    "dataset_path",
    "genes",
    "result_path",
    "assets_dir",
    "teacher_base_path",
    "uni_checkpoint",
    "embedding_cache_dir",
    "gene_voc_path",
}


def resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def entrypoint(params: dict) -> str:
    dataset = params["dataset_name"]
    if dataset not in COHORTS:
        raise ValueError(f"Unsupported paper cohort: {dataset}")
    xenium = dataset.startswith("xenium_")
    model = params["model"]
    if model == "StemUNI":
        return "stem_xenium_patient.py" if xenium else "stem_visium_patient.py"
    if model == "THItoGene":
        if xenium:
            raise ValueError("THItoGene Xenium results are not part of this release.")
        return "main2_thi_her2.py" if dataset == "her2" else "main2_thi_hist2st.py"
    if model in {"TinyEffNet_kd", "TinySTNet", "TinyHisToGene"}:
        return "main2_xenium_patient_kd.py" if xenium else "main2_kd.py"
    if model not in {
        "STNet",
        "VGG",
        "HisToGene",
        "Effnet",
        "TCGN",
        "UNI",
        "STPath",
        "STPathZero",
        "STPathSmall",
        "STPathSmallScratch",
        "GigaPathFlash",
        "EfficientNetTinyStudent",
    }:
        raise ValueError(f"Unsupported paper model: {model}")
    return "main2_xenium_patient.py" if xenium else "main2.py"


def check_inputs(params: dict) -> list[str]:
    errors = []
    data = Path(params["dataset_path"])
    if not data.is_dir():
        errors.append(f"Missing data directory: {data}")
    else:
        h5_ids = {p.stem for p in data.glob("*.h5")}
        ad_ids = {p.stem for p in data.glob("*.h5ad")}
        ids = params.get("patient_ids") or params.get("sample_ids")
        if not ids:
            manifest = (
                ROOT / "code_dataprocessing/manifests" / f"{params['dataset_name']}.csv"
            )
            with manifest.open() as handle:
                ids = [row["sample_id"] for row in csv.DictReader(handle)]
        if ids:
            missing = sorted(set(ids) - (h5_ids & ad_ids))
            if missing:
                errors.append("Missing H5/H5AD pairs: " + ", ".join(missing))
    genes = Path(params["genes"])
    if genes.suffix != ".npy":
        genes = genes / f"{params['dataset_name']}.npy"
    if not genes.is_file():
        errors.append(f"Missing target gene list: {genes}")
    weights = []
    assets = Path(params.get("assets_dir", ROOT / "assets"))
    if params["model"] == "UNI":
        weights.append(assets / "uni/pytorch_model.bin")
    elif params["model"] == "StemUNI":
        weights.append(Path(params["uni_checkpoint"]))
    elif params["model"] == "GigaPathFlash":
        weights.append(
            assets
            / "gigapath_flash"
            / params.get("gigapath_flash_ckpt", "pytorch_model.bin")
        )
    elif params["model"] in {
        "STPath",
        "STPathZero",
        "STPathSmall",
        "STPathSmallScratch",
    }:
        weights.append(
            assets / "gigapath" / params.get("gigapath_ckpt", "pytorch_model.bin")
        )
        if params["model"] != "STPathSmallScratch":
            weights.append(assets / "stpath" / params.get("stpath_ckpt", "stfm.pth"))
            vocabulary = Path(
                params.get(
                    "gene_voc_path",
                    ROOT
                    / "code_model_training/models/STPath/utils_data/symbol2ensembl.json",
                )
            )
            if not vocabulary.is_file():
                errors.append(f"Missing gene vocabulary: {vocabulary}")
    if params.get("use_kd"):
        teacher = params.get("teacher_base_path")
        if not teacher or "REPLACE_WITH_TEACHER_RUN" in teacher:
            errors.append(
                "Provide --teacher-run with the matching teacher's completed run directory."
            )
        else:
            weights.extend(
                Path(teacher) / f"fold_{i}/best_model.pth"
                for i in range(1, int(params.get("k_folds", 5)) + 1)
            )
    errors.extend(f"Missing checkpoint: {p}" for p in weights if not p.is_file())
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="JSON path relative to the repository root, or absolute",
    )
    parser.add_argument(
        "--data-dir", help="Override the directory of prepared H5/H5AD pairs"
    )
    parser.add_argument("--assets-dir", help="Override the pretrained assets root")
    parser.add_argument("--output-dir", help="Override result_path")
    parser.add_argument(
        "--teacher-run", help="Completed teacher run containing fold_1 through fold_5"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without requiring data or starting training",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check input paths without starting training",
    )
    args = parser.parse_args()
    try:
        params = json.loads(resolve_path(args.config).read_text())
        for argument, key in [
            (args.data_dir, "dataset_path"),
            (args.assets_dir, "assets_dir"),
            (args.output_dir, "result_path"),
            (args.teacher_run, "teacher_base_path"),
        ]:
            if argument is not None:
                params[key] = argument
        if args.assets_dir and params["model"] == "StemUNI":
            params["uni_checkpoint"] = str(
                resolve_path(args.assets_dir) / "uni/pytorch_model.bin"
            )
        for key in PATH_KEYS & params.keys():
            params[key] = str(resolve_path(params[key]))
        script = ROOT / "code_model_training" / entrypoint(params)
        if not script.is_file():
            raise FileNotFoundError(script)
        if args.dry_run:
            print(
                json.dumps({"entrypoint": str(script), "parameters": params}, indent=2)
            )
            return 0
        errors = check_inputs(params)
        if errors:
            parser.exit(
                2,
                "Input check failed:\n" + "\n".join(f"  - {e}" for e in errors) + "\n",
            )
        if args.check:
            print(
                "Required paths exist. Use code_dataprocessing/validate_data.py to check data contents."
            )
            return 0
        # Training saves its own experiment_info.json with these resolved parameters.
        with tempfile.TemporaryDirectory(prefix="tinyhist-config-") as temp:
            resolved = Path(temp) / "config.json"
            resolved.write_text(json.dumps(params, indent=2) + "\n")
            return subprocess.call(
                [sys.executable, str(script), "--config", str(resolved)],
                cwd=ROOT / "code_model_training",
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
