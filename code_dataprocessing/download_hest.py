"""Download only the HEST patch/expression pairs selected for a paper cohort.

Accept the upstream access terms and authenticate Hugging Face before running.
Images and expression matrices remain outside version control.
"""
import argparse
import csv
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("xenium_breast_patient", "xenium_lung_patient"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="HEST dataset commit or branch; record the commit for reproducibility")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    # Resolve mutable branches once so all files come from the same revision.
    repo_id = "MahmoodLab/hest"
    commit = HfApi().dataset_info(repo_id, revision=args.revision).sha
    with (Path(__file__).parent / "manifests" / f"{args.cohort}.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    revision_file = args.out_dir / "HEST_REVISION.txt"
    if not revision_file.exists() and any(
        (args.out_dir / row[column]).exists()
        for row in rows for column in ("patch_file", "expression_file")
    ):
        raise ValueError("Existing sample files have no recorded HEST revision; use a new output directory")
    if revision_file.exists() and revision_file.read_text().strip() != commit:
        raise ValueError("Output contains a different recorded HEST revision; use a new output directory")
    revision_file.write_text(commit + "\n")
    for row in rows:
        for folder, column in (("patches", "patch_file"), ("st", "expression_file")):
            target = args.out_dir / row[column]
            if target.exists():
                print(f"Exists: {target}")
                continue
            filename = f"{folder}/{row[column]}"
            downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                                         revision=commit, local_dir=args.out_dir / ".download")
            shutil.move(downloaded, target)
            print(f"Saved: {target}")
    print(f"HEST revision: {commit}. Run validate_data.py before training.")


if __name__ == "__main__":
    main()
