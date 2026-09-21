# TinyHistModels

Training code accompanying **A Scale-Aware Study of Spatial Gene Expression Prediction from Histology Images**.

This repository collects the paper's experiment adapters, model implementations, patient-wise training scripts, ordered target-gene lists, and 62 saved experiment configurations. It covers **Hist2ST, HER2, HEST Xenium Breast, and HEST Xenium Lung**.

**Reproduction status:** the recorded configurations and available training source are included, but the archive has gaps and differences from the manuscript. In particular, the Hist2ST raw-data converter and historical STPath training entrypoint were absent; a STPath adapter has been reconstructed. Read [reproduction notes](docs/REPRODUCIBILITY.md) and [data preparation](docs/DATA_PREPARATION.md) before attempting to reproduce reported numbers.

## Installation

Use Python 3.10 on Linux. The release was smoke-checked with PyTorch 2.6.0, torchvision 0.21.0, and timm 1.0.22. These are the available validation-environment versions, not a complete historical environment lockfile.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 12.6 wheels, matching the release validation environment:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cu126-cp310-cp310-linux_x86_64.whl'
python -m pip install -r requirements.txt
```

For CPU-only use, replace `cu126` with `cpu` in **both** wheel URLs above. For other supported platforms, follow the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) and [torchsort installation instructions](https://github.com/teddykoker/torchsort#install); its compiled wheel must match Python, PyTorch, and CUDA. `torchsort` is required by the preserved training-time correlation code. Full training is intended for a GPU; Stem configurations explicitly require CUDA.

To prepare HER2 from raw images, also run `python -m pip install -r requirements-data.txt`.

## Data and weights

Download data from the [original sources](docs/DATA_SOURCES.md) and follow the [preparation guide](docs/DATA_PREPARATION.md). The expected layout is:

```text
data/
  her2/             A1.h5, A1.h5ad, ...
  hist2st/          23209_C1.h5, 23209_C1.h5ad, ...
  xenium_breast/    NCBI783.h5, NCBI783.h5ad, ...
  xenium_lung/      NCBI856.h5, NCBI856.h5ad, ...
assets/
  uni/pytorch_model.bin       # UNI and Stem only
  gigapath/pytorch_model.bin  # STPath only
  stpath/stfm.pth             # STPath only
```

The HEST downloader selects only the manuscript's sample IDs. Accept the upstream terms and authenticate with your own Hugging Face account first:

```bash
hf auth login
python code_dataprocessing/download_hest.py --cohort xenium_breast_patient --out-dir data/xenium_breast
python code_dataprocessing/download_hest.py --cohort xenium_lung_patient --out-dir data/xenium_lung
```

Only target-gene definitions and sample manifests are bundled. Data, pretrained weights, embeddings, checkpoints, logs, and training outputs are ignored by Git. Single-sample 10x breast/lung demonstrations and pancreas/liver experiments are outside this release.

## Training

Run these commands from the repository root. The launcher resolves all paths in a config relative to the repository root, selects the appropriate original training script, and writes outputs under `outputs/`.

```bash
# Inspect a run without downloading data, importing torch, or starting training:
python scripts/train.py --config configs/her2/effnet.json --dry-run

# Check required input paths:
python scripts/train.py --config configs/her2/effnet.json --check

# Baseline, compound scaling, and specialized model examples:
python scripts/train.py --config configs/her2/effnet.json
python scripts/train.py --config configs/hist2st/effnet_5.json
python scripts/train.py --config configs/her2/thitogene.json
python scripts/train.py --config configs/xenium_breast/stem.json
```

Use `--data-dir /path/to/prepared/data`, `--assets-dir /path/to/assets`, and `--output-dir /path/to/results` to keep large files elsewhere. `--check` verifies filenames and prerequisites; use the preparation guide's validator for image shapes, barcodes, and target genes.

| Experiment | Entry point |
| --- | --- |
| HER2 / Hist2ST regression and EfficientNet scaling | `code_model_training/main2.py` |
| HER2 / Hist2ST distillation | `code_model_training/main2_kd.py` |
| HER2 / Hist2ST THItoGene | `main2_thi_her2.py` / `main2_thi_hist2st.py` |
| Xenium patient-wise regression and scaling | `main2_xenium_patient.py` |
| Xenium patient-wise distillation | `main2_xenium_patient_kd.py` |
| Stem with frozen UNI features | `stem_visium_patient.py` / `stem_xenium_patient.py` |

The [configuration index](configs/README.md) lists all baseline, scaling, and KD runs. `VGG` implements the DeepSpaCE comparison; `Effnet` is EfficientNet-B0; `EfficientNetTinyStudent` supplies EffNet-3/5/10/20. EffNet-100 uses the same `effnet.json` as the full baseline. THItoGene is included for the two spot-level cohorts only.

## Knowledge distillation

Train the full-size teacher for the **same cohort and folds** first. The teacher's run directory must contain `fold_1/best_model.pth` through `fold_5/best_model.pth`. Then pass that directory explicitly:

```bash
python scripts/train.py --config configs/her2/effnet_kd.json \
  --teacher-run outputs/her2/effnet/REPLACE_WITH_COMPLETED_RUN
```

Use `stnet.json` before `stnet_kd.json` and `histogene.json` before `histogene_kd.json`. The launcher rejects the placeholder until a real teacher run is supplied. Fold identities, gene order, and input preprocessing must match the teacher. These source files use the archived centered distillation loss described in the reproduction notes.

## Outputs and release scope

Training generates timestamped run folders, configuration snapshots, fold checkpoints, and training/validation metrics. Validation used for checkpoint selection remains part of training. Standalone testing/evaluation, significance analysis, figure generation, notebooks, old run directories, and the manuscript PDF are intentionally omitted.

The paper's main model families are implemented in `code_model_training/models/`; copying the unrelated complete baseline repositories is unnecessary. Essential upstream source is retained with [provenance and licensing notes](THIRD_PARTY.md). No project-wide license has been selected; the missing upstream permissions listed there should be resolved before public distribution.

See [validation performed for this release](docs/VALIDATION.md).
