# TinyHistModels

Code for **A Scale-Aware Study of Spatial Gene Expression Prediction from Histology Images**.

## Setup

Python 3.10. Package versions in `requirements.txt` are taken from our `isbi` environment.

```bash
# Skip this line if the isbi environment already exists.
conda create -n isbi python=3.10 -y
conda activate isbi
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cu126-cp310-cp310-linux_x86_64.whl'
python -m pip install -r requirements.txt
```

These wheel commands are for Linux with CUDA 12.6. For CPU, replace `cu126` with `cpu` in both URLs. Other platforms need a matching [torchsort wheel](https://github.com/teddykoker/torchsort#install).

## Data

We use Hist2ST, HER2, Xenium Breast and Xenium Lung. [Download and preparation](docs/DATA.md).

```text
data/
  hist2st/         # 23209_C1.h5, 23209_C1.h5ad, ...
  her2/            # A1.h5, A1.h5ad, ...
  xenium_breast/   # NCBI783.h5, NCBI783.h5ad, ...
  xenium_lung/     # NCBI856.h5, NCBI856.h5ad, ...
```

Keep datasets and pretrained encoders outside Git. Ordered gene lists are in `code_model_training/gene_outputs/`.

## Models

ST-Net, DeepSpaCE, HisToGene, EfficientNet-B0, TCGN, THItoGene, UNI, Stem, STPath, STPathSmallScratch and GigaPathFlash. We also include scaled EfficientNet and knowledge-distillation models.

- **STPath:** real spatial coordinates, neighbouring patches, masked gene input and organ/technology tokens.
- **STPathSmallScratch:** frozen pretrained GigaPath encoder with a randomly initialized compact spatial transformer and regression head. No STFM checkpoint.
- **GigaPathFlash:** pretrained small tile encoder with a linear gene-regression head.

See [model details and weight paths](docs/MODELS.md).

## Training

Run from the repository root:

```bash
python scripts/train.py --config configs/hist2st/effnet_5.json
python scripts/train.py --config configs/hist2st/stpath.json
python scripts/train.py --config configs/hist2st/stpath_small_scratch.json
python scripts/train.py --config configs/her2/gigapath_flash.json
```

Use `--data-dir`, `--assets-dir` or `--output-dir` to override paths. `--dry-run` prints the resolved config. Outputs are saved under `outputs/`.

For KD, first train the matching full-size teacher, then pass its run directory:

```bash
python scripts/train.py --config configs/her2/effnet_kd.json \
  --teacher-run outputs/her2/effnet/YOUR_RUN
```

The teacher run must contain `fold_1/best_model.pth` through `fold_5/best_model.pth` and use the same folds and gene order. [Configuration guide](configs/README.md).

## Supplementary models

`configs/supplementary/` contains the earlier zero-input STPath and compact models trained without a teacher. [Details](configs/supplementary/README.md).

## Demo

[demo/predict_patch.ipynb](demo/predict_patch.ipynb) includes an EffNet-5 fold-5 checkpoint, one held-out Hist2ST patch and its gene counts. It prints patch-wise Pearson, Spearman, MAE, MSE and MAPE. [Run the demo](demo/README.md).

Only training validation and this single-patch demo are included. There are no standalone evaluation or significance-analysis scripts. Third-party sources are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
