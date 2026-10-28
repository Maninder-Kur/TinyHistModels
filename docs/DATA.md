# Data

| Cohort | Samples / patient groups | Genes | Download |
| --- | --- | --- | --- |
| Hist2ST | 68 / 23 | 845 | [Mendeley Data v2](https://data.mendeley.com/datasets/29ntw7sh4r/2) |
| HER2 | 36 / 8 | 785 | [HER2ST](https://github.com/almaan/her2st) |
| Xenium Breast | 5 / 5 | 500 | [HEST](https://huggingface.co/datasets/MahmoodLab/hest) |
| Xenium Lung | 19 / 19 | 541 | [HEST](https://huggingface.co/datasets/MahmoodLab/hest) |

Sample IDs and patient groups are in [manifests](../code_dataprocessing/manifests/). Use the supplied ordered [gene lists](../code_model_training/gene_outputs/).

## Prepared files

Each sample needs `<sample_id>.h5` and `<sample_id>.h5ad` in its cohort folder. H5 stores RGB uint8 patches with shape `[N, 224, 224, 3]`. H5AD stores raw counts and gene names; training selects the panel, normalizes to 1e6 and applies log1p.

- Hist2ST/HER2 image key: `patches`. Keep spot IDs, image-row order and pixel coordinates.
- HEST image key: `img`, with `barcode` `[N, 1]` and `coords`. Expression rows are aligned by barcode.
- STPath needs real coordinates (`pixel_x/pixel_y`, H5 `coords`, or the matching H5AD spatial fields).

The current HEST inputs are 224-pixel spatial patches. Do not replace them with the separate single-sample 10x demonstration data. The original Hist2ST raw-to-H5 converter is unavailable; use matching prepared pairs. The validator checks file structure and target genes, not image-processing equivalence.

## HEST download

Accept the upstream dataset terms, then:

```bash
hf auth login
python code_dataprocessing/download_hest.py --cohort xenium_breast_patient --out-dir data/xenium_breast
python code_dataprocessing/download_hest.py --cohort xenium_lung_patient --out-dir data/xenium_lung
```

Only the selected pairs are downloaded. The script records the dataset revision; the historical download revision was not recorded.

## HER2 preparation

```bash
python -m pip install -r requirements-data.txt
python code_dataprocessing/prepare_her2.py --base_dir /path/to/her2st/data --out_dir data/her2 --patch_size 224
```

The input directory must contain `ST-cnts`, `ST-spotfiles` and `ST-imgs`. The converter keeps RGB crops and matching raw counts.

## Check prepared inputs

```bash
python code_dataprocessing/validate_data.py --cohort hist2st_845 --data-dir data/hist2st
```

Other cohort names: `her2`, `xenium_breast_patient`, `xenium_lung_patient`. HEST sample IDs are used as patient groups, following the training code. Keep the original order for matching folds.
