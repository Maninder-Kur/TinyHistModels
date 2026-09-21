# Preparing training inputs

Download locations and upstream access conditions are in
[DATA_SOURCES.md](DATA_SOURCES.md). No dataset, expression matrix, image, trained
checkpoint, or experiment output is distributed in this repository.

## Cohorts and filenames

Each training input directory must contain one matching `<sample_id>.h5` and
`<sample_id>.h5ad` pair per row of the corresponding manifest.

| Configuration dataset name | Samples | Patient groups | Target genes | Manifest |
| --- | ---: | ---: | ---: | --- |
| `hist2st_845` | 68 | 23 | 845 | [Hist2ST](../code_dataprocessing/manifests/hist2st_845.csv) |
| `her2` | 36 | 8 | 785 | [HER2](../code_dataprocessing/manifests/her2.csv) |
| `xenium_breast_patient` | 5 | 5 | 500 | [Xenium Breast](../code_dataprocessing/manifests/xenium_breast_patient.csv) |
| `xenium_lung_patient` | 19 | 19 | 541 | [Xenium Lung](../code_dataprocessing/manifests/xenium_lung_patient.csv) |

The manifests preserve the sample IDs and grouping rules used by the existing
training code: Hist2ST uses the prefix before `_`, HER2 uses the initial letter,
and the Xenium trainers treat each listed HEST sample as a separate patient
group. The latter is the implementation's grouping convention, not a separate
verification of donor identity from upstream clinical metadata. These manifests
identify samples and groups; they do not assign fold numbers or alter the
splitter used by a training entry point.

The four [gene lists](../code_model_training/gene_outputs/README.md) preserve the
experiment target order. Do not substitute newly selected highly variable genes
when reproducing these configurations.

## HER2

The original HER2 converter is included as
[`prepare_her2.py`](../code_dataprocessing/prepare_her2.py) with a configurable
input directory. It expects the upstream structure:

```text
her2st/data/
  ST-cnts/A1.tsv.gz
  ST-spotfiles/A1_selection.tsv
  ST-imgs/A/A1/...
```

From the repository root:

```bash
python code_dataprocessing/prepare_her2.py \
  --base_dir /path/to/her2st/data \
  --out_dir /path/to/prepared/her2 \
  --patch_size 224
```

Omitting `--sample_ids` processes all 36 manifest samples. For one section add
`--sample_ids A1`. Use a fresh output directory; the converter writes generated
files using the sample IDs.

The converter matches expression rows to spot coordinates using grid `x,y`,
converts OpenCV BGR images to RGB, takes 224 × 224 crops around integer pixel
centres, and excludes crops extending outside the image. It saves unnormalized
counts in the retained image-row order. Original `spot_id` and spatial columns
remain in the H5AD observation table. The converter needs `opencv-python`,
`scanpy`, `pandas`, `numpy`, `h5py`, and `tqdm`.

## HEST Xenium cohorts

The multi-patient experiments read HEST's prepared `patches/<id>.h5` and
`st/<id>.h5ad` files. After obtaining access to the
[HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest), authenticate
with your Hugging Face account and run:

```bash
python code_dataprocessing/download_hest.py \
  --cohort xenium_breast_patient \
  --out-dir /path/to/prepared/xenium_breast

python code_dataprocessing/download_hest.py \
  --cohort xenium_lung_patient \
  --out-dir /path/to/prepared/xenium_lung
```

The downloader requires `huggingface_hub`. It requests only the manifest file
pairs, puts each pair directly in the output directory, and writes the resolved
dataset commit to `HEST_REVISION.txt`. Pass `--revision <commit>` to request a
specific dataset revision. Existing files are skipped. The original experiment
directory did not record a HEST dataset commit, so a new upstream download is
not claimed to be byte-identical to the experiment inputs.

There is a material manuscript/source discrepancy: the paper describes
128 × 128 cell-centred Xenium patches, while every inspected multi-patient input
contains `img` arrays with shape `[N, 224, 224, 3]`, and the supplied multi-patient
trainer resizes to 224 × 224. HEST barcodes such as `000x000`, array-coordinate
fields, and patch metadata indicate the prepared HEST spatial-bin/spot format.
The [upstream HEST card](https://huggingface.co/datasets/MahmoodLab/hest)
describes its standard patches as 224 pixels at 0.5 micrometres per pixel around
ST locations. This release preserves the observed multi-patient pipeline; it
does not silently relabel these inputs as individual cells or resample them to
128 pixels. The manuscript description and intended analysis unit need to be
reconciled before claiming exact reproduction of the described cell-level task.

The separate single-sample 10x demonstration pipelines in the research folder
are not substitutes for these cohorts and are omitted, including the pancreas
and liver experiments.

## Hist2ST

Use the [linked original cohort](DATA_SOURCES.md) and the 68-sample manifest.
Prepared training pairs must follow the schema below; for example,
`23209_C1.h5` and `23209_C1.h5ad`.

The supplied research folder contains the processed pairs and exploratory
notebooks, but no recoverable converter for this exact 68-section cohort.
Consequently this release supports training from existing prepared pairs and
provides their validation contract. It does **not** claim a complete, verified
raw-download-to-training reconstruction for Hist2ST. A converter matching the
original coordinate transforms, image scaling, spot filtering, and RGB handling
is still needed to close that reproducibility gap. The HER2 converter must not
be used on Hist2ST files without checking their different source schema.

## Data contract and validation

The image HDF5 must contain unsigned 8-bit RGB pixels in `[N, H, W, 3]` order.
The source files inspected for all four cohorts were 224 × 224 pixels. H5AD
`X` contains raw expression values, `var_names` contains matching gene
identifiers, and the supplied target list must be present in every sample.
The training loaders select targets and then apply their existing normalization;
do not normalize `X` again during preparation.

| Cohort | Image key | Alignment and spatial fields |
| --- | --- | --- |
| Hist2ST | `patches` | Image rows match H5AD rows; H5 `spot_id` matches H5AD observation IDs/`spot_id`; retain `pixel_x`, `pixel_y`, `coords` and H5AD `X`, `Y` observation columns for spatial baselines. |
| HER2 | `patches` | Image rows match H5AD rows; retain H5 and H5AD `x`, `y`, `pixel_x`, `pixel_y`, and H5AD `spot_id`. |
| HEST Xenium | `img` | H5 `barcode` has shape `[N, 1]` and is matched to H5AD `obs_names`; retain H5 `coords` and H5AD spatial metadata. H5AD may have more rows than the image file. |

Run the validator before training:

```bash
python code_dataprocessing/validate_data.py \
  --cohort hist2st_845 --data-dir /path/to/prepared/hist2st
python code_dataprocessing/validate_data.py \
  --cohort her2 --data-dir /path/to/prepared/her2
python code_dataprocessing/validate_data.py \
  --cohort xenium_breast_patient --data-dir /path/to/prepared/xenium_breast
python code_dataprocessing/validate_data.py \
  --cohort xenium_lung_patient --data-dir /path/to/prepared/xenium_lung
```

This tool reads metadata, gene identifiers, spatial row identifiers and barcodes;
it does not load image tensors or expression matrices. It checks required file
pairs, RGB shape/dtype, target availability, uniqueness, and available alignment
fields. Passing does not establish raw-data provenance, correct image colour
conversion, or equality of model results. Add `--report /path/to/report.json`
to save details outside the repository.

During release preparation, metadata checks passed for all 128 local file pairs.
The lung sample `NCBI865` had one image barcode absent from its H5AD; the original
Xenium loader skips such unmatched images, and the validator reports a warning.
No datasets were copied to the release folder and no complete model training
was rerun for this check.
