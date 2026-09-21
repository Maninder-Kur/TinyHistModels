# Data sources

The paper, *A Scale-Aware Study of Spatial Gene Expression Prediction from Histology Images*, compares four cohorts. This source release contains gene lists and sample manifests; download the image and expression data separately. Paths and input preparation are documented in [DATA_PREPARATION.md](DATA_PREPARATION.md).

| Paper cohort | Config dataset name | Patients / sections in paper | Target genes | Original source |
| --- | --- | --- | --- | --- |
| Hist2ST | `hist2st_845` | 23 / 68 | 845 | [Mendeley Data, version 2](https://data.mendeley.com/datasets/29ntw7sh4r/2) |
| HER2 | `her2` | 8 / 36 | 785 | [HER2ST project](https://github.com/almaan/her2st) |
| Xenium Breast | `xenium_breast_patient` | 5 / 5 | 500 | [HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest) |
| Xenium Lung | `xenium_lung_patient` | 19 / 19 | 541 | [HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest) |

“Hist2ST” is the dataset label used in this paper and its code. The underlying source is the 23-patient *Human breast cancer in situ capturing transcriptomics* collection. This label should not be confused with the separate Hist2ST prediction model. Mendeley version 2 excludes the defective section `23901_C1` and supplies patient/replicate metadata. Its dataset page specifies CC BY 4.0. Use the released [sample manifests](../code_dataprocessing/manifests/) and ordered gene lists to preserve the experiment's cohort and prediction targets.

The HER2ST source provides the spot counts, coordinates, and histology used to prepare the HER2 inputs. Its repository explains the `data/ST-cnts`, `data/ST-spotfiles`, and image directories. See the preparation guide for the included HER2 converter and its expected raw layout. The paper groups these two cohorts under “Visium”; the original data sources should be consulted for the precise assay terminology.

## HEST cohorts

Accept the dataset's access conditions using your own Hugging Face account before downloading. The [HEST dataset card](https://huggingface.co/datasets/MahmoodLab/hest) provides authentication and subset download instructions, and lists the dataset license as CC BY-NC-SA 4.0. Use an access token with the required read access; keep it outside this repository.

Download only the selected IDs:

- Breast: `NCBI783`, `NCBI785`, `TENX94`, `TENX95`, `TENX99`.
- Lung: `NCBI856`, `NCBI857`, `NCBI858`, `NCBI859`, `NCBI860`, `NCBI861`, `NCBI864`, `NCBI865`, `NCBI866`, `NCBI867`, `NCBI870`, `NCBI873`, `NCBI876`, `NCBI879`, `NCBI880`, `NCBI881`, `NCBI882`, `NCBI883`, `NCBI884`.

For each ID, HEST distributes `patches/<ID>.h5` and `st/<ID>.h5ad`. Retain matching image/expression barcodes and gene identifiers when preparing these pairs. The [HEST code and documentation](https://github.com/mahmoodlab/HEST) describe the data organization and upstream processing.

The archived patient inputs inspected for this release contain 224 × 224 RGB patches under `img`, plus `barcode` and `coords`. This matches HEST's documented standard patch format. The manuscript instead describes 128 × 128 patches centered on individual cells. These are different specifications: this release preserves the archived HEST input contract. It does not assert that the supplied HEST pairs implement the manuscript's cell-centroid preprocessing. See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the observed formats and remaining provenance limitations.

The separate single-sample 10x breast, lung, pancreas, and liver demonstrations in the working archive are outside this release. In particular, the HEST IDs beginning with `TENX` are still part of the paper's five-sample breast cohort; they should not be dropped because of their prefix.

## Pretrained weights

Weights are downloaded independently of the datasets. Follow each model's own access conditions; no weights or authentication tokens are included here.

| Experiments | Original weight source | Expected local file |
| --- | --- | --- |
| UNI regression and Stem's frozen image encoder | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | `assets/uni/pytorch_model.bin` |
| STPath's frozen GigaPath tile encoder | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | `assets/gigapath/pytorch_model.bin` |
| STPath prediction module | [tlhuang/STPath](https://huggingface.co/tlhuang/STPath) | `assets/stpath/stfm.pth` |

UNI and GigaPath require individual acceptance of their Hugging Face access conditions. UNI's card specifies CC BY-NC-ND 4.0 and additional research-use conditions; GigaPath's card lists Apache-2.0 and its own use notices. STPath's card did not specify a model license when checked. Refer to [THIRD_PARTY.md](../THIRD_PARTY.md) for source provenance and license status. The selected weight files must match the architectures expected by the archived adapters; newly released model variants are not interchangeable.

Primary source pages checked on 2026-09-21. No gated data or weight downloads were performed while preparing the release.
