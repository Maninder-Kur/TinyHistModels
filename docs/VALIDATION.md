# Release validation

Checks performed on 2026-09-21. These checks establish packaging and basic execution behavior, not reproduction of the paper's final scores.

## Passed

- All Python source parses and all JSON configurations load.
- All 62 configuration-to-entrypoint mappings match the experiment registry; all launcher dry runs work from outside the repository.
- Missing data and unresolved KD teacher paths fail before training. The launcher checks the complete cohort manifest when configs omit explicit sample IDs.
- All eight training entrypoints accept `--help` in the Python 3.10 validation environment.
- CPU forward passes for EfficientNet-B0, ST-Net, DeepSpaCE/VGG, HisToGene, TCGN, THItoGene, and the three compressed architecture families.
- Backward passes for a compact EfficientNet and the Stem diffusion loss; STPath's STFM component forward pass and organ/technology token lookup.
- Recovered compressed-model parameter counts match the archived figure registry for all four cohorts. HER2 TCGN checkpoint shapes recover the reported 25,138,985-parameter architecture.
- A temporary synthetic HER2 dataset exercised `main2.py` for one epoch over two patient folds, including data loading, optimization, scheduler, train/validation metrics, and checkpoint writes. This used random data and a compact EfficientNet. Temporary data and outputs were removed.
- Metadata validation passed for all 128 existing input pairs: 68 Hist2ST, 36 HER2, 5 Xenium Breast, and 19 Xenium Lung. The existing loader's handling of one unmatched NCBI865 barcode is documented in the preparation guide.
- Four gene arrays are byte-identical to the originals. Readable gene lists preserve their order. Sample manifests agree with training sample order, and Stem/direct spot-level validation-fold memberships agree under scikit-learn 1.7.2.
- No datasets, pretrained/trained weights, archived results, notebooks, evaluation suites, or nested Git repositories are included. Local machine paths and credential patterns were checked.

The environment used Python 3.10.19, PyTorch 2.6.0+cu126, torchvision 0.21.0+cu126, torchsort 0.1.10, timm 1.0.22, NumPy 1.26.4, and scikit-learn 1.7.2. Smoke execution used CPU. Direct dependency versions are recorded in `requirements.txt`.

## Not performed

No full cohort training, final paper-table regeneration, CPU benchmark reproduction, significance analysis, fresh-environment installation, gated data/weight downloads, or complete pretrained UNI/STPath inference was performed. The HER2 raw converter was inspected and its CLI checked, but not rerun on the complete raw cohort. Hist2ST raw conversion remains unavailable. See [reproduction notes](REPRODUCIBILITY.md) for scientific and provenance limitations.
