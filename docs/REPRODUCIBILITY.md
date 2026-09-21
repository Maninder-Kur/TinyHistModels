# Reproduction notes

This is a curated training-source release, not a claim that every reported number has been independently regenerated. No full training runs were performed during packaging.

## Configuration provenance

The 62 JSON configurations were recovered from the saved `experiment_info.json` files referenced by the manuscript figure-input registry. Four additional EffNet-100 entries reused the corresponding full EfficientNet run and are represented by `effnet.json`. [experiments.json](../configs/experiments.json) records source run IDs and SHA-256 hashes of the original configuration files.

Numerical hyperparameters are preserved. Absolute machine paths were replaced with repository-relative paths; unused machine-path fields were removed. Non-KD entrypoints have `use_kd` set to false where an unused stale flag existed. KD teacher paths are explicit placeholders for newly trained fold-matched teachers. Gene NPY files preserve their original bytes and ordering. Current source files were available, but immutable source snapshots for every historical run were not.

The HER2 TCGN config additionally records `embed_dims: [52, 104, 208, 416]`, recovered from the archived teacher-free baseline checkpoint's four patch-embedding tensor shapes. The current generic source default had smaller widths and did not match that reported run. The recovered setting gives the paper's 25,138,985 trainable parameters; the other cohorts retain the existing default widths. No checkpoint is distributed.

## Differences between the manuscript and available implementation

| Topic | Manuscript | Archived configuration / source used here |
| --- | --- | --- |
| Scheduler patience | 30 | 5 for the recovered regression/KD runs. |
| Batch size | Generally 128 | UNI 64; THItoGene 1; STPath 16 on HER2, 64 on Hist2ST, 128 on Xenium. |
| THItoGene input | Config image dimensions 224 | Dedicated THItoGene transforms actually resize to 112 × 112; batch size 1 represents a whole-slide graph. |
| EfficientNet scaling | Depth, width, and resolution scaling | The available student implementation scales depth and width; trainer input resolution stays 224 × 224. |
| STPath training | Generic setup | Spot-level configs use 100 epochs and learning rate 5e-5; early-stopping patience 20. See the individual configurations. |
| Seed | 123 | Model/subset seed 123; spot-level patient grouping uses GroupKFold with shuffle and seed 42. Stem reseeds model training with 123 plus the fold index. Preserve the existing entrypoint behavior. |
| Xenium images | 128-pixel cell-centred patches | Archived HEST patient inputs and patient training transforms use 224-pixel region patches. The released input contract preserves the archive; it does not establish the manuscript's cell-level interpretation. |
| KD loss | Direct prediction MSE against ground truth and teacher | `KDLoss` uses 0.8 × ground-truth MSE plus 0.2 × MSE between per-sample mean-centred student/teacher predictions. `lambda_reg` is not this mixing coefficient. |
| Stem training | Generic optimizer/checkpoint description | The dedicated source uses AdamW, fixed epochs and EMA, without the direct-regression scheduler/early stopping. Its `best_model.pth` is the final EMA model, not an SRCC-selected epoch. |

Some historical configs contain `pretrained: true` for HisToGene, TCGN, or scaled EfficientNet, but their corresponding constructors ignore that field. The CNN baseline configs explicitly disable pretrained initialization. UNI, STPath, and Stem's image encoder require their specified pretrained assets.

## Data and splits

The release contains the exact selected sample IDs and target genes. HER2 sections are grouped by the first character of the sample ID; Hist2ST sections by the numeric prefix. Xenium training uses the configured ordered patient IDs, five folds, and deterministic spot subsets. Preserve the sample order, barcode alignment, subset seed, normalization, gene order, and library versions. Do not substitute row-number alignment for barcode alignment.

The inherited preprocessing normalizes target-gene expression to a total of 1,000,000 and applies log1p. The dataset adapters retain their existing barcode handling and transforms; this packaging work does not redefine those algorithms. The HEST downloader records its resolved dataset revision, but the exact revision of the original downloaded archive was unavailable. A current upstream download is therefore not guaranteed byte-identical to the training inputs.

The raw Hist2ST-to-H5/H5AD conversion script was absent from the archive. The source links, exact sample manifest, observed input specification, and validator are provided; reproducing the exact prepared Hist2ST inputs still requires those paired files or recovery of the original converter. See [DATA_PREPARATION.md](DATA_PREPARATION.md).

## STPath limitation

The STPath model wrapper, upstream components, checkpoints paths, and historical experiment configurations were available, but the historical training entrypoint was not. The released regression scripts now connect the existing wrapper using those recorded settings. This is a reconstructed adapter and cannot be treated as a verified reconstruction of the reported STPath runs.

The archived wrapper also looked up a nonexistent tokenizer `voc` attribute and silently used padding tokens for organ/technology labels. The release uses the available tokenizer API to resolve the configured labels. This is an explicit behavior correction; it can change outputs relative to that faulty wrapper. The frozen GigaPath extractor and trainable STPath module still require their original external weights. Complete weight-backed STPath inference and retraining have not been validated during packaging.

## Portability changes

Imports resolve from source-file locations. Ambiguous THItoGene imports were made explicit, unused eager optional imports were removed, and machine-specific checkpoint paths were replaced by configurable paths. Device-specific allocations in vendored TCGN helpers were made input-device aware. Training loops and model math were retained except for documented compatibility fixes. [VALIDATION.md](VALIDATION.md) records the checks actually performed.

The THItoGene Hist2ST loader now accepts either dense or sparse AnnData expression storage, without changing values. The obsolete scheduler `verbose` keyword was removed for compatibility with newer PyTorch.

The pinned direct dependencies reflect the available release-check environment, not a historical lockfile. The source's training-time SRCC uses `torchsort` soft ranks and is not interchangeable with the standalone evaluation pipeline that was omitted at the author's request. Training metrics should not be presented as regenerated final paper tables.
