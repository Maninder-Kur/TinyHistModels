# Configurations

Use `configs/<cohort>/<model>.json`. Cohorts: `her2`, `hist2st`, `xenium_breast`, `xenium_lung`.

- Baselines: `stnet`, `deepspace`, `histogene`, `effnet`, `tcgn`, `uni`, `stem`, `stpath`, `stpath_small_scratch`, `gigapath_flash`.
- THItoGene: `thitogene` for HER2 and Hist2ST.
- EfficientNet scaling: `effnet_3`, `effnet_5`, `effnet_10`, `effnet_20`. `effnet` is the full model.
- KD: `stnet_kd`, `histogene_kd`, `effnet_kd`.
- Supplementary runs: [supplementary/README.md](supplementary/README.md).

```bash
python scripts/train.py --config configs/xenium_breast/stpath.json --dry-run
```

Paths are relative to the repository root. The launcher selects the required training script. Config sources and hashes are recorded in [experiments.json](experiments.json); a config records settings, not a claim that all folds finished.
