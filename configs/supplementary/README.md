# Supplementary runs

Each cohort has four configs:

| Config | Setup |
| --- | --- |
| `stpath_zero.json` | Earlier STPath wrapper: zero coordinates and gene inputs, zero organ/technology IDs, independent attention group for each patch, task MLP head. |
| `effnet_no_teacher.json` | EffNet-5 trained with MSE; same architecture and scaling factor as the corresponding KD student. |
| `stnet_no_teacher.json` | TinySTNet trained with MSE. |
| `histogene_no_teacher.json` | TinyHisToGene trained with MSE. |

The no-teacher configs use `use_kd: false` and need no teacher checkpoint. The zero-input wrapper is kept as trained; it is separate from the spatial STPath baseline.

```bash
python scripts/train.py --config configs/supplementary/hist2st/stnet_no_teacher.json
python scripts/train.py --config configs/supplementary/hist2st/stpath_zero.json
```

Results from these runs belong in the supplementary material. Result tables are not included here.
