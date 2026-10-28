# Single-patch demo

```bash
conda activate isbi
python -m pip install -r requirements-demo.txt
python -m ipykernel install --user --name isbi --display-name "Python (isbi)"
jupyter lab demo/predict_patch.ipynb
```

Run all cells with the **Python (isbi)** kernel. CPU is the default.

Included files:

- `weights/effnet5_hist2st_fold5.pth`: the requested fold-5 `best_model.pth` (about 1.3 MB).
- `sample/patch.png`: patch 0 from Hist2ST slide `23287_C1`, held out in fold 5.
- `sample/expression.csv`: matching raw counts in the 845-gene panel.
- `sample/metadata.json`: checkpoint hash, source run and sample details.

The notebook prints Pearson, Spearman, MAE, MSE and MAPE across this patch's genes. These are single-patch scores, not fold-level results. Change `PATCH_PATH` and `TARGET_PATH` in the setup cell to use another example. The image-preview cell is optional. Without targets, set `TARGET_PATH = None` for prediction only.

The sample comes from [Mendeley Data v2](https://data.mendeley.com/datasets/29ntw7sh4r/2), CC BY 4.0. The patch is included unchanged; counts are restricted to the model's target genes.
