# Hist2ST demos

```bash
conda activate isbi
python -m pip install 'jupyterlab>=4,<5'
python -m ipykernel install --user --name isbi --display-name "Python (isbi)"
jupyter lab demo/
```

Open either notebook and run all cells with the **Python (isbi)** kernel. CPU is the default.

Included files:

- `weights/effnet5_hist2st_fold5.pth`: the requested fold-5 `best_model.pth` (about 1.3 MB).
- `sample/patch.png`: patch 0 from Hist2ST slide `23287_C1`, held out in fold 5.
- `sample/expression.csv`: matching raw counts in the 845-gene panel.
- `sample/metadata.json`: checkpoint hash, source run and sample details.
- `predict_stpath_small_flash.ipynb`: the STPath Small–Flash spatial model on fold-3 Hist2ST slide `23270_D2`.
- `weights/stpath_small_flash_hist2st_fold3_head.pth`: fold-3 trained spatial layers (about 18 MB); the frozen Flash encoder is omitted.
- `sample/stpath_flash_patch.png` and `sample/stpath_flash_expression.csv`: target patch 0 and its raw gene counts.
- `sample/stpath_flash_window.npz`: saved GigaPath-Flash embeddings (`features`, 256 × 384), normalized coordinates (`coords`, 256 × 2), spot IDs and source patch indices (about 0.4 MB).
- `sample/stpath_flash_metadata.json`: hashes and provenance for the window and spatial model.

Both notebooks print Pearson, Spearman, MAE, MSE and MAPE across the target patch's genes. These are single-patch scores, not fold-level results. In `predict_patch.ipynb`, change `PATCH_PATH` and `TARGET_PATH` in the setup cell to use another example. The image-preview cell is optional. Without targets, set `TARGET_PATH = None` for prediction only.

`run_demo()` uses the saved embeddings and runs without a Hugging Face login. To extract embeddings from the original patches, download prepared `23270_D2.h5` and `23270_D2.h5ad` into `data/hist2st/`, accept the [GigaPath-Flash access terms](https://huggingface.co/prov-gigapath/prov-gigapath-flash), run `hf auth login`, then change the notebook's final call to `run_demo(source="images", data_dir=ROOT / "data/hist2st")`. The notebook downloads the tile weights into `assets/gigapath_flash/` if they are missing, extracts 256 patch embeddings, and calls the same spatial model. The upstream weights and full Hist2ST slide are not stored in Git.

The sample comes from [Mendeley Data v2](https://data.mendeley.com/datasets/29ntw7sh4r/2), CC BY 4.0. The patch is included unchanged; counts are restricted to the model's target genes.
