# Models

| Config name | Implementation |
| --- | --- |
| `stnet` | DenseNet-121 regression |
| `deepspace` | VGG-16 regression |
| `histogene` | HisToGene transformer |
| `effnet` | EfficientNet-B0 |
| `effnet_3`, `effnet_5`, `effnet_10`, `effnet_20` | Compact EfficientNet; depth and width scaling, 224-pixel input |
| `tcgn` | CNN, graph and transformer model |
| `thitogene` | Whole-slide graph model; 112-pixel patches, HER2/Hist2ST only |
| `uni` | UNI encoder with a linear head |
| `stem` | Diffusion model conditioned on frozen UNI features |
| `stpath` | Spatial windows, pretrained STFM and its gene prediction head |
| `stpath_small_scratch` | Random compact spatial transformer and MLP head; frozen pretrained GigaPath |
| `gigapath_flash` | Small pretrained GigaPath tile encoder and linear head |

## Pretrained files

Download each encoder separately and accept its upstream access terms.

| Model | Download | File under `assets/` |
| --- | --- | --- |
| UNI / Stem | [UNI](https://huggingface.co/MahmoodLab/UNI) | `uni/pytorch_model.bin` |
| All STPath variants | [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `gigapath/pytorch_model.bin` |
| STPath / STPathZero | [STPath](https://huggingface.co/tlhuang/STPath) | `stpath/stfm.pth` |
| GigaPathFlash | [GigaPath-Flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash) | `gigapath_flash/pytorch_model.bin` |

`STPathSmallScratch` does not load `stfm.pth`. Its trainable part uses width 512, one spatial block, four attention heads and a 1536-unit MLP head. GigaPathFlash uses only the tile encoder; its slide encoder and FlashAttention package are not required.

## Training settings

Use the JSON settings for each experiment. Model seed is 123; patient-fold seed is 42. The saved regression runs use LR-scheduler patience 5. Stem uses its separate EMA diffusion-training loop. KD uses `0.8 * MSE(student, target) + 0.2 * MSE(center(student), center(teacher))`, where centering is across each patch's genes.

Configs preserve saved run settings. The pretrained flags in older HisToGene, TCGN and compact EfficientNet configs are unused by those constructors. HER2 TCGN widths `[52, 104, 208, 416]` match its saved checkpoint.
