# Third-party sources and pretrained models

This document identifies the upstream work used by the source release. The original authors retain ownership of their code, data, and model weights. No project-wide license has been selected for the authors' own experiment code.

## Vendored model source

The commits below were recovered from the nested Git repositories in the working archive. Local adaptations may differ from those commits; a commit identifies the upstream checkout, not a guarantee that every adapted file is unchanged.

| Component / included location | Upstream | Archived upstream commit | License evidence |
| --- | --- | --- | --- |
| THItoGene, `code_model_training/models/THItoGene/` | [yrjia1015/THItoGene](https://github.com/yrjia1015/THItoGene) | `13daf4d9c2531a74937dfedeeb4cb905cf0bf6cb` | No repository license file found in this commit or the inspected upstream listing. |
| TCGN, `code_model_training/models/TCGN/` | [lugia-xiao/TCGN](https://github.com/lugia-xiao/TCGN) | `7faf955b3fbf3bfbdb9b5c83eefa70afcfb85202` | No repository license file found in this commit or the inspected upstream listing. |
| STPath, `code_model_training/models/STPath/` | [Graph-and-Geometric-Learning/STPath](https://github.com/Graph-and-Geometric-Learning/STPath) | `3b10e6d67ad72e06b2291e2ca2ab801b2c0c511c` | No repository license file found in this commit or the inspected upstream listing. |
| Stem, `code_model_training/models/Stem/` | [SichenZhu/Stem](https://github.com/SichenZhu/Stem) | `cbc3c7c558cd097d93a4c37318838bb4cd780a1e` | MIT; original copyright and license retained in [licenses/Stem-MIT.txt](licenses/Stem-MIT.txt). |

The Stem license was recovered verbatim from `LICENSE` at the archived commit and checked against the [upstream file at that commit](https://github.com/SichenZhu/Stem/blob/cbc3c7c558cd097d93a4c37318838bb4cd780a1e/LICENSE). Stem's diffusion implementation also contains attribution to upstream diffusion projects in its file headers; these notices are retained.

No license has been invented for THItoGene, TCGN, or STPath. Their redistribution permission remains unresolved in the available source record. The release owner should establish that permission before publicly distributing those vendored components. Public visibility and a research citation alone do not supply a repository license. This source inventory is not a legal determination about permissions that the authors may separately hold.

The original implementations and papers should be cited when using these models:

- THItoGene: Jia et al., *THItoGene: a deep learning method for predicting spatial transcriptomics from histological images* ([upstream citation](https://github.com/yrjia1015/THItoGene#citation)).
- TCGN: Xiao et al., *Transformer with Convolution and Graph-Node co-embedding* ([upstream project](https://github.com/lugia-xiao/TCGN)).
- STPath: Huang et al., *STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images* ([upstream project](https://github.com/Graph-and-Geometric-Learning/STPath)).
- Stem: Zhu et al., *Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images* ([upstream project](https://github.com/SichenZhu/Stem)).

## Pretrained weights

| Weights | Official download page | Access / license as stated upstream |
| --- | --- | --- |
| UNI | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | Gated access; CC BY-NC-ND 4.0 and model-specific research-use conditions. Each user must obtain their own access. |
| GigaPath tile encoder | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | Gated access; model card lists Apache-2.0 and additional use notices. |
| STPath | [tlhuang/STPath](https://huggingface.co/tlhuang/STPath) | No model license declaration found in the inspected card. |

No pretrained weights are bundled. The trainable Stem module is initialized by the training code; its frozen UNI encoder requires the UNI weights. STPath requires both the STPath and GigaPath weights. See [data and weight paths](docs/DATA_SOURCES.md).

## Other model families and dependencies

ST-Net/DenseNet, DeepSpaCE/VGG, EfficientNet, and HisToGene are exposed through experiment adapters in `code_model_training/models/`. The release also contains the paper's compact EfficientNet and student architectures. PyTorch/torchvision, timm, PyTorch Geometric, and other installed packages retain their own licenses; installation does not relicense those dependencies. The unneeded baseline checkout copies, model checkpoints, and example datasets from the working archive are excluded.

Source listings and model cards checked on 2026-09-21. Missing-license entries describe the evidence available at that date.
