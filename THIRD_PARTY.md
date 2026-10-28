# Third-party code

| Component | Source | Included files / license |
| --- | --- | --- |
| THItoGene | [yrjia1015/THItoGene](https://github.com/yrjia1015/THItoGene) | Model layers under `models/THItoGene`; no license file in the source checkout. |
| TCGN | [lugia-xiao/TCGN](https://github.com/lugia-xiao/TCGN) | Model blocks under `models/TCGN`; no license file in the source checkout. |
| STPath | [Graph-and-Geometric-Learning/STPath](https://github.com/Graph-and-Geometric-Learning/STPath) | Model, tokenizer and vocabulary files; no license file in the source checkout. |
| Stem | [SichenZhu/Stem](https://github.com/SichenZhu/Stem) | [MIT license](licenses/Stem-MIT.txt). |
| GigaPath-Flash | [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) | Tile encoder only; [Apache-2.0 license](licenses/GigaPath-Apache-2.0.txt). |

Paths above are under `code_model_training/`. Local changes cover import paths, device handling and the gene-regression wrappers. Original notices are retained.

Pretrained encoder weights are downloaded separately; see [weight links](docs/MODELS.md). The only bundled checkpoint is the project's own compact EffNet-5 demo model. The demo sample is from [Mendeley Data v2](https://data.mendeley.com/datasets/29ntw7sh4r/2), CC BY 4.0.

A project license has not been selected. Confirm permission for the upstream components without license files before public distribution.
