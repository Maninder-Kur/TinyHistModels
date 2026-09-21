# Experiment configurations

These 62 configurations were recovered from saved `experiment_info.json` files linked by the manuscript figure inputs. Numeric hyperparameters are retained. HER2 TCGN widths were additionally recovered from its checkpoint tensor shapes. Paths are portable; unused machine paths are removed; `use_kd` is false for non-KD entrypoints. `EffNet-100` is the same run as `effnet.json`. The source run identifiers and original config hashes are in [experiments.json](experiments.json).

Run from the repository root with `python scripts/train.py --config <path>`. Config paths are always relative to the repository root. See [reproduction notes](../docs/REPRODUCIBILITY.md) before interpreting agreement with the paper.

| Cohort | Paper model | Config | Training script |
| --- | --- | --- | --- |
| her2 | EfficientNet-B0 | [effnet](her2/effnet.json) | `main2.py` |
| her2 | ST-Net | [stnet](her2/stnet.json) | `main2.py` |
| her2 | DeepSpaCE | [deepspace](her2/deepspace.json) | `main2.py` |
| her2 | HisToGene | [histogene](her2/histogene.json) | `main2.py` |
| her2 | TCGN | [tcgn](her2/tcgn.json) | `main2.py` |
| her2 | THItoGene | [thitogene](her2/thitogene.json) | `main2_thi_her2.py` |
| her2 | UNI | [uni](her2/uni.json) | `main2.py` |
| her2 | STpath | [stpath](her2/stpath.json) | `main2.py` |
| her2 | Stem | [stem](her2/stem.json) | `stem_visium_patient.py` |
| her2 | EffNet-C | [effnet_kd](her2/effnet_kd.json) | `main2_kd.py` |
| her2 | ST-Net-C | [stnet_kd](her2/stnet_kd.json) | `main2_kd.py` |
| her2 | HisToGene-C | [histogene_kd](her2/histogene_kd.json) | `main2_kd.py` |
| hist2st | EfficientNet-B0 | [effnet](hist2st/effnet.json) | `main2.py` |
| hist2st | ST-Net | [stnet](hist2st/stnet.json) | `main2.py` |
| hist2st | DeepSpaCE | [deepspace](hist2st/deepspace.json) | `main2.py` |
| hist2st | HisToGene | [histogene](hist2st/histogene.json) | `main2.py` |
| hist2st | TCGN | [tcgn](hist2st/tcgn.json) | `main2.py` |
| hist2st | THItoGene | [thitogene](hist2st/thitogene.json) | `main2_thi_hist2st.py` |
| hist2st | UNI | [uni](hist2st/uni.json) | `main2.py` |
| hist2st | STpath | [stpath](hist2st/stpath.json) | `main2.py` |
| hist2st | Stem | [stem](hist2st/stem.json) | `stem_visium_patient.py` |
| hist2st | EffNet-C | [effnet_kd](hist2st/effnet_kd.json) | `main2_kd.py` |
| hist2st | ST-Net-C | [stnet_kd](hist2st/stnet_kd.json) | `main2_kd.py` |
| hist2st | HisToGene-C | [histogene_kd](hist2st/histogene_kd.json) | `main2_kd.py` |
| xenium_breast | EfficientNet-B0 | [effnet](xenium_breast/effnet.json) | `main2_xenium_patient.py` |
| xenium_breast | ST-Net | [stnet](xenium_breast/stnet.json) | `main2_xenium_patient.py` |
| xenium_breast | DeepSpaCE | [deepspace](xenium_breast/deepspace.json) | `main2_xenium_patient.py` |
| xenium_breast | HisToGene | [histogene](xenium_breast/histogene.json) | `main2_xenium_patient.py` |
| xenium_breast | TCGN | [tcgn](xenium_breast/tcgn.json) | `main2_xenium_patient.py` |
| xenium_breast | UNI | [uni](xenium_breast/uni.json) | `main2_xenium_patient.py` |
| xenium_breast | STpath | [stpath](xenium_breast/stpath.json) | `main2_xenium_patient.py` |
| xenium_breast | Stem | [stem](xenium_breast/stem.json) | `stem_xenium_patient.py` |
| xenium_breast | EffNet-C | [effnet_kd](xenium_breast/effnet_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_breast | ST-Net-C | [stnet_kd](xenium_breast/stnet_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_breast | HisToGene-C | [histogene_kd](xenium_breast/histogene_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_breast | EffNet-3 | [effnet_3](xenium_breast/effnet_3.json) | `main2_xenium_patient.py` |
| xenium_breast | EffNet-5 | [effnet_5](xenium_breast/effnet_5.json) | `main2_xenium_patient.py` |
| xenium_breast | EffNet-10 | [effnet_10](xenium_breast/effnet_10.json) | `main2_xenium_patient.py` |
| xenium_breast | EffNet-20 | [effnet_20](xenium_breast/effnet_20.json) | `main2_xenium_patient.py` |
| xenium_lung | EfficientNet-B0 | [effnet](xenium_lung/effnet.json) | `main2_xenium_patient.py` |
| xenium_lung | ST-Net | [stnet](xenium_lung/stnet.json) | `main2_xenium_patient.py` |
| xenium_lung | DeepSpaCE | [deepspace](xenium_lung/deepspace.json) | `main2_xenium_patient.py` |
| xenium_lung | HisToGene | [histogene](xenium_lung/histogene.json) | `main2_xenium_patient.py` |
| xenium_lung | TCGN | [tcgn](xenium_lung/tcgn.json) | `main2_xenium_patient.py` |
| xenium_lung | UNI | [uni](xenium_lung/uni.json) | `main2_xenium_patient.py` |
| xenium_lung | STpath | [stpath](xenium_lung/stpath.json) | `main2_xenium_patient.py` |
| xenium_lung | Stem | [stem](xenium_lung/stem.json) | `stem_xenium_patient.py` |
| xenium_lung | EffNet-C | [effnet_kd](xenium_lung/effnet_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_lung | ST-Net-C | [stnet_kd](xenium_lung/stnet_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_lung | HisToGene-C | [histogene_kd](xenium_lung/histogene_kd.json) | `main2_xenium_patient_kd.py` |
| xenium_lung | EffNet-3 | [effnet_3](xenium_lung/effnet_3.json) | `main2_xenium_patient.py` |
| xenium_lung | EffNet-5 | [effnet_5](xenium_lung/effnet_5.json) | `main2_xenium_patient.py` |
| xenium_lung | EffNet-10 | [effnet_10](xenium_lung/effnet_10.json) | `main2_xenium_patient.py` |
| xenium_lung | EffNet-20 | [effnet_20](xenium_lung/effnet_20.json) | `main2_xenium_patient.py` |
| her2 | EffNet-3 | [effnet_3](her2/effnet_3.json) | `main2.py` |
| her2 | EffNet-5 | [effnet_5](her2/effnet_5.json) | `main2.py` |
| her2 | EffNet-10 | [effnet_10](her2/effnet_10.json) | `main2.py` |
| her2 | EffNet-20 | [effnet_20](her2/effnet_20.json) | `main2.py` |
| hist2st | EffNet-3 | [effnet_3](hist2st/effnet_3.json) | `main2.py` |
| hist2st | EffNet-5 | [effnet_5](hist2st/effnet_5.json) | `main2.py` |
| hist2st | EffNet-10 | [effnet_10](hist2st/effnet_10.json) | `main2.py` |
| hist2st | EffNet-20 | [effnet_20](hist2st/effnet_20.json) | `main2.py` |
