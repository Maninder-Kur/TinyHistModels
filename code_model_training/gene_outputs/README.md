# Target gene lists

These four small NumPy arrays are copied byte-for-byte from the experiment
directory. They contain gene identifiers only, without expression values.
The text files contain the same identifiers in the same order for inspection.
`provenance.json` records the original array hashes and target counts.

| File | Number of genes | Identifier format |
| --- | ---: | --- |
| `hist2st_845.npy` | 845 | Ensembl gene identifiers |
| `her2.npy` | 785 | Gene symbols |
| `xenium_breast_patient.npy` | 500 | Gene symbols |
| `xenium_lung_patient.npy` | 541 | Gene symbols |

The order defines the prediction columns. Keep it identical when comparing
models, loading teacher checkpoints, and preparing model outputs. The Xenium
arrays preserve the original object dtype and are loaded by the supplied
training code with `allow_pickle=True`; use only trusted files.

Training applies target-gene selection and normalization. Do not pre-normalize
the expression matrices before passing them to these loaders.
