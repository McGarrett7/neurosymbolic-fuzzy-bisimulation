# Neuro-symbolic fuzzy bisimulation on fuzzy knowledge graphs

Code and data accompanying the paper *Learning Behavioral Equivalence in Fuzzy
Knowledge Graphs via Neuro-Symbolic Methods for Smart Grid Fault Diagnosis*.

<!-- After the first Zenodo release, paste the DOI badge here. -->

## Repository layout

| Path | Content |
|---|---|
| `fuzzy_bisimulation/` | Python package: fuzzy KG loading, exact fuzzy bisimulation, sampling, R-GCN model, training, evaluation, experiment suites. See [`fuzzy_bisimulation/README.md`](fuzzy_bisimulation/README.md) for the full pipeline and implementation notes. |
| `data/` | Benchmark knowledge graphs FB15k-237, WN18RR and YAGO3-10 (`train.txt`, `entities.dict`, `relations.dict`, entity/relation text). |
| `kaggle/kaggle_train.ipynb` | Notebook used to run the experiments on Kaggle GPUs. |
| `results/` | Experiment outputs (`<suite>_<dataset>.{json,md}`). |
| `checkpoints/` | Model checkpoints (not distributed; produced by training). |

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

```bash
# sanity check: vectorised operator == literal Eq. (1)
python -m fuzzy_bisimulation.exact_bisimulation --selftest

# 1. supervision and evaluation references
python -m fuzzy_bisimulation.generate_labels --dataset fb15k-237
# 2. training
python -m fuzzy_bisimulation.train --labels data/FB15k-237/labels/<tag>.pt --semantic tfidf
# 3. evaluation
python -m fuzzy_bisimulation.evaluate --checkpoint checkpoints/<run>.pt
# full experiment suites (5 seeds, all baselines)
python -m fuzzy_bisimulation.experiments main --dataset fb15k-237
```

Available datasets: `fb15k-237`, `wn18rr`, `yago3-10`. Other suites:
`ablation, sampling, latency, reference, train-ref, confidence, alpha,
encoders, semantics, stability, assumptions`. A GPU (`--device cuda`) is
recommended for FB15k-237 and YAGO3-10.

The paper's semantic channel uses `openai:text-embedding-3-large:768`, which
requires an `OPENAI_API_KEY` environment variable. The code defaults to the
open-source `st:all-mpnet-base-v2` (also 768-d, no API key); `tfidf` runs offline.

## Data sources

The datasets in `data/` are public benchmarks redistributed unchanged for
reproducibility. Please cite the original works and respect their licenses:

- **FB15k-237**: K. Toutanova and D. Chen, "Observed versus latent features for
  knowledge base completion", CVSC Workshop, 2015.
- **WN18RR**: T. Dettmers, P. Minervini, P. Stenetorp and S. Riedel,
  "Convolutional 2D knowledge graph embeddings", AAAI, 2018.
- **YAGO3-10**: F. Mahdisoltani, J. Biega and F. Suchanek, "YAGO3: A knowledge
  base from multilingual Wikipedias", CIDR, 2015.
- Entity/relation text files for FB15k-237 and WN18RR come from KG-BERT:
  L. Yao, C. Mao and Y. Luo, "KG-BERT: BERT for knowledge graph completion",
  arXiv:1909.03193, 2019.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The
benchmark datasets remain under the licenses of their original publishers.

## Citation

See [`CITATION.cff`](CITATION.cff).
