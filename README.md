# INDIGENA

**IND**uctive **DI**sease–**GEN**e **A**ssociation prediction using
phenotype ontologies. INDIGENA combines knowledge graph embeddings of
the UPheno cross-species phenotype ontology with an explicit
phenotype-level Best-Match Average (BMA) aggregation, allowing it to
predict gene–disease associations for diseases (sets of phenotypes)
that were never seen during training.

If you use this code, please cite:

> Zhapa-Camacho, F. and Hoehndorf, R. INDIGENA: inductive prediction
> of disease–gene associations using phenotype ontologies. *Bioinformatics*
> (under review).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fig/graphs_dark_bg.png">
  <source media="(prefers-color-scheme: light)" srcset="fig/graphs_white_bg.png">
  <img alt="Graph Structures">
</picture>

## Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Data](#data)
- [Reproducing the paper](#reproducing-the-paper)
- [Running inference on a trained model](#running-inference-on-a-trained-model)
- [Repository layout](#repository-layout)

## Dependencies

- Groovy 4.0.26 and Java 8+ (for the semantic-similarity baselines,
  which use the Semantic Measures Library / SLIB)
- Python 3.10 with `mowl`, `pykeen`, `torch`, `wandb`, `pandas`,
  `tqdm`, `click` (for the KGE methods)

The simplest way to install the Python side is via the provided
conda environment file.

## Installation

```bash
git clone https://github.com/bio-ontology-research-group/indigena.git
cd indigena/
conda env create -f environment.yml
conda activate indigena
```

Then unpack the ontology file:

```bash
cd data
gunzip upheno.owl.gz
cd ..
```

The KGE training scripts log to Weights & Biases. Before running any
`kge_*.py` script, edit the `wandb.init(...)` call and set `entity`
to your W&B username (or set `WANDB_MODE=offline` if you do not want
to use W&B).

## Data

All data files used in the paper ship with the repository under
`data/`:

| File                       | Source        | Version (used in paper) |
|----------------------------|---------------|-------------------------|
| `upheno.owl.gz`            | UPheno (obophenotype/upheno-dev) | release `v2025-07-21` |
| `MGI_GenePheno.rpt`        | Mouse Genome Informatics (MGI)   | downloaded 2025-08-20 |
| `MGI_Geno_DiseaseDO.rpt`   | Mouse Genome Informatics (MGI)   | downloaded 2025-07-20 |
| `phenotype.hpoa`           | Human Phenotype Ontology (HPO)   | downloaded 2025-08-20 |
| `gene_phenotypes.csv`      | parsed from `MGI_GenePheno.rpt`  | – |
| `disease_phenotypes.csv`   | parsed from `phenotype.hpoa`     | – |
| `gene_diseases.csv`        | parsed from `MGI_Geno_DiseaseDO.rpt` | – |
| `gene_disease_folds/`      | 10 disease-disjoint CV folds     | – |

To regenerate the parsed CSVs and the cross-validation folds from
scratch:

```bash
python data.py
python generate_inductive_dataset.py
```

## Reproducing the paper

The experiments in the paper are organized as a 10-fold
cross-validation over diseases. For each fold we run (a) the
semantic similarity baselines and (b) the KGE methods on Graphs 1–4
(inductive) and 3T/4T (transductive).

### Semantic similarity baselines

```bash
# Resnik / Lin with BMA and BMM, all 10 folds
bash semantic_similarity_folds.sh

# SimGIC, all 10 folds
bash semantic_similarity_simgic_folds.sh
```

A single configuration of a single fold can be run directly:

```bash
groovy semantic_similarity.groovy -r data -ic resnik -pw resnik -gw bma -fold 0
groovy semantic_similarity_simgic.groovy -r data -ic resnik -fold 0
```

Parameters:

- `-r, --root_dir`: data directory (default `data`)
- `-ic, --ic_measure`: information content measure (`resnik`, `sanchez`)
- `-pw, --pairwise_measure`: pairwise measure (`resnik`, `lin`)
- `-gw, --groupwise_measure`: groupwise measure (`bma`, `bmm`)
- `-fold`: cross-validation fold number (0–9)

Results are written to `data/baseline_results/`. To compute aggregated
metrics across folds:

```bash
python evaluate_sem_sim.py data/baseline_results/<results_file>
python aggregated_sem_sim_metrics.py
```

### Knowledge graph embedding methods

A single training run looks like:

```bash
python kge_transd.py --fold 0 --mode inductive \
  --embedding_dim 100 --batch_size 4096 --learning_rate 0.001 \
  --graph4 --no_sweep
```

The hyperparameters used for each model and graph configuration are:

<details>
<summary>Hyperparameter table</summary>

| Model | Embedding dim | Batch size | Learning rate | Num filters |
|-------|---------------|------------|---------------|-------------|
| **Transductive G3** | | | | |
| TransE | 100 | 8192 | 0.001 | – |
| TransH | 200 | 2048 | 0.001 | – |
| TransD | 400 | 2048 | 0.001 | – |
| ConvKB | 100 | 8192 | 0.0001 | 100 |
| ConvKB-D | 100 | 8192 | 0.0001 | 100 |
| **Transductive G4** | | | | |
| TransE | 100 | 2048 | 0.001 | – |
| TransH | 200 | 1024 | 0.001 | – |
| TransD | 100 | 2048 | 0.001 | – |
| ConvKB | 100 | 8192 | 0.0001 | 100 |
| ConvKB-D | 100 | 8192 | 0.00001 | 200 |
| **Inductive G1** | | | | |
| TransD | 400 | 8192 | 0.001 | – |
| ConvKB-D | 100 | 8192 | 0.0001 | 100 |
| **Inductive G2** | | | | |
| TransD | 400 | 4096 | 0.001 | – |
| ConvKB-D | 100 | 4096 | 0.0001 | 100 |
| **Inductive G3** | | | | |
| TransD | 400 | 8192 | 0.001 | – |
| ConvKB-D | 100 | 8192 | 0.0001 | 100 |
| **Inductive G4** | | | | |
| TransD | 400 | 8192 | 0.001 | – |
| ConvKB-D | 100 | 2048 | 0.0001 | 200 |

</details>

Common KGE-script parameters:

- `--fold`: cross-validation fold number (0–9)
- `--mode`: `inductive` or `transductive`
- `--graph2 / --graph3 / --graph4`: include gene–phenotype,
  disease–phenotype, gene–disease edges respectively
- `--embedding_dim`, `--batch_size`, `--learning_rate`: training
  hyperparameters
- `--pretrained_model`: features to initialize ConvKB embeddings
  (`transe` for ConvKB, `transd` for ConvKB-D)
- `--only_test`: skip training and evaluate an existing model
- `--no_sweep`: disable W&B sweep mode

### Statistical significance and plots

```bash
# Wilcoxon signed-rank test (Lin-BMA vs ConvKB-D / Graph 4)
Rscript p_value.r            # or: python p_value.py

# Box plot comparing Lin-BMA vs ConvKB-D (inductive, Graph 4)
python plot_boxplot.py

# UMAP projections of learned embeddings (Figure 2)
bash plot_umap.sh
python plot_umap_models_grid.py
```

## Running inference on a trained model

Once a KGE model has been trained (e.g. TransD on Graph 4, fold 0)
its checkpoint is stored under `data/models/`. You can skip training
and run evaluation directly using `--only_test`, which loads the
trained model and scores the gene–disease pairs in the fold's
`test.csv` file (located under `data/gene_disease_folds/fold_<N>/`):

```bash
python kge_transd.py --fold 0 --mode inductive --graph4 \
  --no_sweep --only_test
```

Each `test.csv` contains `Gene,Disease` pairs that were held out
during training. The script ranks candidate genes for each test
disease and writes results under `data/results/`.

## Repository layout

```
data.py                          # parse MGI/HPO source files into CSVs
generate_inductive_dataset.py    # build 10-fold disease-disjoint splits
semantic_similarity.groovy       # Resnik/Lin + BMA/BMM baselines (SLIB)
semantic_similarity_simgic.*     # SimGIC baseline
evaluate_sem_sim.py              # per-fold metric computation
aggregated_sem_sim_metrics.py    # aggregate baseline metrics across folds
kge_transe.py                    # TransE training / evaluation
kge_transh.py                    # TransH training / evaluation
kge_transd.py                    # TransD training / evaluation
kge_convkb.py                    # ConvKB / ConvKB-D training / evaluation
extract_metrics_from_sweep.py    # parse W&B sweep results
plot_boxplot.py                  # box plot comparing Lin-BMA vs ConvKB-D
plot_umap.py / plot_umap_models_grid.py  # UMAP visualizations
p_value.py / p_value.r           # Wilcoxon signed-rank test
check_data_leakage.py            # sanity check that test diseases are unseen
sweeps/                          # W&B sweep configs
data/                            # input data and outputs
fig/                             # figures used in the README and paper
```

## Problems running the models?

Please open a GitHub issue or pull request.
