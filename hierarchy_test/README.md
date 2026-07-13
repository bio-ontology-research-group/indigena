# hierarchy_test: is INDIGENA hierarchy-aware?

We test whether INDIGENA's phenotype-embedding similarity degrades gracefully when we generalise
disease query phenotypes to their HP ancestors, and we compare it against a Resnik+BMA
semantic-similarity baseline (Groovy/SLIB). We match the two methods: both score the same
candidate gene pool, the same held-out test pairs, and the same de-propagated, leak-filtered gene
profiles.

## Key result (10-fold cross-validation)

Leaf-only graph, leak-strict, matched candidate pool (~4,500 genes per fold) and matched held-out
test pairs. `k` denotes the HP-ancestor abstraction depth of the disease query (`k=0` is
unperturbed). We report the mean and standard deviation over the 10 disease-disjoint folds.

Leaked scoring uses the full gene profile at scoring time, so `k=0` is the control: both methods
recover the causal gene from specific phenotypes.

| method | k | MRR | H@1 | AUC | MR |
|---|---|---|---|---|---|
| INDIGENA | 0 | 0.702 ± 0.037 | 0.621 ± 0.040 | 0.996 ± 0.002 | 18.8 ± 9.9 |
| INDIGENA | 1 | 0.357 ± 0.067 | 0.267 ± 0.064 | 0.980 ± 0.012 | 94.6 ± 57.0 |
| INDIGENA | 2 | 0.107 ± 0.061 | 0.047 ± 0.032 | 0.938 ± 0.061 | 282.3 ± 273.2 |
| INDIGENA | 3 | 0.031 ± 0.017 | 0.004 ± 0.004 | 0.894 ± 0.097 | 478.8 ± 436.5 |
| Resnik | 0 | 0.725 ± 0.018 | 0.640 ± 0.022 | 0.999 ± 0.000 | 7.2 ± 2.3 |
| Resnik | 1 | 0.599 ± 0.015 | 0.498 ± 0.020 | 0.997 ± 0.001 | 16.0 ± 5.3 |
| Resnik | 2 | 0.460 ± 0.020 | 0.350 ± 0.023 | 0.993 ± 0.002 | 34.2 ± 11.2 |
| Resnik | 3 | 0.300 ± 0.016 | 0.198 ± 0.018 | 0.986 ± 0.002 | 63.3 ± 9.5 |

At `k=0` the two methods match (MRR 0.702 vs 0.725, H@1 0.621 vs 0.640). Under abstraction INDIGENA
collapses (MRR falls 23-fold, H@1 from 0.621 to 0.004) while Resnik degrades gracefully (MRR falls
2.4-fold). At `k=3` Resnik ranks the causal gene 10-fold higher on MRR (0.300 vs 0.031). This gap
quantifies hierarchy-awareness, and Resnik holds a tight fold-to-fold variance (standard deviation
near 0.017 at every `k`). The collapse is not a propagation artifact: the de-propagated fold-0
curve (0.682 to 0.042) matches the original propagated curve (0.765 to 0.049).

Under leak-strict scoring (leak-free), both methods sit at the floor at every `k` (INDIGENA MRR
near 0.010, Resnik near 0.025, MR above 890 of ~4,500). Because the leak-free setting leaves no
baseline signal for either method, we use the leaked `k=0` as the control.

Reproduce the table: `python scripts/aggregate_sweep.py` (reads `data/results` and
`data/baseline_results`); the full output is saved to `results/degradation_sweep_fold0-9.txt`.

## The data

Original INDIGENA and this experiment share the same phenotype ontology backbone (UPheno owl2vec*
projection: HP and MP classes linked by `subClassOf`) and the same disease side (OMIM disease
`has_symptom` HP phenotype, from HPOA). The difference is the gene side.

| | Original INDIGENA | This experiment (Model H, human) |
|---|---|---|
| Gene nodes | mouse genes (MGI), ~16k | human genes (NCBIGene), 4,837 |
| Gene to phenotype | mouse `has_phenotype` MP terms, from MGI lab phenotyping | human `has_phenotype` HP terms, from HPOA `phenotype_to_genes` |
| Provenance | experimental, independent of disease curation | derived from disease curation (the gene inherits a disease's phenotypes), therefore leaky |
| True-path propagation | no (0% carry the MP root; ~14 terms per gene) | yes in the raw HPOA source (86% carry the HP root `HP_0000001`; ~100 terms per gene); Model H de-propagates it to leaf-only, ~27 terms per gene (Issue 2) |
| Cross-species step | HP query to mouse MP via UPheno, then gene to human via HCOP | none; ranks human genes directly |

### Why 4,837 human genes and not ~20k

Human `gene to phenotype` annotations exist only for genes with a curated Mendelian (OMIM)
disease, because HPOA propagates the phenotypes from the disease to the gene. A gene without a
disease has no annotation, therefore no profile, and the model cannot score it. The human
candidate pool is capped at disease genes (4,837). The mouse side reaches ~16k genes because MGI
phenotyping is experimental and independent of disease.

### Issue 1: the leakage problem (`--leak_filter`)

Human `gene to phenotype` edges are `gene to disease to phenotype` collapsed into one hop: a gene
carries a phenotype only because a disease it causes carries that phenotype. Each edge stores its
origin in `AttributedFromDiseases`. The inductive split holds out a test disease and represents
it only by its query phenotypes, so a causal gene that still carries the phenotypes it inherited
from that test disease matches trivially, which is the leak.

`--leak_filter strict` drops a `gene to phenotype` edge if any attributing disease belongs to the
test fold. We apply the filter in two places, and both are required:

- the training graph, so the embeddings never train on leaked edges, and
- the BMA scoring set `gene2pheno` (`SCORING_LEAK_FILTER`, default equal to `--leak_filter`).
  Without the second, the causal gene keeps the test disease's phenotypes at scoring time and BMA
  matches a copy of the answer: the causal gene contained 100% of the query terms for all 333
  fold-0 pairs. Set `SCORING_LEAK_FILTER=none` to reproduce the leaked `k=0` control.

A gene whose entire profile came from test diseases drops out of the graph, because a gene enters
the graph only through a surviving `has_phenotype` edge and a `gene to disease` edge cannot re-add
it. We exclude test pairs whose causal gene drops out.

### Issue 2: true-path propagation (leaf-only de-propagation)

HPOA `phenotype_to_genes` follows the true-path rule: it annotates each gene to its specific terms
and to all their ancestors up to the root. This inflates the human gene side (86% of genes carry
`HP_0000001`; ~100 terms per gene) while the disease query side and the mouse gene side stay
leaf-only. When we then abstract the query to ancestors, it matches terms that almost every gene
already carries, which confounds the hierarchy test.

`depropagate_gene_phenotypes.py` removes the propagated ancestors and keeps only the most specific
terms per gene (it drops 73.3%: 481,778 to 128,501 edges; ~100 to ~27 terms per gene). It
preserves provenance, so the leak filter still applies. Model H uses the leaf-only file
`gene_phenotypes_human_leafonly.csv`.

### Data sources (raw inputs)

`scripts/00_download.sh` fetches these; base derived files come from the main INDIGENA workflow.

| File | Source |
|---|---|
| `phenotype_to_genes.txt`, `genes_to_disease.txt`, `hp.obo` | HPO release (`HP_REL`, default `v2025-05-06`) |
| `HMD_HumanPhenotype.rpt` | MGI (human-mouse orthologs) |
| `upheno.owl`, `upheno_owl2vecstar_edges.tsv` | UPheno owl2vec* projection (base workflow) |
| `disease_phenotypes.csv`, `gene_phenotypes.csv` (mouse), `gene_diseases.csv`, `gene_disease_folds/`, `phenotype.hpoa` | base INDIGENA workflow (`data.py`, `generate_inductive_dataset.py`) |

### Reproducible, self-contained data build

The build writes everything under `hierarchy_test/data/` and never symlinks to, or writes into,
the shared main-repo data directory. It copies base derived files from `$BASE_DATA`, then
downloads and derives the rest. The folds are deterministic, because the unified 10-fold split
keys on the existing mouse fold partition, so a rebuild reproduces the same folds.

```bash
cd hierarchy_test
BASE_DATA=/path/to/main/indigena/data PY=$(which python) bash build_hierarchy_test_data.sh
```

Steps: assemble base files, `00_download.sh`, `merge_human_gene_phenotypes.py`,
`build_orthologs_and_folds.py`, `precompute_hpo_ancestors.py`, `build_abstracted_phenotypes.py`,
`depropagate_gene_phenotypes.py`.

## Workflow (single fold, e.g. fold 0)

```bash
# 1. Train INDIGENA on the leaf-only, leak-strict graph; dump the matched pool and pairs.
env GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv RUN_TAG=_leafonly \
    DUMP_MATCHED_DIR=data/matched_fold0 \
  python scripts/kge_transd_species.py --species human --leak_filter strict --fold 0 \
    --mode inductive --graph4 --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep

# 2. INDIGENA eval grid: scoring in {leaked, strict} times k in {0..3} (reuses the model, --only_test).
bash run_indigena_matched_evals.sh 0

# 3. Resnik matched grid on the same pool and pairs (needs groovy and JAVA_HOME).
bash run_resnik_matched.sh 0

# 4. Table.
python scripts/aggregate_matched_fold0.py
```

## 10-fold sweep

We split the sweep by hardware: Ibex (SLURM, GPU) trains and evaluates INDIGENA, and the
workstation runs the CPU-only Resnik baseline where groovy and SLIB are already configured.

On Ibex, `slurm_sweep.sh` runs one GPU task per fold (train, then INDIGENA eval grid) using the
`indiga` conda environment. Each task also writes the matched pool and pairs to
`data/matched_fold{f}/`.

```bash
# Ibex (glogin): build once, then submit the array. Requires --constraint=[v100|a100].
cd /ibex/user/zhapacfp/indigena/hierarchy_test
BASE_DATA=/ibex/user/zhapacfp/indigena/data PY=/home/zhapacfp/miniforge3/envs/indiga/bin/python \
  bash build_hierarchy_test_data.sh
sbatch slurm_sweep.sh                          # array 0-9: train + INDIGENA eval
```

On the workstation, `run_resnik_sweep_workstation.sh` pulls each fold's matched dump from Ibex
(as the GPU job writes it) and runs the matched Resnik grid locally:

```bash
bash run_resnik_sweep_workstation.sh           # folds 0-9: sync dump from Ibex, run Resnik locally
```

To aggregate, co-locate both result sets on one host, because Ibex holds the INDIGENA result
files (`data/results/`) and the workstation holds the Resnik result files
(`data/baseline_results/`). Copy the INDIGENA results from Ibex to the workstation, then run:

```bash
python scripts/aggregate_sweep.py              # mean and sd across folds; runs mid-sweep as folds land
```

## Scripts

| Script | Role |
|---|---|
| `build_hierarchy_test_data.sh` | reproducible, self-contained data build (source to leaf-only, folds, abstracted queries) |
| `depropagate_gene_phenotypes.py` | leaf-only de-propagation (drop propagated ancestors, keep provenance) |
| `merge_human_gene_phenotypes.py` | HPOA to human gene-phenotype edges with `AttributedFromDiseases` |
| `build_orthologs_and_folds.py` | human gene-disease, HCOP orthologs, deterministic unified 10-fold split |
| `precompute_hpo_ancestors.py` | `hp.obo` to depth-indexed HP ancestor map |
| `build_abstracted_phenotypes.py` | replace each disease phenotype by its k-th HP ancestor (k=0..3) |
| `kge_transd_species.py` | train and evaluate TransD; `--leak_filter` (graph) and `SCORING_LEAK_FILTER` (BMA set); `GENE_PHENO_CSV`, `RUN_TAG`, `DUMP_MATCHED_DIR`, `EVAL_DISEASE_PHENO_CSV`, `EVAL_TAG` |
| `semantic_similarity_human.groovy` | Resnik+BMA (SLIB); env `GENE_PHENO_CSV`, `LEAK_FILTER`, `CANDIDATES_FILE`, `TEST_PAIRS_CSV`, `PERTURB_DISEASE_CSV`, `PERTURB_TAG` |
| `run_indigena_matched_evals.sh <fold>` | INDIGENA eval grid (leaked and strict, k=0..3) |
| `run_resnik_matched.sh <fold>` | Resnik matched grid (leaked and strict, k=0..3), workstation |
| `run_resnik_sweep_workstation.sh` | workstation sweep: pull each fold's dump from Ibex, run Resnik locally |
| `slurm_sweep.sh` | Ibex SLURM array (folds 0-9): train and INDIGENA eval |
| `aggregate_matched_fold0.py`, `aggregate_sweep.py` | single-fold table, cross-fold mean and sd |

## Key data artifacts (under `data/`, all self-contained)

`gene_phenotypes_human.csv` (propagated, with provenance), `gene_phenotypes_human_leafonly.csv`
(used by Model H), `gene_diseases_human.csv`, `gene_disease_folds_unified/fold_{0..9}/`,
`hpo_ancestors.json`, `disease_phenotypes_hp_k{0..3}.csv`, `matched_fold{f}/` (candidate pool and
test pairs, written by training and read by Resnik).
