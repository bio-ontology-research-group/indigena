# hierarchy_test/ — is INDIGENA hierarchy-aware?

A **from-scratch, reproducible** benchmark testing whether INDIGENA's phenotype-embedding
similarity degrades gracefully when disease query phenotypes are **generalised to their HP
ancestors**, compared with a Resnik+BMA semantic-similarity baseline (Groovy/SLIB).

The comparison is done **matched**: both methods score the *same* candidate gene pool and the
*same* held-out test pairs, on the *same* (de-propagated, leak-filtered) gene profiles.

---

## Key result (fold 0)

**Leaf-only, leak-strict graph, matched pool = 4,491 genes / 330 pairs.**
`k` = HP-ancestor abstraction depth of the disease query (`k=0` = unperturbed).

**LEAKED scoring** (full gene profile at scoring time — `k=0` is the *control*: both methods
find the gene from specific phenotypes):

| method | k | MRR | H@1 | AUC | MR |
|---|---|---|---|---|---|
| INDIGENA | 0 | 0.682 | 0.609 | 0.995 | 22.3 |
| INDIGENA | 3 | **0.042** | 0.000 | 0.953 | 217.4 |
| Resnik | 0 | 0.707 | 0.609 | 0.998 | 11.7 |
| Resnik | 3 | **0.298** | 0.194 | 0.983 | 82.0 |

- At `k=0` the two methods are **matched** (MRR 0.68 vs 0.71, H@1 identical at 0.609).
- Under abstraction **INDIGENA collapses** (MRR ↓16×, H@1 0.61→0.00) while **Resnik degrades
  gracefully** (MRR ↓2.4×). At `k=3` Resnik is ~7× ahead on MRR. This is the hierarchy-awareness gap.
- The collapse is **not** a propagation artifact: the de-propagated curve (0.682→0.042) matches
  the original propagated one (0.765→0.049).

**LEAK-STRICT scoring** (leak-free): both methods sit at the floor at every `k`
(INDIGENA MRR ~0.01, Resnik ~0.03, MR ~800/4491). With the leak removed there is **no baseline
signal for either method**, which is why the leaked `k=0` is used as the control.

Aggregate over all 10 folds (mean±sd): `python scripts/aggregate_sweep.py`.

---

## The data

Original INDIGENA and this experiment share the **same phenotype ontology backbone** (UPheno
owl2vec* projection: HP/MP classes linked by `subClassOf`) and the **same disease side** (OMIM
disease `--has_symptom-->` HP phenotype, from HPOA). The difference is entirely the **gene side**.

| | **Original INDIGENA** | **This experiment (“Model H”, human)** |
|---|---|---|
| Gene nodes | **Mouse** genes (MGI), ~16k | **Human** genes (NCBIGene), 4,837 |
| Gene→phenotype | mouse `has_phenotype` **MP** terms, from **MGI** lab phenotyping | human `has_phenotype` **HP** terms, from HPOA `phenotype_to_genes` |
| Provenance | **experimental**, independent of disease curation | **derived** from disease curation (gene inherits a disease's phenotypes) → *leaky* |
| True-path propagation | **no** (0% of genes carry the MP root; ~14 terms/gene) | **yes** (86% carry HP root `HP_0000001`; ~100 terms/gene) |
| Cross-species step | HP query ↔ mouse MP via UPheno; map gene→human via **HCOP** | none — ranks human genes directly |

### Why only 4,837 human genes (not ~20k)
Human `gene→phenotype` annotations exist **only for genes with a curated Mendelian (OMIM)
disease** (the phenotypes are propagated *from* the disease). No disease ⇒ no annotation ⇒ the
gene has no profile and cannot be scored. So the human candidate pool is capped at disease genes
(4,837). The mouse side gets ~16k genes because MGI phenotyping is experimental and
disease-independent.

### Issue 1 — the leakage problem (`--leak_filter`)
Human `gene→phenotype` edges are `gene→disease→phenotype` collapsed into one hop: a gene has a
phenotype **only because** a disease it causes has that phenotype. Each edge stores its origin in
`AttributedFromDiseases`. In the inductive split a test disease is held out and represented only
by its query phenotypes, so a causal gene still wearing the phenotypes it inherited *from that
test disease* is matched trivially — a leak.

`--leak_filter strict` drops a `gene→phenotype` edge if **any** attributing disease is in the test
fold. **This filter is applied in two places (both required):**
- the **training graph** (embeddings are never trained on leaked edges), and
- the **BMA scoring set** `gene2pheno` (`SCORING_LEAK_FILTER`, default = `--leak_filter`).
  Without the second, the true gene keeps the test disease's phenotypes at scoring time and BMA
  matches a copy of the answer (empirically: the true gene contained **100%** of the query terms
  for 333/333 fold-0 pairs). Set `SCORING_LEAK_FILTER=none` to reproduce the leaked `k=0` control.

Genes whose *entire* profile came from test diseases drop out of the graph entirely (a gene enters
the graph only via a surviving `has_phenotype` edge; a `gene→disease` edge cannot re-add it). Test
pairs whose true gene drops out are excluded from evaluation.

### Issue 2 — true-path propagation (leaf-only de-propagation)
HPOA `phenotype_to_genes` is distributed under the **true-path rule**: each gene is annotated to
its specific terms *and all their ancestors* up to the root. This inflates the human gene side
(86% of genes carry `HP_0000001`; ~100 terms/gene) while the disease query side and the mouse gene
side are leaf-only. Abstracting the query onto ancestors then just matches terms nearly every gene
already carries — confounding the hierarchy test.

`depropagate_gene_phenotypes.py` removes the propagated ancestors, keeping only the **most specific
terms** per gene (drops 73.3%: 481,778 → 128,501 edges; ~100 → ~27 terms/gene). Provenance is
preserved so the leak filter still works. **Model H uses the leaf-only file
`gene_phenotypes_human_leafonly.csv`.**

### Data sources (raw inputs)
`scripts/00_download.sh` fetches these; base *derived* files come from the main INDIGENA pipeline.

| File | Source |
|---|---|
| `phenotype_to_genes.txt`, `genes_to_disease.txt`, `hp.obo` | HPO release (`HP_REL`, default `v2025-05-06`) |
| `HMD_HumanPhenotype.rpt` | MGI (human↔mouse orthologs) |
| `upheno.owl`, `upheno_owl2vecstar_edges.tsv` | UPheno owl2vec* projection (base pipeline) |
| `disease_phenotypes.csv`, `gene_phenotypes.csv` (mouse), `gene_diseases.csv`, `gene_disease_folds/`, `phenotype.hpoa` | base INDIGENA pipeline (`data.py` / `generate_inductive_dataset.py`) |

### Reproducible, self-contained data build
Builds **everything under `hierarchy_test/data/`** — nothing symlinked to, or written into, the
shared main-repo data dir. Base derived files are copied in from `$BASE_DATA`; the rest is
downloaded and derived. Folds are **deterministic** (the unified 10-fold split is keyed on the
existing mouse fold partition), so a rebuild reproduces the same folds.

```bash
cd hierarchy_test
BASE_DATA=/path/to/main/indigena/data PY=$(which python) bash build_hierarchy_test_data.sh
```
Steps: assemble base files → `00_download.sh` → `merge_human_gene_phenotypes.py` →
`build_orthologs_and_folds.py` → `precompute_hpo_ancestors.py` → `build_abstracted_phenotypes.py`
→ `depropagate_gene_phenotypes.py`.

---

## Pipeline (single fold, e.g. fold 0)

```bash
# 1. Train INDIGENA on the leaf-only, leak-strict graph; dump the matched pool + pairs.
env GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv RUN_TAG=_leafonly \
    DUMP_MATCHED_DIR=data/matched_fold0 \
  python scripts/kge_transd_species.py --species human --leak_filter strict --fold 0 \
    --mode inductive --graph4 --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep

# 2. INDIGENA eval grid: scoring ∈ {leaked, strict} × k ∈ {0..3} (reuses the model, --only_test).
bash run_indigena_matched_evals.sh 0

# 3. Resnik matched grid on the SAME pool/pairs (needs groovy + JAVA_HOME).
bash run_resnik_matched.sh 0

# 4. Table.
python scripts/aggregate_matched_fold0.py
```

## 10-fold sweep on Ibex (SLURM)

`slurm_sweep.sh` runs one GPU task per fold (train → INDIGENA eval grid → Resnik grid), using the
`indiga` conda env (direct interpreter path) and system `java-11` for groovy.

```bash
cd /ibex/user/zhapacfp/indigena/hierarchy_test
BASE_DATA=/ibex/user/zhapacfp/indigena/data PY=/home/zhapacfp/miniforge3/envs/indiga/bin/python \
  bash build_hierarchy_test_data.sh          # once
sbatch slurm_sweep.sh                         # array 0-9
python scripts/aggregate_sweep.py             # mean±sd once folds land (runs mid-sweep too)
```

---

## Scripts

| Script | Role |
|---|---|
| `build_hierarchy_test_data.sh` | reproducible, self-contained data build (source → leaf-only + folds + abstracted) |
| `depropagate_gene_phenotypes.py` | leaf-only de-propagation (drop propagated ancestors, keep provenance) |
| `merge_human_gene_phenotypes.py` | HPOA → human gene→phenotype edges + `AttributedFromDiseases` |
| `build_orthologs_and_folds.py` | human gene→disease + HCOP + deterministic unified 10-fold split |
| `precompute_hpo_ancestors.py` | `hp.obo` → depth-indexed HP ancestor map |
| `build_abstracted_phenotypes.py` | replace each disease phenotype by its k-th HP ancestor (k=0..3) |
| `kge_transd_species.py` | train/eval TransD; `--leak_filter` (graph) + `SCORING_LEAK_FILTER` (BMA set); `GENE_PHENO_CSV`, `RUN_TAG`, `DUMP_MATCHED_DIR`, `EVAL_DISEASE_PHENO_CSV`, `EVAL_TAG` |
| `semantic_similarity_human.groovy` | Resnik+BMA (SLIB); env: `GENE_PHENO_CSV`, `LEAK_FILTER`, `CANDIDATES_FILE`, `TEST_PAIRS_CSV`, `PERTURB_DISEASE_CSV`, `PERTURB_TAG` |
| `run_indigena_matched_evals.sh <fold>` | INDIGENA eval grid (leaked+strict × k=0..3) |
| `run_resnik_matched.sh <fold>` | Resnik matched grid (leaked+strict × k=0..3) |
| `slurm_sweep.sh` | SLURM array (folds 0-9): train + both eval grids |
| `aggregate_matched_fold0.py` / `aggregate_sweep.py` | single-fold table / cross-fold mean±sd |

## Key data artifacts (under `data/`, all self-contained)

`gene_phenotypes_human.csv` (propagated, with provenance) · `gene_phenotypes_human_leafonly.csv`
(**used by Model H**) · `gene_diseases_human.csv` · `gene_disease_folds_unified/fold_{0..9}/` ·
`hpo_ancestors.json` · `disease_phenotypes_hp_k{0..3}.csv` · `matched_fold{f}/` (candidate pool +
test pairs dumped by training, consumed by Resnik).
