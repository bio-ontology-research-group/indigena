# hierarchy_test/ — is INDIGENA hierarchy-aware?

A **from-scratch, reproducible** pipeline to test whether INDIGENA's embedding
similarity degrades gracefully when disease query phenotypes are **generalised
to their HP ancestors**, compared with a Resnik+BMA semantic-similarity baseline
(computed with the **Groovy/SLIB** engine, the validated implementation).

---

## How the data differs from original INDIGENA

Original INDIGENA and this experiment share the **same phenotype ontology
backbone** (UPheno owl2vec* projection: HP/MP/… classes linked by `subClassOf`)
and the **same disease side** (OMIM disease `--has_symptom-->` HP phenotype, from
HPOA). The difference is entirely on the **gene side**.

| | **Original INDIGENA** (published) | **This experiment (“Model H”, human)** |
|---|---|---|
| Gene nodes | **Mouse** genes (MGI) | **Human** genes (NCBIGene) |
| Gene→phenotype edges | mouse `--has_phenotype-->` **MP** terms, from **MGI** lab phenotyping | human `--has_phenotype-->` **HP** terms, from **HPOA** `phenotype_to_genes` |
| Provenance of gene→phenotype | **Independent** of disease curation (experimental) | **Derived from disease curation** (gene inherits a disease's phenotypes) → *leaky* |
| Leak handling | not needed | **`leak_filter=strict`**: drop a gene→phenotype edge if its attributing disease is in the test fold |
| Cross-species step | human HP query ↔ mouse MP gene phenotypes via UPheno; map gene→human via **HCOP ortholog** | none — ranks human genes directly |
| Gene-side density | dense (~16k genes) | **sparse** after leak filtering (~4.8k genes) |
| Optional | — | **`ancestor_augment_depth=d`**: also add gene/disease `--…-->` depth-≤d HP ancestors (an attempted robustness fix) |

**Why this matters for the hierarchy test.** Because the human gene→phenotype
edges are disease-derived (and then leak-stripped), the human gene side is sparse
and tied to *specific* (leaf) HP terms. Whether generalising the query to ancestors changes the ranking, and whether the
sparse leaf-tied human gene side behaves differently from the denser mouse side, is
what this pipeline measures. No outcome is assumed.

The **perturbation** itself is identical for both methods: each disease's query
phenotypes are replaced by their *k*-th HP ancestor (`k = 0,1,2,3`); `k=0` is the
unperturbed baseline.

---

## The leakage problem and `leak_filter`

Human gene->phenotype annotations (from HPOA) are **disease-derived**: a gene is linked to a
phenotype only because a disease it is causally associated with has that phenotype
(gene -> disease -> phenotype). They are therefore circular with the `gene->disease` and
`disease->phenotype` edges. In the **inductive** setting a test disease is held out and
represented only by its query phenotypes, so if a causal gene still carries the phenotype
edges it acquired *from that test disease*, the model can match it trivially -- a leak.

Each gene->phenotype edge stores its provenance in `AttributedFromDiseases` (the set of
diseases that produced it, built by `merge_human_gene_phenotypes.py`). `--leak_filter`
removes test-leaking edges, keyed on the test-disease set of the fold:

- **strict** (used for Model H): drop the edge if *any* attributing disease is in the test
  fold. A training co-attribution does **not** save it (conservative; zero test leakage).
  In fold 0 this drops ~47.8k / 481.8k (~10%) gene->phenotype edges.
- **light**: drop only if *all* attributing diseases are test diseases; an edge also
  supported by a training disease is kept.
- **none**: keep everything.

Only gene->phenotype *edges* are removed -- phenotype nodes remain (via the ontology,
training-disease `has_symptom` edges, and other genes). Separately, the inductive split
removes each test disease's own `gene->disease` and `disease->phenotype` edges entirely.

Note: the **mouse** gene side (MGI MP phenotypes, used by the published INDIGENA) is
experimental and independent of disease curation -- no such leak, no filter needed. Whether the human (leak-stripped) and mouse gene sides differ under ancestor-abstraction
is what the experiment measures.


## Inputs (sources)

`scripts/00_download.sh` fetches the raw inputs below. The *derived* base files
(upheno edges, disease_phenotypes.csv, mouse gene_phenotypes.csv, gene_diseases.csv,
gene_disease_folds/, phenotype.hpoa) come from the INDIGENA repo data pipeline
(`data.py` / `generate_inductive_dataset.py` / ontology projection); symlink them into `data/`.

| File | Source | Already on workstation? |
|---|---|---|
| `upheno_owl2vecstar_edges.tsv` | UPheno owl2vec* projection (release v2025-07-21) | yes (`indigena/data/`) |
| `disease_phenotypes.csv` | parsed from HPOA `phenotype.hpoa` (OMIM only) | yes |
| `phenotype_to_genes.txt` | HPOA release (`hpo-annotations`) | **download** |
| `genes_to_disease.txt` | HPOA release | yes |
| `HMD_HumanPhenotype.rpt` | MGI (human↔mouse homology) | yes |
| `hp.obo` / `hp.owl` | HPO release | yes |

---

## Pipeline (from scratch)

```bash
cd hierarchy_test
# 0. (optional) download any missing inputs (HPOA phenotype_to_genes.txt, hp.obo)
bash scripts/00_download.sh   # HPOA phenotype_to_genes/genes_to_disease + hp.obo + MGI HMD_HumanPhenotype.rpt

# 1. Build the HUMAN gene→phenotype edges with per-edge disease provenance
python scripts/merge_human_gene_phenotypes.py     # -> data/gene_phenotypes_human.csv

# 2. Human gene→disease + orthologs + 10 disease-disjoint folds (mouse folds preserved)
python scripts/build_orthologs_and_folds.py       # -> data/gene_diseases_human.csv, data/gene_disease_folds_unified/

# 3. Precompute HP ancestor closure (for the abstraction perturbation + aug training)
python scripts/precompute_hpo_ancestors.py        # -> data/hpo_ancestors.json

# 4. Train INDIGENA on the human graph (ONE fold first; then 0..9)
python scripts/kge_transd_species.py --species human --leak_filter strict \
    --fold 0 --mode inductive --graph4 \
    --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep
#   variant: add --ancestor_augment_depth 2   (the "aug-d2" robustness attempt)

# 5. Build abstracted disease-phenotype query sets at depths k=0..3
python scripts/build_abstracted_phenotypes.py --max-depth 3   # TODO -> data/disease_phenotypes_hp_k{0..3}.csv

# 6a. Evaluate INDIGENA under abstraction (similarity / BMA scoring)
python scripts/eval_indigena_abstraction.py --fold 0          # TODO

# 6b. Evaluate Resnik+BMA baseline under abstraction (Groovy/SLIB)
bash   scripts/run_resnik_abstraction.sh --fold 0             # TODO (adapts semantic_similarity.groovy to human genes)

# 7. Aggregate the INDIGENA-vs-Resnik degradation curve (k=0..3)
python scripts/aggregate_hierarchy.py                        # TODO
```

Start with **fold 0 only** to confirm the effect; then loop folds 0–9.

---

## Scripts

| Script | Status | Role |
|---|---|---|
| `merge_human_gene_phenotypes.py` | vendored | HPOA → human gene→phenotype edges + `AttributedFromDiseases` |
| `build_orthologs_and_folds.py` | vendored | human gene→disease + HCOP + unified 10-fold split |
| `precompute_hpo_ancestors.py` | vendored | `hp.obo` → depth-indexed HP ancestor map |
| `kge_transd_species.py` | vendored | train TransD (human/mouse, leak filter, optional ancestor-augment) |
| `extract_for_hapli_species.py` | vendored | export a hapli-style bundle (optional) |
| `data.py` | vendored | `create_train_val_split` helper |
| `00_download.sh` | done | fetch raw inputs (HPOA, hp.obo, MGI HMD) from canonical releases |
| `build_abstracted_phenotypes.py` | done | replace each disease phenotype by its k-th HP ancestor |
| `eval_indigena_abstraction.py` | TODO | INDIGENA BMA ranking per depth |
| `run_resnik_abstraction.sh` + groovy | TODO | Resnik+BMA (SLIB) per depth, human candidate genes |
| `aggregate_hierarchy.py` | TODO | MR/MRR/Hits/AUC degradation table, INDIGENA vs Resnik |

---

## Status

- Vendored the proven data-gen / training / extraction scripts (steps 1–4).
- TODO: download script, abstraction builder, the two evaluators (INDIGENA + Groovy-Resnik), aggregator.
- Goal: measure, from scratch, how INDIGENA (human leak-strict) and Resnik degrade under
  HP-ancestor abstraction, and whether `--ancestor_augment_depth 2` changes it.
