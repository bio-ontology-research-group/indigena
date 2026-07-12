#!/usr/bin/env bash
# Reproducibly build ALL hierarchy_test data, SELF-CONTAINED under hierarchy_test/data/.
# Nothing is symlinked to (or written into) the shared main-repo data dir.
#
# Requires a conda env with the INDIGENA deps (pandas, mowl, pykeen, torch) -- default 'indiga'.
# Base *derived* files from the main INDIGENA pipeline (upheno projection, mouse gene side,
# disease_phenotypes, mouse folds, hpoa) are COPIED in from $BASE_DATA (not regenerated here).
#
# Usage:
#   BASE_DATA=/ibex/user/zhapacfp/indigena/data PY=$(which python) bash build_hierarchy_test_data.sh
#
set -euo pipefail
cd "$(dirname "$0")"                       # hierarchy_test/
PY=${PY:-python}
BASE_DATA=${BASE_DATA:-../data}
mkdir -p data logs data/results data/models data/baseline_results

echo "== [1/6] assemble base derived files as SELF-CONTAINED copies from $BASE_DATA =="
for f in upheno.owl upheno_owl2vecstar_edges.tsv disease_phenotypes.csv \
         gene_phenotypes.csv gene_diseases.csv phenotype.hpoa; do
  if [ -e "data/$f" ]; then echo "  have data/$f"; else cp -v "$BASE_DATA/$f" "data/$f"; fi
done
if [ -d data/gene_disease_folds ]; then echo "  have data/gene_disease_folds"; \
  else cp -rv "$BASE_DATA/gene_disease_folds" data/gene_disease_folds; fi

echo "== [2/6] download raw inputs (hp.obo, HPOA phenotype_to_genes/genes_to_disease, MGI HMD) =="
bash scripts/00_download.sh

echo "== [3/6] human gene->phenotype edges + AttributedFromDiseases provenance =="
$PY scripts/merge_human_gene_phenotypes.py            # -> data/gene_phenotypes_human.csv

echo "== [4/6] human gene->disease + HCOP orthologs + unified 10-fold split (deterministic, keyed on mouse folds) =="
$PY scripts/build_orthologs_and_folds.py              # -> data/gene_diseases_human.csv, data/gene_disease_folds_unified/

echo "== [5/6] HP ancestor closure (depth<=3) + abstracted disease queries k=0..3 =="
$PY scripts/precompute_hpo_ancestors.py --obo data/hp.obo --out data/hpo_ancestors.json --depth-max 3
$PY scripts/build_abstracted_phenotypes.py --max-depth 3   # -> data/disease_phenotypes_hp_k{0..3}.csv

echo "== [6/6] LEAF-ONLY de-propagation of the human gene side (removes true-path ancestor inflation) =="
$PY scripts/depropagate_gene_phenotypes.py \
    --obo data/hp.obo \
    --in  data/gene_phenotypes_human.csv \
    --out data/gene_phenotypes_human_leafonly.csv

echo "== DONE. hierarchy_test/data is self-contained. Key artifacts: =="
ls -la data/gene_phenotypes_human.csv data/gene_phenotypes_human_leafonly.csv \
       data/gene_diseases_human.csv data/hpo_ancestors.json \
       data/disease_phenotypes_hp_k1.csv data/gene_disease_folds_unified 2>&1 | sed 's/^/  /'
