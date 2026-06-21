#!/usr/bin/env bash
# Download the RAW inputs that hierarchy_test adds on top of the base INDIGENA data.
#
# Base (derived) files are produced by the INDIGENA repo's own pipeline and reused here:
#   upheno.owl, upheno_owl2vecstar_edges.tsv, disease_phenotypes.csv,
#   gene_phenotypes.csv (mouse), gene_diseases.csv, gene_disease_folds/, phenotype.hpoa
# Provide them under hierarchy_test/data/ (e.g. symlink ../../data/<f>) or regenerate via
# the repo's data.py / generate_inductive_dataset.py / ontology projection first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
HP_REL="${HP_REL:-v2025-05-06}"   # HPO release matching the UPheno HP version used in training
HPO="https://github.com/obophenotype/human-phenotype-ontology/releases/download/${HP_REL}"
echo "[dl] HPO ${HP_REL}: phenotype_to_genes.txt, genes_to_disease.txt, hp.obo"
curl -fsSL -o data/phenotype_to_genes.txt "${HPO}/phenotype_to_genes.txt"
curl -fsSL -o data/genes_to_disease.txt   "${HPO}/genes_to_disease.txt"
curl -fsSL -o data/hp.obo                  "${HPO}/hp.obo"
echo "[dl] MGI: HMD_HumanPhenotype.rpt (human<->mouse orthologs)"
curl -fsSL -o data/HMD_HumanPhenotype.rpt  https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt
echo "[dl] done:"; wc -l data/phenotype_to_genes.txt data/genes_to_disease.txt data/hp.obo data/HMD_HumanPhenotype.rpt
