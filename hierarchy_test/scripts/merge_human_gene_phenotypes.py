"""
Build a human gene-phenotype edge file from HPOA's phenotype_to_genes.txt
with per-edge disease provenance, suitable for INDIGENA training with
fold-aware leak filtering.

Inputs:
    data/phenotype_to_genes.txt   HPOA: hpo_id, hpo_name, ncbi_gene_id, gene_symbol, disease_id
    data/disease_phenotypes.csv   INDIGENA's training disease set (URI form)

Output:
    data/gene_phenotypes_human.csv  columns: Gene, Phenotype, AttributedFromDiseases
        Gene URI:     http://mowl.borg/NCBIGene_<id>   (distinct from MGI mouse genes)
        Phenotype:    http://purl.obolibrary.org/obo/HP_NNNNNNN
        Diseases:     ;-joined http://mowl.borg/OMIM_NNN URIs
                      (only OMIM kept; ORPHA / DECIPHER dropped — INDIGENA trains on OMIM only)

Provenance design: every (gene, phenotype) edge tracks the *full set* of
diseases that attributed it via HPOA. This lets the training script drop
edges per-fold without losing edges supported by other (training) diseases.

Sanity counters at exit:
    - input rows
    - input rows kept (OMIM, disease in INDIGENA training set)
    - input rows dropped (non-OMIM, disease unknown)
    - distinct (gene, phenotype) edges output
    - distinct genes
    - distinct attributing diseases
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

INPUT_HPOA = Path("data/phenotype_to_genes.txt")
INPUT_DISEASE_SET = Path("data/disease_phenotypes.csv")
OUTPUT = Path("data/gene_phenotypes_human.csv")

OMIM_PREFIX = "http://mowl.borg/OMIM_"
NCBI_PREFIX = "http://mowl.borg/NCBIGene_"
HP_PREFIX = "http://purl.obolibrary.org/obo/HP_"


def main() -> None:
    # 1. Load INDIGENA's training disease set (URI form)
    indigena_diseases: set[str] = set()
    with INPUT_DISEASE_SET.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            indigena_diseases.add(row[0])
    print(f"[load] INDIGENA disease set: {len(indigena_diseases)} unique diseases")

    # 2. Stream HPOA file, group (gene, phenotype) → attributing diseases
    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_in = n_kept = n_drop_non_omim = n_drop_unknown_disease = 0

    with INPUT_HPOA.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_in += 1
            disease_raw = row["disease_id"]
            # OMIM-only filter (INDIGENA trains on OMIM disease set)
            if not disease_raw.startswith("OMIM:"):
                n_drop_non_omim += 1
                continue
            disease_uri = OMIM_PREFIX + disease_raw[len("OMIM:"):]
            if disease_uri not in indigena_diseases:
                # OMIM disease not in INDIGENA's training set — skip; an edge
                # whose only attribution is an unknown disease can't be
                # leak-filtered consistently. Document the drop count.
                n_drop_unknown_disease += 1
                continue
            ncbi_id = row["ncbi_gene_id"]
            hpo_id = row["hpo_id"]
            if not ncbi_id or not hpo_id.startswith("HP:"):
                continue
            gene_uri = NCBI_PREFIX + ncbi_id
            pheno_uri = HP_PREFIX + hpo_id[len("HP:"):]
            edges[(gene_uri, pheno_uri)].add(disease_uri)
            n_kept += 1

    n_genes = len({g for g, _ in edges})
    n_phenos = len({p for _, p in edges})
    n_diseases = len({d for ds in edges.values() for d in ds})

    # 3. Write the merged edge file
    with OUTPUT.open("w") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "Phenotype", "AttributedFromDiseases"])
        for (g, p), ds in sorted(edges.items()):
            w.writerow([g, p, ";".join(sorted(ds))])

    print(f"[counts] HPOA input rows                : {n_in:>10}")
    print(f"[counts]   dropped (non-OMIM)           : {n_drop_non_omim:>10}")
    print(f"[counts]   dropped (OMIM not in INDIGENA): {n_drop_unknown_disease:>10}")
    print(f"[counts]   kept                          : {n_kept:>10}")
    print(f"[output]  unique (gene, phenotype) edges : {len(edges):>10}")
    print(f"[output]  unique genes                   : {n_genes:>10}")
    print(f"[output]  unique phenotypes              : {n_phenos:>10}")
    print(f"[output]  unique attributing diseases    : {n_diseases:>10}")
    print(f"[output]  written to {OUTPUT}")

    # 4. Sanity check: BRCA1 (NCBI 672), TP53 (7157), NF1 (4763)
    print("\n[sanity] sample gene profiles:")
    for label, ncbi in [("BRCA1", "672"), ("TP53", "7157"), ("NF1", "4763")]:
        gene_uri = NCBI_PREFIX + ncbi
        gene_edges = [(p, ds) for (g, p), ds in edges.items() if g == gene_uri]
        diseases_for_gene = sorted({d for _, ds in gene_edges for d in ds})
        print(f"  {label} (NCBI:{ncbi}): {len(gene_edges)} phenotype edges, "
              f"attributed via {len(diseases_for_gene)} diseases (e.g. {diseases_for_gene[:3]})")


if __name__ == "__main__":
    main()
