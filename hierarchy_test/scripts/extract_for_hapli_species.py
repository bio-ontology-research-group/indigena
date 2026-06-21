"""Species-aware extractor — produces ONE bundle per (species, fold) trained model.

Usage:
    python extract_for_hapli_species.py --species mouse --fold 0
    python extract_for_hapli_species.py --species human --fold 0 --leak-filter strict

Outputs to ./hapli_bundle/{species}_fold{fold}{_leak-X}/
    entity_embeddings.pt
    entity_to_id.json
    gene2pheno.json
    disease2pheno.json
    metadata.json

The bundle layout is identical to v1 (single-species). Late fusion in hapli
loads two such bundles + the HCOP table.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mowl
mowl.init_jvm("4g")
from mowl.projection import Edge
from pykeen.models import TransD
import torch as th
import pandas as pd

from data import create_train_val_split

EDGES_FILE = "data/upheno_owl2vecstar_edges.tsv"
EMBEDDING_DIM = 400
BATCH_SIZE = 8192
LR = 0.001
RANDOM_SEED = 0
FOLD_BASE = "data/gene_disease_folds_unified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=["mouse", "human"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--leak-filter", choices=["none", "light", "strict"], default="none")
    ap.add_argument("--ancestor-augment-depth", type=int, default=0)
    ap.add_argument("--ancestors-json", type=str, default="data/hpo_ancestors.json")
    args = ap.parse_args()

    species = args.species
    fold = args.fold
    leak = args.leak_filter
    aug_depth = args.ancestor_augment_depth

    if species == "mouse":
        gene_pheno_file = "data/gene_phenotypes.csv"
        gene_disease_file = "data/gene_diseases.csv"
    else:
        gene_pheno_file = "data/gene_phenotypes_human.csv"
        gene_disease_file = "data/gene_diseases_human.csv"

    # Load HPO ancestors if augmentation requested (must reproduce training triples exactly)
    ancestors_per_term: dict[str, list[list[str]]] = {}
    if aug_depth > 0:
        with open(args.ancestors_json) as f:
            anc_data = json.load(f)
        ancestors_per_term = anc_data["ancestors_by_depth"]
        print(f"[extract] loaded ancestor map for {len(ancestors_per_term)} HP terms; depth={aug_depth}")

    leak_tag = f"_leak-{leak}" if (species == "human" and leak != "none") else ""
    aug_tag = f"_aug-d{aug_depth}" if aug_depth > 0 else ""
    file_id = (f"transd_{species}_inductive_fold_{fold}_seed_{RANDOM_SEED}"
               f"_dim_{EMBEDDING_DIM}_bs_{BATCH_SIZE}_lr_{LR}_graph4{leak_tag}{aug_tag}")
    model_file = f"data/models/{file_id}.pt"
    out_dir = Path(f"hapli_bundle/{species}_fold{fold}{leak_tag}{aug_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] species={species} fold={fold} leak={leak}")
    print(f"[extract] model_file={model_file}")
    print(f"[extract] out_dir={out_dir}")

    # Rebuild triples in the same order the training script used
    triples = []
    entities = set()
    with open(EDGES_FILE) as f:
        for line in f:
            s, r, d = line.strip().split("\t")
            triples.append((s, r, d))
            entities.add(s); entities.add(d)

    gene_phenotypes = pd.read_csv(gene_pheno_file)
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")
    train_disease_genes = pd.read_csv(f"{FOLD_BASE}/fold_{fold}/{species}_train.csv")
    test_disease_genes = pd.read_csv(f"{FOLD_BASE}/fold_{fold}/{species}_test.csv")
    tr, val = create_train_val_split(train_disease_genes, val_ratio=0.1, random_seed=RANDOM_SEED)
    test_diseases = set(test_disease_genes['Disease'])

    # G2: gene -> phenotype, with leak filter for human
    for _, row in gene_phenotypes.iterrows():
        gene = row['Gene']; pheno = row['Phenotype']
        if species == "human" and leak != "none":
            attributing = set(str(row['AttributedFromDiseases']).split(';'))
            if leak == 'light' and attributing.issubset(test_diseases):
                continue
            if leak == 'strict' and (attributing & test_diseases):
                continue
        if pheno not in entities:
            continue
        triples.append((gene, 'has_phenotype', pheno))
        entities.add(gene)
        # Augmentation: must reproduce exactly what training did
        if aug_depth > 0 and pheno in ancestors_per_term:
            for d in range(min(aug_depth, len(ancestors_per_term[pheno]))):
                for anc in ancestors_per_term[pheno][d]:
                    if anc in entities:
                        triples.append((gene, 'has_phenotype', anc))

    # G3: disease -> phenotype (skip test diseases in inductive)
    for _, row in disease_phenotypes.iterrows():
        if row['Disease'] in test_diseases:
            continue
        triples.append((row['Disease'], 'has_symptom', row['Phenotype']))
        entities.add(row['Disease'])
        if aug_depth > 0 and row['Phenotype'] in ancestors_per_term:
            for d in range(min(aug_depth, len(ancestors_per_term[row['Phenotype']]))):
                for anc in ancestors_per_term[row['Phenotype']][d]:
                    if anc in entities:
                        triples.append((row['Disease'], 'has_symptom', anc))

    # G4: gene -> disease (training only)
    for _, row in train_disease_genes.iterrows():
        if row['Gene'] not in entities or row['Disease'] not in entities:
            continue
        triples.append((row['Gene'], 'associated_with', row['Disease']))

    triples = sorted(triples)
    mowl_triples = [Edge(s, r, d) for s, r, d in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)
    print(f"[extract] entities={len(triples_factory.entity_to_id)} "
          f"relations={len(triples_factory.relation_to_id)} triples={len(triples)}")

    model = TransD(
        triples_factory=triples_factory,
        embedding_dim=EMBEDDING_DIM, relation_dim=EMBEDDING_DIM,
        random_seed=RANDOM_SEED,
    ).to("cpu")
    state = th.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    entity_ids = th.tensor(list(triples_factory.entity_to_id.values()))
    entity_embeddings = model.entity_representations[0](indices=entity_ids).detach().cpu()
    th.save(entity_embeddings, out_dir / "entity_embeddings.pt")
    with (out_dir / "entity_to_id.json").open("w") as f:
        json.dump(triples_factory.entity_to_id, f)

    gene2pheno: dict[str, list[str]] = {}
    for _, row in gene_phenotypes.iterrows():
        gene2pheno.setdefault(row['Gene'], []).append(row['Phenotype'])
    with (out_dir / "gene2pheno.json").open("w") as f:
        json.dump(gene2pheno, f)

    disease2pheno: dict[str, list[str]] = {}
    for _, row in disease_phenotypes.iterrows():
        disease2pheno.setdefault(row['Disease'], []).append(row['Phenotype'])
    with (out_dir / "disease2pheno.json").open("w") as f:
        json.dump(disease2pheno, f)

    n_eval_genes = len(set(pd.read_csv(gene_disease_file)['Gene']))
    metadata = {
        "training": {
            "method": "TransD", "graph": "G4", "mode": "inductive",
            "species": species, "fold": fold, "leak_filter": leak,
            "embedding_dim": EMBEDDING_DIM, "batch_size": BATCH_SIZE,
            "learning_rate": LR, "random_seed": RANDOM_SEED,
        },
        "shapes": {
            "entity_embeddings": list(entity_embeddings.shape),
            "num_entities": len(triples_factory.entity_to_id),
            "num_genes_in_gene2pheno": len(gene2pheno),
            "num_diseases_in_disease2pheno": len(disease2pheno),
            "num_eval_genes": n_eval_genes,
        },
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[extract] bundle written:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
