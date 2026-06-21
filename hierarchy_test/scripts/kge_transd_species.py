"""Species-aware INDIGENA training script.

Forked from kge_transd.py. Adds --species {mouse,human} and switches:
  - gene-phenotype source file (mouse: data/gene_phenotypes.csv,
                                 human: data/gene_phenotypes_human.csv)
  - eval_genes source           (mouse: data/gene_diseases.csv,
                                 human: data/gene_diseases_human.csv)
  - fold pair files             (data/gene_disease_folds_unified/fold_N/
                                  {species}_{train,test}.csv)
  - file_identifier prefix      (transd_{species}_...)

For human + --leak_filter strict|light, applies fold-aware edge filtering
on gene-phenotype edges using the AttributedFromDiseases provenance column.
For mouse, the leak filter is a no-op (MGI knockouts are direct experimental
observations, not derived from disease phenotypes).
"""
import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from mowl.datasets import PathDataset
from mowl.utils.random import seed_everything
from pykeen.models import TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import StopperTrainingCallback
import torch as th
from torch.optim import Adam

import os
import click as ck
import pandas as pd
import wandb

from data import create_train_val_split
from pykeen_utils import ValidationStopper
from evaluation import evaluate_model

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

FOLD_BASE = "data/gene_disease_folds_unified"


def model_resolver(triples_factory, embedding_dim, random_seed):
    return TransD(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        relation_dim=embedding_dim,
        random_seed=random_seed,
    )


def projector_resolver(projector_name):
    if projector_name.lower() == "owl2vecstar":
        return "data/upheno_owl2vecstar_edges.tsv", OWL2VecStarProjector(bidirectional_taxonomy=True)
    raise ValueError(f"Projector {projector_name} not supported.")


@ck.command()
@ck.option("--species", type=ck.Choice(["mouse", "human"]), required=True,
           help="Train Model M (mouse) or Model H (human)")
@ck.option("--fold", type=int, default=0)
@ck.option("--graph2", is_flag=True)
@ck.option("--graph3", is_flag=True)
@ck.option("--graph4", is_flag=True)
@ck.option("--projector_name", type=ck.Choice(["owl2vecstar"]), default="owl2vecstar")
@ck.option("--mode", type=ck.Choice(["inductive", "transductive"]), default="inductive")
@ck.option("--embedding_dim", type=int, default=400)
@ck.option("--batch_size", type=int, default=8192)
@ck.option("--learning_rate", type=float, default=0.001)
@ck.option("--random_seed", type=int, default=0)
@ck.option("--only_test", "-ot", is_flag=True)
@ck.option("--description", type=str, default="")
@ck.option("--leak_filter", type=ck.Choice(["none", "light", "strict"]), default="none",
           help="Human only: drop gene-phenotype edges where attributing diseases overlap test fold")
@ck.option("--ancestor_augment_depth", type=int, default=0,
           help="Add (entity, has_phenotype, anc) edges for each (entity, has_phenotype, P) "
                "where anc is at HPO ancestor depth 1..N of P. 0=off, 2=parent+grandparent. "
                "Teaches the model ontology generalization for query-time abstraction.")
@ck.option("--ancestors_json", type=str, default="data/hpo_ancestors.json",
           help="Path to precomputed HPO ancestor map (from precompute_hpo_ancestors.py)")
@ck.option("--no_sweep", is_flag=True)
def main(species, fold, graph2, graph3, graph4, projector_name, mode,
         embedding_dim, batch_size, learning_rate,
         random_seed, only_test, description, leak_filter,
         ancestor_augment_depth, ancestors_json, no_sweep):

    wandb.init(entity="ferzcam", project="indigena", name=description)
    if no_sweep:
        wandb.log({"embedding_dim": embedding_dim, "batch_size": batch_size,
                   "learning_rate": learning_rate, "fold": fold, "mode": mode,
                   "species": species, "leak_filter": leak_filter})
    else:
        embedding_dim = wandb.config.embedding_dim
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        fold = wandb.config.fold
        mode = wandb.config.mode

    seed_everything(random_seed)

    if graph4: graph3 = True
    if graph3: graph2 = True

    # Load HPO ancestor map if augmentation requested
    ancestors_per_term: dict[str, list[list[str]]] = {}
    if ancestor_augment_depth > 0:
        import json as _json
        with open(ancestors_json) as f:
            anc_data = _json.load(f)
        ancestors_per_term = anc_data["ancestors_by_depth"]
        logger.info(f"[ancestor_augment] loaded ancestor map for {len(ancestors_per_term)} HP terms; "
                    f"depth={ancestor_augment_depth}")

    # Per-species data sources
    if species == "mouse":
        gene_pheno_file = "data/gene_phenotypes.csv"
        gene_disease_file = "data/gene_diseases.csv"
    else:
        gene_pheno_file = "data/gene_phenotypes_human.csv"
        gene_disease_file = "data/gene_diseases_human.csv"

    train_disease_genes = pd.read_csv(f"{FOLD_BASE}/fold_{fold}/{species}_train.csv")
    train_disease_genes, val_disease_genes = create_train_val_split(
        train_disease_genes, val_ratio=0.1, random_seed=random_seed
    )
    train_diseases = sorted(set(train_disease_genes['Disease']))
    val_diseases = sorted(set(val_disease_genes['Disease']))
    non_test_diseases = set(train_diseases) | set(val_diseases)

    test_disease_genes = pd.read_csv(f"{FOLD_BASE}/fold_{fold}/{species}_test.csv")
    test_diseases = set(test_disease_genes['Disease'])

    edges_file, projector = projector_resolver(projector_name)
    if not os.path.exists(edges_file):
        ds = PathDataset("data/upheno.owl")
        train_edges = projector.project(ds.ontology)
        with open(edges_file, "w") as f:
            for edge in train_edges:
                f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")

    triples = []
    entities = set()
    relations = set()
    with open(edges_file, "r") as f:
        for line in f:
            src, rel, dst = line.strip().split("\t")
            triples.append((src, rel, dst))
            entities.add(src); entities.add(dst); relations.add(rel)

    gene_phenotypes = pd.read_csv(gene_pheno_file)
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")

    if graph2:
        n_in = n_kept = n_drop_leak = n_drop_unknown_pheno = n_anc_added = 0
        for _, row in gene_phenotypes.iterrows():
            n_in += 1
            gene = row['Gene']; phenotype = row['Phenotype']

            # Leak filter (human only — column is absent for mouse)
            if species == "human" and leak_filter != "none":
                attributing = set(str(row['AttributedFromDiseases']).split(';'))
                if leak_filter == 'light':
                    if attributing.issubset(test_diseases):
                        n_drop_leak += 1; continue
                else:  # strict
                    if attributing & test_diseases:
                        n_drop_leak += 1; continue

            if phenotype not in entities:
                n_drop_unknown_pheno += 1; continue
            triples.append((gene, 'has_phenotype', phenotype))
            entities.add(gene)
            n_kept += 1

            # Ancestor augmentation: add (gene, has_phenotype, anc) for ancestors of P
            if ancestor_augment_depth > 0 and phenotype in ancestors_per_term:
                per_depth = ancestors_per_term[phenotype]
                for d in range(min(ancestor_augment_depth, len(per_depth))):
                    for anc in per_depth[d]:
                        if anc in entities:
                            triples.append((gene, 'has_phenotype', anc))
                            n_anc_added += 1
        logger.info(f"[graph2/{species}] in={n_in} kept={n_kept} "
                    f"drop_leak={n_drop_leak} drop_unknown_pheno={n_drop_unknown_pheno} "
                    f"leak_filter={leak_filter} ancestor_edges_added={n_anc_added}")

    if graph3:
        n_g3_added = n_g3_anc_added = 0
        for _, row in disease_phenotypes.iterrows():
            disease = row['Disease']; phenotype = row['Phenotype']
            assert phenotype in entities, f"Phenotype {phenotype} not in entities"
            if mode == "inductive" and disease in test_diseases:
                continue
            triples.append((disease, 'has_symptom', phenotype))
            entities.add(disease)
            n_g3_added += 1
            # Augment disease side too: (disease, has_symptom, anc(P))
            if ancestor_augment_depth > 0 and phenotype in ancestors_per_term:
                per_depth = ancestors_per_term[phenotype]
                for d in range(min(ancestor_augment_depth, len(per_depth))):
                    for anc in per_depth[d]:
                        if anc in entities:
                            triples.append((disease, 'has_symptom', anc))
                            n_g3_anc_added += 1
        logger.info(f"[graph3] kept={n_g3_added} ancestor_edges_added={n_g3_anc_added}")

    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"
    if mode == "inductive":
        assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"

    if graph4:
        for _, row in train_disease_genes.iterrows():
            disease = row['Disease']; gene = row['Gene']
            if gene not in entities or disease not in entities:
                # In human mode some genes may have only leak-filtered edges; skip gracefully
                continue
            triples.append((gene, 'associated_with', disease))

    entities = sorted(entities); relations = sorted(relations)
    triples = sorted(triples)
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)
    model = model_resolver(triples_factory, embedding_dim, random_seed).to("cuda")

    graph_status = "graph4" if graph4 else "graph3" if graph3 else "graph2" if graph2 else "graph1"
    leak_tag = f"_leak-{leak_filter}" if (species == "human" and leak_filter != "none") else ""
    aug_tag = f"_aug-d{ancestor_augment_depth}" if ancestor_augment_depth > 0 else ""
    file_identifier = (f"transd_{species}_{mode}_fold_{fold}_seed_{random_seed}"
                       f"_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_{graph_status}"
                       f"{leak_tag}{aug_tag}")
    model_out_filename = f"data/models/{file_identifier}.pt"

    # Build gene2pheno + disease2pheno restricted to phenotypes that exist in
    # the trained triples_factory's entity_to_id. Without this filter, the
    # evaluator's `entity_to_id[p]` lookup KeyError's on phenotypes that were
    # filtered out of training (leak filter, unknown-pheno skip, or never in
    # UPheno entity set).
    known_entities = set(triples_factory.entity_to_id.keys())
    gene2pheno = dict()
    n_g2p_in = n_g2p_kept = 0
    for _, row in gene_phenotypes.iterrows():
        n_g2p_in += 1
        if row['Phenotype'] not in known_entities:
            continue
        gene2pheno.setdefault(row['Gene'], []).append(row['Phenotype'])
        n_g2p_kept += 1
    disease2pheno = dict()
    n_d2p_in = n_d2p_kept = 0
    for _, row in disease_phenotypes.iterrows():
        n_d2p_in += 1
        if row['Phenotype'] not in known_entities:
            continue
        disease2pheno.setdefault(row['Disease'], []).append(row['Phenotype'])
        n_d2p_kept += 1
    logger.info(f"[g2p] in={n_g2p_in} kept={n_g2p_kept} (dropped {n_g2p_in-n_g2p_kept} unknown phenos)")
    logger.info(f"[d2p] in={n_d2p_in} kept={n_d2p_kept} (dropped {n_d2p_in-n_d2p_kept} unknown phenos)")
    # Sanity: every eval gene must have at least one phenotype, otherwise BMA
    # divides by zero. Filter eval_genes accordingly later.

    eval_genes_all = set(pd.read_csv(gene_disease_file)['Gene'])
    # Filter to genes that (a) have phenotypes after the known_entities filter
    # and (b) are themselves entities in the trained graph
    eval_genes = sorted(
        g for g in eval_genes_all
        if g in gene2pheno and len(gene2pheno[g]) > 0 and g in known_entities
    )
    logger.info(f"[eval_genes/{species}] in={len(eval_genes_all)} kept={len(eval_genes)} "
                f"(dropped {len(eval_genes_all)-len(eval_genes)} genes without scorable phenotypes or absent from graph)")

    # Also filter test_disease_genes: drop test pairs whose disease has no
    # phenotypes after filter (would crash evaluate_model with disease2pheno KeyError)
    n_test_pairs_in = len(test_disease_genes)
    test_disease_genes = test_disease_genes[
        test_disease_genes['Disease'].isin(disease2pheno.keys()) &
        test_disease_genes['Gene'].isin(eval_genes)
    ].reset_index(drop=True)
    logger.info(f"[test_pairs/{species}] in={n_test_pairs_in} kept={len(test_disease_genes)}")

    validation_stopper = ValidationStopper(
        model, triples_factory, file_identifier, val_disease_genes,
        gene2pheno, disease2pheno, eval_genes, mode, graph3, graph4,
        tolerance=5, model_out_filename=model_out_filename,
    )
    validation_callback = StopperTrainingCallback(
        stopper=validation_stopper, triples_factory=triples_factory,
        best_epoch_model_file_path=model_out_filename
    )
    optimizer = Adam(params=model.get_grad_params(), lr=learning_rate)

    if not only_test:
        training_loop = SLCWATrainingLoop(
            model=model, triples_factory=triples_factory, optimizer=optimizer
        )
        _ = training_loop.train(
            triples_factory=triples_factory, num_epochs=1000,
            batch_size=batch_size, callbacks=[validation_callback],
        )

    print(f"Training complete. Loading best model {model_out_filename}")
    model.load_state_dict(th.load(model_out_filename, weights_only=True))

    output_prefix = f"data/results/kge_results_{file_identifier}"
    if os.environ.get("EVAL_TAG"):
        output_prefix += "_" + os.environ["EVAL_TAG"]
    _pert = os.environ.get("EVAL_DISEASE_PHENO_CSV", "")
    if _pert:
        _pdf = pd.read_csv(_pert)
        _ov = {}; _skip = 0; _tot = 0
        for _r in _pdf.itertuples(index=False):
            _tot += 1
            if _r.Phenotype not in known_entities:
                _skip += 1; continue
            _ov.setdefault(_r.Disease, []).append(_r.Phenotype)
        disease2pheno = _ov
        print(f"[PERTURB] disease2pheno <- {_pert}: {len(_ov)} diseases, {_skip}/{_tot} terms OOV (not in graph)")
    (inductive_bma_macro_metrics, inductive_bmm_macro_metrics,
     transductive_sim_macro_metrics, transductive_function_macro_metrics) = evaluate_model(
        model=model, test_disease_genes=test_disease_genes,
        gene2pheno=gene2pheno, disease2pheno=disease2pheno,
        eval_genes=eval_genes, triples_factory=triples_factory,
        mode=mode, graph3=graph3, graph4=graph4,
        output_file_prefix=output_prefix, verbose=True
    )

    metrics = ['mr', 'mrr', 'auc', 'hits@1', 'hits@3', 'hits@10', 'hits@100']
    wandb.log({f"test_imac_bma_{k}": v for k, v in inductive_bma_macro_metrics.items() if k in metrics})
    wandb.log({f"test_imac_bmm_{k}": v for k, v in inductive_bmm_macro_metrics.items() if k in metrics})


if __name__ == "__main__":
    main()
