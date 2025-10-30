import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from mowl.datasets import PathDataset
from mowl.utils.random import seed_everything
from pykeen.models import TransE
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

def model_resolver(triples_factory, embedding_dim, random_seed, scoring_fct_norm=1):
    model = TransE(
        triples_factory=triples_factory, 
        embedding_dim=embedding_dim, 
        random_seed=random_seed,
        scoring_fct_norm=scoring_fct_norm
    )
    return model

def projector_resolver(projector_name):
    if projector_name.lower() == "owl2vecstar":
        edges_file = "data/upheno_owl2vecstar_edges.tsv"
        projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
    else:
        raise ValueError(f"Projector {projector_name} not supported.")

    return edges_file, projector

@ck.command()
@ck.option("--fold", type=int, default=0, help="Fold number for the dataset")
@ck.option("--graph2", is_flag=True, help="Use graph2")
@ck.option("--graph3", is_flag=True, help="Use graph3")
@ck.option("--graph4", is_flag=True, help="Use graph4")
@ck.option("--projector_name", type=ck.Choice(["owl2vecstar"]), default="owl2vecstar", help="Projector to use for ontology projection")
@ck.option("--mode", type=ck.Choice(["inductive", "transductive"]), default="inductive", help="Inductive or transductive setting")
@ck.option("--embedding_dim", type=int, default=100, help="Embedding dimension for the KGE model")
@ck.option("--batch_size", type=int, default=2048, help="Batch size for training")
@ck.option("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
@ck.option("--num_epochs", type=int, default=1000, help="Number of training epochs")
@ck.option("--scoring_fct_norm", type=int, default=2, help="Norm for TransE scoring function (1 or 2)")
@ck.option("--random_seed", type=int, default=0, help="Random seed for reproducibility")
@ck.option("--only_test", "-ot", is_flag=True, help="Only test the model")
@ck.option("--description", type=str, default="", help="Description for the wandb run")
@ck.option("--no_sweep", is_flag=True, help="Disable wandb sweep mode")
def main(fold, graph2, graph3, graph4, projector_name, mode, embedding_dim, 
         batch_size, learning_rate, num_epochs, scoring_fct_norm, random_seed, 
         only_test, description, no_sweep):

    wandb.init(entity="ferzcam", project="indigena", name=description)                
    if no_sweep:
        wandb.log({"embedding_dim": embedding_dim,
                   "batch_size": batch_size,
                   "learning_rate": learning_rate,
                   "num_epochs": num_epochs,
                   "scoring_fct_norm": scoring_fct_norm,
                   "fold": fold,
                   "mode": mode
                   })
    else:
        embedding_dim = wandb.config.embedding_dim
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        num_epochs = wandb.config.num_epochs
        scoring_fct_norm = wandb.config.scoring_fct_norm
        fold = wandb.config.fold
        mode = wandb.config.mode
    
    seed_everything(random_seed)
    
    if graph4:
        graph3 = True
    if graph3:
        graph2 = True


    train_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/train.csv")

    # Split into train and validation ensuring all validation entities are in training
    train_disease_genes, val_disease_genes = create_train_val_split(train_disease_genes, val_ratio=0.1, random_seed=random_seed)

    train_diseases = sorted(list(set(train_disease_genes['Disease'].values)))
    val_diseases = sorted(list(set(val_disease_genes['Disease'].values)))

    non_test_diseases = set(train_diseases) | set(val_diseases)
    
    test_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/test.csv")
    test_diseases = set(test_disease_genes['Disease'].values)

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
            entities.add(src)
            entities.add(dst)
            relations.add(rel)
            
    gene_phenotypes = pd.read_csv("data/gene_phenotypes.csv")
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")
    
    if graph2:
        for _, row in gene_phenotypes.iterrows():
            gene = row['Gene']
            phenotype = row['Phenotype']
            assert phenotype in entities, f"Phenotype {phenotype} not in entities"
            triples.append((gene, 'has_phenotype', phenotype))
            entities.add(gene)

    if graph3:
        for _, row in disease_phenotypes.iterrows():
            disease = row['Disease']
            phenotype = row['Phenotype']
            assert phenotype in entities, f"Phenotype {phenotype} not in entities"
            if mode == "inductive":
                if disease in test_diseases:
                    continue
            triples.append((disease, 'has_symptom', phenotype))
            entities.add(disease)


    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"

    if mode == "inductive":
        assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"
    elif mode == "transductive":
        assert len(test_diseases & entities) == len(test_diseases), f"Some test diseases not in train diseases. Num entities: {len(entities)}, Num test diseases in entities: {len(test_diseases & entities)}, Total test diseases: {len(test_diseases)}"
    else:
        raise ValueError(f"Mode {mode} not supported.")

    
    if graph4:
        for _, row in train_disease_genes.iterrows():
            disease = row['Disease']
            gene = row['Gene']
            triples.append((gene, 'associated_with', disease))
            assert gene in entities, f"Gene {gene} not in entities"
            assert disease in entities, f"Disease {disease} not in entities"
            

    entities = sorted(list(entities))
    relations = sorted(list(relations))

    triples = sorted(triples)
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)

    model = model_resolver(triples_factory, embedding_dim, random_seed, scoring_fct_norm).to("cuda")

    graph_status = "graph4" if graph4 else "graph3" if graph3 else "graph2" if graph2 else "graph1"

    file_identifier = f"transe_{mode}_fold_{fold}_seed_{random_seed}_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_norm_{scoring_fct_norm}_{graph_status}"
    model_out_filename = f"data/models/{file_identifier}_epochs_{num_epochs}.pt"

    # Build gene2pheno and disease2pheno mappings (needed for validation and testing)
    gene2pheno = dict()
    for _, row in gene_phenotypes.iterrows():
        gene = row['Gene']
        phenotype = row['Phenotype']
        if gene not in gene2pheno:
            gene2pheno[gene] = []
        gene2pheno[gene].append(phenotype)

    disease2pheno = dict()
    for _, row in disease_phenotypes.iterrows():
        disease = row['Disease']
        phenotype = row['Phenotype']
        if disease not in disease2pheno:
            disease2pheno[disease] = []
        disease2pheno[disease].append(phenotype)

    all_gene_diseases = pd.read_csv("data/gene_diseases.csv")
    eval_genes = set(all_gene_diseases['Gene'].values)
    logger.info(f"Number of evaluation genes: {len(eval_genes)}")
    eval_genes = sorted(list(eval_genes))

    tolerance = 5
    validation_stopper = ValidationStopper(
        model,
        triples_factory,
        file_identifier,
        val_disease_genes,
        gene2pheno,
        disease2pheno,
        eval_genes,
        mode,
        graph3,
        graph4,
        tolerance,
        model_out_filename,
    )

    validation_callback = StopperTrainingCallback(stopper=validation_stopper, triples_factory=triples_factory, best_epoch_model_file_path=model_out_filename)

    optimizer = Adam(params=model.get_grad_params(), lr=learning_rate)

    if not only_test:

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer=optimizer
        )

        _ = training_loop.train(
            triples_factory=triples_factory,
            num_epochs=1000,
            batch_size=batch_size,
            callbacks=[validation_callback],
        )

    print("Training complete. Loading best model for testing...")


    model.load_state_dict(th.load(model_out_filename, weights_only=True))

    # Evaluate on test set
    output_prefix = f"data/results/kge_results_{file_identifier}"

    (inductive_results, inductive_micro_metrics, inductive_macro_metrics,
     transductive_results, transductive_micro_metrics, transductive_macro_metrics) = evaluate_model(
         model=model,
         test_disease_genes=test_disease_genes,
         gene2pheno=gene2pheno,
         disease2pheno=disease2pheno,
         eval_genes=eval_genes,
         triples_factory=triples_factory,
         mode=mode,
         graph3=graph3,
         graph4=graph4,
         output_file_prefix=output_prefix,
         verbose=True
    )

    # Log test metrics to wandb
    metrics = ['mr', 'mrr', 'auc', 'hits@1', 'hits@3', 'hits@10', 'hits@100']
    macro_to_log = {f"test_imac_{k}": v for k, v in inductive_macro_metrics.items() if k in metrics}
    micro_to_log = {f"test_imic_{k}": v for k, v in inductive_micro_metrics.items() if k in metrics}
    wandb.log({**macro_to_log, **micro_to_log})

    if mode == "transductive":
        macro_to_log = {f"test_tmac_{k}": v for k, v in transductive_macro_metrics.items() if k in metrics}
        micro_to_log = {f"test_tmic_{k}": v for k, v in transductive_micro_metrics.items() if k in metrics}
        wandb.log({**macro_to_log, **micro_to_log})


if __name__ == "__main__":
    main()
