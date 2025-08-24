import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from mowl.datasets import PathDataset
from mowl.evaluation import BaseRankingEvaluator
from mowl.utils.random import seed_everything
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, TransD
from pykeen.training import SLCWATrainingLoop


import torch as th
from torch.optim import Adam


import os
import click as ck
import pandas as pd
import time
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

def model_resolver(model_name, triples_factory, embedding_dim=100):
    if model_name == "TransE":
        model = TransE(triples_factory=triples_factory, embedding_dim=embedding_dim, random_seed=RANDOM_SEED) 
    elif model_name == "TransD":
        model = TransD(triples_factory=triples_factory, embedding_dim=embedding_dim, random_seed=RANDOM_SEED)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

         
    


@ck.command()
@ck.option("--fold", type=int, default=0, help="Fold number for the dataset")
@ck.option("--graph1", is_flag=True, help="Use graph1")
@ck.option("--graph2", is_flag=True, help="Use graph2")
@ck.option("--graph3", is_flag=True, help="Use graph3")
@ck.option("--graph4", is_flag=True, help="Use graph4")
@ck.option("--model_name", type=ck.Choice(["TransE"]), default="TransE", help="Knowledge graph embedding model to use")
@ck.option("--only_test", "-ot", is_flag=True, help="Only test the model")
def main(fold, graph1, graph2, graph3, graph4, model_name, only_test):
    if graph4:
        graph3 = True
    if graph3:
        graph2 = True
    if graph2:
        graph1 = True

    edges_file = "data/upheno_owl2vecstar_edges.tsv"

    if not os.path.exists(edges_file):
        ds = PathDataset("data/upheno.owl")

        # Project ontology using OWL2VecStarProjector
        projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
        train_edges = projector.project(ds.ontology)
        
        # Write edges to a file
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
            triples.append((disease, 'has_phenotype', phenotype))
            entities.add(disease)
            
    if graph4:
        disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/train.csv")
        for _, row in disease_genes.iterrows():
            disease = row['Disease']
            gene = row['Gene']
            extra_triples.append((gene, 'associated_with', disease))
            assert gene in entities, f"Gene {gene} not in entities"
            assert disease in entities, f"Disease {disease} not in entities"
            

    entities = sorted(list(entities))
    relations = sorted(list(relations))

    triples = sorted(triples)
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)
    
    model = model_resolver(model_name=model_name, triples_factory=triples_factory, embedding_dim=100).to("cuda:1")

    optimizer = Adam(params=model.get_grad_params())

    if not only_test:
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer=optimizer,
        )

        num_epochs = 1  # Set higher for better results
        batch_size = 1000
        start_time = time.time()
        _ = training_loop.train(
            triples_factory=triples_factory,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        th.save(model.state_dict(), f"data/model_{model_name}_fold_{fold}.pt")
    

    model.load_state_dict(th.load(f"data/model_{model_name}_fold_{fold}.pt"))
    
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
        
    test_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/test.csv")

    test_pairs = []
    for _, row in test_disease_genes.iterrows():
        disease = row['Disease']
        gene = row['Gene']
        test_pairs.append((disease, gene))

    logger.info(f"Number of test pairs: {len(test_pairs)}")
    logger.info(f"Example test pair: {test_pairs[0]}")


        
    entity_ids = th.tensor(list(triples_factory.entity_to_id.values()))
    entity_embeddings = model.entity_representations[0](indices=entity_ids).cpu().detach()
    entity_to_id = triples_factory.entity_to_id

    logger.info("Pre-computing gene phenotype vectors...")
    gene_to_pheno_vectors = {}
    for gene in eval_genes:
        phenos = gene2pheno[gene]
        pheno_ids = [entity_to_id[p] for p in phenos]
        pheno_vectors = entity_embeddings[th.tensor(pheno_ids)]
        gene_to_pheno_vectors[gene] = pheno_vectors

    # Number of processes for gene parallelization
    num_processes = 40
    
    results = []
    # Process each test pair (disease) serially
    with tqdm(total=len(test_pairs), desc='Evaluating test diseases') as pbar:
        for test_pair in test_pairs:
            test_disease, test_gene = test_pair
            disease_phenos = disease2pheno[test_disease]
            
            # Get disease phenotype vectors
            pheno_ids = [entity_to_id[pheno] for pheno in disease_phenos if pheno in entity_to_id]
            if not pheno_ids:
                # No phenotypes found for this disease
                results.append((test_gene, test_disease, -1, [0.0] * len(eval_genes)))
                pbar.update()
                continue
            
            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
            gene_index = eval_genes.index(test_gene)

            indexed_genes = [(i, gene) for i, gene in enumerate(eval_genes)]
            
            # Process gene chunks in parallel
            with get_context("spawn").Pool(num_processes) as pool:
                disease_results = []
                with tqdm(total=len(eval_genes), desc=f'Processing disease {test_disease}', leave=False) as pbar_gene:
                    for output in pool.imap_unordered(partial(process_gene, gene_to_pheno_vectors, disease_phenos_vectors), indexed_genes, chunksize=20):
                        disease_results.append(output)
                        pbar_gene.update()

            scores = sorted(disease_results, key=lambda x: x[0])
            results.append((test_gene, test_disease, gene_index, scores))
            pbar.update()

    results_out_file = f"data/kge_results_{model_name}_fold_{fold}_results.txt"
    with open(results_out_file, "w") as f:
        for gene, disease, gene_index, scores in results:
            scores_str = "\t".join([str(score) for score in scores])
            f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")

            

def process_gene(gene_to_pheno_vectors, disease_phenos_vectors, indexed_gene):
    """Process a chunk of genes against a disease's phenotype vectors."""
    index, gene = indexed_gene
    
    gene_phenos_vectors = gene_to_pheno_vectors[gene]
    score = compare(gene_phenos_vectors, disease_phenos_vectors)
    return index, score
        
    
def compare(gene_phenos_vectors, disease_phenos_vectors, criterion="bma"):
    # Compute similarity matrix more efficiently
    sim_matrix = th.sigmoid(gene_phenos_vectors @ disease_phenos_vectors.T)
    
    if criterion == "bma":
        # Use torch operations for better performance
        gene_centric_score = sim_matrix.max(dim=1)[0].mean().item()
        disease_centric_score = sim_matrix.max(dim=0)[0].mean().item()
        score = (gene_centric_score + disease_centric_score) / 2
    
    return score
    
    
if __name__ == "__main__":
    main()
