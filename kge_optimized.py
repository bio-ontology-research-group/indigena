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

    embedding_dim = entity_embeddings.shape[1]
    
    logger.info("Pre-computing gene phenotype vectors...")
    
    max_pheno_count = 0
    gene_pheno_counts = []
    
    for gene in eval_genes:
        phenos = gene2pheno.get(gene, [])
        count = len(phenos)
        gene_pheno_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    logger.info(f"Maximum number of phenotypes per gene: {max_pheno_count}")
    
    all_genes_pheno_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)
    
    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno.get(gene, [])
        if phenos:
            pheno_ids = [entity_to_id[p] for p in phenos if p in entity_to_id]
            if pheno_ids:
                pheno_vectors = entity_embeddings[th.tensor(pheno_ids)]
                all_genes_pheno_vectors[i, :len(pheno_ids), :] = pheno_vectors

    gene_pheno_counts = th.tensor(gene_pheno_counts, dtype=th.float32).unsqueeze(1)
                
    # Create gene indices mapping for faster lookup
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    
    results = []
    unique_test_diseases = sorted(list(set([d for d,g in test_pairs])))
    
    with tqdm(total=len(unique_test_diseases), desc='Evaluating test diseases') as pbar:
        for test_disease in unique_test_diseases:
            disease_phenos = disease2pheno.get(test_disease, [])
            pheno_ids = [entity_to_id[p] for p in disease_phenos if p in entity_to_id]

            if not pheno_ids:
                scores = [0.0] * len(eval_genes)
            else:
                disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
                scores = compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts)
                scores = scores.tolist()
            
            # Find all genes associated with this disease in the test set
            for gene_in_test, disease_in_test in test_pairs:
                if disease_in_test == test_disease:
                    gene_index = gene_to_index[gene_in_test]
                    results.append((gene_in_test, test_disease, gene_index, scores))
            pbar.update()

    results_out_file = f"data/kge_results_{model_name}_fold_{fold}_results.txt"
    with open(results_out_file, "w") as f:
        for gene, disease, gene_index, scores in results:
            scores_str = "\t".join([str(score) for score in scores])
            f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")


def compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma"):
    """
    Compute similarity between a disease and all genes in a vectorized manner.

    :param all_genes_pheno_vectors: Padded tensor of shape (num_genes, max_phenos, emb_dim)
    :param disease_phenos_vectors: Tensor of shape (num_disease_phenos, emb_dim)
    :param gene_pheno_counts: Tensor of shape (num_genes, 1) with counts of real phenotypes for each gene.
    :param criterion: Similarity criterion.
    """
    if disease_phenos_vectors.shape[0] == 0:
        return th.zeros(all_genes_pheno_vectors.shape[0])

    num_genes, max_phenos, emb_dim = all_genes_pheno_vectors.shape
    num_disease_phenos = disease_phenos_vectors.shape[0]

    # Reshape for matrix multiplication: (num_genes * max_phenos, a) x (a,b)
    sim_matrix = th.sigmoid(th.matmul(
        all_genes_pheno_vectors.view(-1, emb_dim),
        disease_phenos_vectors.T
    )) # Resulting shape: (num_genes*max_phenos, num_disease_phenos)

    sim_matrix = sim_matrix.view(num_genes, max_phenos, num_disease_phenos)
    
    if criterion == "bma":
        # Gene-centric scores
        gene_max_sim, _ = sim_matrix.max(dim=2)
        gene_centric_sum = gene_max_sim.sum(dim=1).unsqueeze(1)
        # For genes with 0 phenotypes, gene_pheno_counts is 0, this will result in NaN. Avoid division by zero.
        # We replace 0 counts with 1 to avoid division by zero. The sum is 0 so the score will be 0.
        gene_pheno_counts_safe = th.max(gene_pheno_counts, th.tensor(1.0))
        gene_centric_scores = gene_centric_sum / gene_pheno_counts_safe
        
        # Disease-centric scores
        disease_max_sim, _ = sim_matrix.max(dim=1)
        disease_centric_sum = disease_max_sim.sum(dim=1)
        disease_centric_scores = disease_centric_sum / num_disease_phenos

        scores = (gene_centric_scores.squeeze() + disease_centric_scores) / 2
        return scores
    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")

    
if __name__ == "__main__":
    main()
