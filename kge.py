import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge, CategoricalProjector
from mowl.datasets import PathDataset
from mowl.utils.random import seed_everything
from pykeen.models import TransE, TransD
from pykeen.training import SLCWATrainingLoop
import torch as th
from torch.optim import Adam

import os
import click as ck
import pandas as pd
import time
from tqdm import tqdm
import wandb

from evaluate_sem_sim import compute_metrics, print_as_tex

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class OrderE(TransE):
    def __init__(self, *args, **kwargs):
        super(OrderE, self).__init__(*args, **kwargs)

    def forward(self, h_indices, r_indices, t_indices, mode = None):
        h, _, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        order_loss = th.linalg.norm(th.relu(t-h), dim=1)
        return -order_loss

    def score_hrt(self, hrt_batch, mode = None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r = hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return -th.linalg.norm(th.relu(t-h), dim=1)


def model_resolver(model_name, triples_factory, embedding_dim, random_seed):
    if model_name.lower() == "transe":
        model = TransE(triples_factory=triples_factory, embedding_dim=embedding_dim, random_seed=random_seed)
    elif model_name.lower() == "transd":
        model = TransD(triples_factory=triples_factory, embedding_dim=embedding_dim, random_seed=random_seed)
    elif model_name.lower() == "ordere":
        model = OrderE(triples_factory=triples_factory, embedding_dim=embedding_dim, random_seed=random_seed)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def projector_resolver(projector_name):
    if projector_name.lower() == "owl2vecstar":
        edges_file = "data/upheno_owl2vecstar_edges.tsv"
        projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
    elif projector_name.lower () == "categorical":
        edges_file = "data/upheno_categorical_edges.tsv"
        projector = CategoricalProjector("str")
        
    else:
        raise ValueError(f"Projector {projector_name} not supported.")

    return edges_file, projector

@ck.command()
@ck.option("--fold", type=int, default=0, help="Fold number for the dataset")
@ck.option("--graph2", is_flag=True, help="Use graph2")
@ck.option("--graph3", is_flag=True, help="Use graph3")
@ck.option("--graph4", is_flag=True, help="Use graph4")
@ck.option("--projector_name", type=ck.Choice(["owl2vecstar", "categorical"]), default="owl2vecstar", help="Projector to use for ontology projection")
@ck.option("--model_name", type=ck.Choice(["transe", "transd", "ordere"]), default="transd", help="Knowledge graph embedding model to use")
@ck.option("--embedding_dim", type=int, default=100, help="Embedding dimension for the KGE model")
@ck.option("--batch_size", type=int, default=128, help="Batch size for training")
@ck.option("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
@ck.option("--num_epochs", type=int, default=100, help="Number of training epochs")
@ck.option("--random_seed", type=int, default=0, help="Random seed for reproducibility")
@ck.option("--only_test", "-ot", is_flag=True, help="Only test the model")
@ck.option("--description", type=str, default="", help="Description for the wandb run")
@ck.option("--no_sweep", is_flag=True, help="Disable wandb sweep mode")
def main(fold, graph2, graph3, graph4, projector_name, model_name,
         embedding_dim, batch_size, learning_rate, num_epochs,
         random_seed, only_test, description, no_sweep):

    wandb.init(entity="ferzcam", project="indiga", name=description)                
    if no_sweep:
        wandb.log({"embedding_dim": embedding_dim,
                   "batch_size": batch_size,
                   "learning_rate": learning_rate,
                   "num_epochs": num_epochs,
                   "fold": fold,
                   })
    else:
        embedding_dim = wandb.config.embedding_dim
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        num_epochs = wandb.config.num_epochs
        fold = wandb.config.fold

    
    seed_everything(random_seed)
    
    if graph4:
        graph3 = True
    if graph3:
        graph2 = True

    
    edges_file, projector = projector_resolver(projector_name)

    if not os.path.exists(edges_file):
        ds = PathDataset("data/upheno.owl")

        # Project ontology using OWL2VecStarProjector
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

    train_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/train.csv")
    train_diseases = set(train_disease_genes['Disease'].values)
        
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
    
    model = model_resolver(model_name, triples_factory, embedding_dim, random_seed).to("cuda")


    graph_status = "graph4" if graph4 else "graph3" if graph3 else "graph2" if graph2 else "graph1"
    
    model_out_filename = f"data/models/projector_{projector_name}_model_{model_name}_fold_{fold}_seed_{random_seed}_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_epochs_{num_epochs}_{graph_status}.pt"
    results_out_file = f"data/results/kge_results_{projector_name}_{model_name}_fold_{fold}_seed_{random_seed}_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_epochs_{num_epochs}_{graph_status}.tsv"

    
    optimizer = Adam(params=model.get_grad_params(), lr=learning_rate)

    if not only_test:
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer=optimizer,
            
        )
 
        batch_size = batch_size
        start_time = time.time()
        _ = training_loop.train(
            triples_factory=triples_factory,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        th.save(model.state_dict(), model_out_filename)
    

    model.load_state_dict(th.load(model_out_filename))
    
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
    test_diseases = set(test_disease_genes['Disease'].values)

    assert len(test_diseases & train_diseases) == 0, "Test diseases overlap with train diseases"
    
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
        phenos = gene2pheno[gene]
        count = len(phenos)
        gene_pheno_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    logger.info(f"Maximum number of phenotypes per gene: {max_pheno_count}")
    
    all_genes_pheno_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)
    
    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno[gene]
        pheno_ids = [entity_to_id[p] for p in phenos]
        pheno_vectors = entity_embeddings[th.tensor(pheno_ids)]
        all_genes_pheno_vectors[i, :len(pheno_ids), :] = pheno_vectors

    gene_pheno_counts = th.tensor(gene_pheno_counts, dtype=th.float32)
                
    # Create gene indices mapping for faster lookup
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    logger.info(f"Example gene to index mapping: {list(gene_to_index.items())[:5]}")
    
    results = []
    
    with tqdm(total=len(test_pairs), desc='Evaluating test diseases') as pbar:
        for test_disease, test_gene in test_pairs:
            
            disease_phenos = disease2pheno[test_disease]
            pheno_ids = [entity_to_id[p] for p in disease_phenos]

            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
            scores = compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts)
            assert scores.shape == (len(eval_genes),), f"Scores shape {scores.shape} does not match number of genes {len(eval_genes)}"
            scores = scores.tolist()
            
            results.append((test_gene, test_disease, gene_to_index[test_gene], scores))
            pbar.update()

    with open(results_out_file, "w") as f:
        for gene, disease, gene_index, scores in results:
            scores_str = "\t".join([str(score) for score in scores])
            f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")


    micro_metrics, macro_metrics = compute_metrics(results_out_file)
    metrics = ['mr', 'mrr', 'auc', 'hits@1', 'hits@3', 'hits@10', 'hits@100']

    macro_to_log = {f"mac_{k}": v for k, v in macro_metrics.items() if k in metrics}
    micro_to_log = {f"mic_{k}": v for k, v in micro_metrics.items() if k in metrics}

    wandb.log({**macro_to_log, **micro_to_log})
    print_as_tex(micro_metrics, macro_metrics)

    
def compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma"):
    """
    Compute similarity between a disease and all genes in a vectorized manner.

    :param all_genes_pheno_vectors: Padded tensor of shape (num_genes, max_phenos, emb_dim)
    :param disease_phenos_vectors: Tensor of shape (num_disease_phenos, emb_dim)
    :param gene_pheno_counts: Tensor of shape (num_genes, 1) with counts of real phenotypes for each gene.
    :param criterion: Similarity criterion.
    """
            
    num_genes, max_phenos, emb_dim = all_genes_pheno_vectors.shape
    num_disease_phenos = disease_phenos_vectors.shape[0]

    # Reshape for matrix multiplication: (num_genes * max_phenos, a) x (a,b)
    sim_matrix = th.matmul(
        all_genes_pheno_vectors.view(-1, emb_dim),
        disease_phenos_vectors.T
    )

    # before sigmoid make 0s very negative
    sim_matrix[sim_matrix == 0] = -th.inf
    
    sim_matrix = th.sigmoid(sim_matrix) # Resulting shape: (num_genes*max_phenos, num_disease_phenos)

    sim_matrix = sim_matrix.view(num_genes, max_phenos, num_disease_phenos)
    
    if criterion == "bma":
        # Gene-centric scores
        logger.debug(f"Sim matrix shape: {sim_matrix.shape}")
        gene_max_sim, _ = sim_matrix.max(dim=-1)
        logger.debug(f"Gene max sim shape: {gene_max_sim.shape}")
        gene_centric_sum = gene_max_sim.sum(dim=-1)
        logger.debug(f"Gene centric sum shape: {gene_centric_sum.shape}")
        # For genes with 0 phenotypes, gene_pheno_counts is 0, this will result in NaN. Avoid division by zero.
        # We replace 0 counts with 1 to avoid division by zero. The sum is 0 so the score will be 0.
        gene_pheno_counts_safe = th.max(gene_pheno_counts, th.tensor(1.0))
        logger.debug(f"Gene pheno counts shape: {gene_pheno_counts_safe.shape}")
        gene_centric_scores = gene_centric_sum / gene_pheno_counts_safe
        logger.debug(f"Gene centric scores shape: {gene_centric_scores.shape}")

        assert th.all(gene_centric_scores >= 0) and th.all(gene_centric_scores <= 1), "Gene centric scores out of range [0, 1]"
        
        # Disease-centric scores
        disease_max_sim, _ = sim_matrix.max(dim=1)
        disease_centric_scores = disease_max_sim.mean(dim=-1)
        # disease_centric_scores = disease_centric_sum / num_disease_phenos

        assert th.all(disease_centric_scores >= 0) and th.all(disease_centric_scores <= 1), "Disease centric scores out of range [0, 1]"
        assert gene_centric_scores.shape == disease_centric_scores.shape == (num_genes,), f"Scores shape mismatch: {gene_centric_scores.shape}, {disease_centric_scores.shape}, expected {(num_genes,)}"
        
        scores = (gene_centric_scores + disease_centric_scores) / 2
        return scores
    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")

    
if __name__ == "__main__":
    main()
