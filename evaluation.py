import torch as th
from tqdm import tqdm
from evaluate_sem_sim import compute_metrics, print_as_tex

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_model(model, test_disease_genes, gene2pheno, disease2pheno, eval_genes,
                   triples_factory, mode, graph3, graph4, output_file_prefix=None, verbose=False):
    """
    Evaluate the model on a given test set.

    Args:
        model: The trained KGE model
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns
        gene2pheno: Dictionary mapping genes to phenotypes
        disease2pheno: Dictionary mapping diseases to phenotypes
        eval_genes: List of genes to evaluate
        triples_factory: PyKEEN triples factory
        mode: 'inductive' or 'transductive'
        graph3: Boolean indicating if graph3 mode is active
        graph4: Boolean indicating if graph4 mode is active
        output_file_prefix: Optional prefix for output files. If None, results are not saved.

    Returns:
        tuple: (inductive_results, inductive_micro_metrics, inductive_macro_metrics,
                transductive_results, transductive_micro_metrics, transductive_macro_metrics)
               transductive results/metrics are None if mode != 'transductive'
    """
    entity_ids = th.tensor(list(triples_factory.entity_to_id.values()))
    entity_embeddings = model.entity_representations[0](indices=entity_ids).cpu().detach()
    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id
    embedding_dim = entity_embeddings.shape[1]

    logger.debug("Pre-computing gene phenotype vectors...")

    max_pheno_count = 0
    gene_pheno_counts = []

    for gene in eval_genes:
        phenos = gene2pheno[gene]
        count = len(phenos)
        gene_pheno_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    logger.debug(f"Maximum number of phenotypes per gene: {max_pheno_count}")

    all_genes_pheno_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)

    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno[gene]
        pheno_ids = [entity_to_id[p] for p in phenos]
        pheno_vectors = entity_embeddings[th.tensor(pheno_ids)]
        all_genes_pheno_vectors[i, :len(pheno_ids), :] = pheno_vectors

    gene_pheno_counts = th.tensor(gene_pheno_counts, dtype=th.float32)

    # Create gene indices mapping for faster lookup
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    logger.debug(f"Example gene to index mapping: {list(gene_to_index.items())[:5]}")

    test_pairs = []
    for _, row in test_disease_genes.iterrows():
        disease = row['Disease']
        gene = row['Gene']
        test_pairs.append((disease, gene))

    logger.debug(f"Number of test pairs: {len(test_pairs)}")
    logger.debug(f"Example test pair: {test_pairs[0]}")

    inductive_results = []
    transductive_results = []

    with tqdm(total=len(test_pairs), desc='Evaluating', leave=False) as pbar:
        for test_disease, test_gene in test_pairs:

            disease_phenos = disease2pheno[test_disease]
            pheno_ids = [entity_to_id[p] for p in disease_phenos]

            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
            inductive_scores = compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts)
            assert inductive_scores.shape == (len(eval_genes),), f"Scores shape {inductive_scores.shape} does not match number of genes {len(eval_genes)}"
            inductive_scores = inductive_scores.tolist()

            inductive_results.append((test_gene, test_disease, gene_to_index[test_gene], inductive_scores))

            if mode == "transductive":
                gene_ids = th.tensor([entity_to_id[gene] for gene in eval_genes])
                disease_id = th.tensor([entity_to_id[test_disease]]).repeat(len(eval_genes))
                if graph4:
                    relation_id = th.tensor([relation_to_id['associated_with']]).repeat(len(eval_genes))
                    triple_tensor = th.stack([gene_ids, relation_id, disease_id], dim=1).to("cuda")
                    assert triple_tensor.shape == (len(eval_genes), 3), f"Triple tensor shape {triple_tensor.shape} does not match expected {(len(eval_genes), 3)}"
                with th.no_grad():
                    if graph4:
                        transductive_scores = model.score_hrt(triple_tensor).cpu().detach().squeeze().tolist()
                    elif graph3:
                        gene_embeddings = model.entity_representations[0](indices=gene_ids.to("cuda"))
                        disease_embeddings = model.entity_representations[0](indices=disease_id.to("cuda"))
                        transductive_scores = th.sigmoid(th.sum(gene_embeddings * disease_embeddings, dim=1)).cpu().detach().squeeze().tolist()
                    assert len(transductive_scores) == len(eval_genes), f"Transductive scores length {len(transductive_scores)} does not match number of genes {len(eval_genes)}"
                    transductive_results.append((test_gene, test_disease, gene_to_index[test_gene], transductive_scores))

            pbar.update()

    # Compute metrics
    inductive_micro_metrics = None
    inductive_macro_metrics = None
    transductive_micro_metrics = None
    transductive_macro_metrics = None

    if output_file_prefix:
        inductive_results_out_file = f"{output_file_prefix}_inductive.tsv"
        with open(inductive_results_out_file, "w") as f:
            for gene, disease, gene_index, scores in inductive_results:
                scores_str = "\t".join([str(score) for score in scores])
                f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")

        inductive_micro_metrics, inductive_macro_metrics = compute_metrics(inductive_results_out_file, verbose=verbose)
        if verbose:
            print(f"Inductive results saved to {inductive_results_out_file}")
            print_as_tex(inductive_micro_metrics, inductive_macro_metrics)

        if mode == "transductive":
            transductive_results_out_file = f"{output_file_prefix}_transductive.tsv"
            with open(transductive_results_out_file, "w") as f:
                for gene, disease, gene_index, scores in transductive_results:
                    scores_str = "\t".join([str(score) for score in scores])
                    f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")

            transductive_micro_metrics, transductive_macro_metrics = compute_metrics(transductive_results_out_file)
            if verbose:
                print(f"Transductive results saved to {transductive_results_out_file}")
                print_as_tex(transductive_micro_metrics, transductive_macro_metrics)

    return (inductive_results, inductive_micro_metrics, inductive_macro_metrics,
            transductive_results, transductive_micro_metrics, transductive_macro_metrics)


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
