import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from pykeen.models import TransD
import pandas as pd
import torch as th
import click as ck
import matplotlib.pyplot as plt
import umap
import numpy as np
from data import create_train_val_split

@ck.command()
@ck.option("--graph", type=ck.Choice(["graph1", "graph2", "graph3", "graph4"]), required=True)
@ck.option("--fold", type=str, required=True, help="Path to the trained model file")
@ck.option("--edges_file", type=str, default="data/upheno_owl2vecstar_edges.tsv", help="Path to edges file")
@ck.option("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
@ck.option("--max_samples", type=int, default=50000, help="Maximum number of samples per phenotype type")
def main(graph, fold, edges_file, n_neighbors, max_samples):
    """
    Generate a UMAP plot of Mouse (MP_) and Human (HP_) phenotype embeddings.
    """
    random_seed = 0
    # Load the graph structure
    print("Loading graph structure...")
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

    graph1 = False
    graph2 = False
    graph3 = False
    graph4 = False
    
    if graph == "graph4":
        graph4 = True
        graph3 = True
        graph2 = True
        graph1 = True
    if graph == "graph3":
        graph3 = True
        graph2 = True
        graph1 = True


    if graph == "graph2":
        graph2 = True
        graph1 = True

    if graph == "graph1":
        graph1 = True


    train_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/train.csv")

    # Split into train and validation ensuring all validation entities are in training
    train_disease_genes, val_disease_genes = create_train_val_split(train_disease_genes, val_ratio=0.1, random_seed=0)

    train_diseases = sorted(list(set(train_disease_genes['Disease'].values)))
    val_diseases = sorted(list(set(val_disease_genes['Disease'].values)))

    non_test_diseases = set(train_diseases) | set(val_diseases)

    test_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/test.csv")
    test_diseases = set(test_disease_genes['Disease'].values)


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
            if disease in test_diseases:
                    continue
            triples.append((disease, 'has_symptom', phenotype))
            entities.add(disease)


    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"

    assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"
            
            

    
    if graph4:
        for _, row in train_disease_genes.iterrows():
            disease = row['Disease']
            gene = row['Gene']
            triples.append((gene, 'associated_with', disease))
            assert gene in entities, f"Gene {gene} not in entities"
            assert disease in entities, f"Disease {disease} not in entities"
            

    entities = sorted(list(entities))
    relations = sorted(list(relations))


            
    entities = sorted(list(entities))
    relations = sorted(list(relations))
    triples = sorted(triples)

    # Create triples factory
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)

    # Initialize model
    print("Initializing TransD model...")

    lr = 0.001
    dim = 400
    if graph in ["graph1", "graph3", "graph4"]:
        bs = 8192
        
    elif graph == "graph2":
        bs = 4096
        
        
    model = TransD(
        triples_factory=triples_factory,
        embedding_dim=dim,
        relation_dim=dim,
        random_seed=random_seed,
    )

    model_path = f"data/models/transd_inductive_fold_{fold}_seed_0_dim_{dim}_bs_{bs}_lr_{lr}_{graph}.pt"
    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(th.load(model_path, weights_only=True, map_location=th.device('cpu')))
    model.eval()

    # Extract entity embeddings
    print("Extracting entity embeddings...")
    entity_to_id = triples_factory.entity_to_id

    # Get phenotypes associated with test diseases
    test_disease_phenotypes = set()
    if graph3 or graph4:
        for _, row in disease_phenotypes.iterrows():
            if row['Disease'] in test_diseases:
                test_disease_phenotypes.add(row['Phenotype'])

    
    # Filter entities for Mouse and Human phenotypes, genes, and diseases
    mp_entities = []
    hp_entities = []
    gene_entities = []
    disease_entities = []
    other_entities = []

    for entity in entities:
        if "MP_" in entity:
            mp_entities.append(entity)
        elif "HP_" in entity:
            hp_entities.append(entity)
        elif "MGI_" in entity and graph2:
            gene_entities.append(entity)
        elif "OMIM_" in entity and graph3:
            disease_entities.append(entity)
        else:
            other_entities.append(entity)

    print(f"Found {len(mp_entities)} Mouse phenotypes (MP_)")
    print(f"Found {len(hp_entities)} Human phenotypes (HP_)")
    if graph2:
        print(f"Found {len(gene_entities)} Genes (MGI_)")
    if graph3:
        print(f"Found {len(disease_entities)} Diseases (OMIM_)")
    print(f"Found {len(other_entities)} Other entities")

    # Sample if needed
    if len(mp_entities) > max_samples:
        np.random.seed(random_seed)
        mp_entities = list(np.random.choice(mp_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Mouse phenotypes")

    if len(hp_entities) > max_samples:
        np.random.seed(random_seed + 1)
        hp_entities = list(np.random.choice(hp_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Human phenotypes")

    if graph2 and len(gene_entities) > max_samples:
        np.random.seed(random_seed + 2)
        gene_entities = list(np.random.choice(gene_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Genes")

    if graph3 and len(disease_entities) > max_samples:
        np.random.seed(random_seed + 3)
        disease_entities = list(np.random.choice(disease_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Diseases")

    if len(other_entities) > max_samples:
        np.random.seed(random_seed + 4)
        other_entities = list(np.random.choice(other_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Other entities")

    # Get embeddings
    selected_entities = mp_entities + hp_entities + gene_entities + disease_entities + other_entities
    selected_ids = th.tensor([entity_to_id[entity] for entity in selected_entities])

    with th.no_grad():
        embeddings = model.entity_representations[0](indices=selected_ids).cpu().numpy()

    print(f"Embeddings shape: {embeddings.shape}")

    # Apply UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(selected_entities) - 1))
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create plot
    print("Creating plot...")
    plt.figure(figsize=(12, 10))

    # Plot Mouse phenotypes
    mp_count = len(mp_entities)
    hp_count = len(hp_entities)
    gene_count = len(gene_entities)
    disease_count = len(disease_entities)
    other_count = len(other_entities)

    offset = 0
    tiny_size = 0.05
    small_size = 0.4
    large_size = 0.8
    plt.scatter(embeddings_2d[offset:offset+mp_count, 0], embeddings_2d[offset:offset+mp_count, 1],
                c='blue', label='Mouse Phenotypes (MP_)', alpha=small_size, s=10)

    # Plot Human phenotypes
    offset += mp_count
    plt.scatter(embeddings_2d[offset:offset+hp_count, 0], embeddings_2d[offset:offset+hp_count, 1],
                c='orange', label='Human Phenotypes (HP_)', alpha=small_size, s=10)

    # Plot Genes
    if graph2 and gene_count > 0:
        offset += hp_count
        plt.scatter(embeddings_2d[offset:offset+gene_count, 0], embeddings_2d[offset:offset+gene_count, 1],
                    c='green', label='Genes (MGI_)', alpha=small_size, s=10)

    # Plot Diseases
    if graph3 and disease_count > 0:
        offset += gene_count
        plt.scatter(embeddings_2d[offset:offset+disease_count, 0], embeddings_2d[offset:offset+disease_count, 1],
                    c='purple', label='Diseases (OMIM_)', alpha=small_size, s=10)

    # Plot Other entities
    if other_count > 0:
        offset += disease_count
        plt.scatter(embeddings_2d[offset:offset+other_count, 0], embeddings_2d[offset:offset+other_count, 1],
                    c='gray', label='Other entities', alpha=tiny_size, s=10)

    plt.title('UMAP Visualization of Phenotypes, Genes, and Diseases', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output = f"umap/transd_{graph}_{fold}"
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output}")

    # Also display some statistics
    print("\n=== Statistics ===")
    print(f"Human phenotypes plotted: {len(hp_entities)}")
    if graph2:
        print(f"Genes plotted: {len(gene_entities)}")
    if graph3:
        print(f"Diseases plotted: {len(disease_entities)}")
    print(f"Other entities plotted: {len(other_entities)}")
    print(f"Total entities plotted: {len(selected_entities)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
