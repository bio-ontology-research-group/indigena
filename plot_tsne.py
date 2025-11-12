import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from pykeen.models import TransD
import torch as th
import click as ck
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

@ck.command()
@ck.option("--model_path", type=str, required=True, help="Path to the trained model file")
@ck.option("--edges_file", type=str, default="data/upheno_owl2vecstar_edges.tsv", help="Path to edges file")
@ck.option("--output", type=str, default="tsne_plot.png", help="Output file for the plot")
@ck.option("--embedding_dim", type=int, default=100, help="Embedding dimension")
@ck.option("--random_seed", type=int, default=0, help="Random seed for t-SNE")
@ck.option("--perplexity", type=int, default=30, help="Perplexity parameter for t-SNE")
@ck.option("--max_samples", type=int, default=1000, help="Maximum number of samples per phenotype type")
def main(model_path, edges_file, output, embedding_dim, random_seed, perplexity, max_samples):
    """
    Generate a t-SNE plot of Mouse (MP_) and Human (HP_) phenotype embeddings.
    """

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

    entities = sorted(list(entities))
    relations = sorted(list(relations))
    triples = sorted(triples)

    # Create triples factory
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)

    # Initialize model
    print("Initializing TransD model...")
    model = TransD(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        relation_dim=embedding_dim,
        random_seed=random_seed
    )

    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(th.load(model_path, weights_only=True))
    model.eval()

    # Extract entity embeddings
    print("Extracting entity embeddings...")
    entity_to_id = triples_factory.entity_to_id

    # Filter entities for Mouse and Human phenotypes
    mp_entities = []
    hp_entities = []

    for entity in entities:
        if "MP_" in entity:
            mp_entities.append(entity)
        elif "HP_" in entity:
            hp_entities.append(entity)

    print(f"Found {len(mp_entities)} Mouse phenotypes (MP_)")
    print(f"Found {len(hp_entities)} Human phenotypes (HP_)")

    # Sample if needed
    if len(mp_entities) > max_samples:
        np.random.seed(random_seed)
        mp_entities = list(np.random.choice(mp_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Mouse phenotypes")

    if len(hp_entities) > max_samples:
        np.random.seed(random_seed + 1)
        hp_entities = list(np.random.choice(hp_entities, max_samples, replace=False))
        print(f"Sampled {max_samples} Human phenotypes")

    # Get embeddings
    selected_entities = mp_entities + hp_entities
    selected_ids = th.tensor([entity_to_id[entity] for entity in selected_entities])

    with th.no_grad():
        embeddings = model.entity_representations[0](indices=selected_ids).cpu().numpy()

    print(f"Embeddings shape: {embeddings.shape}")

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=min(perplexity, len(selected_entities) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    print("Creating plot...")
    plt.figure(figsize=(12, 10))

    # Plot Mouse phenotypes
    mp_count = len(mp_entities)
    plt.scatter(embeddings_2d[:mp_count, 0], embeddings_2d[:mp_count, 1],
                c='blue', label='Mouse Phenotypes (MP_)', alpha=0.6, s=20)

    # Plot Human phenotypes
    plt.scatter(embeddings_2d[mp_count:, 0], embeddings_2d[mp_count:, 1],
                c='red', label='Human Phenotypes (HP_)', alpha=0.6, s=20)

    plt.title('t-SNE Visualization of Mouse and Human Phenotype Embeddings', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output}")

    # Also display some statistics
    print("\n=== Statistics ===")
    print(f"Mouse phenotypes plotted: {len(mp_entities)}")
    print(f"Human phenotypes plotted: {len(hp_entities)}")
    print(f"Total entities plotted: {len(selected_entities)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
