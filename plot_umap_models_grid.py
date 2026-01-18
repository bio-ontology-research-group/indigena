import mowl
mowl.init_jvm("10g")

from mowl.projection import Edge
from mowl.utils.random import seed_everything
from pykeen.models import TransE, TransH, TransD
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import umap
import numpy as np
from data import create_train_val_split
import click as ck


@ck.command()
@ck.option("--edges_file", type=str, default="data/upheno_owl2vecstar_edges.tsv", help="Path to edges file")
@ck.option("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
@ck.option("--max_samples", type=int, default=5000, help="Maximum number of samples per phenotype type")
def main(edges_file, n_neighbors, max_samples):
    """
    Generate a 10x3 grid of UMAP plots: 10 folds × 3 models (TransE, TransH, TransD).
    Shows MP Phenotypes, HP Phenotypes, Genes, and Diseases.
    """
    random_seed = 0
    seed_everything(random_seed)

    models = ["transe", "transh", "transd"]
    model_classes = {"transe": TransE, "transh": TransH, "transd": TransD}
    model_display_names = {"transe": "TransE", "transh": "TransH", "transd": "TransD"}
    folds = [str(i) for i in range(10)]

    transparent = 0.5

    # Colorblind-friendly palette (Wong palette, softer tones)
    colors = {
        'mp': '#56B4E9',      # Sky blue
        'hp': '#E69F00',      # Orange
        'genes': '#009E73',   # Bluish green
        'diseases': '#CC79A7', # Reddish purple
        'other': '#999999'    # Gray
    }

    # Create figure with subplots (10 folds x 3 models)
    fig, axes = plt.subplots(10, 3, figsize=(15, 50))

    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Processing fold {fold}...")
        print(f"{'='*60}")

        # Load data for this fold
        train_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/train.csv")
        train_disease_genes, val_disease_genes = create_train_val_split(train_disease_genes, val_ratio=0.1, random_seed=0)

        test_disease_genes = pd.read_csv(f"data/gene_disease_folds/fold_{fold}/test.csv")
        test_diseases = set(test_disease_genes['Disease'].values)

        gene_phenotypes = pd.read_csv("data/gene_phenotypes.csv")
        disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")

        test_disease_phenotypes = set()
        for _, row in disease_phenotypes.iterrows():
            if row['Disease'] in test_diseases:
                test_disease_phenotypes.add(row['Phenotype'])

        # Build graph4 triples and entities
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

        # Add gene-phenotype edges (graph2)
        for _, row in gene_phenotypes.iterrows():
            gene = row['Gene']
            phenotype = row['Phenotype']
            assert phenotype in entities, f"Phenotype {phenotype} not in entities"
            triples.append((gene, 'has_phenotype', phenotype))
            entities.add(gene)

        # Add disease-phenotype edges (graph3)
        for _, row in disease_phenotypes.iterrows():
            disease = row['Disease']
            phenotype = row['Phenotype']
            assert phenotype in entities, f"Phenotype {phenotype} not in entities"
            triples.append((disease, 'has_symptom', phenotype))
            entities.add(disease)

        # Add gene-disease edges (graph4)
        for _, row in train_disease_genes.iterrows():
            disease = row['Disease']
            gene = row['Gene']
            triples.append((gene, 'associated_with', disease))
            assert gene in entities, f"Gene {gene} not in entities"
            assert disease in entities, f"Disease {disease} not in entities"

        entities = sorted(list(entities))
        relations = sorted(list(relations))
        triples = sorted(triples)

        # Create triples factory
        mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
        triples_factory = Edge.as_pykeen(mowl_triples)
        entity_to_id = triples_factory.entity_to_id

        # Classify entities
        mp_entities = []
        hp_entities = []
        gene_entities = []
        disease_entities = []
        other_entities = []

        for entity in entities:
            if "MP_" in entity:
                mp_entities.append(entity)
            elif "HP_" in entity:
                if entity in test_disease_phenotypes:
                    hp_entities.append(entity)
                else:
                    other_entities.append(entity)
            elif "MGI_" in entity:
                gene_entities.append(entity)
            elif "OMIM_" in entity:
                disease_entities.append(entity)
            else:
                other_entities.append(entity)

        # Sample if needed
        if len(mp_entities) > max_samples:
            np.random.seed(random_seed)
            mp_entities = list(np.random.choice(mp_entities, max_samples, replace=False))

        if len(hp_entities) > max_samples:
            np.random.seed(random_seed + 1)
            hp_entities = list(np.random.choice(hp_entities, max_samples, replace=False))

        if len(gene_entities) > max_samples:
            np.random.seed(random_seed + 2)
            gene_entities = list(np.random.choice(gene_entities, max_samples, replace=False))

        if len(disease_entities) > max_samples:
            np.random.seed(random_seed + 3)
            disease_entities = list(np.random.choice(disease_entities, max_samples, replace=False))

        if len(other_entities) > max_samples:
            np.random.seed(random_seed + 4)
            other_entities = list(np.random.choice(other_entities, max_samples, replace=False))

        print(f"MP phenotypes: {len(mp_entities)}")
        print(f"HP phenotypes: {len(hp_entities)}")
        print(f"Genes: {len(gene_entities)}")
        print(f"Diseases: {len(disease_entities)}")
        print(f"Other entities: {len(other_entities)}")

        mp_count = len(mp_entities)
        hp_count = len(hp_entities)
        gene_count = len(gene_entities)
        disease_count = len(disease_entities)
        other_count = len(other_entities)

        all_entities = mp_entities + hp_entities + gene_entities + disease_entities + other_entities
        all_ids = th.tensor([entity_to_id[entity] for entity in all_entities])
        total_count = len(all_entities)

        for model_idx, model_name in enumerate(models):
            print(f"Processing {model_display_names[model_name]}...")

            ax = axes[fold_idx, model_idx]

            if model_name == "transe":
                dim = 100
                model_path = f"data/models/transe_transductive_fold_{fold}_seed_0_dim_100_bs_2048_lr_0.001_norm_2_graph4.pt"
            elif model_name == "transh":
                dim = 200
                model_path = f"data/models/transh_transductive_fold_{fold}_seed_0_dim_200_bs_1024_lr_0.001_graph4.pt"
            elif model_name == "transd":
                dim = 100
                model_path = f"data/models/transd_transductive_fold_{fold}_seed_0_dim_100_bs_2048_lr_0.001_graph4.pt"

            model_class = model_classes[model_name]
            if model_name == "transd":
                model = model_class(
                    triples_factory=triples_factory,
                    embedding_dim=dim,
                    relation_dim=dim,
                    random_seed=random_seed,
                )
            else:
                model = model_class(
                    triples_factory=triples_factory,
                    embedding_dim=dim,
                    random_seed=random_seed,
                )

            print(f"Loading model from {model_path}...")
            model.load_state_dict(th.load(model_path, weights_only=True, map_location=th.device('cpu')))
            model.eval()

            # Get embeddings
            with th.no_grad():
                embeddings = model.entity_representations[0](indices=all_ids).cpu().numpy()

            # Apply UMAP
            print(f"Applying UMAP...")
            reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, total_count - 1))
            embeddings_2d = reducer.fit_transform(embeddings)

            # Plot MP phenotypes
            offset = 0
            ax.scatter(embeddings_2d[offset:offset+mp_count, 0], embeddings_2d[offset:offset+mp_count, 1],
                       c=colors['mp'], alpha=transparent, s=3)

            # Plot HP phenotypes
            offset += mp_count
            ax.scatter(embeddings_2d[offset:offset+hp_count, 0], embeddings_2d[offset:offset+hp_count, 1],
                       c=colors['hp'], alpha=transparent, s=3)

            # Plot Genes
            if gene_count > 0:
                offset += hp_count
                ax.scatter(embeddings_2d[offset:offset+gene_count, 0], embeddings_2d[offset:offset+gene_count, 1],
                           c=colors['genes'], alpha=transparent, s=3)

            # Plot Diseases
            if disease_count > 0:
                offset += gene_count
                ax.scatter(embeddings_2d[offset:offset+disease_count, 0], embeddings_2d[offset:offset+disease_count, 1],
                           c=colors['diseases'], alpha=transparent, s=3)

            # Plot Other entities
            if other_count > 0:
                offset += disease_count
                ax.scatter(embeddings_2d[offset:offset+other_count, 0], embeddings_2d[offset:offset+other_count, 1],
                           c=colors['other'], alpha=0.1, s=2)

            ax.set_title(f'Fold {fold} - {model_display_names[model_name]}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])

    # Create legend
    legend_elements = [
        Patch(facecolor=colors['mp'], alpha=1, label='MP Phenotypes'),
        Patch(facecolor=colors['hp'], alpha=1, label='HP Phenotypes'),
        Patch(facecolor=colors['genes'], alpha=1, label='Genes'),
        Patch(facecolor=colors['diseases'], alpha=1, label='Diseases'),
        Patch(facecolor=colors['other'], alpha=1, label='Other Entities'),
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=5, fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

    output = "umap/graph4_all_folds_models.png"
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*60}")
    print(f"Plot saved to {output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
