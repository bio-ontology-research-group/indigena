import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_train_val_split(df, val_ratio=0.1, random_seed=0):
    """
    Split dataframe into train and validation sets by diseases (tail entities).
    All pairs with the same disease will be in the same split.

    This follows the same logic as generate_inductive_dataset.py where
    the split is done by partitioning diseases, not pairs.

    Args:
        df: DataFrame with 'Disease' and 'Gene' columns
        val_ratio: Ratio of validation samples (default 0.1 for 10%)
        random_seed: Random seed for reproducibility

    Returns:
        train_df, val_df: Training and validation DataFrames
    """
    import numpy as np
    np.random.seed(random_seed)

    # Group pairs by disease (tail entity)
    disease_to_genes = {}
    for _, row in df.iterrows():
        disease = row['Disease']
        gene = row['Gene']
        if disease not in disease_to_genes:
            disease_to_genes[disease] = []
        disease_to_genes[disease].append(gene)

    # Get list of unique diseases and shuffle them
    diseases = list(disease_to_genes.keys())
    np.random.shuffle(diseases)

    # Calculate split point based on number of diseases
    n_val_diseases = int(len(diseases) * val_ratio)
    n_train_diseases = len(diseases) - n_val_diseases

    # Split diseases
    train_diseases = diseases[:n_train_diseases]
    val_diseases = diseases[n_train_diseases:]

    # Create train and validation dataframes
    train_pairs = []
    for disease in train_diseases:
        for gene in disease_to_genes[disease]:
            train_pairs.append({'Gene': gene, 'Disease': disease})

    val_pairs = []
    for disease in val_diseases:
        for gene in disease_to_genes[disease]:
            val_pairs.append({'Gene': gene, 'Disease': disease})

    train_df = pd.DataFrame(train_pairs)
    val_df = pd.DataFrame(val_pairs)

    logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    logger.info(f"Train diseases: {len(set(train_df['Disease']))}, Val diseases: {len(set(val_df['Disease']))}")
    logger.info(f"Train genes: {len(set(train_df['Gene']))}, Val genes: {len(set(val_df['Gene']))}")

    # Verify no disease overlap between train and validation
    assert len(set(train_df['Disease']) & set(val_df['Disease'])) == 0, "Disease overlap between train and validation"

    return train_df, val_df
