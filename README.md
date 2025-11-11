# INDIGENA: 


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fig/graphs_dark_bg.png">
  <source media="(prefers-color-scheme: light)" srcset="fig/graphs_white_bg.png">
  <img alt="Graph Structures">
</picture>

# Dependencies

- Groovy 4.0.26
- Java 8+

## Python KGE Methods
- Python 3.10
- mowl
- pykeen
- torch
- wandb
- pandas
- tqdm
- click


# Installation
```
git clone https://github.com/bio-ontology-research-group/indigena.git
cd indigena/
conda env create -f environment.yml
conda activate indigena
```

# Usage

## 1. Uncompress UPheno file

First, extract the data archive:

```bash
cd data
gunzip upheno.owl.gz
```

## 2. Semantic Similarity Baselines (Groovy)

These baseline methods compute ontology-based semantic similarity between genes and diseases using the SLIB library.

### Run with default parameters:
```bash
groovy semantic_similarity.groovy -r data -fold 0
```

### Run with custom parameters:
```bash
groovy semantic_similarity.groovy -r data -ic resnik -pw resnik -gw bma -fold 0
```

### Run SimGIC variant:
```bash
groovy semantic_similarity_simgic.groovy -r data -ic resnik -fold 0
```
### Parameters:
- `-r, --root_dir`: Data directory (default: `data`)
- `-ic, --ic_measure`: Information content measure (`resnik`, `sanchez`)
- `-pw, --pairwise_measure`: Pairwise measure (`resnik`, `lin`)
- `-gw, --groupwise_measure`: Groupwise measure (`bma`, `bmm`)
- `-fold`: Cross-validation fold number (default: 0)

**Output:** Results saved to `data/baseline_results/`


### Evaluate results:
```bash
python evaluate_sem_sim.py data/baseline_results/<results_file>
```

## 3. Knowledge Graph Embeddings (Python)

This approach uses mOWL to project ontology into triples and PyKEEN to
train KGE models.  We use W&B to track experiments. Therefore, before
running `kge.py`, change the entity name in `wandb.init` to you W&B
username.

### Run basic KGE model:
```bash
python kge_transe.py --fold 0 --model_name transd --mode inductive --graph2 --no_sweep
```

### Run with hyperparameters:
```bash
python kge_transd.py --fold 0 --model_name transd --mode inductive \
  --embedding_dim 100 --batch_size 128 --learning_rate 0.001 \
  --graph2 
```

### Parameters:

You can look at the hyperparameters in each script. They usually look like this:

- `--fold`: Cross-validation fold number
- `--model_name`: KGE model (`transe`, `transd`, `convkb`)
- `--mode`: Evaluation mode (`inductive`, `transductive`)
- `--graph2`: Add gene-phenotype edges
- `--graph3`: Add disease-phenotype edges
- `--graph4`: Add gene-disease training edges
- `--embedding_dim`: Embedding dimensions
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--only_test`: Only test existing model (skip training)
- `--description`: Weights & Biases run description
- `--no_sweep`: Disable W&B sweep mode
- `--pretrained_model`: Features to initialize ConvKB embeddings (`transe`, `transd`). 

## Problems running the models?

Please create a Github issue or PR!
