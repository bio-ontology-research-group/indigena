import mowl
mowl.init_jvm("10g")
import click as ck
import pandas as pd
import logging
from jpype import *
import jpype.imports
import os
import wget
import sys
import random

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import OWLClasses
from org.semanticweb.owlapi.model import IRI
from util import load_gene_phenotypes, load_disease_phenotypes
import java
from java.util import HashSet

adapter = OWLAPIAdapter()
manager = adapter.owl_manager
factory = adapter.data_factory

random.seed(42)

@ck.command()
@ck.option(
    '--save_dir', '-s', default='data', help='Directory to save the data')
def main(save_dir):

    out_dir = os.path.abspath(save_dir)
    logger.info(f'Saving data to {out_dir}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Cheking if the data is already downloaded")

    if not os.path.exists(os.path.join(out_dir, 'upheno.owl')):
        logger.error("File upheno.owl not found. Downloading it...")
        wget.download("https://purl.obolibrary.org/obo/upheno/v2/upheno.owl", out=out_dir)

    if not os.path.exists(os.path.join(out_dir, 'MGI_GenePheno.rpt')):
        logger.error("File MGI_GenePheno.rpt not found. Downloading it for Gene-Phenotype associations")
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt", out=out_dir)
        
    if not os.path.exists(os.path.join(out_dir, 'phenotype.hpoa')):
        logger.error("File phenotype.hpoa not found. Downloading it for Disease-Phenotype associations")
        wget.download("http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa", out=out_dir)
                
    if not os.path.exists(os.path.join(out_dir, 'MGI_Geno_DiseaseDO.rpt')):
        logger.error("File MGI_Geno_DiseaseDO.rpt not found. Downloading it for Gene-Disease associations")
        
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_Geno_DiseaseDO.rpt", out=out_dir)
        
    
    logger.info("Loading ontology")
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(os.path.join(out_dir, 'upheno.owl')))
    classes = set(OWLClasses(ont.getClassesInSignature()).as_str)

    existing_mp_phenotypes = set()
    existing_hp_phenotypes = set()
    for cls in classes:
        if "MP_" in cls:
            existing_mp_phenotypes.add(cls)
        elif "HP_" in cls:
            existing_hp_phenotypes.add(cls)
    logger.info(f"Existing MP phenotypes in ontology: {len(existing_mp_phenotypes)}")
    logger.info(f"Existing HP phenotypes in ontology: {len(existing_hp_phenotypes)}")
    
    has_symptom = "http://mowl.borg/has_symptom" # relation between diseases and phenotypes
    has_phenotype = "http://mowl.borg/has_phenotype" # relation between genes and phenotypes
    associated_with = "http://mowl.borg/associated_with" # relation between genes and diseases


    logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")
    
    gene_phenotypes = load_gene_phenotypes(os.path.join(out_dir, 'MGI_GenePheno.rpt'), preexisting_phenotypes=existing_mp_phenotypes)
    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"\tE.g. {gene_phenotypes[0]}")

    with open(os.path.join(out_dir, 'gene_phenotypes.csv'), 'w') as f:
        f.write("Gene,Phenotype\n")
        for gene, phenotype in gene_phenotypes:
            f.write(f"{gene},{phenotype}\n")

    logger.info("Obtaining Disease-Phenotype associations from phenotype.hpoa")
    disease_phenotypes = load_disease_phenotypes(os.path.join(out_dir, 'phenotype.hpoa'), preexisting_phenotypes=existing_hp_phenotypes)
    logger.info(f"Disease-Phenotype associations: {len(disease_phenotypes)}")
    logger.info(f"\tE.g. {disease_phenotypes[0]}")

    with open(os.path.join(out_dir, 'disease_phenotypes.csv'), 'w') as f:
        f.write("Disease,Phenotype\n")
        for disease, phenotype in disease_phenotypes:
            f.write(f"{disease},{phenotype}\n")
    
    logger.info("Obtaining Gene-Disease associations from MGI_Geno_DiseaseDO.rpt.rpt. Genes are represented as MGI IDs and Diseases are represented as OMIM IDs")
    mgi_geno_diseasedo = pd.read_csv(os.path.join(out_dir, 'MGI_Geno_DiseaseDO.rpt'), sep='\t')
    mgi_geno_diseasedo.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "DO ID", "MIM ID"]
    gene_disease = []
    for index, row in mgi_geno_diseasedo.iterrows():
        gene = row["MGI ID"]
        diseases = row["MIM ID"]
        if pd.isna(gene):
            logger.warning(f"Gene not found for {row}")
            continue
        if pd.isna(diseases):
            logger.warning(f"Disease not found for {row}")
            continue
        assert diseases.startswith("OMIM:")
        
        diseases = diseases.split("|")
        gene = "http://mowl.borg/" + str(gene).replace(":", "_")
        for disease in diseases:
            assert disease.startswith("OMIM:")
            disease = "http://mowl.borg/" + disease.replace(":", "_")
            gene_disease.append((gene, disease))

    gene_disease = list(set(gene_disease))  # Remove duplicates
            
    logger.info(f"Gene-Disease associations: {len(gene_disease)}")
    logger.info(f"\tE.g.: {gene_disease[0]}")

    logger.info("Filtering out x--phenotype pairs with phenotypes not in the ontology")

    gene_phenotypes = [(gene, phenotype) for gene, phenotype in gene_phenotypes if phenotype in classes]
    disease_phenotypes = [(disease, phenotype) for disease, phenotype in disease_phenotypes if phenotype in classes]
    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"Disease-Phenotype associations: {len(disease_phenotypes)}")
    
    gene_set = set([gene for gene, _ in gene_phenotypes])
    disease_set = set([disease for disease, _ in disease_phenotypes])

    logger.info(f"Updating gene-disease associations")
    gene_disease = [(gene, disease) for gene, disease in gene_disease if gene in gene_set and disease in disease_set]
    logger.info(f"Gene-Disease associations: {len(gene_disease)}")

    logger.info(f"Saving gene-disease associations to {os.path.join(out_dir, 'gene_diseases.csv')} before splitting into folds")
    with open(os.path.join(out_dir, 'gene_diseases.csv'), 'w') as f:
        f.write("Gene,Disease\n")
        for gene, disease in gene_disease:
            f.write(f"{gene},{disease}\n")

    
    logger.info("Splitting the data into training, validation and test sets")

    k = 10
    folds = split_pairs_kfold(gene_disease, k=k, split_by="tail")

    folds_dir= os.path.join(out_dir, "gene_disease_folds")
    for i in range(k):
        fold_dir = os.path.join(folds_dir, f"fold_{i}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        
        train_fold, test_fold = get_fold_splits(folds, i)
        with open(os.path.join(fold_dir, 'train.csv'), 'w') as f:
            f.write("Gene,Disease\n")
            for gene, disease in train_fold:
                f.write(f"{gene},{disease}\n")
        
        with open(os.path.join(fold_dir, 'test.csv'), 'w') as f:
            f.write("Gene,Disease\n")
            for gene, disease in test_fold:
                f.write(f"{gene},{disease}\n")
    
    logger.info("Done")
        
def split_pairs_kfold(pairs, k=10, split_by="tail"):
    logger.info(f"Creating {k}-fold split for {len(pairs)} pairs")
    
    if split_by == "pair":
        raise NotImplementedError
    if split_by == "head":
        raise NotImplementedError
    if split_by == "tail":
        tail_to_heads = dict()
        for head, tail in pairs:
            if tail not in tail_to_heads:
                tail_to_heads[tail] = []
            tail_to_heads[tail].append(head)
        
        tails = list(tail_to_heads.keys())
        random.shuffle(tails)
        
        # Calculate fold sizes
        fold_size = len(tails) // k
        remainder = len(tails) % k
        
        folds = []
        start_idx = 0
        
        for fold_idx in range(k):
            # Some folds get one extra tail if there's a remainder
            current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
            fold_tails = tails[start_idx:start_idx + current_fold_size]
            
            # Convert tails back to pairs
            fold_pairs = [(head, tail) for tail in fold_tails for head in tail_to_heads[tail]]
            folds.append(fold_pairs)
            
            start_idx += current_fold_size
        
        # Verify all pairs are included exactly once across all folds
        total_pairs = sum(len(fold) for fold in folds)
        assert total_pairs == len(pairs)
        
        # Verify no overlap between folds
        all_fold_pairs = set()
        for fold in folds:
            fold_set = set(fold)
            assert len(fold_set & all_fold_pairs) == 0, "Overlapping pairs between folds"
            all_fold_pairs.update(fold_set)
        
        logger.info(f"Created {k} folds with sizes: {[len(fold) for fold in folds]}")
        
        return folds

# Usage function to get train/test splits for each fold
def get_fold_splits(folds, test_fold_idx):
    """Get training and testing sets for a specific fold"""
    test = folds[test_fold_idx]
    train = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            train.extend(fold)
    
    return train, test


if __name__ == '__main__':
    main()
    shutdownJVM()
