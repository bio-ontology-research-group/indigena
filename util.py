import pandas as pd
import os

def load_gene_phenotypes(filename, preexisting_phenotypes=None):
    if preexisting_phenotypes is None:
        preexisting_phenotypes = set()
        
    mgi_gene_pheno = pd.read_csv(filename, sep='\t', header=None)
    mgi_gene_pheno.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "MGI Genotype ID"]

    gene_phenotypes = []
    for index, row in mgi_gene_pheno.iterrows():
        genes = row["MGI ID"]
        phenotype = row["MP Phenotype"]
        assert phenotype.startswith("MP:")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        if not phenotype in preexisting_phenotypes:
            continue

        for gene in genes.split('|'):
            gene = "http://mowl.borg/" + str(gene).replace(":", "_")
            gene_phenotypes.append((gene, phenotype))

    gene_phenotypes = list(set(gene_phenotypes))  # Remove duplicates
    return gene_phenotypes

def load_disease_phenotypes(filename, preexisting_phenotypes=None):
    hpoa = pd.read_csv(filename, sep='\t', comment='#', low_memory=False)

    disease_phenotypes = []
    for index, row in hpoa.iterrows():
        disease = row["database_id"]
        phenotype = row["hpo_id"]
        if not disease.startswith("OMIM:"):
            continue
        assert phenotype.startswith("HP:")
        disease = "http://mowl.borg/" + disease.replace(":", "_")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        if not phenotype in preexisting_phenotypes:
            continue
        disease_phenotypes.append((disease, phenotype))

    disease_phenotypes = list(set(disease_phenotypes))  # Remove duplicates
    return disease_phenotypes
