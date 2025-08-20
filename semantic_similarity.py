import mowl
mowl.init_jvm("20g")
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import rankdata
import wandb
import click as ck
from math import floor
import numpy as np
from multiprocessing import get_context
from functools import partial

from dataset import GDADataset

from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import AxiomType

from slib.sml.sm.core.engine import SM_Engine
from slib.sml.sm.core.measures import Measure_Groupwise
from slib.sml.sm.core.metrics.ic.utils import IC_Conf_Topo
from slib.sml.sm.core.utils import SMConstants
from slib.utils.ex import SLIB_Exception
from slib.sml.sm.core.utils import SMconf
from slib.graph.model.impl.graph.memory import GraphMemory
from slib.graph.io.conf import GDataConf
from slib.graph.io.util import GFormat
from slib.graph.io.loader import GraphLoaderGeneric
from slib.graph.model.impl.repo import URIFactoryMemory
from java.util import HashSet
from java.io import Serializable
from java.io import ObjectInputStream
from java.io import ObjectOutputStream

from jpype.pickle import JPickler, JUnpickler
from jpype import JImplements


import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
                         
@ck.command()
@ck.option("--root_dir", default="data", help="Root directory for data")
@ck.option("--ic_measure", "-ic", default="resnik", help="Information content measure")
@ck.option("--pairwise_measure", "-pw", default="resnik", help="Pairwise measure")
@ck.option("--groupwise_measure", "-gw", default="bma", help="Groupwise measure")
def main(root_dir, ic_measure, pairwise_measure, groupwise_measure):
    wandb_logger = wandb.init(entity="ferzcam", project="indiga", group="semantic_similarity", name=f"{ic_measure}_{pairwise_measure}_{groupwise_measure}")

    ds = GDADataset()
    classes = set(ds.classes.as_str)
    eval_genes, _ = ds.evaluation_classes
    eval_genes = eval_genes.as_str
    eval_gene_to_id = {g: i for i, g in enumerate(eval_genes)}
    logger.info(f"Total evaluation genes: {len(eval_genes)}")

    gene2pheno = ds.gene_phenotypes
    disease2pheno = ds.disease_phenotypes
    
    existing_mp_phenotypes = set()
    existing_hp_phenotypes = set()
    for cls in classes:
        if "MP_" in cls:
            existing_mp_phenotypes.add(cls)
        elif "HP_" in cls:
            existing_hp_phenotypes.add(cls)
            
    test_pairs = ds.get_fold(0)[1]  # Get test pairs from the first fold
    test_genes, test_phenotypes = zip(*test_pairs)

    disease2pheno = ds.disease_phenotypes

    logger.info("Preparing Semantic Similarity Engine")
    factory = URIFactoryMemory.getSingleton()
    graph_uri = factory.getURI("http://purl.obolibrary.org/obo/GDA_")
    factory.loadNamespacePrefix("GDA", graph_uri.toString())
    graph = GraphMemory(graph_uri)

    goConf = GDataConf(GFormat.RDF_XML, os.path.join(root_dir, "upheno.owl"))
    GraphLoaderGeneric.populate(goConf, graph)

    engine = SM_Engine(graph)

    icConf = IC_Conf_Topo(ic_measure_resolver(ic_measure))
    sm_conf_pairwise = SMconf(pairwise_measure_resolver(pairwise_measure))
    sm_conf_pairwise.setICconf(icConf)
    sm_conf_groupwise = SMconf(groupwise_measure_resolver(groupwise_measure))
        

    mr = 0
    mrr = 0
    hits_k = {1: 0, 3: 0, 10: 0, 100: 0}
    ranks = dict()

    # test_pairs = test_pairs[:20]
    test_pairs = [(eval_gene_to_id[g], disease2pheno[d.split("/")[-1].replace("_", ":")]) for g, d in test_pairs]

    
        # indexed_preds = [(i, preds[i]) for i in range(len(preds))]
    
    # with get_context("spawn").Pool(30) as p:
        # results = []
        # with tqdm(total=len(preds)) as pbar:
            # for output in p.imap_unordered(partial(propagate_annots, go=go, terms_dict=terms_dict), indexed_preds, chunksize=200):
                # results.append(output)
                # pbar.update()
        
        # unordered_preds = [pred for pred in results]
        # ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
        # preds = [pred[1] for pred in ordered_preds]


    
    # serialized_gene2pheno = pickler.dumps(gene2pheno)
    with open("/tmp/engine.pkl", "wb") as f:
        serialized_engine = JPickler(f).dump(SerializableWrapper(engine))

    with get_context("spawn").Pool(10) as p:
        ranks_parallel = []
        with tqdm(total=len(test_pairs)) as pbar:
            for rank in p.imap_unordered(partial(score_genes_parallel_wrapper,
                                                 gene2pheno, serialized_engine, sm_conf_pairwise,
                                                 sm_conf_groupwise, factory), test_pairs, chunksize=1):
                ranks_parallel.append(rank)
                pbar.update()

            
    # for gene, disease in tqdm(test_pairs):
        # gene_id = eval_gene_to_id[gene]
        # disease = disease.split("/")[-1].replace("_", ":")
        # scores = score_genes(disease2pheno[disease], gene2pheno,
                             # engine, sm_conf_pairwise, sm_conf_groupwise, factory)
        # ordering = rankdata([-score for score in scores], method='average')
        # rank = ordering[gene_id]

    for rank in ranks_parallel:
        
        mr += rank
        mrr += 1 / rank

        for k in hits_k:
            if rank <= int(k):
                hits_k[k] += 1

        if rank not in ranks:
            ranks[rank] = 0
        ranks[rank] += 1
                                
    metrics = dict()
    
    mr = mr / len(test_pairs)
    mrr = mrr / len(test_pairs)
    auc = compute_rank_roc(ranks, len(eval_genes))
    for k in hits_k:
        hits_k[k] = hits_k[k] / len(test_pairs)
        metrics[f"hits@{k}"] = hits_k[k]

    metrics["mr"] = mr
    metrics["mrr"] = mrr
    metrics["auc"] = auc
                
                    
                                                        
    metrics = {f"test_{k}": v for k, v in metrics.items()}
    wandb_logger.log(metrics)


def score_genes(disease_phenotypes, gene2pheno, engine, sm_conf_pairwise, sm_conf_groupwise, factory):

    j_disease_phenotypes = HashSet()
    for d in disease_phenotypes:
        j_disease_phenotypes.add(factory.getURI(d))

    scores = []

    for gene, gene_phenotypes in gene2pheno.items():
        if len(gene_phenotypes) == 0:
            raise ValueError(f"Gene {gene} has no phenotypes")
        j_gene_phenotypes = HashSet()
        for g in gene_phenotypes:
            j_gene_phenotypes.add(factory.getURI(g))

        score = engine.compare(sm_conf_groupwise, sm_conf_pairwise, j_disease_phenotypes, j_gene_phenotypes)

        scores.append(score)

    return scores



def score_genes_parallel(test_pair, gene2pheno, engine, sm_conf_pairwise, sm_conf_groupwise, factory):

    gene_id, disease_phenotypes = test_pair
    
    j_disease_phenotypes = HashSet()
    for d in disease_phenotypes:
        j_disease_phenotypes.add(factory.getURI(d))

    scores = []

    for gene, gene_phenotypes in gene2pheno.items():
        if len(gene_phenotypes) == 0:
            raise ValueError(f"Gene {gene} has no phenotypes")
        j_gene_phenotypes = HashSet()
        for g in gene_phenotypes:
            j_gene_phenotypes.add(factory.getURI(g))

        score = engine.compare(sm_conf_groupwise, sm_conf_pairwise, j_disease_phenotypes, j_gene_phenotypes)

        scores.append(score)

    ordering = rankdata([-score for score in scores], method='average')
    rank = ordering[gene_id]

        
    return rank

def score_genes_parallel_wrapper(args):
    serialized_gene2pheno, serialized_engine, sm_conf_pairwise, sm_conf_groupwise, factory, test_pair = args
    
        
    # gene2pheno = unpickler.loads(serialized_gene2pheno)
    with open("/tmp/engine.pkl", "rb") as f:
        engine = JUnpickler(f).load().java_object
        
    
    # Call the actual function
    return score_genes_parallel(gene2pheno, engine, sm_conf_pairwise, sm_conf_groupwise, factory, test_pair)


def axioms_to_pairs(ontology):
    pairs = []
    for axiom in ontology.getAxioms():
        if axiom.getAxiomType() == AxiomType.SUBCLASS_OF:
            superclass = axiom.getSuperClass()
            if superclass.getClassExpressionType() == CT.OBJECT_SOME_VALUES_FROM:
                subclass = axiom.getSubClass()
                prop = superclass.getProperty()
                filler = superclass.getFiller()

                gene = str(subclass.toStringID())
                disease = str(filler.toStringID())

                pairs.append((gene, disease))

    logger.info(f"Total evaluation pairs: {len(pairs)}")
    return pairs

def compute_rank_roc(ranks, num_entities):
    n_tails = num_entities
                    
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_tails)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_tails
    return auc


def ic_measure_resolver(measure):
    if measure.lower() == "sanchez":
        return SMConstants.FLAG_ICI_SANCHEZ_2011
    else:
        raise ValueError(f"Invalid IC measure: {measure}")

def pairwise_measure_resolver(measure):
    if measure.lower() == "lin":
        return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_LIN_1998
    else:
        raise ValueError(f"Invalid pairwise measure: {measure}")

def groupwise_measure_resolver(measure):
    if measure.lower() == "bma":
        return SMConstants.FLAG_SIM_GROUPWISE_BMA
    else:
        raise ValueError(f"Invalid groupwise measure: {measure}")
        
    
if __name__ == "__main__":
    main()

    
