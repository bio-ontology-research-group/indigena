import sys
from scipy.stats import rankdata
import numpy as np

import torch as th

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

def print_as_tex(metrics, title):
    header = "MR & MRR & Hits@1 & Hits@3 & Hits@10 & Hits@100 & AUC"
    print(header)
    metrics = [metrics['mr'], metrics['mrr'], metrics['hits@1'], metrics['hits@3'], metrics['hits@10'], metrics['hits@100'], metrics['auc']]
    
    print(title)
    for m in metrics:
        print(f"{m:.3f} & ", end="")

    print("\n")

                
        
                                        
    
def compute_metrics(filename, verbose=False, output_ranks=False):
    with open(filename, "r") as f:
        results = f.readlines()
        results = [x.strip().split("\t") for x in results]

        
        
    if verbose:
        print(f"Number of results: {len(results)}")

    if output_ranks:
        ranks_output_file = filename.split(".")[0] + "_ranks.txt"
        ranks_f = open(ranks_output_file, "w")
        
    mr = 0
    mrr = 0
    hits_k = {1: 0, 3: 0, 10: 0, 100: 0}
    ranks = dict()

    micro_mr = dict()
    micro_mrr = dict()
    micro_hits_k = dict()
    micro_ranks = dict()
    num_results_per_disease = dict()

    # results = results[:2]

    genes_ids = th.arange(len(results[0][3:]))
    if verbose:
        print(f"Number of evaluated genes: {len(genes_ids)}")

    for i in range(len(results)):
        disease = results[i][1]
        position = int(results[i][2])
        scores = results[i][3:]
        scores = [-float(x) for x in scores]

        perm = th.randperm(len(scores))
        scores = th.tensor(scores)

        updated_position = th.where(genes_ids[perm] == position)[0].item()
        scores = scores[perm]

        order = th.argsort(scores, descending=False)
        rank = th.where(order == updated_position)[0].item() + 1
        if output_ranks:
            ranks_f.write(f"{disease}\t{rank}\n")
        
        # ordering = rankdata(scores, method='average')
        # rank = ordering[int(position)]

        mr += rank
        mrr += 1 / rank

        for k in hits_k:
            if rank <= int(k):
                hits_k[k] += 1

        if rank not in ranks:
            ranks[rank] = 0
        ranks[rank] += 1

        if disease not in micro_mr:
            micro_mr[disease] = 0
            micro_mrr[disease] = 0
            micro_hits_k[disease] = {1: 0, 3: 0, 10: 0, 100: 0}
            micro_ranks[disease] = dict()
            num_results_per_disease[disease] = 0

        num_results_per_disease[disease] += 1

        micro_mr[disease] += rank
        micro_mrr[disease] += 1 / rank
        for k in micro_hits_k[disease]:
            if rank <= int(k):
                micro_hits_k[disease][k] += 1

        if rank not in micro_ranks[disease]:
            micro_ranks[disease][rank] = 0
        micro_ranks[disease][rank] += 1



    macro_metrics = dict()



    mr = mr / len(results)
    mrr = mrr / len(results)
    auc = compute_rank_roc(ranks, len(scores))
    for k in hits_k:
        hits_k[k] = hits_k[k] / len(results)
        macro_metrics[f"hits@{k}"] = hits_k[k]

    macro_metrics["mr"] = mr
    macro_metrics["mrr"] = mrr
    macro_metrics["auc"] = auc

    # metrics = {f"test_macro_{k}": v for k, v in macro_metrics.items()}
    # print(metrics)


    micro_mr = {k: v / num_results_per_disease[k] for k, v in micro_mr.items()}
    micro_mrr = {k: v / num_results_per_disease[k] for k, v in micro_mrr.items()}
    micro_auc = {k: compute_rank_roc(micro_ranks[k], len(scores)) for k in micro_ranks}
    for k in micro_hits_k:
        micro_hits_k[k] = {k2: v / num_results_per_disease[k] for k2, v in micro_hits_k[k].items()}

    mean_micro_mr = np.mean(list(micro_mr.values()))
    mean_micro_mrr = np.mean(list(micro_mrr.values()))
    mean_micro_auc = np.mean(list(micro_auc.values()))
    mean_micro_hits_k = dict()
    for k in hits_k:
        mean_micro_hits_k[k] = np.mean([x[k] for x in micro_hits_k.values()])

    micro_metrics = dict()
    micro_metrics["mr"] = mean_micro_mr
    micro_metrics["mrr"] = mean_micro_mrr
    micro_metrics["auc"] = mean_micro_auc
    for k in mean_micro_hits_k:
        micro_metrics[f"hits@{k}"] = mean_micro_hits_k[k]

    if verbose:
        print(f"Number of genes: {len(scores)}")
    return micro_metrics, macro_metrics
        
    # metrics = {f"test_micro_{k}": v for k, v in micro_metrics.items()}
    # print(metrics)


if __name__ == "__main__":
    filename = sys.argv[1]
    micro_metrics, macro_metrics = compute_metrics(filename)
    print_as_tex(micro_metrics, macro_metrics)



