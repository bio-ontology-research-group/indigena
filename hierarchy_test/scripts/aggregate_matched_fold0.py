#!/usr/bin/env python
"""Matched INDIGENA-vs-Resnik degradation table (fold 0), leaf-only, over HP-ancestor
abstraction k=0..3, in both scoring modes:

  leaked : full-profile scoring / no leak filter -> k=0 is the CONTROL baseline
  strict : leak-free scoring                     -> supplementary "floor"

Both methods score the SAME candidate pool (data/matched_fold0/candidates_human_fold0.txt)
and SAME test pairs (data/matched_fold0/pairs_human_fold0.csv). Run from hierarchy_test/.
"""
import os, sys
sys.path.insert(0, "scripts")
from evaluate_sem_sim import compute_metrics

FID = "transd_human_inductive_fold_0_seed_0_dim_400_bs_8192_lr_0.001_graph4_leak-strict_leafonly"

def ind_path(tag, k):
    return f"data/results/kge_results_{FID}_{tag}_k{k}_inductive_bma.tsv"

def rsk_path(tag, k):
    return f"data/baseline_results/resnik_resnik_bma_fold0_matched_{tag}_k{k}_results.txt"

def metrics(path):
    if not os.path.exists(path):
        return None
    _, mac = compute_metrics(path)
    return mac

def row(name, k, m):
    if m is None:
        return f"{name:20s}{k:>3d}   MISSING"
    return (f"{name:20s}{k:>3d}{m['mrr']:>8.3f}{m['hits@1']:>7.3f}"
            f"{m['hits@10']:>7.3f}{m['hits@100']:>8.3f}{m['auc']:>7.3f}{m['mr']:>9.1f}")

hdr = f"{'method':20s}{'k':>3s}{'MRR':>8s}{'H@1':>7s}{'H@10':>7s}{'H@100':>8s}{'AUC':>7s}{'MR':>9s}"
for scoring in ("leaked", "strict"):
    title = ("LEAKED scoring (k=0 = control baseline)" if scoring == "leaked"
             else "LEAK-STRICT scoring (leak-free; floor)")
    print(f"\n=== {title} — leaf-only, matched pool=4491 pairs=330 ===")
    print(hdr)
    for name, fn in [("INDIGENA", ind_path), ("Resnik", rsk_path)]:
        for k in (0, 1, 2, 3):
            print(row(f"{name}({scoring})", k, metrics(fn(scoring, k))))
