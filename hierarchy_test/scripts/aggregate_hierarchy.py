#!/usr/bin/env python
"""INDIGENA vs Resnik-BMA degradation over HP-abstraction depth k=0..3 (fold 0).
Run from hierarchy_test/:  python scripts/aggregate_hierarchy.py"""
import os, sys
from evaluate_sem_sim import compute_metrics
RES, BR = "data/results", "data/baseline_results"
FID = "transd_human_inductive_fold_0_seed_0_dim_400_bs_8192_lr_0.001_graph4_leak-strict"
def ind(k): return f"{RES}/kge_results_{FID}{'' if k==0 else f'_hp_k{k}'}_inductive_bma.tsv"
def rsk(k): return f"{BR}/resnik_resnik_bma_fold0_hp_k{k}_results.txt"
def m(path):
    if not os.path.exists(path): return None
    _, mac = compute_metrics(path); return mac
print(f"{'method':9s}{'k':>3s}{'MRR':>8s}{'H@1':>7s}{'H@10':>7s}{'H@100':>8s}{'AUC':>7s}{'MR':>9s}")
for name, fn in [("INDIGENA", ind), ("Resnik", rsk)]:
    for k in (0, 1, 2, 3):
        mac = m(fn(k))
        if mac is None:
            print(f"{name:9s}{k:>3d}   MISSING  {fn(k)}"); continue
        print(f"{name:9s}{k:>3d}{mac['mrr']:>8.3f}{mac['hits@1']:>7.3f}{mac['hits@10']:>7.3f}{mac['hits@100']:>8.3f}{mac['auc']:>7.3f}{mac['mr']:>9.1f}")
