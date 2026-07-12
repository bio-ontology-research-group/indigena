#!/usr/bin/env python
"""Cross-fold matched INDIGENA-vs-Resnik degradation (mean +/- sd over available folds),
leaf-only, HP-ancestor abstraction k=0..3, both scoring modes (leaked control / leak-strict floor).

Reads whatever folds are present under data/results and data/baseline_results, so it can be
run mid-sweep. Run from hierarchy_test/:  python scripts/aggregate_sweep.py [--folds 0-9]
"""
import os, sys, argparse, statistics
sys.path.insert(0, "scripts")
from evaluate_sem_sim import compute_metrics

FID = "transd_human_inductive_fold_{f}_seed_0_dim_400_bs_8192_lr_0.001_graph4_leak-strict_leafonly"

def ind_path(f, tag, k):
    return f"data/results/kge_results_{FID.format(f=f)}_{tag}_k{k}_inductive_bma.tsv"

def rsk_path(f, tag, k):
    return f"data/baseline_results/resnik_resnik_bma_fold{f}_matched_{tag}_k{k}_results.txt"

def metric_over_folds(path_fn, folds, tag, k, key):
    vals = []
    for f in folds:
        p = path_fn(f, tag, k)
        if os.path.exists(p):
            try:
                _, m = compute_metrics(p)
                vals.append(m[key])
            except Exception:
                pass
    return vals

def fmt(vals):
    if not vals:
        return "   --   "
    if len(vals) == 1:
        return f"{vals[0]:.3f}    "
    return f"{statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-9")
    args = ap.parse_args()
    a, b = (args.folds.split("-") + [args.folds])[:2]
    folds = list(range(int(a), int(b) + 1))

    for scoring in ("leaked", "strict"):
        title = ("LEAKED scoring (k=0 = control baseline)" if scoring == "leaked"
                 else "LEAK-STRICT scoring (leak-free floor)")
        print(f"\n=== {title} — leaf-only, matched pool/pairs, mean±sd over folds {folds[0]}..{folds[-1]} ===")
        print(f"{'method':10s}{'k':>3s}{'MRR':>14s}{'H@1':>14s}{'H@10':>14s}{'AUC':>14s}{'MR':>12s}{'n':>4s}")
        for name, fn in [("INDIGENA", ind_path), ("Resnik", rsk_path)]:
            for k in (0, 1, 2, 3):
                n = len(metric_over_folds(fn, folds, scoring, k, "mrr"))
                row = f"{name:10s}{k:>3d}"
                for key, w in [("mrr", 14), ("hits@1", 14), ("hits@10", 14), ("auc", 14), ("mr", 12)]:
                    row += f"{fmt(metric_over_folds(fn, folds, scoring, k, key)):>{w}s}"
                row += f"{n:>4d}"
                print(row)

if __name__ == "__main__":
    main()
