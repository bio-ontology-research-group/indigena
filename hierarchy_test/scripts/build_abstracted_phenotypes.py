#!/usr/bin/env python
"""Build abstracted disease->phenotype query sets at depths k=0..K.
k=0 is the original (unperturbed). For k>=1, each phenotype is replaced by its
depth-k HP ancestors (replace-all union) from data/hpo_ancestors.json; if a term
has no depth-k ancestor (reached root), it is kept as-is. Same Disease,Phenotype
schema as disease_phenotypes.csv so both INDIGENA (EVAL_DISEASE_PHENO_CSV) and
the Groovy Resnik (PERTURB_DISEASE_CSV) consume identical inputs.

  python scripts/build_abstracted_phenotypes.py --max-depth 3
    -> data/disease_phenotypes_hp_k{0,1,2,3}.csv
"""
import json, csv, argparse, statistics, collections
ap = argparse.ArgumentParser()
ap.add_argument("--max-depth", type=int, default=3)
ap.add_argument("--disease-phenotypes", default="data/disease_phenotypes.csv")
ap.add_argument("--ancestors", default="data/hpo_ancestors.json")
a = ap.parse_args()

anc = json.load(open(a.ancestors))["ancestors_by_depth"]
rows = list(csv.DictReader(open(a.disease_phenotypes)))
base = collections.defaultdict(set)
for r in rows:
    base[r["Disease"]].add(r["Phenotype"])

for k in range(0, a.max_depth + 1):
    out = []
    pert = collections.defaultdict(set)
    kept = 0
    for r in rows:
        d, p = r["Disease"], r["Phenotype"]
        if k == 0:
            out.append((d, p)); pert[d].add(p); continue
        levels = anc.get(p, [])
        terms = levels[k - 1] if len(levels) >= k else []
        if terms:
            for t in terms:
                out.append((d, t)); pert[d].add(t)
        else:
            out.append((d, p)); pert[d].add(p); kept += 1
    out = sorted(set(out))
    jac = [len(base[d] & pert[d]) / len(base[d] | pert[d]) for d in base if (base[d] | pert[d])]
    fn = f"data/disease_phenotypes_hp_k{k}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Disease", "Phenotype"]); w.writerows(out)
    print(f"k={k}: rows={len(out)} kept_at_root={kept} mean_jaccard_vs_base={statistics.mean(jac):.3f} -> {fn}")
