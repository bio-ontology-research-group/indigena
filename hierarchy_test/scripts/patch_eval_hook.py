"""Add eval-time disease-phenotype override (perturbation) to kge_transd_species.py.
Inserts, right before the final evaluate_model call: EVAL_TAG suffix on output_prefix
and a disease2pheno override from EVAL_DISEASE_PHENO_CSV, filtered to in-graph
known_entities. Only affects the final eval (perturbation), not training."""
import os
p = "scripts/kge_transd_species.py"
s = open(p).read()
anchor = '    output_prefix = f"data/results/kge_results_{file_identifier}"'
assert s.count(anchor) == 1, "anchor not found/unique"
block = anchor + '''
    if os.environ.get("EVAL_TAG"):
        output_prefix += "_" + os.environ["EVAL_TAG"]
    _pert = os.environ.get("EVAL_DISEASE_PHENO_CSV", "")
    if _pert:
        _pdf = pd.read_csv(_pert)
        _ov = {}; _skip = 0; _tot = 0
        for _r in _pdf.itertuples(index=False):
            _tot += 1
            if _r.Phenotype not in known_entities:
                _skip += 1; continue
            _ov.setdefault(_r.Disease, []).append(_r.Phenotype)
        disease2pheno = _ov
        print(f"[PERTURB] disease2pheno <- {_pert}: {len(_ov)} diseases, {_skip}/{_tot} terms OOV (not in graph)")'''
s = s.replace(anchor, block)
assert s.count("[PERTURB]") == 1
open(p, "w").write(s)
print("patched kge_transd_species.py with EVAL_DISEASE_PHENO_CSV/EVAL_TAG hook")
