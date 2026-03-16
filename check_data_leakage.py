import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Print individual disease names in addition to summaries")
args = parser.parse_args()

FOLDS_DIR = Path("data/gene_disease_folds")
DISEASE_PHENOTYPES = Path("data/disease_phenotypes.csv")

# Load phenotypic profiles: disease -> frozenset of phenotypes
pheno_df = pd.read_csv(DISEASE_PHENOTYPES)
disease_to_phenotypes = (
    pheno_df.groupby("Disease")["Phenotype"]
    .apply(frozenset)
    .to_dict()
)

fold_dirs = sorted(FOLDS_DIR.iterdir())

# --- Global check: diseases with identical phenotype profile AND identical gene set ---
all_folds_df = pd.concat(
    [pd.read_csv(f) for fold_dir in fold_dirs if fold_dir.is_dir() for f in [fold_dir / "train.csv", fold_dir / "test.csv"]],
    ignore_index=True,
).drop_duplicates()

disease_to_genes = (
    all_folds_df.groupby("Disease")["Gene"]
    .apply(frozenset)
    .to_dict()
)

# Group diseases by (phenotype_profile, gene_set) tuple
profile_gene_to_diseases: dict[tuple, list[str]] = {}
for disease in set(disease_to_phenotypes) & set(disease_to_genes):
    key = (disease_to_phenotypes[disease], disease_to_genes[disease])
    profile_gene_to_diseases.setdefault(key, []).append(disease)

duplicate_groups = {k: v for k, v in profile_gene_to_diseases.items() if len(v) > 1}

print("=== Global check: diseases with identical phenotype profile AND gene associations ===")
if duplicate_groups:
    print(f"Found {len(duplicate_groups)} group(s) of diseases that are phenotypic+genetic duplicates:")
    if args.verbose:
        for (pheno_profile, gene_set), diseases in sorted(duplicate_groups.items(), key=lambda x: -len(x[1])):
            print(f"  Diseases: {sorted(diseases)}  (phenotypes: {len(pheno_profile)}, genes: {len(gene_set)})")
else:
    print("OK: no diseases share both an identical phenotype profile and identical gene associations.")
print()


disease_leakage_found = False
phenotype_leakage_found = False

for fold_dir in fold_dirs:
    if not fold_dir.is_dir():
        continue

    train_df = pd.read_csv(fold_dir / "train.csv")
    test_df = pd.read_csv(fold_dir / "test.csv")

    train_diseases = set(train_df["Disease"])
    test_diseases = set(test_df["Disease"])

    # Check 1: disease identity leakage
    overlap = test_diseases & train_diseases
    if overlap:
        disease_leakage_found = True
        print(f"[{fold_dir.name}] DISEASE LEAKAGE: {len(overlap)} test disease(s) found in train")
        if args.verbose:
            for d in sorted(overlap):
                print(f"  {d}")
    else:
        print(f"[{fold_dir.name}] Disease identity check: OK (no test disease in train)")

    # Check 2: phenotypic profile identity leakage
    # Build mapping of phenotype frozenset -> list of diseases for training diseases
    train_profile_to_diseases: dict[frozenset, list[str]] = {}
    for d in train_diseases:
        profile = disease_to_phenotypes.get(d)
        if profile is None:
            continue
        train_profile_to_diseases.setdefault(profile, []).append(d)

    profile_leaks = []
    for test_disease in test_diseases:
        test_profile = disease_to_phenotypes.get(test_disease)
        if test_profile is None:
            continue
        if test_profile in train_profile_to_diseases:
            profile_leaks.append((test_disease, train_profile_to_diseases[test_profile]))

    if profile_leaks:
        phenotype_leakage_found = True
        print(f"[{fold_dir.name}] PHENOTYPE PROFILE LEAKAGE: {len(profile_leaks)} / {len(test_diseases)} test disease(s) share identical phenotype profile with a train disease")
        if args.verbose:
            for test_d, train_ds in profile_leaks:
                test_profile = disease_to_phenotypes[test_d]
                print(f"  Test:  {test_d}  (profile size: {len(test_profile)})")
                for td in train_ds:
                    print(f"  Train: {td}")
    else:
        print(f"[{fold_dir.name}] Phenotype profile check: OK (no identical profiles between test and train)")

    # Check 3: identical phenotype profile AND identical gene associations
    train_genes_by_disease = (
        train_df.groupby("Disease")["Gene"].apply(frozenset).to_dict()
    )
    test_genes_by_disease = (
        test_df.groupby("Disease")["Gene"].apply(frozenset).to_dict()
    )

    train_profile_gene_to_diseases: dict[tuple, list[str]] = {}
    for d in train_diseases:
        profile = disease_to_phenotypes.get(d)
        genes = train_genes_by_disease.get(d)
        if profile is None or genes is None:
            continue
        train_profile_gene_to_diseases.setdefault((profile, genes), []).append(d)

    profile_gene_leaks = []
    for test_disease in test_diseases:
        test_profile = disease_to_phenotypes.get(test_disease)
        test_genes = test_genes_by_disease.get(test_disease)
        if test_profile is None or test_genes is None:
            continue
        key = (test_profile, test_genes)
        if key in train_profile_gene_to_diseases:
            profile_gene_leaks.append((test_disease, train_profile_gene_to_diseases[key]))

    if profile_gene_leaks:
        print(f"[{fold_dir.name}] PHENOTYPE+GENE LEAKAGE: {len(profile_gene_leaks)} test disease(s) share identical phenotype profile AND gene associations with a train disease")
        if args.verbose:
            for test_d, train_ds in profile_gene_leaks:
                print(f"  Test:  {test_d}")
                for td in train_ds:
                    print(f"  Train: {td}")
    else:
        print(f"[{fold_dir.name}] Phenotype+gene check: OK (no identical profile+gene pairs between test and train)")

    # Write test_no_leakage.csv: test rows whose disease has no phenotype+gene duplicate in train
    leaked_diseases = {test_d for test_d, _ in profile_gene_leaks}
    clean_test_df = test_df[~test_df["Disease"].isin(list(leaked_diseases))]
    out_path = fold_dir / "test_no_leakage.csv"
    clean_test_df.to_csv(out_path, index=False)
    print(f"[{fold_dir.name}] Wrote {out_path.name}: {len(clean_test_df)} rows ({len(test_df) - len(clean_test_df)} removed)")

print()
if not disease_leakage_found and not phenotype_leakage_found:
    print("All checks passed: no disease identity or phenotype profile leakage detected.")
else:
    if disease_leakage_found:
        print("WARNING: Disease identity leakage detected in one or more folds.")
    if phenotype_leakage_found:
        print("WARNING: Phenotype profile leakage detected in one or more folds.")
