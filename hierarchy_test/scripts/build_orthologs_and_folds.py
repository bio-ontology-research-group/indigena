"""
Day-1 setup script for two-model INDIGENA fusion.

Outputs (all under data/):
    1. hcop_human_mouse.tsv         — 4-col table: human_ncbi, human_symbol, mgi_id, mouse_symbol
                                       parsed from HMD_HumanPhenotype.rpt
    2. gene_diseases_human.csv      — human gene->disease pairs from HPOA, restricted to
                                       diseases in INDIGENA's training disease set (OMIM only)
                                       columns: Gene, Disease (URI form, like mouse file)
    3. gene_disease_folds_unified/  — disease-disjoint 10-fold partition over UNION of
                                       (mouse causal disease set ∪ human causal disease set)
                                       per fold writes:
                                         fold_N/test_diseases.txt  (one OMIM URI per line)
                                         fold_N/train_diseases.txt
                                         fold_N/mouse_train.csv    (Gene,Disease subset)
                                         fold_N/mouse_test.csv
                                         fold_N/human_train.csv
                                         fold_N/human_test.csv

Design:
- Mouse fold partition is preserved as the seed: every disease that was in mouse fold N
  stays in fold N (so existing mouse-only baselines remain reproducible bit-for-bit
  by reading {fold_N/mouse_train.csv, fold_N/mouse_test.csv}).
- Human-only diseases (in HPOA but not in mouse gene_diseases.csv) are distributed by
  hashing OMIM ID stably into one of the 10 folds — disjoint, deterministic.
- Per-fold human pair lists derive from gene_diseases_human.csv by disease membership.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
HMD_FILE = DATA / "HMD_HumanPhenotype.rpt"
HPOA_FILE = DATA / "phenotype_to_genes.txt"
DISEASE_PHENOS_FILE = DATA / "disease_phenotypes.csv"
MOUSE_GD_FILE = DATA / "gene_diseases.csv"
MOUSE_FOLDS_DIR = DATA / "gene_disease_folds"

OUT_HCOP = DATA / "hcop_human_mouse.tsv"
OUT_HUMAN_GD = DATA / "gene_diseases_human.csv"
OUT_FOLDS = DATA / "gene_disease_folds_unified"

OMIM_PREFIX = "http://mowl.borg/OMIM_"
NCBI_PREFIX = "http://mowl.borg/NCBIGene_"


def step1_orthologs() -> None:
    """Parse HMD_HumanPhenotype.rpt into a clean human↔mouse table.

    Format: Human_Symbol \t Human_NCBI_ID \t Mouse_Symbol \t MGI_ID \t MP_terms \t (empty)
    Rows where Human_NCBI is blank or MGI is blank are dropped (no usable mapping).
    """
    n_in = n_out = 0
    with HMD_FILE.open() as f, OUT_HCOP.open("w") as out:
        out.write("human_ncbi\thuman_symbol\tmgi_id\tmouse_symbol\n")
        for line in f:
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            human_symbol, human_ncbi, mouse_symbol, mgi_id = parts[0], parts[1], parts[2], parts[3]
            if not human_ncbi or not mgi_id:
                continue
            out.write(f"{human_ncbi}\t{human_symbol}\t{mgi_id}\t{mouse_symbol}\n")
            n_out += 1
    print(f"[orthologs] in={n_in} out={n_out} -> {OUT_HCOP}")


def step2_human_gene_diseases() -> set[str]:
    """Build human gene->disease pair file from HPOA, restricted to OMIM diseases
    in INDIGENA's training disease set.

    Returns the set of OMIM URIs that have at least one human gene attribution
    (used in step 3 to expand the disease pool beyond mouse-only).
    """
    indigena_diseases: set[str] = set()
    with DISEASE_PHENOS_FILE.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            indigena_diseases.add(row[0])
    print(f"[human_gd] INDIGENA disease set: {len(indigena_diseases)}")

    pairs: set[tuple[str, str]] = set()
    n_in = n_drop_non_omim = n_drop_unknown = 0
    with HPOA_FILE.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n_in += 1
            d = row["disease_id"]
            if not d.startswith("OMIM:"):
                n_drop_non_omim += 1; continue
            disease_uri = OMIM_PREFIX + d[5:]
            if disease_uri not in indigena_diseases:
                n_drop_unknown += 1; continue
            ncbi = row["ncbi_gene_id"]
            if not ncbi:
                continue
            pairs.add((NCBI_PREFIX + ncbi, disease_uri))

    with OUT_HUMAN_GD.open("w") as out:
        w = csv.writer(out)
        w.writerow(["Gene", "Disease"])
        for g, d in sorted(pairs):
            w.writerow([g, d])

    diseases_with_human_gene = {d for _, d in pairs}
    genes = {g for g, _ in pairs}
    print(f"[human_gd] HPOA rows={n_in} dropped(non-OMIM)={n_drop_non_omim} "
          f"dropped(disease unknown to INDIGENA)={n_drop_unknown}")
    print(f"[human_gd] pairs={len(pairs)} unique_genes={len(genes)} "
          f"unique_diseases={len(diseases_with_human_gene)} -> {OUT_HUMAN_GD}")
    return diseases_with_human_gene


def step3_unified_folds(diseases_with_human_gene: set[str]) -> None:
    """Build unified disease-disjoint 10-fold partition.

    1. Read existing mouse fold assignments (disease -> fold).
    2. For diseases only in human side, assign by stable hash to 0..9.
    3. Per fold, write disease lists + per-species (gene, disease) train/test CSVs.
    """
    # 1. Read mouse disease->fold map from existing folds
    disease_fold: dict[str, int] = {}
    for fold in range(10):
        with (MOUSE_FOLDS_DIR / f"fold_{fold}" / "test.csv").open() as f:
            r = csv.reader(f); next(r)
            for row in r:
                disease_fold[row[1]] = fold
    print(f"[folds] mouse-assigned diseases: {len(disease_fold)}")

    # 2. Hash-assign human-only diseases
    human_only = diseases_with_human_gene - disease_fold.keys()
    for d in human_only:
        h = int(hashlib.sha1(d.encode()).hexdigest()[:8], 16)
        disease_fold[d] = h % 10
    print(f"[folds] +human-only diseases: {len(human_only)} (total={len(disease_fold)})")

    # Sanity: per-fold disease counts
    fold_counts = defaultdict(int)
    for d, fk in disease_fold.items():
        fold_counts[fk] += 1
    print(f"[folds] per-fold disease totals: " +
          ", ".join(f"f{i}={fold_counts[i]}" for i in range(10)))

    # 3. Read mouse + human gene-disease pair files
    def read_pairs(path: Path) -> list[tuple[str, str]]:
        out = []
        with path.open() as f:
            r = csv.reader(f); next(r)
            for row in r:
                out.append((row[0], row[1]))
        return out
    mouse_pairs = read_pairs(MOUSE_GD_FILE)
    human_pairs = read_pairs(OUT_HUMAN_GD)
    print(f"[folds] mouse pairs={len(mouse_pairs)} human pairs={len(human_pairs)}")

    OUT_FOLDS.mkdir(parents=True, exist_ok=True)
    for fold in range(10):
        fdir = OUT_FOLDS / f"fold_{fold}"
        fdir.mkdir(exist_ok=True)
        test_diseases = {d for d, fk in disease_fold.items() if fk == fold}
        train_diseases = {d for d, fk in disease_fold.items() if fk != fold}
        with (fdir / "test_diseases.txt").open("w") as f:
            for d in sorted(test_diseases): f.write(d + "\n")
        with (fdir / "train_diseases.txt").open("w") as f:
            for d in sorted(train_diseases): f.write(d + "\n")

        for tag, pairs in [("mouse", mouse_pairs), ("human", human_pairs)]:
            tr = [(g, d) for g, d in pairs if d in train_diseases]
            te = [(g, d) for g, d in pairs if d in test_diseases]
            for split, rows in [("train", tr), ("test", te)]:
                with (fdir / f"{tag}_{split}.csv").open("w") as f:
                    w = csv.writer(f); w.writerow(["Gene", "Disease"])
                    for g, d in rows: w.writerow([g, d])
        # Per-fold counts
        m_tr = sum(1 for _, d in mouse_pairs if d in train_diseases)
        m_te = sum(1 for _, d in mouse_pairs if d in test_diseases)
        h_tr = sum(1 for _, d in human_pairs if d in train_diseases)
        h_te = sum(1 for _, d in human_pairs if d in test_diseases)
        print(f"[folds] fold_{fold}: mouse train/test={m_tr}/{m_te}  human train/test={h_tr}/{h_te}")


def main() -> None:
    step1_orthologs()
    diseases = step2_human_gene_diseases()
    step3_unified_folds(diseases)
    print("[done] day-1 ortholog + folds setup complete")


if __name__ == "__main__":
    main()
