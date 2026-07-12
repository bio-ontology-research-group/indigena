#!/usr/bin/env python3
"""De-propagate a gene->phenotype annotation file to LEAF-ONLY (most-specific) terms.

Motivation
----------
HPOA `phenotype_to_genes` (the source of `gene_phenotypes_human.csv`) is
distributed under the **true-path rule**: every gene is annotated not only to its
most specific HP terms but to *all their ancestors* up to the root. In the human
Model-H build this inflates each gene to ~100 HP terms (86% of genes carry the
root `HP_0000001`), whereas the disease query side and the mouse (MGI/MP) gene
side are leaf-only. That asymmetry confounds the HP-ancestor abstraction
experiment: abstracting a query onto ancestor terms that nearly every gene
already carries destroys ranking discrimination for IC-blind embedding similarity.

This script removes the propagated ancestors, keeping for each gene only the
terms that are **not** a proper ancestor of another term the same gene has (i.e.
the leaves of the gene's induced subgraph). The per-edge provenance column
(`AttributedFromDiseases`) is preserved unchanged for the surviving edges, so the
downstream strict/light `--leak_filter` still works.

Ontology source
---------------
Ancestors are computed from `hp.obo` `is_a` edges (full transitive closure,
NOT depth-bounded), matching `precompute_hpo_ancestors.py`'s parsing. Obsolete
terms are dropped.

Usage
-----
    python scripts/depropagate_gene_phenotypes.py \
        --obo   data/hp.obo \
        --in    data/gene_phenotypes_human.csv \
        --out   data/gene_phenotypes_human_leafonly.csv

Input/Output CSV schema (header required):
    Gene,Phenotype,AttributedFromDiseases      # human (provenance kept)
    Gene,Phenotype                             # also accepted (e.g. mouse)

Deterministic: no randomness, stable row order (input order preserved).
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HP_PREFIX = "http://purl.obolibrary.org/obo/HP_"


def to_uri(hp_id: str) -> str:
    return HP_PREFIX + hp_id.split(":", 1)[1]


def parse_obo_parents(obo_path: Path) -> dict[str, set[str]]:
    """Parse hp.obo -> {term_uri: {parent_uri, ...}} over is_a, skipping obsolete."""
    parents: dict[str, set[str]] = {}
    cur = None
    obsolete = False
    with obo_path.open() as f:
        for raw in f:
            line = raw.rstrip()
            if line == "[Term]":
                cur = None
                obsolete = False
                continue
            if not line or ": " not in line:
                continue
            key, _, val = line.partition(": ")
            if key == "id" and val.startswith("HP:"):
                cur = to_uri(val)
                parents.setdefault(cur, set())
            elif key == "is_obsolete" and val.strip() == "true":
                obsolete = True
                if cur in parents:
                    del parents[cur]
                    cur = None
            elif key == "is_a" and cur and not obsolete:
                p = val.split(" ", 1)[0]
                if p.startswith("HP:"):
                    parents[cur].add(to_uri(p))
    return parents


def build_ancestor_fn(parents: dict[str, set[str]]):
    """Memoized full transitive ancestor closure, cycle-safe."""
    cache: dict[str, set[str]] = {}

    def ancestors(t: str) -> set[str]:
        if t in cache:
            return cache[t]
        cache[t] = set()  # cycle guard
        out: set[str] = set()
        for p in parents.get(t, ()):
            out.add(p)
            out |= ancestors(p)
        cache[t] = out
        return out

    return ancestors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obo", required=True, type=Path, help="Path to hp.obo")
    ap.add_argument("--in", dest="inp", required=True, type=Path,
                    help="Input gene->phenotype CSV (Gene,Phenotype[,AttributedFromDiseases])")
    ap.add_argument("--out", required=True, type=Path, help="Output leaf-only CSV")
    args = ap.parse_args()

    sys.setrecursionlimit(1_000_000)

    parents = parse_obo_parents(args.obo)
    ancestors = build_ancestor_fn(parents)

    # Load, grouping rows per gene while preserving input order.
    with args.inp.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        has_prov = len(header) >= 3
        pcol = 1
        gene_order: list[str] = []
        rows_by_gene: dict[str, list[list[str]]] = defaultdict(list)
        for row in reader:
            if not row:
                continue
            g = row[0]
            if g not in rows_by_gene:
                gene_order.append(g)
            rows_by_gene[g].append(row)

    n_in = n_out = 0
    before = after = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["Gene", "Phenotype", "AttributedFromDiseases"] if has_prov
                   else ["Gene", "Phenotype"])
        for g in gene_order:
            items = rows_by_gene[g]
            terms = {r[pcol] for r in items}
            anc_union: set[str] = set()
            for t in terms:
                anc_union |= ancestors(t)
            keep = terms - anc_union            # most-specific terms only
            before += len(terms)
            after += len(keep)
            for r in items:
                n_in += 1
                if r[pcol] in keep:
                    w.writerow(r if has_prov else r[:2])
                    n_out += 1

    ng = len(gene_order)
    sys.stderr.write(
        f"[depropagate] genes={ng}  edges_in={n_in}  leaf_only={n_out}  "
        f"dropped_ancestors={n_in - n_out} ({100*(n_in-n_out)/max(n_in,1):.1f}%)\n"
        f"[depropagate] mean terms/gene: before={before/max(ng,1):.1f} "
        f"after={after/max(ng,1):.1f}\n"
        f"[depropagate] wrote {args.out}\n"
    )


if __name__ == "__main__":
    main()
