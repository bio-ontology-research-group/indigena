"""Precompute HPO ancestor closure (depth-bounded) for use during INDIGENA training augmentation.

Reads hp.obo, walks `is_a` edges upward, and writes a JSON mapping
each HP URI to its ancestor URIs at each depth level.

Output schema:
    {
        "depth_max": int,
        "ancestors_by_depth": {
            "<hp_uri>": [
                ["<parent_uri>", ...],          # depth 1
                ["<grandparent_uri>", ...],     # depth 2
                ...
            ]
        }
    }

So `ancestors_by_depth[term][0]` is depth-1 ancestors (immediate parents),
`[1]` is depth-2, etc. Lists may be empty if walked past root.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

HP_PREFIX = "http://purl.obolibrary.org/obo/HP_"


def parse_obo(obo_path: Path) -> dict[str, set[str]]:
    """Returns parents map: {HP:NNN: {HP:parent1, HP:parent2, ...}}."""
    parents: dict[str, set[str]] = {}
    cur = None; obsolete = False
    with obo_path.open() as f:
        for raw in f:
            line = raw.rstrip()
            if line == "[Term]":
                cur = None; obsolete = False; continue
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(": ")
            if key == "id" and val.startswith("HP:"):
                cur = val; parents.setdefault(cur, set())
            elif key == "is_obsolete" and val.strip() == "true":
                obsolete = True
                if cur and cur in parents:
                    del parents[cur]
            elif key == "is_a" and cur and not obsolete:
                p = val.split(" ", 1)[0]
                if p.startswith("HP:"):
                    parents[cur].add(p)
    return parents


def parents_at_depth(parents: dict[str, set[str]], term: str, depth: int) -> set[str]:
    """All distinct ancestors at exact depth N (BFS over DAG)."""
    if depth <= 0:
        return set()
    frontier = parents.get(term, set())
    for _ in range(depth - 1):
        next_frontier: set[str] = set()
        for t in frontier:
            next_frontier.update(parents.get(t, ()))
        frontier = next_frontier
        if not frontier:
            break
    return frontier


def to_uri(hp_id: str) -> str:
    return HP_PREFIX + hp_id.split(":", 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--depth-max", type=int, default=3)
    args = ap.parse_args()

    parents = parse_obo(args.obo)
    print(f"[precompute] parsed {len(parents)} HP terms")

    out = {"depth_max": args.depth_max, "ancestors_by_depth": {}}
    n_with_anc = n_total_anc = 0
    for term in parents:
        per_depth = []
        for d in range(1, args.depth_max + 1):
            ancs = parents_at_depth(parents, term, d)
            per_depth.append(sorted(to_uri(a) for a in ancs))
        out["ancestors_by_depth"][to_uri(term)] = per_depth
        if any(per_depth):
            n_with_anc += 1
            n_total_anc += sum(len(d) for d in per_depth)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f)
    print(f"[precompute] {n_with_anc}/{len(parents)} terms have ≥1 ancestor "
          f"(total {n_total_anc} ancestor edges across depths 1..{args.depth_max})")
    print(f"[precompute] wrote {args.out}")


if __name__ == "__main__":
    main()
