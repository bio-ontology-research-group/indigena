#!/usr/bin/env bash
# Reproducible MATCHED Resnik-BMA baseline (fold 0), aligned to leaf-only Model H.
#
# Scores the SAME candidate pool and test pairs INDIGENA evaluated (dumped to
# data/matched_fold0/ by kge_transd_species.py with DUMP_MATCHED_DIR), on the
# de-propagated LEAF-ONLY gene profiles, in TWO leak modes:
#   strict : drop gene->phenotype edges attributed to a fold-0 test disease (leak-free)
#   leaked : keep them (LEAK_FILTER=none) -> k=0 control baseline, matches Model H's setup
#
#   k=0    : raw leaf query          (data/disease_phenotypes.csv)
#   k=1..3 : HP-ancestor-abstracted  (data/disease_phenotypes_hp_k{k}.csv)
#
# Outputs: data/baseline_results/resnik_resnik_bma_fold0_matched_{strict,leaked}_k{k}_results.txt
set -o pipefail
cd ~/Git/indigena/hierarchy_test

export JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED"
G=~/.sdkman/candidates/groovy/current/bin/groovy
M=data/matched_fold0

export GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv
export CANDIDATES_FILE=$M/candidates_human_fold0.txt
export TEST_PAIRS_CSV=$M/pairs_human_fold0.csv

for LK in strict none; do
  tag=$([ "$LK" = none ] && echo leaked || echo strict)
  export LEAK_FILTER=$LK
  for k in 0 1 2 3; do
    if [ "$k" -eq 0 ]; then Q=data/disease_phenotypes.csv; else Q=data/disease_phenotypes_hp_k${k}.csv; fi
    echo "[resnik-matched] leak=$tag k=$k  $(date '+%H:%M:%S')"
    env PERTURB_DISEASE_CSV="$Q" PERTURB_TAG=matched_${tag}_k${k} \
      "$G" scripts/semantic_similarity_human.groovy -r data -ic resnik -pw resnik -gw bma -fold 0 \
      > logs/resnik_matched_${tag}_k${k}.log 2>&1
  done
done
echo "[resnik-matched] ALL DONE $(date '+%H:%M:%S')"
