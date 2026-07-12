#!/usr/bin/env bash
# Usage: run_resnik_matched.sh <fold>
# Matched Resnik-BMA baseline for <fold>: leaf-only profiles, INDIGENA's exact
# candidate pool + test pairs (data/matched_fold<fold>/), both leak modes, k=0..3.
set -o pipefail
cd ~/Git/indigena/hierarchy_test 2>/dev/null || cd "$(dirname "$0")"
FOLD=${1:?fold required}
export JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED"
G=${GROOVY:-groovy}
M=data/matched_fold${FOLD}
export GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv
export CANDIDATES_FILE=$M/candidates_human_fold${FOLD}.txt
export TEST_PAIRS_CSV=$M/pairs_human_fold${FOLD}.csv

for LK in strict none; do
  tag=$([ "$LK" = none ] && echo leaked || echo strict)
  export LEAK_FILTER=$LK
  for k in 0 1 2 3; do
    if [ "$k" -eq 0 ]; then Q=data/disease_phenotypes.csv; else Q=data/disease_phenotypes_hp_k${k}.csv; fi
    echo "[resnik fold$FOLD] leak=$tag k=$k $(date '+%H:%M:%S')"
    env PERTURB_DISEASE_CSV="$Q" PERTURB_TAG=matched_${tag}_k${k} \
      "$G" scripts/semantic_similarity_human.groovy -r data -ic resnik -pw resnik -gw bma -fold ${FOLD} \
      > logs/resnik_matched_fold${FOLD}_${tag}_k${k}.log 2>&1
  done
done
echo "[resnik fold$FOLD] done $(date '+%H:%M:%S')"
