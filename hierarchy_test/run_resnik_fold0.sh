#!/usr/bin/env bash
# Resnik-BMA (Groovy/SLIB), human genes, abstraction k=1,2,3, fold 0, parallel. (k=0 run separately.)
set -o pipefail; cd ~/Git/indigena/hierarchy_test
export JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED"
G=~/.sdkman/candidates/groovy/current/bin/groovy
for k in 1 2 3; do
  env PERTURB_DISEASE_CSV=data/disease_phenotypes_hp_k${k}.csv PERTURB_TAG=hp_k${k} \
    $G scripts/semantic_similarity_human.groovy -r data -ic resnik -pw resnik -gw bma -fold 0 \
    > logs/resnik_k${k}.log 2>&1 &
done
wait; echo "resnik k1-3 done"
