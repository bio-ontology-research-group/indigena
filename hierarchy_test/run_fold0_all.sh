#!/usr/bin/env bash
# Master orchestrator: Resnik k1-3 (parallel) + wait for training + INDIGENA eval k1-3 + aggregate.
set -o pipefail; cd ~/Git/indigena/hierarchy_test
PY=~/miniforge3/envs/multihopgda/bin/python
export JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED"
G=~/.sdkman/candidates/groovy/current/bin/groovy
ts(){ date '+%H:%M:%S'; }
K0=data/results/kge_results_transd_human_inductive_fold_0_seed_0_dim_400_bs_8192_lr_0.001_graph4_leak-strict_inductive_bma.tsv
echo "[$(ts)] launch Resnik k1-3 (Groovy/SLIB, human genes)"
for k in 1 2 3; do
  env PERTURB_DISEASE_CSV=data/disease_phenotypes_hp_k${k}.csv PERTURB_TAG=hp_k${k} \
    $G scripts/semantic_similarity_human.groovy -r data -ic resnik -pw resnik -gw bma -fold 0 > logs/resnik_k${k}.log 2>&1 &
done
echo "[$(ts)] waiting for INDIGENA training (k0 baseline eval -> $K0)"
while [ ! -f "$K0" ]; do sleep 60; done
echo "[$(ts)] training+k0 eval done; INDIGENA eval k1-3"
for k in 1 2 3; do
  echo "[$(ts)]   indigena eval k=$k"
  env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline EVAL_DISEASE_PHENO_CSV=data/disease_phenotypes_hp_k${k}.csv EVAL_TAG=hp_k${k} \
    $PY scripts/kge_transd_species.py --species human --leak_filter strict --fold 0 --mode inductive --graph4 \
    --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep --only_test > logs/eval_indigena_k${k}.log 2>&1
done
echo "[$(ts)] waiting for Resnik k1-3 to finish"; wait
echo "[$(ts)] aggregating"
$PY scripts/aggregate_hierarchy.py | tee results/degradation_fold0.txt
echo "[$(ts)] ===== FOLD0 PIPELINE DONE ====="
