#!/usr/bin/env bash
# INDIGENA eval-only under abstraction k=1,2,3, fold 0 (k=0 = training auto-eval). GPU.
set -o pipefail; cd ~/Git/indigena/hierarchy_test
PY=~/miniforge3/envs/multihopgda/bin/python
ARGS="--species human --leak_filter strict --fold 0 --mode inductive --graph4 --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep --only_test"
for k in 1 2 3; do
  echo "[indigena eval k=$k]"
  env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline EVAL_DISEASE_PHENO_CSV=data/disease_phenotypes_hp_k${k}.csv EVAL_TAG=hp_k${k} \
    $PY scripts/kge_transd_species.py $ARGS > logs/eval_indigena_k${k}.log 2>&1
done
echo "indigena eval k1-3 done"
