#!/usr/bin/env bash
# Evaluate the trained LEAF-ONLY, leak-strict INDIGENA model (fold 0) under
# HP-ancestor abstraction k=0..3, in TWO scoring modes (reuses the saved model
# via --only_test; no retraining). Matched to Resnik on the same pool/pairs.
#   leaked  (SCORING_LEAK_FILTER=none)   : full-profile scoring; k=0 is the control baseline
#   strict  (SCORING_LEAK_FILTER=strict) : leak-free scoring; supplementary "floor"
# The training graph is always leak-strict; only the BMA scoring set changes.
set -o pipefail
cd ~/Git/indigena/hierarchy_test
PY=~/miniforge3/envs/multihopgda/bin/python
COMMON="--species human --leak_filter strict --fold 0 --mode inductive --graph4 --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep --only_test"
export GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv
export RUN_TAG=_leafonly WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0

for mode in none strict; do
  tag=$([ "$mode" = none ] && echo leaked || echo strict)
  for k in 0 1 2 3; do
    if [ "$k" -eq 0 ]; then Q=data/disease_phenotypes.csv; else Q=data/disease_phenotypes_hp_k${k}.csv; fi
    echo "[indigena] scoring=$tag k=$k  $(date '+%H:%M:%S')"
    env SCORING_LEAK_FILTER=$mode EVAL_DISEASE_PHENO_CSV="$Q" EVAL_TAG=${tag}_k${k} \
      $PY scripts/kge_transd_species.py $COMMON > logs/eval_indigena_${tag}_k${k}.log 2>&1
  done
done
echo "[indigena] ALL EVALS DONE $(date '+%H:%M:%S')"
