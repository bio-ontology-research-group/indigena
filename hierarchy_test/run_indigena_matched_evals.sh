#!/usr/bin/env bash
# Usage: run_indigena_matched_evals.sh <fold> [gpu]
# Evaluate the trained LEAF-ONLY leak-strict INDIGENA model for <fold> under
# HP-ancestor abstraction k=0..3, in two scoring modes (leaked / strict).
# Reuses the saved model via --only_test (no retraining).
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")"      # hierarchy_test/ (portable: workstation or Ibex)
FOLD=${1:?fold required}
GPU=${2:-}                                  # explicit device on multi-GPU hosts; leave empty under SLURM
PY=${PY:-$(which python)}
COMMON="--species human --leak_filter strict --fold ${FOLD} --mode inductive --graph4 --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep --only_test"
export GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv
export RUN_TAG=_leafonly WANDB_MODE=offline
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU   # else inherit SLURM's allocated GPU

for mode in none strict; do
  tag=$([ "$mode" = none ] && echo leaked || echo strict)
  for k in 0 1 2 3; do
    if [ "$k" -eq 0 ]; then Q=data/disease_phenotypes.csv; else Q=data/disease_phenotypes_hp_k${k}.csv; fi
    echo "[indigena fold$FOLD] scoring=$tag k=$k $(date '+%H:%M:%S')"
    env SCORING_LEAK_FILTER=$mode EVAL_DISEASE_PHENO_CSV="$Q" EVAL_TAG=${tag}_k${k} \
      $PY scripts/kge_transd_species.py $COMMON > logs/eval_indigena_fold${FOLD}_${tag}_k${k}.log 2>&1
  done
done
echo "[indigena fold$FOLD] evals done $(date '+%H:%M:%S')"
