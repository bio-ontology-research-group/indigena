#!/bin/bash
# SLURM array sweep: matched leaf-only INDIGENA-vs-Resnik over the 10 disease-disjoint
# folds. One GPU task per fold: train (leaf-only, leak-strict) -> INDIGENA eval grid
# (leaked+strict x k=0..3) -> Resnik matched grid. Uses the 'indiga' conda env + sdkman groovy.
# Submit from hierarchy_test/ AFTER build_hierarchy_test_data.sh has produced data/.
#   sbatch slurm_sweep.sh
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J hdig-sweep
#SBATCH --array=0-9
#SBATCH -o logs/hdig.%A_%a.out
#SBATCH -e logs/hdig.%A_%a.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=05:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100|a100]
#SBATCH --cpus-per-task=8
set -o pipefail
HT=/ibex/user/zhapacfp/indigena/hierarchy_test
cd "$HT"
mkdir -p logs data/results data/models data/baseline_results
FOLD=${SLURM_ARRAY_TASK_ID}

# --- environment: indiga conda env + groovy/java (SLIB Resnik) ---
source /home/zhapacfp/miniforge3/etc/profile.d/conda.sh
conda activate indiga
export PY=$(command -v python)
export WANDB_MODE=offline
export JAVA_HOME=/usr/lib/jvm/java-11
export PATH=$JAVA_HOME/bin:$PATH
export GROOVY=/home/zhapacfp/.sdkman/candidates/groovy/current/bin/groovy

echo "[fold $FOLD] node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES py=$PY groovy=$GROOVY $(date)"
$PY -c "import torch;print('[torch] cuda', torch.cuda.is_available())"

echo "[fold $FOLD] TRAIN leaf-only leak-strict $(date)"
env GENE_PHENO_CSV=data/gene_phenotypes_human_leafonly.csv RUN_TAG=_leafonly DUMP_MATCHED_DIR=data/matched_fold${FOLD} \
  $PY scripts/kge_transd_species.py --species human --leak_filter strict --fold ${FOLD} --mode inductive --graph4 \
      --embedding_dim 400 --batch_size 8192 --learning_rate 0.001 --no_sweep || { echo "TRAIN FAILED fold $FOLD"; exit 1; }

echo "[fold $FOLD] INDIGENA eval grid (leaked+strict x k0-3) $(date)"
PY=$PY bash run_indigena_matched_evals.sh ${FOLD}          # no explicit GPU -> inherit SLURM's

# Resnik (CPU-only) runs as a separate per-fold job: slurm_resnik.sh (aftercorr dependency).
echo "[fold $FOLD] GPU DONE (train+eval) $(date)"
