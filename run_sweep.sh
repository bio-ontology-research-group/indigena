#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J indigena
#SBATCH -o out/indigena.%J.out
#SBATCH -e err/indigena.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=0:30:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]


wandb agent --count 1 ferzcam/indigena/$1
