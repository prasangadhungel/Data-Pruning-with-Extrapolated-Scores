#!/bin/bash

#SBATCH -J "Prune"
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH -o "/nfs/homedirs/dhp/unsupervised-data-pruning/logs/slurm/%j-11-17-dual-beta-cifar.out"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:59:00

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate prune

python src/prune/dual.py