#!/bin/bash
#SBATCH --job-name=rbfn_run1
#SBATCH --partition=gpu1          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rbfn_%j.out
#SBATCH --error=logs/rbfn_%j.err

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ls-temporal-gap-filling
mkdir -p logs

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ls-temporal-gap-filling
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"

# --- Environment Setup ---
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lustre/users/$USER/envs/rbfn_env

# --- Diagnostics ---
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"
echo "Using python: $(which python)"

# --- Run Training ---
python -m src.train

echo "Job finished on: $(date)"