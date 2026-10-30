#!/bin/bash
#SBATCH --job-name=compile_thermal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/lustre/users/fngari/ls-temporal-gap-filling/logs/%x_%j.out
#SBATCH --error=/scratch/lustre/users/fngari/ls-temporal-gap-filling/logs/%x_%j.err

# --- Working Directory Setup ---
PROJECT_DIR=/scratch/lustre/users/fngari/ls-temporal-gap-filling
cd "$PROJECT_DIR"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

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

# --- Run Landsat Time-Series Dataset Compilation ---
python -u -m src.compile

echo "Job finished on: $(date)"