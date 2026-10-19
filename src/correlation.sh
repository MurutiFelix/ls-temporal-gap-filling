#!/bin/bash
#SBATCH --job-name=rbfn_corr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300000
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rbfn_corr_%j.out
#SBATCH --error=logs/rbfn_corr_%j.err
# Ensure output logs directory exists
mkdir -p logs

echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"

# Activate Python environment
ENV_PATH="/scratch/lustre/users/fngari/envs/rbfn_env/bin/activate"
if [ -f "$ENV_PATH" ]; then
    source "$ENV_PATH"
    echo "Using python: $(which python)"
else
    echo "Error: Virtual environment not found at $ENV_PATH"
    exit 1
fi

# Run NDVI correlation & SSIM evaluation
python src/evaluate_ndvi_correlation.py

echo "Job finished on: $(date)"