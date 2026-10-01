#!/bin/bash
#SBATCH --job-name=rbfn_predict
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rbfn_predict_%j.out
#SBATCH --error=logs/rbfn_predict_%j.err

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ls-temporal-gap-filling
mkdir -p logs
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

# --- Run Spatial Map Inference ---
# Specify target test months/scenes via CLI flags depending on your predict.py signature
python -u -m src.predict \
    --model-path data/processed/models/rbfn_landsat_gap_filler.pt \
    --scaler-path data/processed/models/rbfn_scalers.joblib \
    --output-dir data/processed/predictions

echo "Job finished on: $(date)"