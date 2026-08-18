#!/bin/bash
#SBATCH --job-name=rbfn_gapfill
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300000
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/rbfn_%j.out
#SBATCH --error=logs/rbfn_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1

module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lustre/users/$USER/envs/gapfill_env

MY_ENV_PACKAGES="/scratch/lustre/users/$USER/envs/gapfill_env/lib/python3.12/site-packages"
export PYTHONPATH="${MY_ENV_PACKAGES}:/scratch/lustre/users/$USER/ls-temporal-gap-filling:${PYTHONPATH}"
export LD_LIBRARY_PATH="${MY_ENV_PACKAGES}/torch/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

cd /scratch/lustre/users/$USER/ls-temporal-gap-filling
mkdir -p logs

echo "Job started on: $(date)"
echo "Using python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

python -m src.train

echo "Job finished on: $(date)"