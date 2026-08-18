# Redirect pip cache off home quota, onto scratch
mkdir -p /scratch/lustre/users/fngari/.cache/pip
mkdir -p /scratch/lustre/users/fngari/.tmp
export PIP_CACHE_DIR=/scratch/lustre/users/fngari/.cache/pip
export TMPDIR=/scratch/lustre/users/fngari/.tmp
rm -rf ~/.cache/pip

# Make this permanent for future sessions
echo 'export PIP_CACHE_DIR=/scratch/lustre/users/$USER/.cache/pip' >> ~/.bashrc
echo 'export TMPDIR=/scratch/lustre/users/$USER/.tmp' >> ~/.bashrc

# Load conda module
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh

# Create the env inside your existing envs/ folder
conda create --prefix /scratch/lustre/users/fngari/envs/rbfn_env python=3.12 -y
conda activate /scratch/lustre/users/fngari/envs/rbfn_env

# Install torch matching cluster CUDA
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of requirements.txt
pip install --no-cache-dir -r /scratch/lustre/users/fngari/ls-temporal-gap-filling/requirements.txt