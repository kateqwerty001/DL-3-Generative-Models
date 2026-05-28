#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH --job-name=cats
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_log_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --mem=64G

source ~/cats_env/bin/activate

rm -rf ~/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONWARNINGS="ignore"

python3 ~/DL-3-Generative-Models/stylegan_cats/scripts/generate_cats_stylegan.py