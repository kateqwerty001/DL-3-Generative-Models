#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH --job-name=stylegan2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_log_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --mem=64G

source ~/DL-3-Generative-Models/cats_env/bin/activate

rm -rf ~/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/DL-3-Generative-Models/stylegan2-ada-pytorch

python train.py \
  --outdir=../cats-gen/stylegan_runs \
  --data=../cats-gen/data_stylegan/cats128.zip \
  --cfg=auto \
  --mirror=1 \
  --aug=noaug \
  --gamma=50 \
  --kimg=1000 \
  --fp32=1 \
  --metrics=none \
  --workers=1