#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH --job-name=fid_all
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_log_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --mem=64G

source ~/DL-3-Generative-Models/cats_env/bin/activate

python -u generate_cats_gpt.py