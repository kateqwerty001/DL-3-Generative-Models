#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH --job-name=export
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_log_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --mem=64G

source ~/cats_env/bin/activate

python3 ~/DL-3-Generative-Models/stylegan_cats/scripts/export_for_stylegan.py