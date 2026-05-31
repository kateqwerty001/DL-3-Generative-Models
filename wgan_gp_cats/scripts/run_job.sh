#!/bin/bash
#SBATCH -A stud-2526-l-03
#SBATCH -p student
#SBATCH --job-name=fid_all
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_log_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --mem=64G

source ~/project_3_run/.venv/bin/activate

python -u generate_cats_wgan.py