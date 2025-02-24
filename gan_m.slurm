#!/bin/bash
#SBATCH --job-name=gan_m_benchmark
#SBATCH --output=gan_output_m.txt
#SBATCH --error=gan_error_m.txt
#SBATCH --partition=ice-gpu   
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00   
#SBATCH --mem=16G         

module load anaconda3
source activate base

python gan_multi.py  # 读取环境变量