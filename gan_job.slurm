#!/bin/bash
#SBATCH --job-name=gan_benchmark
#SBATCH --output=gan_output.txt
#SBATCH --error=gan_error.txt
#SBATCH --partition=ice-gpu   
#SBATCH --gres=gpu:1      
#SBATCH --time=02:00:00   
#SBATCH --mem=16G         

module load anaconda3
source activate base

python gan.py --device ${DEVICE}  # 读取环境变量
