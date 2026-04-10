#!/bin/bash --login
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=newfull_toy_one_model_video_roberta	# create a short name for your job

#SBATCH --output=one_model_video_roberta_toy_full.out
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=90G
#SBATCH --constraint=v100
#SBATCH --time=150:00:00
conda activate /ibex/scratch/projects/c2090/envs/conda-environment-examples/env



srun python pretrain_bert_full.py

