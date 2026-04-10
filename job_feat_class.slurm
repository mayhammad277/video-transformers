#!/bin/bash --login
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=s3d_classify      # create a short name for your job

#SBATCH --output=classify.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-gpu=90GB	# memory/cpu (in MB)
#SBATCH --gpus-per-node=6	 # node count
#SBATCH --cpus-per-gpu=4 
#SBATCH --time=60:00:00
module purge
module load anaconda3
conda activate v-bert

srun python classify_feats.py
