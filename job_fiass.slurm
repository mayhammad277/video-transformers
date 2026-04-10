#!/bin/bash --login
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=faiss_cluster-test2      # create a short name for your job

#SBATCH --output=faiss_cl.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem-per-gpu=90GB	# memory/cpu (in MB)
#SBATCH --gpus-per-node=8        # node count
#SBATCH --cpus-per-gpu=6
#SBATCH --time=300:00:00
module purge
module load anaconda3
conda activate v-bert

srun python faiss_clustering.py

