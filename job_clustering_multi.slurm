#!/bin/bash --login
#SBATCH --job-name=M_cluster-s3dc2_dear	   # create a short name for your job
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --output=cluster.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=62GB	# memory/cpu (in MB)
#SBATCH --cpus-per-task=48	 # node count

#SBATCH --time=300:00:00
module purge
module load anaconda3

conda activate /ibex/scratch/projects/c2090/envs/env

srun python multi_process_kmeans.py
