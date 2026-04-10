#!/bin/bash --login
##SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=align_s3d-test2      # create a short name for your job

#SBATCH --output=align.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=62GB	# memory/cpu (in MB)
#SBATCH --cpus-per-task=48	 # node count

#SBATCH --time=100:00:00
module purge
module load anaconda3
conda activate v-bert

srun python feat_alighment_task.py

