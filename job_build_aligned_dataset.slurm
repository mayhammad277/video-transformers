#!/bin/bash --login
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=align_v_t      # create a short name for your job

#SBATCH --output=align_visual_txt.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=55GB	# memory/cpu (in MB)
#SBATCH --cpus-per-task=35	 # node count

#SBATCH --time=300:00:00
module purge
module load anaconda3
conda activate v-bert

srun python align_visual_text.py

