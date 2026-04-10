#!/bin/bash --login

#SBATCH --job-name=vis_sent_lab	  # create a short name for your job

#SBATCH --output=vis_sent_lab_txt.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=62GB	# memory/cpu (in MB)
#SBATCH --cpus-per-task=48	 # node count

#SBATCH --time=100:00:00
module purge
module load anaconda3
conda activate v-bert

srun python build_sents_vis_pair_data.py

