#!/bin/bash --login
##SBATCH -A conf-gpu-2020.11.23
#SBATCH --job-name=one_model_video_roberta      # create a short name for your job

#SBATCH --output=one_model_video_roberta.out
#SBATCH --nodes=2
#SBATCH --ntasks=2


#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=90G

#SBATCH --time=180:00:00
module purge
module load anaconda3
conda activate  /ibex/scratch/projects/c2090/envs/conda-environment-examples/env
srun python /ibex/scratch/x_hammadm/d3dhelper/MIL-NCE_HowTo100M/count_gpus.py
srun python -m torch.distributed.launch \
    --nproc_per_node 8 roberta_custom_trainer_one_model.py


