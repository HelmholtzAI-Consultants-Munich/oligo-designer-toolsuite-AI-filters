#!/bin/bash
### SLURM specific parameters, for e.g:

#SBATCH -o "/dev/null"
#SBATCH -e ".mlp.err"
#SBATCH -J mlp_training
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:2
#SBATCH -c 2
#SBATCH --mem=10G
#SBATCH -t 01-00


## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:


CUDA_VISIBLE_DEVICES=0,1
python oligo_designer_toolsuite_ai_filters/hybridization_probability/train_model.py -c configs/hybridization_probability/train_MLP.yaml  
