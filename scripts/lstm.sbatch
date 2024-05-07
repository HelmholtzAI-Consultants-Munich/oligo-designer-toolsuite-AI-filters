#!/bin/bash
### SLURM specific parameters, for e.g:

#SBATCH -o ".lstm.out"
#SBATCH -e ".lstm.err"
#SBATCH -J lstm_training
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH -t 04-00

## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
#SBATCH --mail-user=francesco.campi@helmholtz-munich.de
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:

CUDA_VISIBLE_DEVICES=0
python oligo_designer_toolsuite_ai_filters/hybridization_probability/train_model.py -c configs/hybridization_probability/train_LSTM.yaml  
