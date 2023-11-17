#!/bin/bash
### SLURM specific parameters, for e.g:

#SBATCH -o ".rnn-out"
#SBATCH -e ".rnn.err"
#SBATCH -J rnn_training
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH -t 02-00

## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:

python oligo_seq/pipelines/train_model.py -c config/train_RNN.yaml  
