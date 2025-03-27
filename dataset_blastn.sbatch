#!/bin/bash
### SLURM specific parameters, for e.g:

##SBATCH -o ".dataset.out"
##SBATCH -e ".dataset_real.err"
#SBATCH -J dataset_generation
#SBATCH -p cpu_p
#SBATCH -q cpu_normal
#SBATCH -c 10
#SBATCH --mem=300G
#SBATCH -t 03-00

## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:

scource conda activate pombo

srun python oligo_designer_toolsuite_ai_filters/hybridization_probability/generate_real_dataset.py -c configs/hybridization_probability/generate_real_dataset_blastn.yaml

### Exit
exit 0
