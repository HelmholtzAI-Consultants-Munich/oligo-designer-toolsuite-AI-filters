#!/bin/bash
### SLURM specific parameters, for e.g:

##SBATCH -o "/dev/null"
##SBATCH -e ".dataset.err"
#SBATCH -J dataset_generation
#SBATCH -p cpu_p
#SBATCH -q cpu_normal
#SBATCH -c 30
#SBATCH --mem=150G
#SBATCH -t 02-00

## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:

python odt_ai_filters/_hybridization_probability/generate_artificial_dataset.py -c configs/hybridization_probability/generate_artificial_dataset.yaml

### Exit
exit 0
