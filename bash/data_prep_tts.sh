#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000063
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err

# >>> micromamba initialize >>>
export MAMBA_EXE='/projects/m000008/micromamba'
export MAMBA_ROOT_PREFIX='/projects/m000008/micromamba-packages'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup
# <<< micromamba initialize <<<

# Activate environment
micromamba activate speechglue

# Change to your project directory
cd /scratch/m000063/gt/s3prl/s3prl

# Run the data prep script
python downstream/speechglue/data_prep.py --glue-task "$1"