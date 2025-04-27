#!/bin/bash
#SBATCH --job-name=s3prl_run
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=11:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=46G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gretatu@mit.edu

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate speechglue

# Set PYTHONPATH
export PYTHONPATH=/om2/user/gretatu/script_repos/s3prl:$PYTHONPATH

cd /om2/user/gretatu/script_repos/s3prl/s3prl/

# Define variables
TASK=wnli
UPSTREAM=AuriStream100M_RoPE_librilight
OUTDIR=result/${TASK}/${UPSTREAM}

# Run the Python script
python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u ${UPSTREAM} \
  -d speechglue \
  -c downstream/speechglue/config_${TASK}.yaml \
  -a
