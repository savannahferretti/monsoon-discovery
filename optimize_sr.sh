#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --job-name=optimize_sr
#SBATCH --output=logs/%x_%j.log

set -eo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

# EQ=${1:-all}
# SPLITS=${2:-valid}

# scontrol update JobId=${SLURM_JOB_ID} Name=${EQ}
# echo "Optimizing equations: ${EQ}  |  splits: ${SPLITS}  |  workers: ${SLURM_CPUS_PER_TASK}"

python -m scripts.models.sr.optimize --equations all --splits valid