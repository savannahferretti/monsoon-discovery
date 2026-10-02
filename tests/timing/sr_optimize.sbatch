#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=timing_opt
#SBATCH --output=tests/timing/logs/%x_%j.log

set -eo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

VARIANT=${1:?usage: sbatch tests/timing/sr_optimize.sbatch <variant> <equations>}
EQS=${2:?usage: sbatch tests/timing/sr_optimize.sbatch <variant> <equations>}
scontrol update JobId=${SLURM_JOB_ID} Name=timing_opt_${VARIANT}

python tests/timing/sr_optimize.py --variants ${VARIANT} --equations ${EQS} --splits test
