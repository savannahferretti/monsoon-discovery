#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=01:00:00
#SBATCH --job-name=timing_sr
#SBATCH --output=tests/timing/logs/%x_%j.log

set -eo pipefail

JULIA_DEPOT_CFS=/global/cfs/cdirs/m4334/sferrett/.julia
JULIA_DEPOT_SCRATCH=$SCRATCH/.julia
rsync -a --update $JULIA_DEPOT_CFS/ $JULIA_DEPOT_SCRATCH/

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_DEPOT_PATH=$JULIA_DEPOT_SCRATCH
export PYTHON_JULIAPKG_PROJECT=$JULIA_DEPOT_SCRATCH/environments/pyjuliapkg
export UCX_ERROR_SIGNALS=""

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

VARIANT=${1:?usage: sbatch tests/timing/sr_train.sbatch <variant> <run> [extra args]}
RUN=${2:?usage: sbatch tests/timing/sr_train.sbatch <variant> <run> [extra args]}
shift 2
scontrol update JobId=${SLURM_JOB_ID} Name=timing_${VARIANT}_${RUN}

python tests/timing/sr_train.py --variants ${VARIANT} --runs ${RUN} --procs 50 "$@"
